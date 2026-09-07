"""Anthropic SSE emitter — total state machine.

The emitter owns the content_block index and the open-block set. Callers CANNOT
produce a malformed lifecycle because block open/close is not an input — it is
derived from SinkEvent kinds:

  ThinkingDelta → open('thinking') if closed, delta
  TextDelta     → close thinking block if open; open('text') if closed; delta
  ToolCallStart → close any open block; open('tool_use', index)
  ToolCallArgsDelta → delta on tool_use block
  ToolCallEnd   → close tool_use block
  MessageEnd    → close everything, message_delta(stop_reason), message_stop
  SinkError     → close everything, emit error frame
"""
from __future__ import annotations

import json
from typing import Optional, Iterator
from enum import Enum

from v2.ir.events import (
    SinkEvent,
    MessageStart,
    ThinkingDelta,
    TextDelta,
    ToolCallStart,
    ToolCallArgsDelta,
    ToolCallEnd,
    MessageEnd,
    SinkError,
    StopReason,
    is_thinking,
    is_text,
    is_tool_start,
    is_tool_args,
    is_tool_end,
    is_message_end,
    is_error,
)


class BlockType(Enum):
    THINKING = "thinking"
    TEXT = "text"
    TOOL_USE = "tool_use"
    REDACTED_THINKING = "redacted_thinking"


class AnthropicEmitter:
    """Total function: any sequence of SinkEvents yields a protocol-valid
    Anthropic SSE stream.

    Invariants enforced:
    - At most one content block open at any time
    - Block indices monotonically increase
    - No delta after stop
    - message_delta(stop_reason) immediately before message_stop
    """

    def __init__(self, message_id: str | None = None):
        self._message_id = message_id or "msg_" + __import__("uuid").uuid4().hex[:24]
        self._block_index = 0
        self._open_block: Optional[BlockType] = None
        self._open_block_index: Optional[int] = None
        self._started = False
        self._finished = False
        self._tool_call_buffers: dict[int, str] = {}

    def emit(self, event: SinkEvent) -> Iterator[bytes]:
        """Emit SSE frames for a single SinkEvent."""
        if self._finished:
            raise RuntimeError("Emitter already finished")

        if is_error(event):
            yield from self._close_all()
            yield self._frame("error", {"type": "error", "error": {"type": event.kind, "message": event.message}})
            self._finished = True
            return

        if isinstance(event, MessageStart):
            yield from self._emit_message_start(event)
        elif is_thinking(event):
            yield from self._emit_thinking_delta(event)
        elif is_text(event):
            yield from self._emit_text_delta(event)
        elif is_tool_start(event):
            yield from self._emit_tool_call_start(event)
        elif is_tool_args(event):
            yield from self._emit_tool_call_args(event)
        elif is_tool_end(event):
            yield from self._emit_tool_call_end(event)
        elif is_message_end(event):
            yield from self._emit_message_end(event)
            self._finished = True

    def close(self) -> Iterator[bytes]:
        """Idempotent close — also the cancel/abort path."""
        if not self._finished:
            yield from self._close_all()
            self._finished = True

    # ---- internal emission methods ----

    def _emit_message_start(self, event: MessageStart) -> Iterator[bytes]:
        if self._started:
            raise RuntimeError("MessageStart after stream started")
        self._started = True
        self._block_index = 0
        self._open_block = None
        self._open_block_index = None
        yield self._frame("message_start", {
            "type": "message_start",
            "message": {
                "id": self._message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": event.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        })

    def _emit_thinking_delta(self, event: ThinkingDelta) -> Iterator[bytes]:
        if self._open_block != BlockType.THINKING:
            yield from self._close_open_block()
            self._open_block = BlockType.THINKING
            self._open_block_index = self._block_index
            yield self._frame("content_block_start", {
                "type": "content_block_start",
                "index": self._block_index,
                "content_block": {"type": "thinking"}
            })
        if event.text:
            yield self._frame("content_block_delta", {
                "type": "content_block_delta",
                "index": self._open_block_index,
                "delta": {"type": "thinking_delta", "thinking": event.text}
            })

    def _emit_text_delta(self, event: TextDelta) -> Iterator[bytes]:
        if self._open_block == BlockType.THINKING:
            yield from self._close_open_block()
        if self._open_block != BlockType.TEXT:
            self._open_block = BlockType.TEXT
            self._open_block_index = self._block_index
            yield self._frame("content_block_start", {
                "type": "content_block_start",
                "index": self._block_index,
                "content_block": {"type": "text", "text": ""}
            })
        if event.text:
            yield self._frame("content_block_delta", {
                "type": "content_block_delta",
                "index": self._open_block_index,
                "delta": {"type": "text_delta", "text": event.text}
            })

    def _emit_tool_call_start(self, event: ToolCallStart) -> Iterator[bytes]:
        yield from self._close_open_block()
        self._open_block = BlockType.TOOL_USE
        self._open_block_index = event.index
        self._tool_call_buffers[event.index] = ""
        yield self._frame("content_block_start", {
            "type": "content_block_start",
            "index": event.index,
            "content_block": {
                "type": "tool_use",
                "id": event.id,
                "name": event.name,
                "input": {}
            }
        })
        self._block_index = max(self._block_index, event.index + 1)

    def _emit_tool_call_args(self, event: ToolCallArgsDelta) -> Iterator[bytes]:
        if self._open_block != BlockType.TOOL_USE:
            # Should not happen if protocol followed; be defensive
            self._open_block = BlockType.TOOL_USE
            self._open_block_index = event.index
        buf = self._tool_call_buffers.get(event.index, "") + event.partial_json
        self._tool_call_buffers[event.index] = buf
        yield self._frame("content_block_delta", {
            "type": "content_block_delta",
            "index": event.index,
            "delta": {"type": "input_json_delta", "partial_json": event.partial_json}
        })

    def _emit_tool_call_end(self, event: ToolCallEnd) -> Iterator[bytes]:
        if self._open_block == BlockType.TOOL_USE:
            yield from self._close_open_block()
        # No frame for tool_use end — Anthropic closes via next block start or message_delta

    def _emit_message_end(self, event: MessageEnd) -> Iterator[bytes]:
        yield from self._close_all()
        stop_reason_map = {
            StopReason.END_TURN: "end_turn",
            StopReason.MAX_TOKENS: "max_tokens",
            StopReason.TOOL_CALLS: "tool_use",
            StopReason.ERROR: "error",
            StopReason.STOP_SEQUENCE: "stop_sequence",
        }
        sr = stop_reason_map.get(event.stop_reason, "end_turn")
        yield self._frame("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": sr, "stop_sequence": None},
            "usage": event.usage
        })
        yield self._frame("message_stop", {"type": "message_stop"})

    # ---- block management ----

    def _close_open_block(self) -> Iterator[bytes]:
        if self._open_block is not None and self._open_block_index is not None:
            yield self._frame("content_block_stop", {
                "type": "content_block_stop",
                "index": self._open_block_index
            })
            # Next block index is the closed block's index + 1
            self._block_index = self._open_block_index + 1
            self._open_block = None
            self._open_block_index = None

    def _close_all(self) -> Iterator[bytes]:
        if self._open_block is not None:
            yield from self._close_open_block()

    def _frame(self, event_type: str, data: dict) -> bytes:
        """Serialize as SSE frame."""
        return f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()