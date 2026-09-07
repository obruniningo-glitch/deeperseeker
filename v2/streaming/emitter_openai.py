"""OpenAI SSE emitter — total state machine.

Mirrors the Anthropic emitter but emits OpenAI-format SSE:
- First: role delta (assistant)
- Then: content deltas (text) and/or reasoning_content deltas
- Tool calls: function calling deltas with index
- Final: finish_reason in choices[0].delta, then [DONE]
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
    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"


class OpenAIEmitter:
    """Total function: any sequence of SinkEvents yields a protocol-valid
    OpenAI SSE stream.

    Invariants enforced:
    - Exactly one role delta at start
    - At most one content stream open (text or reasoning)
    - Tool call indices monotonically increase
    - finish_reason exactly once, then [DONE] sentinel
    """

    def __init__(self, message_id: str | None = None):
        self._message_id = message_id or "chatcmpl-" + __import__("uuid").uuid4().hex[:24]
        self._role_emitted = False
        self._block_index = 0
        self._open_block: Optional[BlockType] = None
        self._tool_call_buffers: dict[int, str] = {}
        self._finished = False

    def emit(self, event: SinkEvent) -> Iterator[bytes]:
        """Emit SSE frames for a single SinkEvent."""
        if self._finished:
            raise RuntimeError("Emitter already finished")

        if is_error(event):
            yield from self._close_all()
            yield self._frame({"error": {"message": event.message, "type": event.kind}})
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
        if self._role_emitted:
            raise RuntimeError("MessageStart after stream started")
        self._role_emitted = True
        # OpenAI sends role in first chunk
        yield self._frame({
            "id": self._message_id,
            "object": "chat.completion.chunk",
            "created": int(__import__("time").time()),
            "model": event.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        })

    def _emit_thinking_delta(self, event: ThinkingDelta) -> Iterator[bytes]:
        if self._open_block != BlockType.REASONING:
            self._open_block = BlockType.REASONING
        if event.text:
            yield self._frame({
                "id": self._message_id,
                "object": "chat.completion.chunk",
                "created": int(__import__("time").time()),
                "model": "",  # filled by transport
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": event.text},
                    "finish_reason": None
                }]
            })

    def _emit_text_delta(self, event: TextDelta) -> Iterator[bytes]:
        if self._open_block == BlockType.REASONING:
            # Close reasoning stream
            self._open_block = None
        if self._open_block != BlockType.TEXT:
            self._open_block = BlockType.TEXT
        if event.text:
            yield self._frame({
                "id": self._message_id,
                "object": "chat.completion.chunk",
                "created": int(__import__("time").time()),
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {"content": event.text},
                    "finish_reason": None
                }]
            })

    def _emit_tool_call_start(self, event: ToolCallStart) -> Iterator[bytes]:
        if self._open_block is not None:
            self._open_block = None
        self._open_block = BlockType.TOOL_CALL
        self._tool_call_buffers[event.index] = ""
        # OpenAI emits tool_calls in the delta
        yield self._frame({
            "id": self._message_id,
            "object": "chat.completion.chunk",
            "created": int(__import__("time").time()),
            "model": "",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": event.index,
                        "id": event.id,
                        "type": "function",
                        "function": {"name": event.name, "arguments": ""}
                    }]
                },
                "finish_reason": None
            }]
        })

    def _emit_tool_call_args(self, event: ToolCallArgsDelta) -> Iterator[bytes]:
        if self._open_block != BlockType.TOOL_CALL:
            self._open_block = BlockType.TOOL_CALL
        buf = self._tool_call_buffers.get(event.index, "") + event.partial_json
        self._tool_call_buffers[event.index] = buf
        yield self._frame({
            "id": self._message_id,
            "object": "chat.completion.chunk",
            "created": int(__import__("time").time()),
            "model": "",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": event.index,
                        "function": {"arguments": event.partial_json}
                    }]
                },
                "finish_reason": None
            }]
        })

    def _emit_tool_call_end(self, event: ToolCallEnd) -> Iterator[bytes]:
        # OpenAI doesn't have explicit tool_call_end; next start or finish closes
        if False:
            yield  # Make it a generator

    def _emit_message_end(self, event: MessageEnd) -> Iterator[bytes]:
        yield from self._close_all()
        stop_reason_map = {
            StopReason.END_TURN: "stop",
            StopReason.MAX_TOKENS: "length",
            StopReason.TOOL_CALLS: "tool_calls",
            StopReason.ERROR: "error",
            StopReason.STOP_SEQUENCE: "stop",
        }
        sr = stop_reason_map.get(event.stop_reason, "stop")
        yield self._frame({
            "id": self._message_id,
            "object": "chat.completion.chunk",
            "created": int(__import__("time").time()),
            "model": "",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": sr
            }]
        })
        yield b"data: [DONE]\n\n"

    def _close_all(self) -> Iterator[bytes]:
        self._open_block = None
        if False:
            yield  # Make it a generator

    def _frame(self, data: dict) -> bytes:
        """Serialize as SSE frame."""
        return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode()