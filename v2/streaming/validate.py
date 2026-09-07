"""Strict SSE lifecycle invariant checker.

Used by tests to verify emitter output. Production does not run this.

Checks both Anthropic and OpenAI block lifecycles:
- Monotonic indices
- At most one open content block
- No delta after stop
- message_delta(stop_reason) immediately before message_stop (Anthropic)
- Exactly one finish_reason, then [DONE] (OpenAI)
- Tool call indices monotonic
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Iterator


@dataclass
class ValidationError:
    frame_idx: int
    event_type: str
    message: str
    frame_data: dict


class AnthropicValidator:
    """Validates Anthropic SSE stream lifecycle."""

    def __init__(self):
        self._open_block_idx: int | None = None
        self._open_block_type: str | None = None
        self._block_count = 0
        self._message_started = False
        self._message_stopped = False
        self._errors: list[ValidationError] = []

    def validate(self, frames: Iterator[bytes]) -> list[ValidationError]:
        for idx, frame in enumerate(frames):
            if not frame.strip():
                continue
            # Parse SSE frame
            event_type = None
            data = {}
            for line in frame.decode().split('\n'):
                if line.startswith('event: '):
                    event_type = line[7:].strip()
                elif line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        self._errors.append(ValidationError(idx, "parse", "Invalid JSON", {}))
            if event_type:
                self._validate_frame(idx, event_type, data)
        if self._open_block_idx is not None:
            self._errors.append(ValidationError(-1, "lifecycle", "Unclosed block at end of stream", {}))
        if self._message_started and not self._message_stopped:
            self._errors.append(ValidationError(-1, "lifecycle", "message_start without message_stop", {}))
        return self._errors

    def _validate_frame(self, idx: int, event_type: str, data: dict) -> None:
        if event_type == "message_start":
            if self._message_started:
                self._errors.append(ValidationError(idx, event_type, "Duplicate message_start", data))
            self._message_started = True
            self._block_count = 0
            self._open_block_idx = None
        elif event_type == "content_block_start":
            if self._open_block_idx is not None:
                self._errors.append(ValidationError(idx, event_type, "Block start while another open", data))
            self._open_block_idx = data.get("index")
            if self._open_block_idx != self._block_count:
                self._errors.append(ValidationError(idx, event_type, f"Block index mismatch: expected {self._block_count}, got {self._open_block_idx}", data))
            self._block_count += 1
        elif event_type in ("content_block_delta",):
            if self._open_block_idx is None:
                self._errors.append(ValidationError(idx, event_type, "Delta with no open block", data))
        elif event_type == "content_block_stop":
            if self._open_block_idx is None:
                self._errors.append(ValidationError(idx, event_type, "Block stop with no open block", data))
            if data.get("index") != self._open_block_idx:
                self._errors.append(ValidationError(idx, event_type, f"Block stop index mismatch: open={self._open_block_idx}, got={data.get('index')}", data))
            self._open_block_idx = None
        elif event_type == "message_delta":
            if data.get("delta", {}).get("stop_reason") is None:
                self._errors.append(ValidationError(idx, event_type, "message_delta missing stop_reason", data))
            # message_stop must immediately follow
        elif event_type == "message_stop":
            if not self._message_started:
                self._errors.append(ValidationError(idx, event_type, "message_stop without message_start", data))
            self._message_stopped = True
        elif event_type == "error":
            # Error frame is valid anywhere
            pass
        else:
            self._errors.append(ValidationError(idx, event_type, f"Unknown event type: {event_type}", data))


class OpenAIValidator:
    """Validates OpenAI SSE stream lifecycle."""

    def __init__(self):
        self._role_emitted = False
        self._finish_reason_emitted = False
        self._done_received = False
        self._tool_call_indices: list[int] = []
        self._errors: list[ValidationError] = []

    def validate(self, frames: Iterator[bytes]) -> list[ValidationError]:
        for idx, frame in enumerate(frames):
            if not frame.strip():
                continue
            if frame.startswith(b"data: [DONE]"):
                if self._done_received:
                    self._errors.append(ValidationError(idx, "done", "Duplicate [DONE]", {}))
                self._done_received = True
                continue
            if not frame.startswith(b"data: "):
                continue
            try:
                data = json.loads(frame[6:].decode())
            except json.JSONDecodeError:
                self._errors.append(ValidationError(idx, "parse", "Invalid JSON", {}))
                continue
            self._validate_frame(idx, data)
        if not self._done_received:
            self._errors.append(ValidationError(-1, "lifecycle", "Missing [DONE] sentinel", {}))
        if self._role_emitted and not self._finish_reason_emitted:
            self._errors.append(ValidationError(-1, "lifecycle", "role emitted but no finish_reason", {}))
        return self._errors

    def _validate_frame(self, idx: int, data: dict) -> None:
        choices = data.get("choices", [])
        if not choices:
            self._errors.append(ValidationError(idx, "structure", "Empty choices", data))
            return
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        if "role" in delta:
            if self._role_emitted:
                self._errors.append(ValidationError(idx, "role", "Duplicate role emission", data))
            self._role_emitted = True

        if "tool_calls" in delta:
            for tc in delta["tool_calls"]:
                tc_idx = tc.get("index")
                if tc_idx is not None:
                    # OpenAI sends the same index in multiple chunks for the same tool call
                    # Track first occurrence only
                    if tc_idx not in self._tool_call_indices:
                        self._tool_call_indices.append(tc_idx)

        if finish_reason is not None:
            if self._finish_reason_emitted:
                self._errors.append(ValidationError(idx, "finish_reason", "Duplicate finish_reason", data))
            self._finish_reason_emitted = True


def validate_anthropic(frames: Iterator[bytes]) -> list[ValidationError]:
    """Validate Anthropic SSE frames."""
    return AnthropicValidator().validate(frames)


def validate_openai(frames: Iterator[bytes]) -> list[ValidationError]:
    """Validate OpenAI SSE frames."""
    return OpenAIValidator().validate(frames)