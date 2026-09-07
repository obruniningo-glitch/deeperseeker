"""SinkEvent vocabulary — provider-agnostic events the pipeline produces.

These are the canonical streaming events that all emitters translate from.
The emitter state machines derive block lifecycle from these; callers CANNOT
produce a malformed lifecycle because open/close is not an input — it is
derived from event kinds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union
from enum import Enum


class StopReason(Enum):
    """Unified stop reasons across providers."""
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    STOP_SEQUENCE = "stop_sequence"


# ---- Provider-agnostic sink events (what the pipeline produces) ----

@dataclass(frozen=True, slots=True)
class MessageStart:
    """First event of a message."""
    model: str


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """Reasoning/thinking content delta."""
    text: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    """Regular text content delta."""
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """Tool call begins — closes any open content block."""
    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallArgsDelta:
    """Tool call arguments delta (partial JSON)."""
    index: int
    partial_json: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    """Tool call completes."""
    index: int


@dataclass(frozen=True, slots=True)
class MessageEnd:
    """Message completes — emits stop_reason and usage."""
    stop_reason: StopReason
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}


@dataclass(frozen=True, slots=True)
class SinkError:
    """Pipeline error — never leaks exception text (audit S4)."""
    kind: str
    message: str


SinkEvent = Union[
    MessageStart,
    ThinkingDelta,
    TextDelta,
    ToolCallStart,
    ToolCallArgsDelta,
    ToolCallEnd,
    MessageEnd,
    SinkError,
]

# ---- Provider-side events (what adapters yield) ----

class ProviderEvent:
    """Base for provider-side streaming events."""
    pass


@dataclass(frozen=True, slots=True)
class ProviderText(ProviderEvent):
    text: str


@dataclass(frozen=True, slots=True)
class ProviderThinking(ProviderEvent):
    text: str


@dataclass(frozen=True, slots=True)
class ProviderFinished(ProviderEvent):
    pass


@dataclass(frozen=True, slots=True)
class ProviderError(ProviderEvent):
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ProviderToolCall(ProviderEvent):
    """Some providers emit structured tool calls directly."""
    index: int
    id: str
    name: str
    arguments: str  # JSON string


# Helper: type guard
def is_thinking(event: SinkEvent) -> bool:
    return isinstance(event, ThinkingDelta)

def is_text(event: SinkEvent) -> bool:
    return isinstance(event, TextDelta)

def is_tool_start(event: SinkEvent) -> bool:
    return isinstance(event, ToolCallStart)

def is_tool_args(event: SinkEvent) -> bool:
    return isinstance(event, ToolCallArgsDelta)

def is_tool_end(event: SinkEvent) -> bool:
    return isinstance(event, ToolCallEnd)

def is_message_end(event: SinkEvent) -> bool:
    return isinstance(event, MessageEnd)

def is_error(event: SinkEvent) -> bool:
    return isinstance(event, SinkError)