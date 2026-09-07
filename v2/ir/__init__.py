"""IR package exports."""
from __future__ import annotations

from v2.ir.message import (
    Conversation,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    FileBlock,
    UnknownBlock,
    ContentBlock,
    Role,
)
from v2.ir.canonical import canonicalize_conversation, _merge_text
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
    ProviderEvent,
    ProviderText,
    ProviderThinking,
    ProviderFinished,
    ProviderError,
    ProviderToolCall,
)

__all__ = [
    # Message IR
    "Conversation",
    "Message",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "FileBlock",
    "UnknownBlock",
    "ContentBlock",
    "Role",
    # Canonicalization
    "canonicalize_conversation",
    "_merge_text",
    # Streaming events
    "SinkEvent",
    "MessageStart",
    "ThinkingDelta",
    "TextDelta",
    "ToolCallStart",
    "ToolCallArgsDelta",
    "ToolCallEnd",
    "MessageEnd",
    "SinkError",
    "StopReason",
    # Provider events
    "ProviderEvent",
    "ProviderText",
    "ProviderThinking",
    "ProviderFinished",
    "ProviderError",
    "ProviderToolCall",
]