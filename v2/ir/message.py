"""Canonical message IR models (v2 P0) — the sole internal representation.

Everything downstream of the edge adapters (cache, compaction, emitters,
providers) operates on these frozen pydantic models; wire dialects
(OpenAI / Anthropic) exist only at the adapters.

Design notes (from V2_DESIGN.md §2, amended by the 2026-09-06 expert review):
- IR_VERSION enables future schema evolution: persisted IR (cache rows,
  recordings) carries it and can be detected/migrated.
- UnknownBlock is the graceful-degradation path for wire content types the
  IR does not model yet: adapters round-trip them losslessly instead of
  dropping fields.
- Message.extra / Conversation.metadata preserve unknown MESSAGE-level and
  REQUEST-level fields respectively (anti-silent-loss, spec §13).
- tool_result keeps its nested content as blocks (ordering preserved);
  flattening to text happens only at prompt-build time (P4).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

IR_VERSION = 1

Role = Literal["system", "user", "assistant", "tool"]


class _Block(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(_Block):
    type: Literal["thinking"] = "thinking"
    text: str
    # Anthropic redacted_thinking round-trips as an empty, flagged block.
    is_redacted: bool = False


class ToolUseBlock(_Block):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    # dict | list | JSON-string (unparseable fragments from providers stay str)
    arguments: Union[dict, list, str] = Field(default_factory=dict)


# Content allowed inside a tool_result (ordering preserved).
class ToolResultBlock(_Block):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: tuple[Union["TextBlock", "ImageBlock"], ...] = ()
    is_error: bool = False
    # OpenAI carries the tool name on role="tool" messages.
    name: Optional[str] = None

    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))


class ImageBlock(_Block):
    type: Literal["image"] = "image"
    # "url" -> .url set; "base64" -> .data/.media_type set; "file" -> .file_ref
    source: Literal["url", "base64", "file"] = "url"
    url: Optional[str] = None
    data: Optional[str] = None  # base64 payload
    media_type: Optional[str] = None
    file_ref: Optional[str] = None
    # OpenAI image_url detail hint (auto/low/high) — preserved for round-trip.
    detail: Optional[str] = None


class FileBlock(_Block):
    type: Literal["file"] = "file"
    file_ref: Optional[str] = None  # provider-side file id
    data: Optional[str] = None  # base64 payload
    filename: Optional[str] = None
    media_type: Optional[str] = None


class UnknownBlock(_Block):
    """Pass-through for wire content types the IR does not model (yet).

    Adapters store the dialect + original type name + raw payload so a
    round-trip through the IR loses nothing even for shapes this version
    has never seen.
    """

    type: Literal["unknown"] = "unknown"
    dialect: Literal["openai", "anthropic"]
    type_name: str
    raw: dict[str, Any] = Field(default_factory=dict)


ContentBlock = Annotated[
    Union[
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
        ToolResultBlock,
        ImageBlock,
        FileBlock,
        UnknownBlock,
    ],
    Field(discriminator="type"),
]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    blocks: tuple[ContentBlock, ...] = ()
    # OpenAI `name` on user/tool messages (author identity).
    name: Optional[str] = None
    # Unknown message-level wire fields, preserved for round-trip.
    extra: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.role == "tool":
            trs = [b for b in self.blocks if isinstance(b, ToolResultBlock)]
            if not trs or len(trs) != len(self.blocks):
                raise ValueError(
                    "role='tool' messages must contain only ToolResultBlock(s); "
                    "wire adapters must not attach sibling text to tool messages")

    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, TextBlock))


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ir_version: int = IR_VERSION
    # Anthropic system may be a block array; str is the normalized common case.
    system: Optional[Union[str, tuple[TextBlock, ...]]] = None
    messages: tuple[Message, ...] = ()
    # Request-level fields that don't belong to any message (model hints,
    # response_format, stop sequences, ...) — typed nothing, lost nothing.
    metadata: dict[str, Any] = Field(default_factory=dict)

    def system_text(self) -> Optional[str]:
        if self.system is None:
            return None
        if isinstance(self.system, str):
            return self.system or None
        return "\n".join(b.text for b in self.system) or None

    def to_summary(self, limit: int = 120) -> str:
        parts = [f"[v{self.ir_version} msgs={len(self.messages)}]"]
        for m in self.messages[-3:]:
            t = m.text().replace("\n", " ")[:limit]
            parts.append(f"{m.role}: {t}")
        return " | ".join(parts)