"""Edge adapter registry: wire dialect <-> IR."""
from __future__ import annotations

from typing import Any, Literal

from v2.adapters import anthropic, openai
from v2.ir import Conversation

Dialect = Literal["openai", "anthropic"]


def to_ir(body: dict[str, Any], dialect: Dialect) -> Conversation:
    if dialect == "openai":
        return openai.to_ir(body)
    return anthropic.to_ir(body)


def from_ir(conv: Conversation, dialect: Dialect) -> dict[str, Any]:
    if dialect == "openai":
        return openai.from_ir(conv)
    return anthropic.from_ir(conv)
