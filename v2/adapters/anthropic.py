"""Anthropic wire <-> IR adapters (pure functions, side-effect free).

Behavioral reference: v1's app.py convert_anthropic_messages() — but the
v1 bug class (tool_result flattened into user text) is structurally
impossible here: tool_result maps to a ToolResultBlock in a Message whose
role is "tool", and build_prompt reads blocks, never strings.
"""
from __future__ import annotations

from typing import Any

from v2.ir import (
    canonicalize_conversation,
    Conversation,
    FileBlock,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UnknownBlock,
)

_MSG_KEYS = {"role", "content", "name"}


def to_ir(body: dict[str, Any]) -> Conversation:
    """Anthropic /v1/messages request body -> Conversation."""
    system = None
    sys_field = body.get("system")
    if isinstance(sys_field, str):
        system = sys_field if sys_field else None
    elif isinstance(sys_field, list):
        blocks = tuple(TextBlock(text=c.get("text", "")) for c in sys_field
                       if isinstance(c, dict) and c.get("type") == "text")
        system = blocks if blocks else None

    messages: list[Message] = []
    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content")
        blocks: list = []
        if isinstance(content, str):
            if content:
                blocks.append(TextBlock(text=content))
        elif isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type")
                if ctype == "text":
                    if c.get("text"):
                        blocks.append(TextBlock(text=c["text"]))
                elif ctype == "thinking":
                    blocks.append(ThinkingBlock(text=c.get("thinking", "")))
                elif ctype == "redacted_thinking":
                    blocks.append(ThinkingBlock(text="", is_redacted=True))
                elif ctype == "tool_use":
                    blocks.append(ToolUseBlock(id=c.get("id") or "", name=c.get("name") or "",
                                               arguments=c.get("input", {})))
                elif ctype == "tool_result":
                    inner = c.get("content", "")
                    inner_blocks: list = []
                    if isinstance(inner, str):
                        if inner:
                            inner_blocks.append(TextBlock(text=inner))
                    elif isinstance(inner, list):
                        for ic in inner:
                            if isinstance(ic, dict) and ic.get("type") == "text":
                                inner_blocks.append(TextBlock(text=ic.get("text", "")))
                            elif isinstance(ic, dict) and ic.get("type") == "image":
                                src = ic.get("source", {}) or {}
                                if src.get("type") == "base64":
                                    inner_blocks.append(ImageBlock(
                                        source="base64", data=src.get("data"),
                                        media_type=src.get("media_type")))
                    blocks.append(ToolResultBlock(tool_use_id=c.get("tool_use_id") or "",
                                                  content=tuple(inner_blocks),
                                                  is_error=bool(c.get("is_error"))))
                elif ctype == "image":
                    src = c.get("source", {}) or {}
                    if src.get("type") == "base64":
                        blocks.append(ImageBlock(source="base64", data=src.get("data"),
                                                 media_type=src.get("media_type")))
                    elif src.get("type") == "file":
                        blocks.append(ImageBlock(source="file", file_ref=src.get("file_id")))
                    elif src.get("type") == "url":
                        blocks.append(ImageBlock(source="url", url=src.get("url")))
                elif ctype == "document":
                    src = c.get("source", {}) or {}
                    blocks.append(FileBlock(file_ref=src.get("file_id"),
                                            data=src.get("data") if src.get("type") == "base64" else None,
                                            filename=c.get("title"), media_type=src.get("media_type")))
                else:
                    blocks.append(UnknownBlock(dialect="anthropic", type_name=str(ctype), raw=c))
        # Anthropic packs tool_result blocks into user turns; the IR assigns
        # them their own role="tool" messages so downstream never has to
        # pattern-match strings. Text from the same turn stays a user message.
        tool_blocks = [b for b in blocks if isinstance(b, ToolResultBlock)]
        other_blocks = [b for b in blocks if not isinstance(b, ToolResultBlock)]
        extra = {k: v for k, v in m.items() if k not in _MSG_KEYS}
        if other_blocks or not tool_blocks:
            messages.append(Message(role=role, blocks=tuple(other_blocks),
                                    name=m.get("name"), extra=extra))
        for tb in tool_blocks:
            messages.append(Message(role="tool", blocks=(tb,)))
    known_top = {"messages", "system", "model", "stream", "tools", "max_tokens"}
    metadata = {k: v for k, v in body.items() if k not in known_top}
    conv = Conversation(system=system, messages=tuple(messages), metadata=metadata)
    return canonicalize_conversation(conv)


def from_ir(conv: Conversation) -> dict[str, Any]:
    """Conversation -> Anthropic /v1/messages request body."""
    out: dict[str, Any] = {}
    if isinstance(conv.system, tuple):
        out["system"] = [{"type": "text", "text": b.text} for b in conv.system]
    elif conv.system is not None:
        out["system"] = conv.system
    msgs: list[dict[str, Any]] = []
    for m in conv.messages:
        if m.role == "system":
            # No system role in Anthropic messages; fold into the system field.
            existing = out.get("system")
            text = m.text()
            if isinstance(existing, str):
                out["system"] = (existing + "\n" + text) if existing else text
            elif isinstance(existing, list):
                existing.append({"type": "text", "text": text})
            else:
                out["system"] = text
            continue
        content: list[dict[str, Any]] = []
        for b in m.blocks:
            if isinstance(b, TextBlock):
                content.append({"type": "text", "text": b.text})
            elif isinstance(b, ThinkingBlock):
                if b.is_redacted:
                    content.append({"type": "redacted_thinking"})
                else:
                    content.append({"type": "thinking", "thinking": b.text,
                                    "signature": m.extra.get("thinking_signature", "")})
            elif isinstance(b, ToolUseBlock):
                content.append({"type": "tool_use", "id": b.id, "name": b.name,
                                "input": b.arguments if isinstance(b.arguments, dict) else {}})
            elif isinstance(b, ToolResultBlock):
                inner: list[dict[str, Any]] = []
                for ib in b.content:
                    if isinstance(ib, TextBlock):
                        inner.append({"type": "text", "text": ib.text})
                    elif isinstance(ib, ImageBlock) and ib.source == "base64":
                        inner.append({"type": "image", "source": {
                            "type": "base64", "media_type": ib.media_type, "data": ib.data}})
                tr: dict[str, Any] = {"type": "tool_result", "tool_use_id": b.tool_use_id,
                                      "content": inner or b.text()}
                if b.is_error:
                    tr["is_error"] = True
                content.append(tr)
            elif isinstance(b, ImageBlock):
                if b.source == "base64":
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": b.media_type, "data": b.data}})
                elif b.source == "url":
                    content.append({"type": "image", "source": {"type": "url", "url": b.url}})
            elif isinstance(b, FileBlock):
                content.append({"type": "document", "title": b.filename,
                                "source": {"type": "base64", "media_type": b.media_type,
                                           "data": b.data} if b.data else {"type": "file",
                                                                           "file_id": b.file_ref}})
            elif isinstance(b, UnknownBlock) and b.dialect == "anthropic":
                content.append(b.raw)
        if m.role == "tool":
            # tool results ride in a user turn in Anthropic's format
            if msgs and msgs[-1].get("role") == "user":
                msgs[-1]["content"].extend(content)
            else:
                msgs.append({"role": "user", "content": content})
        else:
            msgs.append({"role": m.role, "content": content})
    out["messages"] = msgs
    out.update(conv.metadata)
    return out
