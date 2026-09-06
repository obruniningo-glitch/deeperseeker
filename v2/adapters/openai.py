"""OpenAI wire <-> IR adapters (pure functions, side-effect free).

Behavioral reference: v1's app.py handle_chat message handling, plus the
2026-09-06 expert review round-trip requirements (image detail, refusal,
message `name`, legacy function_call, reasoning_content, unknown parts).
"""
from __future__ import annotations

import json
from typing import Any

from v2.ir import canonicalize_conversation
from v2.ir import (
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


def _parse_args(raw: Any) -> dict | list | str:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw if isinstance(raw, (dict, list)) else {}


def to_ir(body: dict[str, Any]) -> Conversation:
    """OpenAI chat-completions request body -> Conversation."""
    messages: list[Message] = []
    system_texts: list[str] = []
    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content")
        if role == "tool":
            # Handle FIRST: the tool message's content IS the result — never
            # also a TextBlock (the IR forbids sibling text on tool messages).
            blocks: list = [ToolResultBlock(tool_use_id=m.get("tool_call_id") or "",
                                            content=(TextBlock(text=str(m.get("content", ""))),),
                                            name=m.get("name"))]
            messages.append(Message(role="tool", blocks=tuple(blocks),
                                    name=m.get("name"),
                                    extra={k: v for k, v in m.items()
                                           if k not in {"role", "content", "name", "tool_call_id"}}))
            continue
        blocks = []
        # For assistant role, process tool_calls FIRST (canonical order: tools before text)
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                blocks.append(ToolUseBlock(id=tc.get("id") or "", name=fn.get("name") or "",
                                           arguments=_parse_args(fn.get("arguments"))))
            # legacy deprecated top-level function_call
            fc = m.get("function_call")
            if fc and not m.get("tool_calls"):
                blocks.append(ToolUseBlock(id="legacy_fc", name=fc.get("name") or "",
                                           arguments=_parse_args(fc.get("arguments"))))
            if m.get("refusal"):
                blocks.append(TextBlock(text=f"[refusal]: {m['refusal']}"))
            if m.get("reasoning_content"):
                blocks.append(ThinkingBlock(text=m["reasoning_content"]))
        # Then process content (text, images, files) for all roles
        if isinstance(content, str):
            if content:
                blocks.append(TextBlock(text=content))
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    if part.get("text"):
                        blocks.append(TextBlock(text=part["text"]))
                elif ptype == "image_url":
                    iu = part.get("image_url", {}) or {}
                    url = iu.get("url", "")
                    if url.startswith("data:"):
                        header, _, data = url.partition(",")
                        media = header[5:].split(";")[0] if header.startswith("data:") else None
                        blocks.append(ImageBlock(source="base64", data=data or None, media_type=media,
                                                 detail=iu.get("detail")))
                    else:
                        blocks.append(ImageBlock(source="url", url=url, detail=iu.get("detail")))
                elif ptype == "file":
                    f = part.get("file", {}) or {}
                    blocks.append(FileBlock(file_ref=f.get("file_id"), data=f.get("file_data"),
                                            filename=f.get("filename")))
                else:
                    blocks.append(UnknownBlock(dialect="openai", type_name=str(ptype), raw=part))
        if role == "system":
            system_texts.append(m.get("content") if isinstance(m.get("content"), str)
                                else "\n".join(b.text for b in blocks if isinstance(b, TextBlock)))
            continue
        known_keys = {"role", "content", "name", "tool_calls", "tool_call_id",
                      "function_call", "refusal", "reasoning_content"}
        extra = {k: v for k, v in m.items() if k not in known_keys}
        messages.append(Message(role=role, blocks=tuple(blocks), name=m.get("name"), extra=extra))
    known_top = {"messages", "model", "tools", "stream", "system"}
    metadata = {k: v for k, v in body.items() if k not in known_top}
    conv = Conversation(system="\n".join(system_texts) or None, messages=tuple(messages),
                        metadata=metadata)
    return canonicalize_conversation(conv)


def from_ir(conv: Conversation) -> dict[str, Any]:
    """Conversation -> OpenAI chat-completions request body."""
    out: dict[str, Any] = {}
    if conv.system is not None:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": conv.system_text() or ""}]
    else:
        msgs = []
    for m in conv.messages:
        if m.role == "system":
            msgs.append({"role": "system", "content": m.text()})
            continue
        if m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant"}
            text_parts = [b.text for b in m.blocks if isinstance(b, TextBlock)]
            tool_uses = [b for b in m.blocks if isinstance(b, ToolUseBlock)]
            thinking = [b for b in m.blocks if isinstance(b, ThinkingBlock)]
            if text_parts:
                msg["content"] = "\n".join(text_parts)
            elif tool_uses:
                msg["content"] = None
            if thinking:
                msg["reasoning_content"] = "\n".join(t.text for t in thinking)
            if tool_uses:
                msg["tool_calls"] = [
                    {"id": t.id, "type": "function",
                     "function": {"name": t.name,
                                  "arguments": t.arguments if isinstance(t.arguments, str)
                                  else json.dumps(t.arguments)}}
                    for t in tool_uses
                ]
            if m.name:
                msg["name"] = m.name
            msg.update(m.extra)
            msgs.append(msg)
            continue
        if m.role == "tool":
            for b in m.blocks:
                if isinstance(b, ToolResultBlock):
                    entry: dict[str, Any] = {"role": "tool", "tool_call_id": b.tool_use_id,
                                             "content": b.text()}
                    if b.name:
                        entry["name"] = b.name
                    msgs.append(entry)
            continue
        # user (or degraded shapes): parts list when images present, else str
        has_media = any(isinstance(b, (ImageBlock, FileBlock)) for b in m.blocks)
        if has_media:
            parts: list[dict[str, Any]] = []
            for b in m.blocks:
                if isinstance(b, TextBlock):
                    parts.append({"type": "text", "text": b.text})
                elif isinstance(b, ImageBlock):
                    if b.source == "url" and b.url:
                        iu: dict[str, Any] = {"url": b.url}
                        if b.detail:
                            iu["detail"] = b.detail
                        parts.append({"type": "image_url", "image_url": iu})
                    elif b.source == "base64" and b.data:
                        url = f"data:{b.media_type or 'application/octet-stream'};base64,{b.data}"
                        iu = {"url": url}
                        if b.detail:
                            iu["detail"] = b.detail
                        parts.append({"type": "image_url", "image_url": iu})
                elif isinstance(b, FileBlock):
                    f: dict[str, Any] = {}
                    if b.file_ref:
                        f["file_id"] = b.file_ref
                    if b.filename:
                        f["filename"] = b.filename
                    parts.append({"type": "file", "file": f})
                elif isinstance(b, UnknownBlock) and b.dialect == "openai":
                    parts.append(b.raw)
            msg = {"role": m.role, "content": parts}
            if m.name:
                msg["name"] = m.name
            msg.update(m.extra)
            msgs.append(msg)
        else:
            msg = {"role": m.role, "content": m.text()}
            if m.name:
                msg["name"] = m.name
            msg.update(m.extra)
            msgs.append(msg)
    out["messages"] = msgs
    out.update(conv.metadata)
    return out
