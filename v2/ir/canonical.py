"""Canonicalization functions for IR models."""
from __future__ import annotations

from typing import Union

from v2.ir.message import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    FileBlock,
    UnknownBlock,
    Conversation,
    Message,
)


def _block_sort_key(block: ContentBlock) -> int:
    """Return sort priority for canonical block ordering within a message.

    Order matches wire format emission: tool calls first (OpenAI tool_calls field),
    then text content, then media, then thinking/reasoning.
    """
    if isinstance(block, ToolUseBlock):
        return 0
    if isinstance(block, TextBlock):
        return 1
    if isinstance(block, (ImageBlock, FileBlock)):
        return 2
    if isinstance(block, ThinkingBlock):
        return 3
    if isinstance(block, ToolResultBlock):
        return 4
    if isinstance(block, UnknownBlock):
        return 5
    return 99


def _merge_text(blocks: tuple[ContentBlock, ...], sort: bool = True) -> tuple[ContentBlock, ...]:
    """Merge adjacent TextBlocks, drop empty ones, optionally sort to canonical order.

    Sort runs BEFORE merging: sorting can make previously separated same-type
    blocks adjacent (e.g. two ThinkingBlocks split by a ToolUseBlock), and the
    merged result must be a fixpoint or canonicalize_conversation would not be
    idempotent.

    For ThinkingBlocks: merge adjacent non-redacted ones, keep redacted separate.

    Args:
        blocks: Tuple of ContentBlocks to merge
        sort: If True (default), sort blocks to canonical order using _block_sort_key.
              If False, preserve input order (used for ToolResultBlock.content).
    """
    work = list(blocks)
    if sort:
        work.sort(key=_block_sort_key)
    merged: list[ContentBlock] = []
    for b in work:
        if isinstance(b, TextBlock):
            if not b.text:
                continue
            if merged and isinstance(merged[-1], TextBlock):
                merged[-1] = TextBlock(text=merged[-1].text + "\n" + b.text)
                continue
        elif isinstance(b, ThinkingBlock):
            if not b.is_redacted and b.text:
                if merged and isinstance(merged[-1], ThinkingBlock) and not merged[-1].is_redacted:
                    merged[-1] = ThinkingBlock(text=merged[-1].text + "\n" + b.text)
                    continue
        elif isinstance(b, ToolResultBlock):
            # Recursively process ToolResultBlock content WITHOUT sorting
            # to preserve wire format content order
            b = b.model_copy(update={"content": _merge_text(b.content, sort=False)})
        merged.append(b)
    return tuple(merged)


def canonicalize_conversation(conv: Conversation) -> Conversation:
    """Return the canonical form: adjacent/empty text merged, blocks sorted."""
    msgs = tuple(
        m.model_copy(update={"blocks": _merge_text(m.blocks, sort=True)})
        for m in conv.messages
    )
    if conv.messages == msgs:
        return conv
    return conv.model_copy(update={"messages": msgs})