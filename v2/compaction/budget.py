"""TokenBudget service — deterministic compaction at a single pipeline point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import deepseek_tokenizer

from v2.ir import ContentBlock, Conversation, Message, TextBlock, ToolResultBlock, ToolUseBlock, ThinkingBlock


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Compaction policy (from settings/env)."""
    total: int = 24_000
    tool_result_clip: int = 4_000
    recent_turns: int = 6
    min_recent_turns: int = 2
    old_text_clip: int = 200
    head_ratio: float = 0.7


@dataclass(frozen=False, slots=True)
class RenderedPrompt:
    """Prompt after compaction, ready for provider."""
    messages: list[dict]
    model: str = ""
    tools: list[dict] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: dict | None = None
    # Telemetry
    elided_turns: int = 0
    elided_tokens: int = 0


class TokenBudget:
    """Deterministic compaction service.

    Single application site: orchestrator step 4, immediately before adapter.send.
    Pure function of (Conversation, BudgetPolicy).
    """

    def __init__(self, policy: BudgetPolicy | None = None):
        self.policy = policy or BudgetPolicy()

    def measure(self, blocks: Sequence[ContentBlock]) -> int:
        """Count tokens in a sequence of blocks."""
        total = 0
        for b in blocks:
            if isinstance(b, TextBlock):
                total += len(deepseek_tokenizer.ds_token.encode(b.text))
            elif isinstance(b, ToolResultBlock):
                total += len(deepseek_tokenizer.ds_token.encode(b.text()))
            elif isinstance(b, ToolUseBlock):
                # Tool calls: name + JSON args
                import json
                args = json.dumps(b.arguments) if not isinstance(b.arguments, str) else b.arguments
                total += len(deepseek_tokenizer.ds_token.encode(b.name)) + len(deepseek_tokenizer.ds_token.encode(args))
            elif isinstance(b, ThinkingBlock):
                total += len(deepseek_tokenizer.ds_token.encode(b.text))
        return total

    def _clip_text(self, text: str, max_tokens: int, label: str = "") -> str:
        """Head+tail clip with deterministic marker."""
        if len(deepseek_tokenizer.ds_token.encode(text)) <= max_tokens:
            return text
        head_allow = int(max_tokens * self.policy.head_ratio)
        tail_allow = max_tokens - head_allow - 20  # marker overhead
        if tail_allow < 0:
            tail_allow = 0
            head_allow = max_tokens - 20
        head = text[:head_allow * 4]  # rough char estimate
        while len(deepseek_tokenizer.ds_token.encode(head)) > head_allow and head:
            head = head[:len(head) * 3 // 4]
        tail = text[-tail_allow * 4:] if tail_allow else ""
        while count_tokens(tail) > tail_allow and tail:
            tail = tail[len(tail) // 4:]
        marker = f"[... ~{count_tokens(text) - head_allow - tail_allow} tokens of {label} elided ...]"
        return head + marker + tail

    def _clip_tool_result(self, tr: ToolResultBlock) -> ToolResultBlock:
        """Clip a tool result block."""
        clipped_text = self._clip_text(tr.text(), self.policy.tool_result_clip, "tool_result")
        if clipped_text == tr.text():
            return tr
        return ToolResultBlock(
            tool_use_id=tr.tool_use_id,
            content=(TextBlock(text=clipped_text),),
            is_error=tr.is_error,
            name=tr.name,
        )

    def _process_message(self, msg: Message) -> Message:
        """Process one message: clip tool results, preserve structure."""
        blocks = []
        for b in msg.blocks:
            if isinstance(b, ToolResultBlock):
                blocks.append(self._clip_tool_result(b))
            else:
                blocks.append(b)
        return Message(role=msg.role, blocks=tuple(blocks), name=msg.name, extra=msg.extra)

    def plan(self, conv: Conversation, policy: BudgetPolicy | None = None) -> RenderedPrompt:
        """Deterministic tiered compaction.

        Tiers (applied in order until under budget):
        1. Verbatim: system + last N turns (recent_turns) + final user message
        2. Tool-result stubs: older tool results as one-liners
        3. Old text clipped: older text blocks clipped to old_text_clip
        4. Drop oldest: shrink recent window toward min_recent_turns

        System and final user message are NEVER clipped.
        """
        p = policy or self.policy
        messages = list(conv.messages)

        # Separate system
        system_text = conv.system_text() if conv.system else None

        # Find the final user message (never clipped)
        final_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                final_user_idx = i
                break

        def measure_conv(msgs: list[Message]) -> int:
            total = 0
            for m in msgs:
                total += self.measure(m.blocks)
            if system_text:
                total += len(deepseek_tokenizer.ds_token.encode(system_text))
            return total

        # Tier 1: verbatim window
        def build_tier(recent: int) -> list[Message]:
            if recent <= 0:
                return []
            # Always keep final user message
            keep = messages[-recent:] if recent <= len(messages) else messages
            if final_user_idx >= 0 and final_user_idx < len(messages) - recent:
                # Final user is outside window; include it explicitly
                keep = messages[-(recent - 1):] + [messages[final_user_idx]]
            return [self._process_message(m) for m in keep]

        # Try verbatim with full recent window
        for recent in range(p.recent_turns, p.min_recent_turns - 1, -1):
            tier_msgs = build_tier(recent)
            tokens = measure_conv(tier_msgs)
            if tokens <= p.total:
                return RenderedPrompt(
                    messages=[self._to_wire(m) for m in tier_msgs],
                    model="",  # filled by orchestrator
                    elided_turns=len(messages) - len(tier_msgs),
                    elided_tokens=0,
                )

        # Tier 2: tool-result stubs for older tool messages
        # (process all messages, but replace old ToolResultBlocks with name-only stubs)
        def build_with_stubs(recent: int) -> list[Message]:
            out = []
            for i, m in enumerate(messages):
                is_recent = i >= len(messages) - recent
                if m.role == "tool" and not is_recent:
                    # Stub: keep only tool name as a one-liner
                    for b in m.blocks:
                        if isinstance(b, ToolResultBlock):
                            name = b.name or "tool"
                            stub_text = f"[tool:{name} completed]"
                            out.append(Message(
                                role="tool",
                                blocks=(TextBlock(text=stub_text),),
                                name=m.name,
                                extra=m.extra,
                            ))
                            break
                else:
                    out.append(self._process_message(m))
            return out

        for recent in range(p.recent_turns, p.min_recent_turns - 1, -1):
            tier_msgs = build_with_stubs(recent)
            tokens = measure_conv(tier_msgs)
            if tokens <= p.total:
                return RenderedPrompt(
                    messages=[self._to_wire(m) for m in tier_msgs],
                    model="",
                    elided_turns=len(messages) - len(tier_msgs),
                    elided_tokens=0,
                )

        # Tier 3: clip old text blocks
        def build_with_clipped_text(recent: int) -> list[Message]:
            out = []
            for i, m in enumerate(messages):
                is_recent = i >= len(messages) - recent
                new_blocks = []
                for b in m.blocks:
                    if isinstance(b, TextBlock) and not is_recent:
                        clipped = self._clip_text(b.text, p.old_text_clip, "history")
                        if clipped != b.text:
                            new_blocks.append(TextBlock(text=clipped))
                        else:
                            new_blocks.append(b)
                    elif isinstance(b, ToolResultBlock) and not is_recent:
                        new_blocks.append(self._clip_tool_result(b))
                    else:
                        new_blocks.append(b)
                out.append(Message(role=m.role, blocks=tuple(new_blocks), name=m.name, extra=m.extra))
            return out

        for recent in range(p.recent_turns, p.min_recent_turns - 1, -1):
            tier_msgs = build_with_clipped_text(recent)
            tokens = measure_conv(tier_msgs)
            if tokens <= p.total:
                return RenderedPrompt(
                    messages=[self._to_wire(m) for m in tier_msgs],
                    model="",
                    elided_turns=len(messages) - len(tier_msgs),
                    elided_tokens=0,
                )

        # Tier 4: minimum window (should always fit if policy is sane)
        min_msgs = build_with_clipped_text(p.min_recent_turns)
        return RenderedPrompt(
            messages=[self._to_wire(m) for m in min_msgs],
            model="",
            elided_turns=len(messages) - len(min_msgs),
            elided_tokens=0,
        )

    def _to_wire(self, msg: Message) -> dict:
        """Convert IR Message to wire-format dict (provider-agnostic base)."""
        parts = []
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ToolUseBlock):
                parts.append({
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": b.arguments if isinstance(b.arguments, dict) else {},
                })
            elif isinstance(b, ToolResultBlock):
                parts.append({
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "content": b.text(),
                    "is_error": b.is_error,
                })
            elif isinstance(b, ThinkingBlock):
                parts.append({
                    "type": "thinking" if not b.is_redacted else "redacted_thinking",
                    "thinking": b.text,
                })
        return {"role": msg.role, "content": parts, **({"name": msg.name} if msg.name else {})}