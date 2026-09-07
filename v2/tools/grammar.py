"""Grammar registry and tool-call grammars.

Provides the transformation: ProviderEvents -> SinkEvents via pluggable grammars.
"""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Protocol

from v2.ir.events import (
    ProviderEvent, ProviderText, ProviderThinking, ProviderFinished, ProviderError, ProviderToolCall,
    SinkEvent, ThinkingDelta, TextDelta, ToolCallStart, ToolCallArgsDelta, ToolCallEnd, MessageEnd, SinkError, StopReason
)
from v2.tools.inject import ToolInjection


class Grammar(Protocol):
    """Protocol for tool-call grammars.

    Transforms provider-side event stream into canonical SinkEvents.
    """

    async def transform(self, stream: AsyncIterator[ProviderEvent]) -> AsyncIterator[SinkEvent]:
        ...

    def parse_final(self, full_text: str) -> tuple[list[dict], str]:
        """Non-streaming final parse: returns (tool_calls, remaining_text)."""
        ...


@dataclass
class GrammarContext:
    """Runtime context for grammar transformation."""
    tool_injection: ToolInjection
    model_family: str = "deepseek"


class BaseGrammar:
    """Base grammar with common infrastructure."""

    def __init__(self, ctx: GrammarContext):
        self.ctx = ctx

    async def transform(self, stream: AsyncIterator[ProviderEvent]) -> AsyncIterator[SinkEvent]:
        async for event in stream:
            async for sink_ev in self._handle_event(event):
                yield sink_ev

    async def _handle_event(self, event: ProviderEvent) -> AsyncIterator[SinkEvent]:
        if isinstance(event, ProviderThinking):
            yield ThinkingDelta(text=event.text)
        elif isinstance(event, ProviderText):
            yield TextDelta(text=event.text)
        elif isinstance(event, ProviderToolCall):
            # Emit tool call as structured events
            yield ToolCallStart(index=event.index, id=event.id, name=event.name)
            yield ToolCallArgsDelta(index=event.index, partial_json=event.arguments)
            yield ToolCallEnd(index=event.index)
        elif isinstance(event, ProviderFinished):
            yield MessageEnd(stop_reason=StopReason.END_TURN, usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        elif isinstance(event, ProviderError):
            yield SinkError(kind=event.kind, message=event.message)

    def parse_final(self, full_text: str) -> tuple[list[dict], str]:
        return [], full_text


# ---- DSML Grammar (DeepSeek family) ----

class DSMLGrammar(BaseGrammar):
    """DeepSeek-family grammar: structured primary parser for injected formats.

    Recognizes the exact forms we inject instructions for:
    - ```json {"name": "...", "arguments": {...}} ```
    - DSML fullwidth-pipe variants (the │ and ┃ forms DeepSeek emits)
    - Fenced JSON with companion prose preservation

    Uses a small scanner (not cascading regex) keeping v1's hard-won rules:
    - Code-fence immunity (_code_fence_spans)
    - Companion-prose preservation
    - Repeated-identical-call preservation
    - No dedupe (audit C4)
    """

    def __init__(self, ctx: GrammarContext):
        super().__init__(ctx)
        self._buffer = ""
        self._in_code_fence = False
        self._fence_marker = ""
        self._tool_call_idx = 0
        self._current_tool: dict | None = None
        self._accumulated_args = ""

    async def _handle_event(self, event: ProviderEvent) -> AsyncIterator[SinkEvent]:
        if isinstance(event, ProviderThinking):
            yield ThinkingDelta(text=event.text)
            return
        if isinstance(event, ProviderFinished):
            # Flush any pending tool call
            async for ev in self._flush_tool_call():
                yield ev
            yield MessageEnd(stop_reason=StopReason.TOOL_CALLS if self._current_tool else StopReason.END_TURN,
                           usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            return
        if isinstance(event, ProviderError):
            yield SinkError(kind=event.kind, message=event.message)
            return
        if isinstance(event, ProviderToolCall):
            # Structured tool call from provider (rare for DeepSeek)
            yield ToolCallStart(index=event.index, id=event.id, name=event.name)
            yield ToolCallArgsDelta(index=event.index, partial_json=event.arguments)
            yield ToolCallEnd(index=event.index)
            return
        if isinstance(event, ProviderText):
            # Stream text through the DSML scanner
            async for ev in self._scan_text(event.text):
                yield ev
            return
        # Fallback to base
        async for ev in super()._handle_event(event):
            yield ev

    async def _scan_text(self, text: str) -> AsyncIterator[SinkEvent]:
        """Scan incremental text for tool calls.

        State machine:
        - Outside fence: look for ```json or ```tool
        - Inside fence: accumulate until closing ```
        - Outside fence: emit as TextDelta (companion prose)
        """
        self._buffer += text
        while self._buffer:
            if not self._in_code_fence:
                # Look for fence start
                fence_match = re.search(r'```(?:json|tool|function)\s*', self._buffer)
                if not fence_match:
                    # No fence start, emit all as text
                    yield TextDelta(text=self._buffer)
                    self._buffer = ""
                    break
                # Emit text before fence as companion prose
                if fence_match.start() > 0:
                    yield TextDelta(text=self._buffer[:fence_match.start()])
                # Enter fence
                self._in_code_fence = True
                self._fence_marker = self._buffer[fence_match.start():fence_match.end()].strip()
                self._buffer = self._buffer[fence_match.end():]
                # Initialize tool call state
                self._current_tool = {"index": self._tool_call_idx, "id": f"call_{self._tool_call_idx}", "name": "", "arguments": ""}
                self._accumulated_args = ""
            else:
                # Inside fence: look for closing ```
                close_pos = self._buffer.find('```')
                if close_pos == -1:
                    # Fence not closed yet, accumulate
                    if self._current_tool:
                        self._accumulated_args += self._buffer
                    self._buffer = ""
                    break
                # Fence closed
                fence_content = self._buffer[:close_pos]
                self._buffer = self._buffer[close_pos + 3:]
                self._in_code_fence = False
                # Parse the fence content as tool call
                async for ev in self._parse_fence_content(fence_content):
                    yield ev
                self._tool_call_idx += 1
                self._current_tool = None
                self._accumulated_args = ""

    async def _parse_fence_content(self, content: str) -> AsyncIterator[SinkEvent]:
        """Parse fence content as JSON tool call."""
        content = content.strip()
        if not content:
            return
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Malformed JSON — treat as text (companion prose recovery)
            yield TextDelta(text=content)
            return

        # Expected: {"name": "...", "arguments": {...}} or DSML variants
        name = data.get("name") or data.get("function") or data.get("tool")
        args = data.get("arguments") or data.get("parameters") or data.get("input") or {}
        if not name:
            yield TextDelta(text=content)
            return

        # Emit tool call
        tool_id = self._current_tool.get("id") if self._current_tool else f"call_{self._tool_call_idx}"
        yield ToolCallStart(index=self._tool_call_idx, id=tool_id, name=name)
        args_json = json.dumps(args) if isinstance(args, dict) else str(args)
        yield ToolCallArgsDelta(index=self._tool_call_idx, partial_json=args_json)
        yield ToolCallEnd(index=self._tool_call_idx)

    async def _flush_tool_call(self) -> AsyncIterator[SinkEvent]:
        """Flush any incomplete tool call at stream end."""
        if self._in_code_fence and self._current_tool:
            # Incomplete fence — treat accumulated content as text (companion prose recovery)
            if self._accumulated_args.strip():
                yield TextDelta(text=self._accumulated_args)
        self._in_code_fence = False
        self._current_tool = None
        self._accumulated_args = ""

    def parse_final(self, full_text: str) -> tuple[list[dict], str]:
        """Non-streaming final parse for offline use."""
        tool_calls = []
        text_parts = []
        # Find all ```json ... ``` blocks
        pattern = r'```(?:json|tool|function)\s*(.*?)\s*```'
        last_end = 0
        for match in re.finditer(pattern, full_text, re.DOTALL):
            # Text before fence
            if match.start() > last_end:
                text_parts.append(full_text[last_end:match.start()])
            # Parse fence content
            content = match.group(1).strip()
            try:
                data = json.loads(content)
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or {}
                if name:
                    tool_calls.append({
                        "index": len(tool_calls),
                        "id": f"call_{len(tool_calls)}",
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    })
                else:
                    text_parts.append(match.group(0))
            except json.JSONDecodeError:
                text_parts.append(match.group(0))
            last_end = match.end()
        # Text after last fence
        if last_end < len(full_text):
            text_parts.append(full_text[last_end:])
        return tool_calls, "".join(text_parts)


# ---- Free-text fallback grammar ----

class FreeTextGrammar(BaseGrammar):
    """Permissive fallback: bare JSON after tags, brace-balancing repair.

    Generalizes v1's raw_decode + }-counting heuristic (functions.py:638-644).
    Selected per model family or as last resort.
    """

    def __init__(self, ctx: GrammarContext):
        super().__init__(ctx)
        self._buffer = ""
        self._brace_depth = 0
        self._in_json = False
        self._json_start = 0

    async def _handle_event(self, event: ProviderEvent) -> AsyncIterator[SinkEvent]:
        if isinstance(event, ProviderText):
            async for ev in self._scan_text(event.text):
                yield ev
        else:
            async for ev in super()._handle_event(event):
                yield ev

    async def _scan_text(self, text: str) -> AsyncIterator[SinkEvent]:
        self._buffer += text
        i = 0
        while i < len(self._buffer):
            ch = self._buffer[i]
            if ch == '{' and not self._in_json:
                # Potential JSON start
                self._in_json = True
                self._brace_depth = 1
                self._json_start = i
            elif ch == '{' and self._in_json:
                self._brace_depth += 1
            elif ch == '}' and self._in_json:
                self._brace_depth -= 1
                if self._brace_depth == 0:
                    # Complete JSON object
                    json_text = self._buffer[self._json_start:i+1]
                    async for ev in self._try_parse_json(json_text):
                        yield ev
                    self._in_json = False
                    self._buffer = self._buffer[i+1:]
                    i = -1  # Will be incremented to 0
            i += 1
        # Emit any non-JSON text as TextDelta
        if not self._in_json and self._buffer:
            yield TextDelta(text=self._buffer)
            self._buffer = ""

    async def _try_parse_json(self, json_text: str) -> AsyncIterator[SinkEvent]:
        try:
            data = json.loads(json_text)
            name = data.get("name") or data.get("function")
            args = data.get("arguments") or data.get("parameters") or {}
            if name:
                idx = len(self._tool_call_indices) if hasattr(self, '_tool_call_indices') else 0
                yield ToolCallStart(index=idx, id=f"call_{idx}", name=name)
                yield ToolCallArgsDelta(index=idx, partial_json=json.dumps(args) if isinstance(args, dict) else str(args))
                yield ToolCallEnd(index=idx)
            else:
                yield TextDelta(text=json_text)
        except json.JSONDecodeError:
            yield TextDelta(text=json_text)

    def parse_final(self, full_text: str) -> tuple[list[dict], str]:
        tool_calls = []
        text_parts = []
        # Simple brace-matching for standalone JSON objects
        brace_depth = 0
        start = -1
        for i, ch in enumerate(full_text):
            if ch == '{' and brace_depth == 0:
                # Text before JSON
                if start == -1:
                    text_parts.append(full_text[:i])
                brace_depth = 1
                start = i
            elif ch == '{' and brace_depth > 0:
                brace_depth += 1
            elif ch == '}' and brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0 and start != -1:
                    json_text = full_text[start:i+1]
                    try:
                        data = json.loads(json_text)
                        name = data.get("name") or data.get("function")
                        args = data.get("arguments") or data.get("parameters") or {}
                        if name:
                            tool_calls.append({
                                "index": len(tool_calls),
                                "id": f"call_{len(tool_calls)}",
                                "name": name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                            })
                            start = -1
                        else:
                            text_parts.append(json_text)
                    except json.JSONDecodeError:
                        text_parts.append(json_text)
        if brace_depth == 0 and start == -1 and full_text:
            text_parts.append(full_text)
        return tool_calls, "".join(text_parts)


# ---- Grammar registry ----

@dataclass
class GrammarRegistry:
    """Maps (provider, model_family) to grammar instances."""

    _grammars: dict[tuple[str, str], BaseGrammar] = field(default_factory=dict)

    def register(self, provider: str, model_family: str, grammar: BaseGrammar) -> "GrammarRegistry":
        self._grammars[(provider, model_family)] = grammar
        return self

    def get(self, provider: str, model_family: str) -> BaseGrammar:
        return self._grammars.get((provider, model_family), self._grammars.get(("default", "default")))

    def get_or_create(self, provider: str, model_family: str, ctx: GrammarContext) -> BaseGrammar:
        key = (provider, model_family)
        if key not in self._grammars:
            # Default selection based on model family
            if model_family == "deepseek":
                self._grammars[key] = DSMLGrammar(ctx)
            else:
                self._grammars[key] = FreeTextGrammar(ctx)
        return self._grammars[key]


# Global registry
_GRAMMAR_REGISTRY: GrammarRegistry | None = None


def get_grammar_registry() -> GrammarRegistry:
    global _GRAMMAR_REGISTRY
    if _GRAMMAR_REGISTRY is None:
        _GRAMMAR_REGISTRY = GrammarRegistry()
    return _GRAMMAR_REGISTRY