"""FakeProvider — deterministic in-memory provider for offline tests.

Implements the same ProviderAdapter protocol as real providers but with
scripted responses. No network, no PoW, no cookies.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from v2.providers.base import (
    ProviderAdapter,
    Capabilities,
    SessionRef,
    SendOptions,
    ProviderEvent,
    ProviderText,
    ProviderThinking,
    ProviderFinished,
    ProviderHealth,
)
from v2.ir import Conversation, Message, ToolUseBlock, ToolResultBlock, TextBlock


@dataclass
class ScriptedTurn:
    """One scripted provider turn."""
    text: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # {"id", "name", "arguments"}
    error: Exception | None = None
    delay: float = 0.0  # simulate latency


class FakeProvider(ProviderAdapter):
    """Deterministic fake provider for offline testing.

    Scripted responses are consumed sequentially. When exhausted, returns
    a generic "I understand" response. Tool calls in the prompt are echoed
    back as results.
    """

    name = "fake"
    capabilities = Capabilities(
        branching=True,
        thinking=True,
        tools=True,
        attachments=False,
        search=False,
        max_prompt_tokens=100_000,
    )

    def __init__(self):
        self._scripts: list[ScriptedTurn] = []
        self._script_index = 0
        self._lock = asyncio.Lock()
        self._chat_counter = 0

    # ---- scriptable API ----

    def add_script(self, turn: ScriptedTurn) -> "FakeProvider":
        self._scripts.append(turn)
        return self

    def add_scripted(
        self,
        text: str = "",
        thinking: str = "",
        tool_calls: list[dict] | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> "FakeProvider":
        return self.add_script(ScriptedTurn(text, thinking, tool_calls or [], error, delay))

    def clear_scripts(self) -> "FakeProvider":
        self._scripts.clear()
        self._script_index = 0
        return self

    def reset_chat_counter(self) -> "FakeProvider":
        self._chat_counter = 0
        return self

    # ---- ProviderAdapter ----

    async def authenticate(self, token: str) -> None:
        # Always succeeds for fake
        pass

    async def ensure_chat(self, token: str, resume: SessionRef | None) -> SessionRef:
        async with self._lock:
            self._chat_counter += 1
            chat_id = f"fake-chat-{self._chat_counter}"
            parent = resume.parent_message_id if resume else None
        return SessionRef(
            provider=self.name,
            chat_id=chat_id,
            parent_message_id=parent,
            token_id=resume.token_id if resume else 0,
        )

    async def send(
        self,
        session: SessionRef,
        prompt: "RenderedPrompt",
        *,
        opts: SendOptions,
    ) -> AsyncIterator[ProviderEvent]:
        """Yield ProviderEvent stream for the scripted or auto-generated response."""
        script = None
        async with self._lock:
            if self._script_index < len(self._scripts):
                script = self._scripts[self._script_index]
                self._script_index += 1

        if script and script.error:
            raise script.error

        if script and script.delay:
            await asyncio.sleep(script.delay)

        # Emit thinking if present
        if script and script.thinking:
            for chunk in self._chunk(script.thinking):
                yield ProviderThinking(text=chunk)

        # Emit text
        text = script.text if script else self._auto_response(prompt)
        for chunk in self._chunk(text):
            yield ProviderText(text=chunk)

        # Emit tool calls if scripted
        if script and script.tool_calls:
            for i, tc in enumerate(script.tool_calls):
                # In real provider, tool calls come as separate events after text
                # For fake, we just note them in finished
                pass

        yield ProviderFinished()

    def _chunk(self, text: str, chunk_size: int = 10) -> list[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _auto_response(self, prompt: "RenderedPrompt") -> str:
        # Extract tool calls from prompt messages and echo them as results
        tool_results = []
        for msg in prompt.messages:
            if msg.role == "assistant":
                for b in msg.blocks:
                    if isinstance(b, ToolUseBlock):
                        tool_results.append(f"[Result for {b.name}: ok]")
        if tool_results:
            return "Done. " + " ".join(tool_results)
        return "I understand."

    async def upload_file(
        self, token: str, data: bytes, filename: str, media_type: str
    ) -> str:
        return f"fake-file-{hash(filename) % 10000}"

    async def refresh_credentials(self) -> None:
        pass

    async def health(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, details={"chats": self._chat_counter})

    # ---- test helpers ----

    def last_scripted(self) -> Optional[ScriptedTurn]:
        if self._script_index > 0:
            return self._scripts[self._script_index - 1]
        return None


@dataclass
class RenderedPrompt:
    """Compiled prompt ready for provider."""
    messages: list[dict]  # wire-format messages
    model: str
    tools: list[dict] | None = None
    max_tokens: int | None = None
    temperature: float | None = None