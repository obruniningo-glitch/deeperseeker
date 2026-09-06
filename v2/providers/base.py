"""ProviderAdapter protocol and shared types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, NamedTuple, Optional, Protocol, TypedDict


class Capabilities(TypedDict, total=False):
    branching: bool          # server-side conversation tree (DeepSeek: parent_message_id)
    thinking: bool           # provider emits reasoning/thinking deltas
    tools: bool              # False => pipeline uses prompt injection + grammar
    attachments: bool
    search: bool
    max_prompt_tokens: Optional[int]


class SessionRef(NamedTuple):
    provider: str
    chat_id: str
    parent_message_id: Optional[int]
    token_id: int


class SendOptions(TypedDict, total=False):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]


# Provider-side events (what the adapter yields)
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


class ProviderHealth(TypedDict, total=False):
    healthy: bool
    details: dict


class ProviderAdapter(Protocol):
    """Protocol for provider adapters.

    Each adapter implements the wire protocol for a specific provider
    (DeepSeek, Kimi, OpenAI API, etc.) and translates to/from our
    canonical events and session references.
    """

    name: str
    capabilities: Capabilities

    async def authenticate(self, token: str) -> None:
        """Verify token works (called at pool-add time)."""
        ...

    async def ensure_chat(
        self, token: str, resume: SessionRef | None
    ) -> SessionRef:
        """Create or resume a provider-side chat session.

        Returns a SessionRef with the provider's chat_id and the
        parent_message_id for branching.
        """
        ...

    def send(
        self,
        session: SessionRef,
        prompt: "RenderedPrompt",
        *,
        opts: SendOptions,
    ) -> AsyncIterator[ProviderEvent]:
        """Stream provider response as ProviderEvent objects.

        The pipeline will transform these into SinkEvents via the grammar.
        """
        ...

    async def upload_file(
        self, token: str, data: bytes, filename: str, media_type: str
    ) -> str:
        """Upload a file; returns provider file reference."""
        ...

    async def refresh_credentials(self) -> None:
        """Refresh cookies/auth (for cookie-based providers)."""
        ...

    async def health(self) -> ProviderHealth:
        """Health check for this provider."""
        ...


# Pipeline types (what the orchestrator produces)

@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Compiled prompt ready for provider adapter."""
    messages: list[dict]  # wire-format messages for the provider
    model: str
    tools: list[dict] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: dict | None = None