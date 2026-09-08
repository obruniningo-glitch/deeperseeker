"""Qwen provider adapter — implements ProviderAdapter protocol."""
from __future__ import annotations

from typing import AsyncIterator

from v2.providers.base import (
    ProviderAdapter,
    Capabilities,
    SessionRef,
    SendOptions,
    ProviderEvent,
    ProviderText,
    ProviderThinking,
    ProviderFinished,
    ProviderError,
    ProviderHealth,
)
from v2.providers.qwen.wire import send_message, create_new_chat
from v2.providers.qwen.cookies import get_cookies as get_qwen_cookies, _regenerate_cookies
from v2.settings import get_settings


class QwenAdapter(ProviderAdapter):
    """Qwen web-chat provider adapter.

    Implements the ProviderAdapter protocol by wrapping Qwen's wire logic.
    Capabilities: thinking=True, tools=False (grammar injection handles tools),
    attachments=False (deferred), search=True, branching=False,
    max_prompt_tokens=100_000.
    """

    name = "qwen"
    capabilities = Capabilities(
        branching=False,
        thinking=True,
        tools=False,  # Grammar injection handles tools upstream
        attachments=False,  # File upload deferred
        search=True,
        max_prompt_tokens=100_000,
    )

    def __init__(self):
        self._settings = get_settings()

    async def authenticate(self, token: str) -> None:
        """Verify token works by checking cookie availability and token format.

        Qwen uses JWT bearer tokens. Verify the token looks like a JWT
        and that we can get cookies (including baxia token).
        """
        if not token:
            raise RuntimeError("Qwen auth failed: token is empty")

        # Verify token looks like a JWT (starts with "eyJ" and has 3 parts)
        if not token.startswith("eyJ"):
            raise RuntimeError("Qwen auth failed: token does not look like a JWT (should start with 'eyJ')")

        # Check we can get cookies (including baxia token)
        try:
            cookies = await get_qwen_cookies()
            if not cookies:
                # Cookies unavailable; this might be okay if we only need the JWT,
                # but we'll log it. For now, proceed.
                pass
        except Exception as e:
            raise RuntimeError(f"Qwen auth failed: {e}")

    async def ensure_chat(self, token: str, resume: SessionRef | None) -> SessionRef:
        """Create or resume a Qwen chat session.

        Qwen uses explicit chat creation via /api/v2/chats/new.
        If resume is provided, reuse it.
        """
        if resume and resume.chat_id:
            # Resume existing session
            return SessionRef(
                provider=self.name,
                chat_id=resume.chat_id,
                parent_message_id=resume.parent_message_id,
                token_id=resume.token_id,
            )

        # Create new chat session (default model; send() re-resolves per request)
        chat_id = await create_new_chat(token, "qwen3-max")
        return SessionRef(
            provider=self.name,
            chat_id=chat_id,
            parent_message_id=0,
            token_id=resume.token_id if resume else 0,
        )

    async def send(
        self,
        session: SessionRef,
        prompt: "RenderedPrompt",
        *,
        opts: SendOptions,
    ) -> AsyncIterator[ProviderEvent]:
        """Stream a chat completion via Qwen wire protocol.

        Maps wire events to ProviderText/ProviderThinking/ProviderFinished/ProviderError.
        """
        # Build the prompt text from rendered messages
        messages = prompt.messages
        if not messages:
            yield ProviderFinished()
            return

        # Extract the last user message as the prompt
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        if not last_user:
            yield ProviderFinished()
            return

        # Model mapping
        model_type = self._resolve_model(prompt.model)

        # Extract metadata
        thinking = False
        search = False
        parent_message_id = None
        auth_token = ""

        if prompt.metadata:
            thinking = prompt.metadata.get("thinking", False)
            search = prompt.metadata.get("search", False)
            parent_message_id = prompt.metadata.get("parent_message_id")
            auth_token = prompt.metadata.get("auth_token", "")

        # If auth_token is not in metadata, use the session token
        # (the token is passed separately to ensure_chat but not stored in SessionRef)
        # We need the token for send; it should be in metadata
        if not auth_token:
            # Try to get it from the session's token_id (we store token_id but not the token itself)
            # For now, raise a clear error
            raise RuntimeError("auth_token not found in prompt.metadata; ensure_chat must store it")

        try:
            async for event_type, payload in send_message(
                auth_token=auth_token,
                chat_id=session.chat_id,
                message=last_user,
                model_type=model_type,
                thinking=thinking,
                search=search,
                parent_message_id=str(session.parent_message_id) if session.parent_message_id else None,
            ):
                if event_type == "thinking":
                    yield ProviderThinking(text=payload)
                elif event_type == "response":
                    yield ProviderText(text=payload)
                elif event_type == "finished":
                    yield ProviderFinished()
                    return
                elif event_type == "error":
                    yield ProviderError(kind="provider_error", message=payload)
                    return
                else:
                    # Unknown event type
                    yield ProviderError(kind="unknown_event", message=f"Unknown event: {event_type}")
                    return
        except Exception as e:
            yield ProviderError(kind="provider_error", message=str(e))
            return

        yield ProviderFinished()

    async def upload_file(
        self, token: str, data: bytes, filename: str, media_type: str
    ) -> str:
        """Upload a file to Qwen.

        Deferred implementation; not yet supported.
        """
        raise NotImplementedError("File upload not yet implemented for Qwen provider")

    async def refresh_credentials(self) -> None:
        """Refresh Qwen cookies via Playwright."""
        await _regenerate_cookies()

    async def health(self) -> ProviderHealth:
        """Health check for Qwen provider.

        Returns minimal healthy status. Actual API health could be checked
        by making a lightweight API call, but for now return healthy.
        """
        return ProviderHealth(healthy=True, details={})

    def _resolve_model(self, model: str) -> str:
        """Map v2 model alias to Qwen model_type.

        Aliases:
            - "qwen-instant" -> "qwen3-max"
            - "qwen-expert"  -> "qwen3-max-plus"
            - "instant"      -> "qwen3-max"
            - "expert"       -> "qwen3-max-plus"
        """
        model_map = {
            "instant": "qwen3-max",
            "expert": "qwen3-max-plus",
            "qwen-instant": "qwen3-max",
            "qwen-expert": "qwen3-max-plus",
        }
        return model_map.get(model, "qwen3-max")
