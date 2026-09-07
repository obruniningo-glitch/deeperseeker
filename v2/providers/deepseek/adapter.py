"""DeepSeek provider adapter — implements ProviderAdapter protocol.

Ports v1's DeepSeek wire logic (functions.py) into the v2 ProviderAdapter interface.
"""
from __future__ import annotations

from typing import AsyncIterator

from v2.providers.base import ProviderAdapter, Capabilities, SessionRef, SendOptions, ProviderEvent, ProviderText, ProviderThinking, ProviderFinished, ProviderHealth
from v2.providers.deepseek.wire import send_message, create_new_chat, upload_file, get_file_content
from v2.providers.deepseek.cookies import get_cookies as get_deepseek_cookies
from v2.settings import get_settings


class DeepSeekAdapter(ProviderAdapter):
    """DeepSeek web-chat provider adapter.

    Implements the ProviderAdapter protocol by wrapping the v1 wire logic.
    """

    name = "deepseek"
    capabilities = Capabilities(
        branching=True,
        thinking=True,
        tools=True,  # DeepSeek supports tools via prompt injection
        attachments=True,
        search=True,
        max_prompt_tokens=100_000,
    )

    def __init__(self):
        self._settings = get_settings()

    async def authenticate(self, token: str) -> None:
        """Verify token works by making a lightweight API call."""
        # DeepSeek auth is bearer token + PoW + cookies; just check cookie availability
        try:
            await get_deepseek_cookies()
        except Exception as e:
            raise RuntimeError(f"DeepSeek auth failed: {e}")

    async def ensure_chat(self, token: str, resume: SessionRef | None) -> SessionRef:
        """Create or resume a DeepSeek chat session."""
        if resume and resume.chat_id:
            # Resume existing session
            return SessionRef(
                provider=self.name,
                chat_id=resume.chat_id,
                parent_message_id=resume.parent_message_id,
                token_id=resume.token_id,
            )
        # Create new chat session
        chat_id = await create_new_chat(token)
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
        """Stream a chat completion via DeepSeek wire protocol."""
        from v2.providers.deepseek.wire import send_message

        # Build the prompt text from rendered messages
        # For DeepSeek, we send the full conversation as the prompt
        messages = prompt.messages
        if not messages:
            yield ProviderText(text="")
            yield ProviderFinished()
            return

        # Extract the last user message as the prompt
        # DeepSeek expects the full conversation in its web session
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        # Model mapping
        model_type = self._resolve_model(prompt.model)

        try:
            async for chunk in send_message(
                chat_id=session.chat_id,
                auth_token=prompt.metadata.get("auth_token", "") if prompt.metadata else "",
                message=last_user,
                parent_message_id=session.parent_message_id,
                thinking=prompt.metadata.get("thinking", False) if prompt.metadata else False,
                search=prompt.metadata.get("search", False) if prompt.metadata else False,
                model_type=model_type,
                file_ids=prompt.metadata.get("file_ids", []) if prompt.metadata else [],
            ):
                # Determine if chunk is thinking or response
                if chunk.startswith("思考\n") or chunk.startswith("thinking\n"):
                    yield ProviderThinking(text=chunk.replace("思考\n", "").replace("thinking\n", ""))
                elif chunk.startswith("回答\n\n") or chunk.startswith("answer\n\n"):
                    yield ProviderText(text=chunk.replace("回答\n\n", "").replace("answer\n\n", ""))
                else:
                    yield ProviderText(text=chunk)
        except Exception as e:
            yield ProviderError(kind="provider_error", message=str(e))
            return

        yield ProviderFinished()

    async def upload_file(
        self, token: str, data: bytes, filename: str, media_type: str
    ) -> str:
        from v2.providers.deepseek.wire import upload_file
        async for event_type, payload in upload_file(data, filename, media_type, token):
            if event_type == "success":
                return payload["file_id"]
            elif event_type == "error":
                raise RuntimeError(f"File upload failed: {payload}")
        raise RuntimeError("File upload did not complete")

    async def get_file_content(
        self, token: str, file_id: str
    ) -> AsyncIterator[bytes | str]:
        from v2.providers.deepseek.wire import get_file_content
        async for chunk in get_file_content(token, file_id):
            if isinstance(chunk, str):
                yield chunk  # mimetype
            else:
                yield chunk  # bytes

    async def refresh_credentials(self) -> None:
        """Refresh DeepSeek cookies via Playwright."""
        from v2.providers.deepseek.cookies import _regenerate_cookies
        await _regenerate_cookies()

    async def health(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, details={})

    def _resolve_model(self, model: str) -> str:
        """Map v2 model alias to DeepSeek model_type."""
        model_map = {
            "instant": "deepseek-v4-flash",
            "expert": "deepseek-v4-pro",
            "vision": "deepseek-v4-pro",
        }
        return model_map.get(model, "deepseek-v4-flash")


import re