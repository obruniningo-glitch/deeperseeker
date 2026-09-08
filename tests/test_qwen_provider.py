"""Tests for Qwen provider adapter.

Offline tests mocking the HTTP layer to verify adapter behavior.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from v2.providers.base import (
    ProviderText,
    ProviderThinking,
    ProviderFinished,
    ProviderError,
    SessionRef,
)
from v2.providers.qwen.adapter import QwenAdapter
from v2.providers.qwen.wire import send_message, create_new_chat
from v2.providers.registry import get_registry


@pytest.fixture
def qwen_adapter():
    """Create a QwenAdapter instance."""
    return QwenAdapter()


_BAXIA_BUNDLE = {"bx_ua": "test-ua", "bx_umidtoken": "T2gAv_test-baxia", "bx_v": "2.5.37"}


@pytest.fixture
def mock_cookies():
    """Mock baxia token generation to avoid Playwright/network."""
    with patch("v2.providers.qwen.wire.get_baxia_token_bundle", return_value=_BAXIA_BUNDLE),          patch("v2.providers.qwen.wire.get_baxia_token", return_value="T2gAv_test-baxia"):
        yield


class TestQwenAdapterCapabilities:
    """Test QwenAdapter capabilities configuration."""

    def test_capabilities(self, qwen_adapter):
        """Test that capabilities match the design spec."""
        assert qwen_adapter.name == "qwen"
        assert qwen_adapter.capabilities["branching"] is False
        assert qwen_adapter.capabilities["thinking"] is True
        assert qwen_adapter.capabilities["tools"] is False
        assert qwen_adapter.capabilities["attachments"] is False
        assert qwen_adapter.capabilities["search"] is True
        assert qwen_adapter.capabilities["max_prompt_tokens"] == 100_000


class TestQwenAdapterAuthenticate:
    """Test authenticate method."""

    def test_authenticate_with_valid_token(self, qwen_adapter, mock_cookies):
        """Test authenticate with a valid JWT token."""
        token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozilNpM5o"
        # Should not raise
        asyncio.run(qwen_adapter.authenticate(token))

    def test_authenticate_with_empty_token(self, qwen_adapter, mock_cookies):
        """Test authenticate with empty token raises error."""
        with pytest.raises(RuntimeError, match="token is empty"):
            asyncio.run(qwen_adapter.authenticate(""))

    def test_authenticate_with_invalid_jwt_format(self, qwen_adapter, mock_cookies):
        """Test authenticate with non-JWT token raises error."""
        token = "not-a-jwt-token"
        with pytest.raises(RuntimeError, match="does not look like a JWT"):
            asyncio.run(qwen_adapter.authenticate(token))


class TestQwenAdapterEnsureChat:
    """Test ensure_chat method."""

    @patch("v2.providers.qwen.adapter.create_new_chat")
    def test_ensure_chat_creates_new_session(self, mock_create_chat, qwen_adapter, mock_cookies):
        """Test ensure_chat creates a new chat session."""
        mock_create_chat.return_value = "chat-123"
        token = "eyJtesttoken"

        session = asyncio.run(qwen_adapter.ensure_chat(token, None))

        assert session.provider == "qwen"
        assert session.chat_id == "chat-123"
        assert session.parent_message_id == 0
        assert session.token_id == 0
        mock_create_chat.assert_called_once_with(token, "qwen3-max")

    @patch("v2.providers.qwen.wire.create_new_chat")
    def test_ensure_chat_resumes_existing_session(self, mock_create_chat, qwen_adapter, mock_cookies):
        """Test ensure_chat resumes an existing session when resume is provided."""
        token = "eyJtesttoken"
        resume = SessionRef(
            provider="qwen",
            chat_id="existing-chat-456",
            parent_message_id=5,
            token_id=42,
        )

        session = asyncio.run(qwen_adapter.ensure_chat(token, resume))

        assert session.provider == "qwen"
        assert session.chat_id == "existing-chat-456"
        assert session.parent_message_id == 5
        assert session.token_id == 42
        mock_create_chat.assert_not_called()


class TestQwenAdapterSend:
    """Test send method."""

    @pytest.fixture
    def sample_prompt(self):
        """Create a sample RenderedPrompt."""
        from v2.providers.base import RenderedPrompt
        return RenderedPrompt(
            messages=[
                {"role": "user", "content": "Hello, world!"}
            ],
            model="instant",
            metadata={
                "auth_token": "eyJtesttoken",
                "thinking": True,
                "search": False,
            }
        )

    @pytest.fixture
    def sample_session(self):
        """Create a sample SessionRef."""
        return SessionRef(
            provider="qwen",
            chat_id="chat-123",
            parent_message_id=0,
            token_id=0,
        )

    @patch("v2.providers.qwen.adapter.send_message")
    def test_send_maps_thinking_to_provider_thinking(
        self, mock_send, qwen_adapter, sample_session, sample_prompt, mock_cookies
    ):
        """Test that thinking events map to ProviderThinking."""
        async def mock_send_iterator():
            yield ("thinking", "Let me think about that...")
            yield ("response", "Here is my answer.")
            yield ("finished", None)

        mock_send.return_value = mock_send_iterator()

        events = []
        async def collect():
            async for event in qwen_adapter.send(sample_session, sample_prompt, opts={}):
                events.append(event)

        asyncio.run(collect())

        assert len(events) == 3
        assert isinstance(events[0], ProviderThinking)
        assert events[0].text == "Let me think about that..."
        assert isinstance(events[1], ProviderText)
        assert events[1].text == "Here is my answer."
        assert isinstance(events[2], ProviderFinished)

    @patch("v2.providers.qwen.adapter.send_message")
    def test_send_maps_error_to_provider_error(
        self, mock_send, qwen_adapter, sample_session, sample_prompt, mock_cookies
    ):
        """Test that error events map to ProviderError."""
        async def mock_send_iterator():
            yield ("error", "Something went wrong")

        mock_send.return_value = mock_send_iterator()

        events = []
        async def collect():
            async for event in qwen_adapter.send(sample_session, sample_prompt, opts={}):
                events.append(event)

        asyncio.run(collect())

        assert len(events) == 1
        assert isinstance(events[0], ProviderError)
        assert events[0].kind == "provider_error"
        assert events[0].message == "Something went wrong"

    @patch("v2.providers.qwen.wire.send_message")
    def test_send_handles_empty_messages(
        self, mock_send, qwen_adapter, sample_session, mock_cookies
    ):
        """Test send with empty messages yields ProviderFinished."""
        from v2.providers.base import RenderedPrompt
        prompt = RenderedPrompt(
            messages=[],
            model="instant",
            metadata={"auth_token": "eyJtesttoken"}
        )

        events = []
        async def collect():
            async for event in qwen_adapter.send(sample_session, prompt, opts={}):
                events.append(event)

        asyncio.run(collect())

        assert len(events) == 1
        assert isinstance(events[0], ProviderFinished)
        mock_send.assert_not_called()

    @patch("v2.providers.qwen.wire.send_message")
    def test_send_handles_no_user_message(
        self, mock_send, qwen_adapter, sample_session, mock_cookies
    ):
        """Test send with no user message yields ProviderFinished."""
        from v2.providers.base import RenderedPrompt
        prompt = RenderedPrompt(
            messages=[
                {"role": "assistant", "content": "I am an assistant"}
            ],
            model="instant",
            metadata={"auth_token": "eyJtesttoken"}
        )

        events = []
        async def collect():
            async for event in qwen_adapter.send(sample_session, prompt, opts={}):
                events.append(event)

        asyncio.run(collect())

        assert len(events) == 1
        assert isinstance(events[0], ProviderFinished)
        mock_send.assert_not_called()

    @patch("v2.providers.qwen.adapter.send_message")
    def test_send_passes_metadata_to_wire(
        self, mock_send, qwen_adapter, sample_session, sample_prompt, mock_cookies
    ):
        """Test that metadata parameters are passed to send_message."""
        async def mock_send_iterator():
            yield ("finished", None)

        mock_send.return_value = mock_send_iterator()

        events = []
        async def collect():
            async for event in qwen_adapter.send(sample_session, sample_prompt, opts={}):
                events.append(event)

        asyncio.run(collect())

        # Verify send_message was called with correct parameters
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["auth_token"] == "eyJtesttoken"
        assert call_kwargs["chat_id"] == "chat-123"
        assert call_kwargs["message"] == "Hello, world!"
        assert call_kwargs["thinking"] is True
        assert call_kwargs["search"] is False
        assert call_kwargs["model_type"] == "qwen3-max"


class TestQwenAdapterModelResolution:
    """Test model resolution."""

    def test_resolve_model_aliases(self, qwen_adapter):
        """Test that model aliases resolve correctly."""
        assert qwen_adapter._resolve_model("instant") == "qwen3-max"
        assert qwen_adapter._resolve_model("expert") == "qwen3-max-plus"
        assert qwen_adapter._resolve_model("qwen-instant") == "qwen3-max"
        assert qwen_adapter._resolve_model("qwen-expert") == "qwen3-max-plus"
        assert qwen_adapter._resolve_model("unknown") == "qwen3-max"


class TestQwenAdapterHealth:
    """Test health check."""

    def test_health_returns_healthy(self, qwen_adapter):
        """Test that health check returns healthy status."""
        health = asyncio.run(qwen_adapter.health())
        assert health["healthy"] is True
        assert "details" in health


class TestQwenAdapterRefreshCredentials:
    """Test refresh_credentials."""

    @patch("v2.providers.qwen.adapter._regenerate_cookies")
    def test_refresh_credentials_calls_regenerate(self, mock_regenerate, qwen_adapter):
        """Test that refresh_credentials calls _regenerate_cookies."""
        mock_regenerate.return_value = None
        asyncio.run(qwen_adapter.refresh_credentials())
        mock_regenerate.assert_called_once()


class TestRegistryIntegration:
    """Test that QwenAdapter is registered in the registry."""

    def test_registry_resolves_qwen_instant(self):
        """Test that the registry resolves "qwen-instant" to QwenAdapter."""
        registry = get_registry()
        adapter, config = registry.resolve("qwen-instant")
        assert isinstance(adapter, QwenAdapter)
        assert config.provider == "qwen"
        assert config.model_name == "qwen3-max"
        assert config.alias == "qwen-instant"

    def test_registry_resolves_qwen_expert(self):
        """Test that the registry resolves "qwen-expert" to QwenAdapter."""
        registry = get_registry()
        adapter, config = registry.resolve("qwen-expert")
        assert isinstance(adapter, QwenAdapter)
        assert config.provider == "qwen"
        assert config.model_name == "qwen3-max-plus"
        assert config.alias == "qwen-expert"

    def test_registry_has_qwen_adapter_registered(self):
        """Test that QwenAdapter is registered in the registry."""
        registry = get_registry()
        adapter = registry.get("qwen")
        assert isinstance(adapter, QwenAdapter)


class _SSEContent:
    """Async-iterable stand-in for aiohttp response.content."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


def _sse(chunks):
    return _SSEContent(chunks)


class TestWireFunctions:
    """Test the wire protocol functions."""

    @patch("aiohttp.ClientSession.post")
    def test_create_new_chat_success(self, mock_http, mock_cookies):
        """Test create_new_chat success path."""

        # Mock the response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True, "data": {"id": "chat-abc123"}})
        mock_http.return_value.__aenter__.return_value = mock_response

        chat_id = asyncio.run(create_new_chat("eyJtesttoken", "qwen3-max"))
        assert chat_id == "chat-abc123"

    @patch("aiohttp.ClientSession.post")
    def test_create_new_chat_with_data_id(self, mock_http, mock_cookies):
        """Test create_new_chat with response shape {"data": {"id": "..."}}."""

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"id": "chat-def456"}})
        mock_http.return_value.__aenter__.return_value = mock_response

        chat_id = asyncio.run(create_new_chat("eyJtesttoken", "qwen3-max"))
        assert chat_id == "chat-def456"

    @patch("aiohttp.ClientSession.post")
    def test_create_new_chat_with_chat_id(self, mock_http, mock_cookies):
        """Test create_new_chat with response shape {"chat_id": "..."}."""

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"chat_id": "chat-ghi789"})
        mock_http.return_value.__aenter__.return_value = mock_response

        chat_id = asyncio.run(create_new_chat("eyJtesttoken", "qwen3-max"))
        assert chat_id == "chat-ghi789"

    @patch("aiohttp.ClientSession.post")
    def test_create_new_chat_failure(self, mock_http, mock_cookies):
        """Test create_new_chat failure with non-200 response."""

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_http.return_value.__aenter__.return_value = mock_response

        with pytest.raises(RuntimeError, match="HTTP 500"):
            asyncio.run(create_new_chat("eyJtesttoken", "qwen3-max"))

    @patch("aiohttp.ClientSession.post")
    def test_send_message_streaming(self, mock_http, mock_cookies):
        """Test send_message streaming events."""

        # Create a mock response with async iterable content
        mock_response = AsyncMock()
        mock_response.status = 200

        # Simulate SSE events
        sse_events = [
            b'data: {"choices": [{"delta": {"reasoning_content": "Let me think..."}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "First step"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "Second step"}}]}\n\n',
            b'data: {"choices": [{"finish_reason": "stop"}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        # Create a class that implements async iteration
        class AsyncIterableContent:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        # Set up the content with the async iterable
        mock_response.content = AsyncIterableContent(sse_events)
        mock_http.return_value.__aenter__.return_value = mock_response

        events = []
        async def collect():
            async for event_type, payload in send_message(
                auth_token="eyJtesttoken",
                chat_id="chat-123",
                message="Hello",
                model_type="qwen3-max",
                thinking=True,
                search=False,
            ):
                events.append((event_type, payload))

        asyncio.run(collect())

        assert len(events) >= 4
        assert events[0] == ("thinking", "Let me think...")
        assert events[1] == ("response", "First step")
        assert events[2] == ("response", "Second step")
        # The finish_reason triggers a finished event
        assert events[3][0] == "finished"

    @patch("aiohttp.ClientSession.post")
    def test_send_message_http_error(self, mock_http, mock_cookies):
        """Test send_message handles HTTP error gracefully."""

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value="Rate limit exceeded")
        mock_http.return_value.__aenter__.return_value = mock_response

        events = []
        async def collect():
            async for event_type, payload in send_message(
                auth_token="eyJtesttoken",
                chat_id="chat-123",
                message="Hello",
            ):
                events.append((event_type, payload))

        asyncio.run(collect())

        assert len(events) == 1
        assert events[0][0] == "error"
        assert "HTTP 429" in events[0][1]


class TestQwenAdapterUploadFile:
    """Test upload_file method (deferred)."""

    def test_upload_file_not_implemented(self, qwen_adapter):
        """Test that upload_file raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            asyncio.run(qwen_adapter.upload_file("token", b"data", "file.txt", "text/plain"))


class TestQwenBaxiaTokens:
    """Baxia token generation: cache, wu.json fallback, header wiring."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        from v2.providers.qwen import cookies as qc

        qc._baxia_token_cache = None
        qc._baxia_cache_time = 0.0
        yield
        qc._baxia_token_cache = None
        qc._baxia_cache_time = 0.0

    def test_cache_reuse_within_ttl(self):
        """Second call within 25 min reuses the cached bundle (no browser)."""
        from v2.providers.qwen import cookies as qc

        bundle = {"bx_ua": "ua1", "bx_umidtoken": "T2gAv_cached", "bx_v": "2.5.37"}
        with patch.object(qc, "_primary_baxia_path", new=AsyncMock(return_value=bundle)) as prim, \
             patch.object(qc, "_fallback_wu_json_path", new=AsyncMock(return_value=None)) as fb:
            first = asyncio.run(qc.get_baxia_tokens())
            second = asyncio.run(qc.get_baxia_tokens())
        assert first == bundle and second == bundle
        assert prim.await_count == 1
        assert fb.await_count == 0

    def test_fallback_wu_json_on_primary_failure(self):
        """When the browser path fails, the wu.json path is used."""
        from v2.providers.qwen import cookies as qc

        fb_bundle = {"bx_ua": "231!T2gAv_fb", "bx_umidtoken": "T2gAv_fb", "bx_v": "2.5.36"}
        with patch.object(qc, "_primary_baxia_path", new=AsyncMock(return_value=None)), \
             patch.object(qc, "_fallback_wu_json_path", new=AsyncMock(return_value=fb_bundle)):
            got = asyncio.run(qc.get_baxia_tokens())
        assert got == fb_bundle

    def test_wu_json_regex_extraction(self):
        """wu.json body `try{umx.wu('T2gA...');}catch(e){}` yields the umid token."""
        from v2.providers.qwen.cookies import _extract_wu_token

        body = "try{umx.wu('T2gAv_abc123xyz');}catch(e){} try{__fycb('T2gAv_other');}catch(e){}"
        assert _extract_wu_token(body) == "T2gAv_abc123xyz"


class TestQwenWireShapes:
    """Request shapes per the qwen2api reference (core.js)."""

    @patch("aiohttp.ClientSession.post")
    def test_create_posts_full_body_and_headers(self, mock_http, mock_cookies):
        """chats/new is a POST with title/models/chat_mode + bx headers."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True, "data": {"id": "c1"}})
        mock_http.return_value.__aenter__.return_value = mock_response

        asyncio.run(create_new_chat("eyJtok", "qwen3-max"))

        _, kwargs = mock_http.call_args
        sent = json.loads(kwargs["json"]) if isinstance(kwargs.get("json"), str) else kwargs.get("json")
        assert sent["models"] == ["qwen3-max"]
        assert sent["chat_mode"] == "normal"
        assert "timestamp" in sent
        headers = kwargs["headers"]
        assert headers["bx-umidtoken"] == "T2gAv_test-baxia"
        assert headers["source"] == "web"
        assert "x-request-id" in headers
        assert mock_http.call_args[0][0].endswith("/chats/new")

    @patch("aiohttp.ClientSession.post")
    def test_completions_uses_query_param_and_message_envelope(self, mock_http, mock_cookies):
        """completions POSTs to ?chat_id= with the fid/childrenIds envelope."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content = _sse([b"data: [DONE]\n\n"])
        mock_http.return_value.__aenter__.return_value = mock_response

        async def collect():
            async for _ in send_message(auth_token="eyJtok", chat_id="chat-9", message="hi"):
                pass

        asyncio.run(collect())

        url = mock_http.call_args[0][0]
        assert url.endswith("/chat/completions?chat_id=chat-9")
        sent = mock_http.call_args[1]["json"]
        assert sent["chat_mode"] == "normal"
        assert sent["version"] == "2.1"
        msg = sent["messages"][0]
        assert msg["role"] == "user" and msg["content"] == "hi"
        assert "fid" in msg and "childrenIds" in msg
