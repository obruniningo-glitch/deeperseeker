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


@pytest.fixture
def mock_cookies():
    """Mock cookie functions to avoid Playwright."""
    with patch("v2.providers.qwen.adapter.get_qwen_cookies") as mock_get_cookies:
        mock_get_cookies.return_value = "T2gAv_test-token"
        with patch("v2.providers.qwen.wire.get_cookies") as mock_wire_cookies:
            mock_wire_cookies.return_value = "T2gAv_test-token"
            with patch("v2.providers.qwen.wire.get_baxia_token") as mock_baxia:
                mock_baxia.return_value = "T2gAv_test-baxia"
                with patch("v2.providers.qwen.adapter._regenerate_cookies"):
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
        mock_create_chat.assert_called_once_with(token)

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


class TestWireFunctions:
    """Test the wire protocol functions."""

    @patch("v2.providers.qwen.wire.get_baxia_token")
    @patch("v2.providers.qwen.wire.get_cookies")
    @patch("aiohttp.ClientSession.get")
    def test_create_new_chat_success(self, mock_get, mock_cookies, mock_baxia):
        """Test create_new_chat success path."""
        mock_cookies.return_value = "T2gAv_test-token"
        mock_baxia.return_value = "T2gAv_test-baxia"

        # Mock the response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"id": "chat-abc123"})
        mock_get.return_value.__aenter__.return_value = mock_response

        chat_id = asyncio.run(create_new_chat("eyJtesttoken"))
        assert chat_id == "chat-abc123"

    @patch("v2.providers.qwen.wire.get_baxia_token")
    @patch("v2.providers.qwen.wire.get_cookies")
    @patch("aiohttp.ClientSession.get")
    def test_create_new_chat_with_data_id(self, mock_get, mock_cookies, mock_baxia):
        """Test create_new_chat with response shape {"data": {"id": "..."}}."""
        mock_cookies.return_value = "T2gAv_test-token"
        mock_baxia.return_value = "T2gAv_test-baxia"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"id": "chat-def456"}})
        mock_get.return_value.__aenter__.return_value = mock_response

        chat_id = asyncio.run(create_new_chat("eyJtesttoken"))
        assert chat_id == "chat-def456"

    @patch("v2.providers.qwen.wire.get_baxia_token")
    @patch("v2.providers.qwen.wire.get_cookies")
    @patch("aiohttp.ClientSession.get")
    def test_create_new_chat_with_chat_id(self, mock_get, mock_cookies, mock_baxia):
        """Test create_new_chat with response shape {"chat_id": "..."}."""
        mock_cookies.return_value = "T2gAv_test-token"
        mock_baxia.return_value = "T2gAv_test-baxia"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"chat_id": "chat-ghi789"})
        mock_get.return_value.__aenter__.return_value = mock_response

        chat_id = asyncio.run(create_new_chat("eyJtesttoken"))
        assert chat_id == "chat-ghi789"

    @patch("v2.providers.qwen.wire.get_baxia_token")
    @patch("v2.providers.qwen.wire.get_cookies")
    @patch("aiohttp.ClientSession.get")
    def test_create_new_chat_failure(self, mock_get, mock_cookies, mock_baxia):
        """Test create_new_chat failure with non-200 response."""
        mock_cookies.return_value = "T2gAv_test-token"
        mock_baxia.return_value = "T2gAv_test-baxia"

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_get.return_value.__aenter__.return_value = mock_response

        with pytest.raises(RuntimeError, match="HTTP 500"):
            asyncio.run(create_new_chat("eyJtesttoken"))

    @patch("v2.providers.qwen.wire.get_baxia_token")
    @patch("v2.providers.qwen.wire.get_cookies")
    @patch("aiohttp.ClientSession.post")
    def test_send_message_streaming(self, mock_post, mock_cookies, mock_baxia):
        """Test send_message streaming events."""
        mock_cookies.return_value = "T2gAv_test-token"
        mock_baxia.return_value = "T2gAv_test-baxia"

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
        mock_post.return_value.__aenter__.return_value = mock_response

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

    @patch("v2.providers.qwen.wire.get_baxia_token")
    @patch("v2.providers.qwen.wire.get_cookies")
    @patch("aiohttp.ClientSession.post")
    def test_send_message_http_error(self, mock_post, mock_cookies, mock_baxia):
        """Test send_message handles HTTP error gracefully."""
        mock_cookies.return_value = "T2gAv_test-token"
        mock_baxia.return_value = "T2gAv_test-baxia"

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value="Rate limit exceeded")
        mock_post.return_value.__aenter__.return_value = mock_response

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
