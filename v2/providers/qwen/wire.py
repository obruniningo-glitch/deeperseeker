"""Qwen wire protocol — async functions for Qwen's chat completion API.

Uses standard OpenAI-compatible SSE streaming at /api/v2/chat/completions.
No PoW required; uses JWT bearer token + baxia anti-bot header.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import aiohttp

from v2.providers.qwen.cookies import get_baxia_token, get_cookies
from v2.settings import get_settings


_QWEN_API_BASE = "https://chat.qwen.ai/api/v2"
_BAXIA_HEADER = "x-baxia-token"  # Header name for the anti-bot token


async def create_new_chat(auth_token: str) -> str:
    """Create a new Qwen chat session.

    GET /api/v2/chats/new with Authorization: Bearer <token>.
    Returns the new chat id from the JSON response.
    Inspects response shape tolerantly: keys like "id", "chat_id", "data.id".
    """
    settings = get_settings()
    url = f"{_QWEN_API_BASE}/chats/new"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Add baxia token if available
    baxia = await get_baxia_token()
    if baxia:
        headers[_BAXIA_HEADER] = baxia

    # Add cookies for session context
    cookies_str = await get_cookies()
    cookie_jar = aiohttp.CookieJar()
    if cookies_str:
        # Parse semicolon-separated cookies
        for pair in cookies_str.split("; "):
            if "=" in pair:
                key, value = pair.split("=", 1)
                cookie_jar.update_cookies({key: value})

    async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status} creating chat: {error_text}")

            data = await response.json()

    # Tolerantly extract chat id from various response shapes
    # Expected: {"id": "..."} or {"chat_id": "..."} or {"data": {"id": "..."}}
    chat_id = None
    if isinstance(data, dict):
        if "id" in data and isinstance(data["id"], str):
            chat_id = data["id"]
        elif "chat_id" in data and isinstance(data["chat_id"], str):
            chat_id = data["chat_id"]
        elif "data" in data and isinstance(data["data"], dict):
            if "id" in data["data"] and isinstance(data["data"]["id"], str):
                chat_id = data["data"]["id"]
            elif "chat_id" in data["data"] and isinstance(data["data"]["chat_id"], str):
                chat_id = data["data"]["chat_id"]

    if not chat_id:
        raise RuntimeError(f"Could not extract chat id from response: {data}")

    return chat_id


async def send_message(
    auth_token: str,
    chat_id: str,
    message: str,
    model_type: str = "qwen3-max",
    thinking: bool = False,
    search: bool = False,
    parent_message_id: str | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Send a message to Qwen and stream the response via SSE.

    POST /api/v2/chat/completions with streaming enabled.
    Yields tuples: ("thinking", text) for reasoning content,
                    ("response", text) for final answer content,
                    ("finished", None) on stream end,
                    ("error", message) on failures.
    No exceptions cross the iterator boundary — all failures become ("error", ...).

    Args:
        auth_token: JWT bearer token (starts with "eyJ")
        chat_id: Chat session ID from create_new_chat
        message: User message text
        model_type: Qwen model name (e.g., "qwen3-max", "qwen3-max-plus")
        thinking: Enable reasoning/thinking deltas
        search: Enable web search
        parent_message_id: Optional parent message ID for branching

    Yields:
        Stream events as (event_type, payload) tuples.
    """
    settings = get_settings()
    url = f"{_QWEN_API_BASE}/chat/completions"

    # Build request body
    body = {
        "stream": True,
        "incremental_output": True,
        "model": model_type,
        "messages": [{"role": "user", "content": message}],
        "chat_id": chat_id,
        "chat_mode": "normal",
    }

    if thinking:
        body["thinking_enabled"] = True

    if search:
        body["search_enabled"] = True

    if parent_message_id:
        body["parent_message_id"] = parent_message_id

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    # Add baxia token if available
    baxia = await get_baxia_token()
    if baxia:
        headers[_BAXIA_HEADER] = baxia

    # Add cookies for session context
    cookies_str = await get_cookies()
    cookie_jar = aiohttp.CookieJar()
    if cookies_str:
        for pair in cookies_str.split("; "):
            if "=" in pair:
                key, value = pair.split("=", 1)
                cookie_jar.update_cookies({key: value})

    try:
        async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
            async with session.post(
                url,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ("error", f"HTTP {response.status}: {error_text}")
                    return

                async for line in response.content:
                    if not line:
                        continue

                    try:
                        decoded = line.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        continue

                    if not decoded.startswith("data: "):
                        continue

                    data_str = decoded[6:].strip()
                    if data_str == "[DONE]":
                        yield ("finished", None)
                        return

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Parse OpenAI-compatible delta format
                    # Example: {"choices": [{"delta": {"reasoning_content": "...", "content": "..."}}]}
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    if not delta:
                        continue

                    # Check for reasoning/thinking content
                    reasoning = delta.get("reasoning_content")
                    if reasoning and isinstance(reasoning, str) and reasoning.strip():
                        yield ("thinking", reasoning)

                    # Check for response content
                    content = delta.get("content")
                    if content and isinstance(content, str) and content.strip():
                        yield ("response", content)

                    # Check for finish_reason
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        yield ("finished", None)
                        return

                # If we reach here without a [DONE] or finish_reason, still finish
                yield ("finished", None)

    except asyncio.TimeoutError:
        yield ("error", "Request timeout")
    except aiohttp.ClientError as e:
        yield ("error", f"Connection error: {str(e)}")
    except Exception as e:
        yield ("error", f"Unexpected error: {str(e)}")
