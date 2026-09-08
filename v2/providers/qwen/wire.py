"""Qwen wire protocol — async functions for Qwen's chat completion API.

Uses standard OpenAI-compatible SSE streaming at /api/v2/chat/completions.
No PoW required; uses JWT bearer token + baxia anti-bot header.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

import aiohttp

from v2.providers.qwen.cookies import get_baxia_tokens, get_baxia_token
from v2.settings import get_settings


_QWEN_API_BASE = "https://chat.qwen.ai/api/v2"
_BAXIA_HEADER = "x-baxia-token"  # Header name for the anti-bot token
_DEFAULT_BX_V = "2.5.37"
_DEFAULT_BX_V_FALLBACK = "2.5.36"


def _build_base_headers(bx_ua: str, bx_umidtoken: str, bx_v: str) -> dict:
    """Build the standard header set for Qwen API calls."""
    settings = get_settings()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "bx-ua": bx_ua,
        "bx-umidtoken": bx_umidtoken,
        "bx-v": bx_v,
        "Origin": "https://chat.qwen.ai",
        "source": "web",
        "version": "0.2.83",
        "Referer": "https://chat.qwen.ai/",
        "User-Agent": settings.QWEN_USER_AGENT or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        ),
        "Accept-Language": settings.QWEN_ACCEPT_LANGUAGE or "zh-CN,zh;q=0.9,en;q=0.8",
        "x-request-id": uuid.uuid4().hex,
    }


async def get_baxia_token_bundle() -> Optional[dict]:
    """Fetch baxia token bundle: {bx_ua, bx_umidtoken, bx_v}.

    Mirrors the get_baxia_tokens() from cookies.py. Returns None on failure.
    """
    tokens = await get_baxia_tokens()
    if tokens is not None:
        return {"bx_ua": tokens["bx_ua"], "bx_umidtoken": tokens["bx_umidtoken"], "bx_v": tokens.get("bx_v", _DEFAULT_BX_V)}
    return None


async def create_new_chat(auth_token: str, model_type: str) -> str:
    """Create a new Qwen chat session.

    POST https://chat.qwen.ai/api/v2/chats/new with the full header set
    (Accept, Content-Type, bx-ua, bx-umidtoken, bx-v, Cookie when available,
     Origin, source='web', version='0.2.83', Referer=https://chat.qwen.ai/,
     realistic desktop User-Agent, Accept-Language, x-request-id=<uuid>)
    and JSON body {title, models:[model_type], chat_mode:'normal', timestamp:<ms>,
    project_id:''}.

    Parse response {success, data:{id}} tolerantly (keep the old tolerant id
    extraction as fallback). Raise RuntimeError with status+body preview on failure.

    Args:
        auth_token: JWT bearer token (starts with "eyJ")
        model_type: Qwen model name (e.g., "qwen3-max", "qwen3-max-plus")

    Returns:
        Chat session ID string extracted from the response.

    Raises:
        RuntimeError: On HTTP error or failure to extract chat id.
    """
    # Get baxia token bundle
    bx_tokens = await get_baxia_token_bundle()
    if bx_tokens is None:
        # Fallback: try to get just the umid token via the wrapper
        bx_umidtoken = await get_baxia_token()
        bx_ua = f"231!{bx_umidtoken}" if bx_umidtoken else ""
        bx_v = _DEFAULT_BX_V_FALLBACK
    else:
        bx_ua = bx_tokens["bx_ua"]
        bx_umidtoken = bx_tokens["bx_umidtoken"]
        bx_v = bx_tokens.get("bx_v", _DEFAULT_BX_V)

    settings = get_settings()
    url = f"{_QWEN_API_BASE}/chats/new"

    # Build JSON body per core.js createChatSession
    timestamp = int(time.time() * 1000)  # milliseconds
    body = {
        "title": "新建对话",
        "models": [model_type],
        "chat_mode": "normal",
        "timestamp": timestamp,
        "project_id": "",
    }

    # Build headers
    headers = _build_base_headers(bx_ua, bx_umidtoken, bx_v)

    # Add Authorization header last (overrides any duplicate from _build_base_headers)
    headers["Authorization"] = f"Bearer {auth_token}"

    # Add Cookie jar if available
    cookies_str = await get_baxia_tokens_from_cookies()
    cookie_jar = aiohttp.CookieJar()
    if cookies_str:
        for pair in cookies_str.split("; "):
            if "=" in pair:
                key, value = pair.split("=", 1)
                cookie_jar.update_cookies({key: value})

    async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
        async with session.post(
            url,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"HTTP {response.status} creating chat: {error_text[:200]}"
                )

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


async def get_baxia_tokens_from_cookies() -> Optional[str]:
    """Get cookie string from the cookie cache (helper for wire.py)."""
    from v2.providers.qwen.cookies import get_cookies
    return await get_cookies()


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

    POST https://chat.qwen.ai/api/v2/chat/completions?chat_id=<chat_id> with
    the same header set plus 'x-accel-buffering':'no';
    body {stream:true, version:'2.1', incremental_output:true, chat_id,
    chat_mode:'normal', model:model_type, parent_id:None, messages:[{fid:<uuid>,
    parentId:None, childrenIds:[<uuid>], role:'user', content:message,
    user_action:'chat', files:[], timestamp:<ms>, models:[model_type]}]
    plus thinking_enabled/search_enabled flags when set.

    Keeps the existing SSE delta parsing (reasoning_content/content, [DONE],
    finish_reason) and the ("error",...) no-exceptions contract unchanged.

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
    # Get baxia token bundle
    bx_bundle = await get_baxia_token_bundle()
    if bx_bundle is None:
        # Fallback: try to get just the umid token via the wrapper
        bx_umidtoken = await get_baxia_token()
        bx_ua = f"231!{bx_umidtoken}" if bx_umidtoken else ""
        bx_v = _DEFAULT_BX_V_FALLBACK
    else:
        bx_ua = bx_bundle["bx_ua"]
        bx_umidtoken = bx_bundle["bx_umidtoken"]
        bx_v = bx_bundle.get("bx_v", _DEFAULT_BX_V)

    settings = get_settings()
    url = f"{_QWEN_API_BASE}/chat/completions?chat_id={chat_id}"

    # Build request body per core.js
    timestamp = int(time.time() * 1000)  # milliseconds
    body = {
        "stream": True,
        "incremental_output": True,
        "version": "2.1",
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model_type,
        "parent_id": None,
        "messages": [
            {
                "fid": uuid.uuid4().hex,
                "parentId": None,
                "childrenIds": [uuid.uuid4().hex],
                "role": "user",
                "content": message,
                "user_action": "chat",
                "files": [],
                "timestamp": timestamp,
                "models": [model_type],
            }
        ],
        "chat_mode": "normal",
    }

    if thinking:
        body["thinking_enabled"] = True

    if search:
        body["search_enabled"] = True

    if parent_message_id:
        body["parent_message_id"] = parent_message_id

    headers = _build_base_headers(bx_ua, bx_umidtoken, bx_v)
    headers["x-accel-buffering"] = "no"
    headers["Authorization"] = f"Bearer {auth_token}"
    headers["Accept"] = "text/event-stream"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ("error", f"HTTP {response.status}: {error_text[:200]}")
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

    except asyncio.TimeoutError:
        yield ("error", "Request timeout")
    except aiohttp.ClientError as e:
        yield ("error", f"Connection error: {str(e)}")
    except Exception as e:
        yield ("error", f"Unexpected error: {str(e)}")