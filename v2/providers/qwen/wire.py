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

from curl_cffi.requests import AsyncSession

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


def _parse_sse_line(raw_line: bytes) -> list:
    """Parse one raw SSE line into wire events.

    Tolerates split/joined chunks ("data:" with or without trailing space,
    multi-line JSON is handled by the caller's buffer, one line at a time).
    Returns a list of ("thinking"|"response"|"finished", payload) tuples.
    """
    try:
        decoded = raw_line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return []
    if not decoded.startswith("data:"):
        return []
    data_str = decoded[5:].strip()
    if data_str == "[DONE]":
        return [("finished", None)]
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return []
    # Parse OpenAI-compatible delta format
    # Example: {"choices": [{"delta": {"reasoning_content": "...", "content": "..."}}]}
    choices = data.get("choices", [])
    if not choices:
        return []
    events = []
    delta = choices[0].get("delta", {})
    if delta:
        reasoning = delta.get("reasoning_content")
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            events.append(("thinking", reasoning))
        content = delta.get("content")
        if content and isinstance(content, str) and content.strip():
            events.append(("response", content))
    finish_reason = choices[0].get("finish_reason")
    if finish_reason:
        events.append(("finished", None))
    return events


def _parse_cookies(cookie_str: str) -> dict:
    """Convert a semicolon-separated cookie string into a dict for curl_cffi."""
    ck: dict = {}
    if not cookie_str:
        return ck
    for pair in cookie_str.split(";"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            if key.strip() and value.strip():
                ck[key.strip()] = value.strip()
    return ck


async def _http_post_json(
    session: AsyncSession,
    url: str,
    headers: dict,
    json_body: dict,
    timeout: float,
) -> dict:
    """Perform a POST with JSON body using curl_cffi, returning the parsed JSON response.

    Raises RuntimeError on non-200 status, mirroring the original aiohttp behavior.
    """
    resp = await session.post(
        url,
        headers=headers,
        json=json_body,
        timeout=timeout,
    )
    if resp.status_code != 200:
        error_text = resp.text[:200]
        raise RuntimeError(
            f"HTTP {resp.status_code} creating chat: {error_text}"
        )
    return resp.json()


_PUNISH_MARKERS = (b"_____tmd_____/punish", b"FAIL_SYS_USER_VALIDATE", b"RGV587")


def is_punish_response(body: bytes) -> bool:
    """Detect the Alibaba WAF slider-challenge page in a response body."""
    return any(m in body for m in _PUNISH_MARKERS)


async def _http_post_stream(
    session: AsyncSession,
    url: str,
    headers: dict,
    json_body: dict,
    timeout: float,
    on_event,
) -> None:
    """Perform a POST with streaming SSE response using curl_cffi.

    Feeds received bytes through the existing _parse_sse_line buffered parser
    unchanged. Calls on_event(event_type, payload) for each parsed event.

    Timeout yields ("error", "Request timeout") on asyncio.TimeoutError.
    Curl errors yield ("error", f"Connection error: ...").
    """
    resp = await session.post(
        url,
        headers=headers,
        json=json_body,
        timeout=timeout,
        stream=True,
    )
    if resp.status_code != 200:
        error_text = resp.text[:200]
        on_event(("error", f"HTTP {resp.status_code}: {error_text}"))
        return

    buf = b""
    async for raw_line in resp.aiter_lines():
        buf += raw_line + b"\n"
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            for event in _parse_sse_line(line):
                if event[0] == "finished":
                    on_event(event)
                    return
                on_event(event)

    # Flush any remaining buffer
    if buf.strip():
        for event in _parse_sse_line(buf):
            if event[0] == "finished":
                on_event(event)
                return
            on_event(event)

    # If we reach here without a [DONE] or finish_reason, still finish
    on_event(("finished", None))


async def get_baxia_token_bundle() -> Optional[dict]:
    """Fetch baxia token bundle: {bx_ua, bx_umidtoken, bx_v, cookies}.

    Mirrors the get_baxia_tokens() from cookies.py. Returns None on failure.
    """
    tokens = await get_baxia_tokens()
    if tokens is not None:
        return {"bx_ua": tokens["bx_ua"], "bx_umidtoken": tokens["bx_umidtoken"],
                "bx_v": tokens.get("bx_v", _DEFAULT_BX_V),
                "cookies": tokens.get("cookies", "")}
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

    # Attach the real-browser cookies harvested alongside the baxia tokens
    # (qwen2api sends document.cookie as the Cookie header).
    cookie_jar_dict = _parse_cookies(
        (bx_tokens or {}).get("cookies", "") if isinstance(bx_tokens, dict) else ""
    )
    if not cookie_jar_dict:
        cookie_str_from_api = await get_baxia_tokens_from_cookies()
        if cookie_str_from_api:
            cookie_jar_dict = _parse_cookies(cookie_str_from_api)

    # Create curl_cffi session with Chrome impersonation and cookies
    async with AsyncSession(impersonate="chrome150", cookies=cookie_jar_dict) as session:
        result = await _http_post_json(
            session, url, headers, body, 20,
        )

    # Tolerantly extract chat id from various response shapes
    # Expected: {"id": "..."} or {"chat_id": "..."} or {"data": {"id": "..."}}
    chat_id = None
    if isinstance(result, dict):
        if "id" in result and isinstance(result["id"], str):
            chat_id = result["id"]
        elif "chat_id" in result and isinstance(result["chat_id"], str):
            chat_id = result["chat_id"]
        elif "data" in result and isinstance(result["data"], dict):
            if "id" in result["data"] and isinstance(result["data"]["id"], str):
                chat_id = result["data"]["id"]
            elif "chat_id" in result["data"] and isinstance(result["data"]["chat_id"], str):
                chat_id = result["data"]["chat_id"]

    if not chat_id:
        raise RuntimeError(f"Could not extract chat id from response: {result}")

    return chat_id


async def get_baxia_tokens_from_cookies() -> Optional[str]:
    """Get cookie string from the cookie cache (helper for wire.py)."""
    from v2.providers.qwen.cookies import get_cookies
    return await get_cookies()


async def _collect_once(session, url, headers, body, timeout) -> tuple[list, bytes]:
    """Single completions attempt: returns (events, raw_body_for_punish_check)."""
    from curl_cffi.requests import AsyncSession as _S  # noqa (type clarity)

    collected: list = []
    raw = b""
    resp = await session.post(url, headers=headers, json=body, timeout=timeout, stream=True)
    if resp.status_code != 200:
        collected.append(("error", f"HTTP {resp.status_code}: {resp.text[:200]}"))
        return collected, raw
    async for chunk in resp.aiter_content():
        raw += chunk
    text = raw.decode("utf-8", "replace")
    for line in text.split("\n"):
        for event in _parse_sse_line(line.encode("utf-8", "replace")):
            collected.append(event)
            if event[0] == "finished":
                return collected, raw
    if not any(e[0] == "finished" for e in collected):
        collected.append(("finished", None))
    return collected, raw


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

    # Same-page cookies: the document.cookie harvested alongside the baxia
    # tokens must ride on the completions call (qwen2api sends it as Cookie).
    cookie_jar_dict = _parse_cookies(
        (bx_bundle or {}).get("cookies", "") if isinstance(bx_bundle, dict) else ""
    )
    if not cookie_jar_dict:
        cookie_str_from_api = await get_baxia_tokens_from_cookies()
        if cookie_str_from_api:
            cookie_jar_dict = _parse_cookies(cookie_str_from_api)

    collected: list = []

    async with AsyncSession(impersonate="chrome150", cookies=cookie_jar_dict) as session:
        collected, raw = await _collect_once(session, url, headers, body, 300)
        if is_punish_response(raw):
            # WAF slider challenge: drop the stale baxia bundle + cookies and
            # re-harvest once, then retry a single time (qwengate recipe).
            from v2.providers.qwen import cookies as _qc

            _qc._baxia_token_cache = None
            _qc._baxia_cache_time = 0.0
            fresh = await get_baxia_token_bundle()
            if fresh is not None:
                headers["bx-ua"] = fresh["bx_ua"]
                headers["bx-umidtoken"] = fresh["bx_umidtoken"]
                headers["bx-v"] = fresh.get("bx_v", _DEFAULT_BX_V)
                jar2 = _parse_cookies(fresh.get("cookies", ""))
                if jar2:
                    cookie_jar_dict = jar2
                    session.cookies.update(jar2)
                collected, raw = await _collect_once(session, url, headers, body, 300)
                if is_punish_response(raw):
                    collected = [("error", "WAF slider challenge (punish) after refresh; "
                                           "solve it in the browser, then retry")]

    for event in collected:
        yield event