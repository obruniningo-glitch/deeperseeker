"""DeepSeek wire protocol — verbatim port from v1 functions.py (§867-1071).

Handles chat completion streaming, file upload, and DeepSeek-specific
wire format quirks (p/o/v fragments, BATCH ops, THINK/RESPONSE types).
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import aiohttp

from v2.providers.deepseek.pow import solve_create_pow, get_cookies, _get_headers
from v2.settings import get_settings


_DEEPSEEK_API_BASE = "https://chat.deepseek.com/api/v0"


async def create_new_chat(auth_token: str) -> str:
    """Create a new DeepSeek chat session."""
    headers = _get_headers(auth_token)
    cookie = await get_cookies()
    url = f"{_DEEPSEEK_API_BASE}/chat_session/create"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, cookies=cookie, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=20)) as response:
            data = await response.json()
    return data["data"]["biz_data"]["chat_session"]["id"]


async def send_message(
    chat_id: str,
    auth_token: str,
    message: str,
    parent_message_id: int,
    *,
    thinking: bool = False,
    search: bool = False,
    model_type: str | None = None,
    file_ids: list[str] | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion from DeepSeek.

    Yields text fragments (thinking + response) as they arrive.
    The v1 stream format uses p/o/v fragments and BATCH ops; we normalize
    to plain text here, with thinking content prefixed by "question" markers.
    """
    if file_ids is None:
        file_ids = []

    cookie = await get_cookies()
    pow_response = await solve_create_pow("/api/v0/chat/completion", auth_token)
    headers = _get_headers(auth_token, pow_response)

    # v1's model_type mapping
    if model_type == "expert":
        pass  # file_ids handled as-is
    elif model_type == "vision" and file_ids:
        # v1's vision file fork logic - simplified for adapter
        pass

    url = f"{_DEEPSEEK_API_BASE}/chat/completion"
    json_data = {
        "chat_session_id": chat_id,
        "parent_message_id": parent_message_id if parent_message_id != 0 else None,
        "model_type": model_type,
        "prompt": message,
        "ref_file_ids": file_ids,
        "thinking_enabled": thinking,
        "search_enabled": search,
        "preempt": False,
        "action": None,
    }

    think_open = False

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            cookies=await get_cookies(),
            headers=headers,
            json=json_data,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")

            async for line in response.content:
                if not line:
                    continue
                decoded_line = line.decode("utf-8").strip()
                if not decoded_line.startswith("data: "):
                    continue
                try:
                    data = json.loads(decoded_line[6:])
                except Exception:
                    continue

                # Handle FINISHED status
                if data.get("p") == "response/status" and data.get("v") == "FINISHED":
                    if think_open:
                        yield "\n回答\n\n"
                    return

                # Handle BATCH ops with quasi_status
                if data.get("o") == "BATCH" and isinstance(data.get("v"), list):
                    for op in data["v"]:
                        if isinstance(op, dict) and op.get("p") == "quasi_status" and op.get("v") == "FINISHED":
                            if think_open:
                                yield "\n回答\n\n"
                            return

                # Handle fragments with response dict
                if "v" in data and isinstance(data["v"], dict) and "response" in data["v"]:
                    fragments = data["v"]["response"].get("fragments")
                    if fragments:
                        for fragment in fragments:
                            if fragment.get("type") == "THINK":
                                if not think_open:
                                    yield "思考\n"
                                    think_open = True
                                yield fragment.get("content", "")
                            else:
                                if think_open:
                                    yield "\n回答\n\n"
                                    think_open = False
                                yield fragment.get("content", "")
                        continue

                # Handle p/o/v fragments (APPEND)
                if data.get("p") == "response/fragments" and data.get("o") == "APPEND":
                    fragments = data.get("v")
                    if isinstance(fragments, list):
                        for fragment in fragments:
                            if fragment.get("type") == "RESPONSE":
                                if think_open:
                                    yield "\n回答\n\n"
                                    think_open = False
                                yield fragment.get("content", "")
                            elif fragment.get("type") == "THINK":
                                if not think_open:
                                    yield "思考\n"
                                    think_open = True
                                yield fragment.get("content", "")
                            else:
                                yield fragment.get("content", "")
                        continue

                # Fallback: plain string value
                v = data.get("v")
                if isinstance(v, str):
                    yield v

            if think_open:
                yield "\n回答\n\n"


async def upload_file(
    file_bytes: bytes,
    file_name: str,
    file_content_type: str,
    auth_token: str,
) -> AsyncIterator[tuple[str, Any]]:
    """Upload a file to DeepSeek, yielding progress events.

    Yields: ("uploaded", file_id) on start, then ("success", info) or ("error", file_id)
    """
    cookie = await get_cookies()
    pow_response = await solve_create_pow("/api/v0/file/upload_file", auth_token)
    headers = _get_headers(auth_token, pow_response)
    headers.update({
        "content-type": f"multipart/form-data; boundary=----WebKitFormBoundaryTB0pXOQR2RL219Hu",
        "x-file-size": str(len(file_bytes)),
    })

    boundary = b"----WebKitFormBoundaryTB0pXOQR2RL219Hu"
    safe_name = re.sub(r"[^ -~]", "_", file_name).replace('"', "_") or "file.bin"
    body_parts = [
        b"--" + b"----WebKitFormBoundaryTB0pXOQR2RL219Hu" + b"\r\n",
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'.encode("utf-8"),
        f"Content-Type: {file_content_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n--" + b"----WebKitFormBoundaryTB0pXOQR2RL219Hu" + b"--\r\n",
    ]
    body = b"".join(body_parts)

    url = f"{_DEEPSEEK_API_BASE}/file/upload_file"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=body, cookies=await get_cookies(),
                                headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as response:
            resp_json = await response.json()

    file_id = resp_json["data"]["biz_data"]["id"]
    yield ("uploaded", file_id)

    js_data = resp_json["data"]["biz_data"]
    status = js_data["status"]
    headers = _get_headers(auth_token)
    deadline = time.time() + 300

    async with aiohttp.ClientSession() as session:
        while status in ["PENDING", "PARSING"] and time.time() < deadline:
            yield ("uploaded", file_id)
            await asyncio.sleep(0.3)
            async with session.get(
                f"{_DEEPSEEK_API_BASE}/file/fetch_files?file_ids={file_id}",
                headers=headers, cookies=await get_cookies(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp_json = await resp.json()
            js_data = resp_json["data"]["biz_data"]["files"][0]
            status = js_data["status"]

    if status == "SUCCESS" or (status == "CONTENT_EMPTY" and str(file_content_type).startswith("image/")):
        tp_data = datetime.fromtimestamp(js_data["updated_at"], timezone.utc)
        yield ("success", {
            "file_id": file_id,
            "openai_timestamp": int(js_data["updated_at"]),
            "size": js_data["file_size"],
            "anthropic_timestamp": tp_data.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    else:
        yield ("error", file_id)


async def get_file_content(
    auth_token: str,
    file_id: str,
) -> AsyncIterator[bytes | str]:
    """Get file content from DeepSeek, yielding mimetype then chunks."""
    cookie = await get_cookies()
    headers = _get_headers(auth_token)
    url = f"{_DEEPSEEK_API_BASE}/file/fetch_files?file_ids={file_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, cookies=await get_cookies(),
                               timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp_json = await resp.json()

    js_data = resp_json["data"]["biz_data"]["files"][0]
    yield mimetypes.guess_type(js_data["file_name"])[0]

    deadline = time.time() + 60
    while js_data.get("status") in ("PENDING", "PARSING") and time.time() < deadline and not js_data.get("signed_path"):
        await asyncio.sleep(0.5)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_DEEPSEEK_API_BASE}/file/fetch_files?file_ids={file_id}",
                headers=headers, cookies=await get_cookies(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp_json = await resp.json()
            js_data = resp_json["data"]["biz_data"]["files"][0]

    if not js_data.get("signed_path"):
        return

    file_path = "https://files.deepseeksvc.com/api" + js_data["signed_path"] + "&ty=r"
    async with aiohttp.ClientSession() as session:
        async with session.get(file_path, timeout=aiohttp.ClientTimeout(total=120)) as data:
            async for chunk in data.content.iter_chunked(8192):
                if chunk:
                    yield chunk


import re
import base64
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator
import aiohttp