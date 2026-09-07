"""DeepSeek PoW solver — verbatim port from v1 functions.py (§797-864).

Uses wasmtime to run the DeepSeekHashV1 reference algorithm packaged as a
.wasm module. Correctness by construction — the .wasm IS the algorithm.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any

import wasmtime

from v2.settings import get_settings

# Path to the DeepSeekHashV1 .wasm (same as v1)
_wasm_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "deepseek_hash_v1.wasm")


_pow_setup: tuple[wasmtime.Engine, wasmtime.Module, wasmtime.Linker] | None = None


def _get_pow_setup() -> tuple[wasmtime.Engine, wasmtime.Module, wasmtime.Linker]:
    global _pow_setup
    if _pow_setup is None:
        engine = wasmtime.Engine()
        with open(_wasm_path, "rb") as f:
            module = wasmtime.Module(engine, f.read())
        _pow_setup = (engine, module, wasmtime.Linker(engine))
    return _pow_setup


def _write_string_pow(text: str, alloc_func: Any, memory: Any, store: Any) -> tuple[int, int]:
    data = text.encode("utf-8")
    ptr = alloc_func(store, len(data))
    mem = memory.data_ptr(store)
    for i in range(len(data)):
        mem[ptr + i] = data[i]
    return ptr, len(data)


def _find_pow_answer_blocking(challenge_data: dict[str, Any]) -> int | None:
    """Blocking PoW solve (runs in thread pool)."""
    engine, module, linker = _get_pow_setup()
    store = wasmtime.Store(engine)
    instance = linker.instantiate(store, module)
    memory = instance.exports(store)["memory"]
    alloc_func = instance.exports(store)["alloc"]
    solve_func = instance.exports(store)["solve_pow"]

    ch_ptr, ch_len = _write_string_pow(challenge_data["challenge"], alloc_func, memory, store)
    salt_ptr, salt_len = _write_string_pow(challenge_data["salt"], alloc_func, memory, store)

    result = solve_func(store, ch_ptr, ch_len, salt_ptr, salt_len,
                        challenge_data["expire_at"], challenge_data["difficulty"])

    if result < 0:
        result = result + 0x10000000000000000
    return result if result != 0xFFFFFFFFFFFFFFFF else None


async def find_pow_answer(challenge_data: dict[str, Any]) -> int | None:
    """Async wrapper for blocking PoW solve."""
    return await asyncio.to_thread(_find_pow_answer_blocking, challenge_data)


async def create_challenge_pow(target_path: str, auth_token: str) -> dict[str, Any]:
    """Request a PoW challenge from DeepSeek."""
    settings = get_settings()
    headers = _get_headers(auth_token)
    cookie = await get_cookies()

    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{settings.DEEPSEEK_API_BASE}/chat/create_pow_challenge",
            cookies=cookie,
            headers=headers,
            json={"target_path": target_path},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            data = await response.json()
    return data["data"]["biz_data"]["challenge"]


async def solve_create_pow(target_path: str, auth_token: str) -> str:
    """Create challenge and solve PoW, return base64-encoded response."""
    challenge = await create_challenge_pow(target_path, auth_token)
    answer = await find_pow_answer(challenge)
    if answer is None:
        raise RuntimeError("PoW solve failed")
    payload = {
        "algorithm": "DeepSeekHashV1",
        "challenge": challenge["challenge"],
        "salt": challenge["salt"],
        "answer": answer,
        "signature": challenge["signature"],
        "target_path": target_path,
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def _get_headers(auth_token: str, pow_response: str | None = None) -> dict[str, str]:
    settings = get_settings()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": settings.DEEPSEEK_ORIGIN,
        "Referer": f"{settings.DEEPSEEK_ORIGIN}/",
        "Content-Type": "application/json",
    }
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    if pow_response:
        headers["x-ds-pow-response"] = pow_response
    return headers


async def get_cookies() -> str:
    """Get valid DeepSeek cookies (cached or regenerated)."""
    try:
        cookie_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aws_cookies_deepseek.json")
        if os.path.exists(cookie_path):
            with open(cookie_path) as f:
                cookies = json.load(f)
            cookie = _decode_cookie_value(cookies.get("cookie"))
            if cookies.get("expiry") is not None and cookies["expiry"] > time.time() and cookie is not None:
                return cookie
    except Exception:
        pass

    # Regenerate
    async with _cookie_lock:
        return await _regenerate_cookies()


_cookie_lock = asyncio.Lock()


async def _regenerate_cookies() -> str:
    """Regenerate cookies via Playwright (headless)."""
    from playwright.async_api import async_playwright

    cookie_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aws_cookies_deepseek.json")

    async with async_playwright() as p:
        launch_kwargs = {"headless": True}
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            launch_kwargs["args"] = ["--no-sandbox"]

        async with p.chromium.launch(**launch_kwargs) as browser:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto("https://chat.deepseek.com/", wait_until="domcontentloaded")
            await page.wait_for_selector("body")
            try:
                await page.wait_for_url("**/sign_in*", timeout=30000)
            except Exception:
                pass
            cookies = await context.cookies()

        final_cookies = {}
        expiry = None
        for c in cookies:
            if c.get("name") == "aws-waf-token":
                expiry = c.get("expires")
            final_cookies[c["name"]] = c["value"]
        final_cookies["ds_cookie_preference"] = "%257B%2522level%2522%253A%2522all%2522%257D"
        if not expiry or expiry < 0:
            expiry = time.time() + 1800

        tmp_path = cookie_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"cookie": _encode_cookie_value(final_cookies), "expiry": expiry}, f)
        os.replace(tmp_path, cookie_path)
        return _encode_cookie_value(final_cookies)


def _encode_cookie_value(cookies: dict[str, str]) -> str:
    """Encode cookie dict as semicolon-separated string."""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def _decode_cookie_value(cookie_str: str | None) -> str | None:
    """Decode semicolon-separated cookie string."""
    if not cookie_str:
        return None
    return cookie_str


import time
import asyncio
import base64
import json
import os
import mimetypes
from datetime import datetime, timezone
from typing import Any, AsyncIterator
import aiohttp
import base64
import re
import mimetypes
from datetime import datetime, timezone