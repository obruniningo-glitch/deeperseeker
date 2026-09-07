"""DeepSeek cookie management — verbatim port from v1 functions.py (§227-290).

Provides cookie retrieval with automatic Playwright-based regeneration.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from playwright.async_api import async_playwright

from v2.settings import get_settings


_cookie_lock = asyncio.Lock()
_cookie_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aws_cookies_deepseek.json")


async def get_cookies() -> str:
    """Get valid DeepSeek cookie string (cached or regenerated)."""
    try:
        if os.path.exists(_cookie_path):
            with open(_cookie_path) as f:
                cookies = json.load(f)
            cookie = _decode_cookie_value(cookies.get("cookie"))
            if cookies.get("expiry") is not None and cookies["expiry"] > time.time() and cookie is not None:
                return cookie
    except Exception:
        pass

    async with _cookie_lock:
        return await _regenerate_cookies()


async def _regenerate_cookies() -> str:
    """Regenerate cookies via Playwright (headless)."""
    settings = get_settings()

    async with async_playwright() as p:
        launch_kwargs = {"headless": True}
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            launch_kwargs["args"] = ["--no-sandbox"]

        async with p.chromium.launch(**launch_kwargs) as browser:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(settings.DEEPSEEK_ORIGIN, wait_until="domcontentloaded")
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

        tmp_path = _cookie_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"cookie": _encode_cookie_value(final_cookies), "expiry": expiry}, f)
        os.replace(tmp_path, _cookie_path)
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
import json
import os