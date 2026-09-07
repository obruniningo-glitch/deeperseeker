"""Qwen cookie management — anti-bot token harvesting via Playwright.

Qwen uses a JWT bearer token and an anti-bot token (baxia token).
The JWT bearer token is NOT stored here — only the baxia token is harvested
and managed. Callers must provide the JWT token separately.
The baxia token has ~25-minute TTL and is prefixed with T2gAv_.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional

from playwright.async_api import async_playwright

from v2.settings import get_settings


_cookie_lock = asyncio.Lock()


def _get_cookie_path() -> str:
    """Get the path to the Qwen cookie cache file."""
    settings = get_settings()
    # If QWEN_COOKIE_PATH is relative, resolve it relative to the repo root
    path = settings.QWEN_COOKIE_PATH
    if not os.path.isabs(path):
        # Resolve relative to the project root (where deeperseeker/ is)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(repo_root, path)
    return path


async def get_cookies() -> str | None:
    """Get valid Qwen cookie string (cached or regenerated).

    Returns the cookie string (semicolon-separated) or None if
    no valid cookies can be obtained. The baxia token is the key
    anti-bot token; the JWT bearer token is NOT stored here.
    """
    cookie_path = _get_cookie_path()
    try:
        if os.path.exists(cookie_path):
            with open(cookie_path) as f:
                data = json.load(f)
            cookie_str = data.get("cookie")
            expiry = data.get("expiry")
            if cookie_str and expiry and expiry > time.time():
                # Verify the cookie string contains a baxia token (T2gAv_ prefix)
                if "T2gAv_" in cookie_str:
                    return cookie_str
    except Exception:
        pass

    async with _cookie_lock:
        return await _regenerate_cookies()


async def _regenerate_cookies() -> str | None:
    """Regenerate cookies via Playwright (headless).

    Navigates to chat.qwen.ai, waits for the page to load, and harvests
    all cookies. Returns the cookie string (semicolon-separated) or None
    if harvest fails.

    This extracts the baxia token (T2gAv_*) which is the anti-bot token.
    The JWT bearer token is NOT extracted here — it's provided separately.
    """
    settings = get_settings()
    cookie_path = _get_cookie_path()

    try:
        async with async_playwright() as p:
            launch_kwargs = {"headless": True}
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                launch_kwargs["args"] = ["--no-sandbox"]

            async with p.chromium.launch(**launch_kwargs) as browser:
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto(settings.QWEN_ORIGIN, wait_until="domcontentloaded")

                # Wait for the page to settle and load any anti-bot mechanisms
                try:
                    await page.wait_for_selector("body", timeout=30000)
                except Exception:
                    pass

                # Wait a bit for anti-bot tokens to be set via JavaScript
                await asyncio.sleep(2)

                cookies = await context.cookies()

        # Extract all cookies, looking for the baxia token (T2gAv_ prefix)
        cookie_dict = {}
        expiry = None
        for c in cookies:
            cookie_dict[c["name"]] = c["value"]
            # Use the earliest expiry among cookies as the cache expiry
            if c.get("expires"):
                c_expiry = c["expires"]
                if expiry is None or c_expiry < expiry:
                    expiry = c_expiry

        # Check if we have a baxia token (T2gAv_*)
        has_baxia = any(
            value.startswith("T2gAv_") for value in cookie_dict.values()
        )
        if not has_baxia:
            # No baxia token found — this might mean the anti-bot mechanism
            # hasn't loaded yet or uses a different prefix
            # Return None gracefully; callers must tolerate None
            return None

        # Encode cookies as semicolon-separated string
        cookie_str = "; ".join(f"{k}={v}" for k, v in cookie_dict.items())

        # If no expiry found or expiry is in the past, set a conservative TTL
        if not expiry or expiry < time.time():
            expiry = time.time() + 1500  # 25 minutes

        # Write to cache
        tmp_path = cookie_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"cookie": cookie_str, "expiry": expiry}, f)
        os.replace(tmp_path, cookie_path)

        return cookie_str

    except Exception:
        # Any failure in Playwright or navigation should return None gracefully
        return None


def _decode_cookie_value(cookie_str: str | None) -> str | None:
    """Decode semicolon-separated cookie string."""
    return cookie_str


def _encode_cookie_value(cookies: dict[str, str]) -> str:
    """Encode cookie dict as semicolon-separated string."""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


async def get_baxia_token() -> str | None:
    """Get the baxia anti-bot token from cookies.

    Returns the token value (string) or None if not found.
    The token is prefixed with T2gAv_ and has ~25-minute TTL.
    """
    cookie_str = await get_cookies()
    if not cookie_str:
        return None

    # Parse semicolon-separated cookie string
    for pair in cookie_str.split("; "):
        if "=" in pair:
            key, value = pair.split("=", 1)
            if value.startswith("T2gAv_"):
                return value

    return None
