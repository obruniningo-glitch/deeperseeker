"""Qwen cookie management — baxia-token generation via Playwright headless Chromium.

The baxia token has ~25-minute TTL and is prefixed with T2gAv_.
The JWT bearer token is NOT stored here — only the baxia token is harvested
and managed. Callers must provide the JWT token separately.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import random
import string
from typing import Optional

import aiohttp
from playwright.async_api import async_playwright

from v2.settings import get_settings


_cookie_lock = asyncio.Lock()
_baxia_token_cache: Optional[dict] = None
_baxia_cache_time: float = 0.0


def _random_suffix(length: int = 5) -> str:
    """Generate a random alphanumeric suffix for fallback tokens."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _get_cookie_path() -> str:
    """Get the path to the Qwen cookie cache file."""
    settings = get_settings()
    path = settings.QWEN_COOKIE_PATH
    if not os.path.isabs(path):
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        path = os.path.join(repo_root, path)
    return path


async def get_baxia_tokens() -> Optional[dict]:
    """Get baxia tokens with cached TTL of ~25 minutes.

    Returns a dict with keys: bx_ua, bx_umidtoken, bx_v
    or None if both primary and fallback paths fail.
    """
    global _baxia_token_cache, _baxia_cache_time

    now = time.time()
    # Return cached token if still valid (25-minute TTL)
    if _baxia_token_cache and (now - _baxia_cache_time) < 25 * 60:
        return _baxia_token_cache

    # Primary path: Playwright headless Chromium
    result = await _primary_baxia_path()
    if result is not None:
        _baxia_token_cache = result
        _baxia_cache_time = now
        return _baxia_token_cache

    # Fallback: HTTP GET to wu.json
    result = await _fallback_wu_json_path()
    if result is not None:
        _baxia_token_cache = result
        _baxia_cache_time = now
        return _baxia_token_cache

    # Both paths failed
    return None


async def get_baxia_token() -> Optional[str]:
    """Get just the bx_umidtoken (umid) value from the baxia token cache.

    Wrapper around get_baxia_tokens() for callers that only need the umid token.
    """
    tokens = await get_baxia_tokens()
    if tokens is not None:
        return tokens.get("bx_umidtoken")
    return None


def _get_profile_dir() -> str:
    """Persistent Chromium profile dir (survives restarts; holds WAF clearance)."""
    settings = get_settings()
    path = settings.QWEN_PROFILE_DIR
    if not os.path.isabs(path):
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        path = os.path.join(repo_root, path)
    os.makedirs(path, exist_ok=True)
    return path


async def _primary_baxia_path() -> Optional[dict]:
    """Primary path: Playwright Chromium (persistent profile) → getFYModule.

    Uses a PERSISTENT profile dir so WAF clearance cookies (acw_tc/acw_sc,
    baxia session) survive across runs. First run needs an interactive
    challenge solve via qwen_login.py; later runs reuse the clearance.
    Returns dict {bx_ua, bx_umidtoken, bx_v, ver, cookies} or None.
    """
    browser = None
    try:
        async with async_playwright() as p:
            launch_kwargs = {"headless": True}
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                launch_kwargs["args"] = ["--no-sandbox"]

            # Explicit launch/close: Browser is not used as a context manager
            # (async with on the launch coroutine never awaits it).
            browser = await p.chromium.launch_persistent_context(
                _get_profile_dir(), **launch_kwargs)
            context = browser
            page = await context.new_page()
            await page.goto("https://chat.qwen.ai/", wait_until="domcontentloaded")

            # The uid token materializes a few seconds after load (~4s observed);
            # poll until it is present instead of a fixed sleep.
            uid = ""
            fy = ""
            ver = ""
            for _ in range(60):  # 60 attempts x 500ms = 30s
                try:
                    js_code = (
                        "(function(){"
                        "var fm = (window.__baxia__||{}).getFYModule;"
                        "if (!fm || !fm.fyObj) return { ready: false };"
                        "var u = ''; var f = '';"
                        "try { u = String(fm.getUidToken()); } catch(e) {}"
                        "try { f = String(fm.getFYToken()); } catch(e) {}"
                        "if (!u || u === 'undefined' || u.length <= 20) return { ready: false };"
                        "return { ready: true, uid: u, fy: f, ver: fm.fyObj.ver || '' };"
                        "})()"
                    )
                    result = await page.evaluate(js_code)
                    if result and result.get("ready"):
                        uid = result.get("uid", "")
                        fy = result.get("fy", "")
                        ver = result.get("ver", "")
                        if uid and re.match(r"^T2gA", uid) and len(uid) > 20:
                            break
                except Exception:
                    pass
                await asyncio.sleep(0.5)

            if not uid:
                return None

            try:
                cookie_str = await context.cookies()
                _ = cookie_str  # harvested below via document.cookie equivalent
            except Exception:
                pass
            try:
                doc_cookie = await page.evaluate("() => document.cookie || ''")
            except Exception:
                doc_cookie = ""

            await context.close()

            # Build result: bx_ua = fy (or '231!'+uid if fy empty),
            # bx_umidtoken = uid, bx_v = '2.5.37'
            bx_ua = fy if fy else f"231!{uid}"
            bx_umidtoken = uid
            bx_v = "2.5.37"

            return {"bx_ua": bx_ua, "bx_umidtoken": bx_umidtoken, "bx_v": bx_v,
                    "ver": ver, "cookies": doc_cookie}

    except Exception:
        return None
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass


def _extract_wu_token(body_text: str) -> str:
    """Extract the umid token from a wu.json body.

    Primary: umx.wu('...'). Fallback: first quoted string.
    Returns "" when nothing is found.
    """
    m = re.search(r"umx\.wu\('([^']+)'\)", body_text)
    if m:
        return m.group(1)
    qm = re.search(r"'([^']+)'", body_text)
    return qm.group(1) if qm else ""


async def _fallback_wu_json_path() -> Optional[dict]:
    """Fallback path: plain HTTP GET https://sg-wum.alibaba.com/w/wu.json.

    Extracts the umid token with regex umx.wu('...').
    Falls back to first quoted string, then etag header.
    Returns dict {bx_ua, bx_umidtoken, bx_v} or None.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://sg-wum.alibaba.com/w/wu.json",
                headers={"User-Agent":
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None

                body_text = await resp.text()
                bx_umidtoken = ""

                # Primary regex: umx.wu('...')
                m = re.search(r"umx\.wu\('([^']+)'\)", body_text)
                if m:
                    bx_umidtoken = m.group(1)
                else:
                    # Fallback 1: first quoted string in the body
                    qm = re.search(r"'([^']+)'", body_text)
                    if qm:
                        bx_umidtoken = qm.group(1)
                    else:
                        # Fallback 2: etag header
                        bx_umidtoken = resp.headers.get("etag", "")

                # Ensure the token starts with T2gA; if not, prepend it
                if bx_umidtoken and not re.match(r"^T2gA", bx_umidtoken):
                    bx_umidtoken = "T2gA" + bx_umidtoken

                # If we have an umid token, build bx_ua and return
                if bx_umidtoken:
                    token_suffix = _random_suffix(5)
                    bx_ua = f"231!{bx_umidtoken}"
                    bx_v = "2.5.36"
                    return {"bx_ua": bx_ua, "bx_umidtoken": bx_umidtoken, "bx_v": bx_v}

        return None

    except Exception:
        return None


async def get_cookies() -> Optional[str]:
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
                if "T2gAv_" in cookie_str:
                    return cookie_str
    except Exception:
        pass

    async with _cookie_lock:
        return await _regenerate_cookies()


async def _regenerate_cookies() -> Optional[str]:
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


async def get_baxia_token_from_cookies() -> Optional[str]:
    """Get the baxia anti-bot token from cookies (deprecated alias).

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