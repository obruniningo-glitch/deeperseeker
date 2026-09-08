"""Interactive Qwen login: solve the slider/CAPTCHA once, keep the clearance.

Opens chat.qwen.ai in a headed browser on the persistent profile
(QWEN_PROFILE_DIR). Log in / solve any challenge in the window, then press
Ctrl+C here. The script harvests the full cookie jar (incl. HttpOnly auth
cookies) + baxia tokens, validates a T2gAv_ token was minted, and writes the
cache file so later headless runs reuse the clearance.

Run: deeperseeker_env/Scripts/python.exe qwen_login.py
Holds no secrets itself; the profile dir + cache file are git-ignored.
"""
import asyncio
import json
import os
import re
import time

from playwright.async_api import async_playwright

from v2.providers.qwen.cookies import _get_cookie_path, _get_profile_dir


async def harvest(page, context):
    """Pull baxia tokens + full cookie jar from the live page."""
    uid, fy, ver = "", "", ""
    try:
        res = await page.evaluate(
            "(function(){"
            "var fm = (window.__baxia__||{}).getFYModule;"
            "if (!fm || !fm.fyObj) return null;"
            "var u=''; var f='';"
            "try { u = String(fm.getUidToken()); } catch(e) {}"
            "try { f = String(fm.getFYToken()); } catch(e) {}"
            "return {uid: u, fy: f, ver: fm.fyObj.ver || ''};"
            "})()"
        )
        if res:
            uid, fy, ver = res.get("uid", ""), res.get("fy", ""), res.get("ver", "")
    except Exception as e:
        print("baxia evaluate failed:", str(e)[:100])
    try:
        jar = await context.cookies()
        cookie_str = "; ".join(f'{c["name"]}={c["value"]}' for c in jar)
    except Exception as e:
        print("cookie harvest failed:", str(e)[:100])
        cookie_str = ""
    return uid, fy, ver, cookie_str


async def main():
    print("Opening chat.qwen.ai — log in and solve any challenge in the window.")
    print("When done, press Ctrl+C here. (Docker/root: works headless too.)")
    async with async_playwright() as p:
        kwargs = {"headless": False}
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            kwargs["args"] = ["--no-sandbox"]
        browser = await p.chromium.launch_persistent_context(_get_profile_dir(), **kwargs)
        context = browser
        page = await context.new_page()
        await page.goto("https://chat.qwen.ai/", wait_until="domcontentloaded")
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        print("Harvesting session...")
        uid, fy, ver, cookie_str = await harvest(page, context)
        names = sorted(n for n in (c.split("=")[0].strip() for c in cookie_str.split(";")) if n)
        print("cookies:", ", ".join(names) if names else "(none)")
        ok = bool(uid and re.match(r"^T2gA", uid) and len(uid) > 20)
        print("baxia uid:", (uid[:16] + "...") if uid else "(missing)",
              "| valid T2gAv_*" if ok else "| NOT FOUND — clearance likely missing")
        if cookie_str:
            path = _get_cookie_path()
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"cookie": cookie_str, "expiry": time.time() + 1500}, f)
            os.replace(tmp, path)
            print("cache written:", path)
        await browser.close()
        if not ok or not cookie_str:
            raise SystemExit("Harvest incomplete — solve the challenge fully and retry.")


asyncio.run(main())
