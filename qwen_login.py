"""One-off: open chat.qwen.ai in a headed browser using the persistent profile.

Solve any CAPTCHA / slider challenge and log in if needed, then close the
window. The WAF clearance cookies persist in QWEN_PROFILE_DIR and later
headless harvests reuse them.

Run: deeperseeker_env/Scripts/python.exe qwen_login.py
Delete this file afterwards (optional — it holds no secrets).
"""
import asyncio
import os

from playwright.async_api import async_playwright

from v2.providers.qwen.cookies import _get_profile_dir

DONE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".qwen_login_done")


async def main():
    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            _get_profile_dir(), headless=False)
        page = await browser.new_page()
        await page.goto("https://chat.qwen.ai/", wait_until="domcontentloaded")
        print("Solve any challenge / log in, then create the file .qwen_login_done")
        print("(or just say 'done' and I will finish it). Waiting...")
        for _ in range(600):  # up to 10 minutes
            if os.path.exists(DONE_FILE):
                break
            await asyncio.sleep(1)
        cookies = await browser.cookies()
        names = sorted(c["name"] for c in cookies)
        print("stored cookies:", ", ".join(names))
        await browser.close()
    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)


asyncio.run(main())
