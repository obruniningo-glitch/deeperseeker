"""Spike: round-trip a message through the coder.qwen.ai page.

Persistent profile, headed so the user can watch. Types the message,
clicks Send, polls for the answer, prints it. One-shot; leaves the
chat in account history.

Run: deeperseeker_env/Scripts/python.exe qwen_spike.py [message]
"""
import asyncio
import sys

from playwright.async_api import async_playwright

from v2.providers.qwen.cookies import _get_profile_dir


async def main():
    msg = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(_get_profile_dir(), headless=False)
        page = await browser.new_page()
        await page.goto("https://coder.qwen.ai/", wait_until="domcontentloaded", timeout=45000)
        # First start initializes the environment; wait until the composer
        # is interactive (textarea enabled and Send control present).
        for _ in range(90):
            ready = await page.evaluate("""() => {
              const ta = document.querySelector('textarea');
              if (!ta || ta.disabled) return false;
              return true;
            }""")
            if ready:
                break
            await asyncio.sleep(2)
        await asyncio.sleep(4)

        # New chat if the button exists, else reuse current view.
        try:
            btn = page.get_by_text("New Chat", exact=False)
            if await btn.count() > 0:
                await btn.first.click()
                await asyncio.sleep(3)
        except Exception as e:
            print("new-chat click skipped:", str(e)[:80])

        box = page.locator("textarea").first
        await box.click()
        await box.fill(msg)
        await asyncio.sleep(1)

        # Send = round icon button (up-arrow) beside the textarea; fall back to Enter.
        try:
            box_owner = page.locator("textarea").first
            send = box_owner.locator(
                "xpath=ancestor::*[self::div or self::form][1]//button").last
            await send.click(timeout=5000)
        except Exception:
            await page.locator("textarea").first.press("Enter")
        print("sent, waiting for answer...")

        answer = ""
        stable = 0
        for _ in range(300):  # up to 10 minutes
            await asyncio.sleep(2)
            try:
                texts = await page.evaluate("""() => {
                  const all = [];
                  document.querySelectorAll('p, li, pre, code, h1, h2, h3').forEach(el => {
                    const t = (el.innerText || '').trim();
                    if (t.length > 20) all.push(t);
                  });
                  return [...new Set(all)];
                }""")
                cands = [t for t in texts if msg[:20] not in t and "Qwen Coder" not in t]
                if cands:
                    longest = max(cands, key=len)
                    if longest == answer:
                        stable += 1
                        if stable >= 3 and len(longest) > 30:
                            break
                    else:
                        answer = longest
                        stable = 0
            except Exception:
                pass
        print("ANSWER:", answer[:1000] if answer else "(none detected)")
        await browser.close()


asyncio.run(main())
