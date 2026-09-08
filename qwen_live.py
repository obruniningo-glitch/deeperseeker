"""Live multi-step page session: send -> open new chat -> read answer.

Usage: deeperseeker_env/Scripts/python.exe qwen_live.py "msg1" "msg2" ...
Headed (watch it work). Prints each answer.
"""
import asyncio
import sys

from playwright.async_api import async_playwright

from v2.providers.qwen.cookies import _get_profile_dir


async def history_titles(page):
    try:
        return await page.evaluate("""() => {
          const all = [];
          document.querySelectorAll('*').forEach(el => {
            if (el.children.length === 0) {
              const t = (el.innerText || '').trim();
              if (t.length > 0 && t.length < 60) all.push(t);
            }
          });
          return [...new Set(all)];
        }""")
    except Exception:
        return []


async def send_message(page, msg: str):
    box = page.locator("textarea").first
    await box.click()
    await box.fill(msg)
    await asyncio.sleep(1)
    try:
        send = box.locator("xpath=ancestor::*[self::div or self::form][1]//button").last
        await send.click(timeout=5000)
    except Exception:
        await box.press("Enter")
    print(f"SENT: {msg}", flush=True)


async def wait_for_new_chat(page, known: set, timeout_s: int = 180):
    """Wait until a history title appears that wasn't there before; open it."""
    for _ in range(timeout_s // 2):
        await asyncio.sleep(2)
        titles = set(await history_titles(page))
        new = titles - known
        # a fresh chat shows as a short title (our message, truncated)
        cands = [t for t in new if len(t) > 3 and "Qwen Coder" not in t
                 and "Initializing" not in t]
        if cands:
            return cands[0]
    return None


async def read_answer_in_chat(page, title: str, sent: str = "") -> str:
    try:
        link = page.get_by_text(title, exact=False).first
        await link.click(timeout=5000)
    except Exception as e:
        print("open-chat click failed:", str(e)[:100])
        return ""
    await asyncio.sleep(4)
    # wait for streaming to settle: answer text stable across polls
    last, stable = "", 0
    for _ in range(150):
        await asyncio.sleep(2)
        try:
            texts = await page.evaluate("""() => {
              const all = [];
              document.querySelectorAll('p, li, pre, code, h1, h2, h3, td').forEach(el => {
                const t = (el.innerText || '').trim();
                if (t.length > 30) all.push(t);
              });
              return [...new Set(all)];
            }""")
        except Exception:
            continue
        cands = [t for t in texts if "Qwen Coder" not in t
                 and "Initializing environment" not in t
                 and sent[:25] not in t and title not in t]
        longest = max(cands, key=len) if cands else ""
        if longest == last and longest:
            stable += 1
            if stable >= 3:
                return longest
        else:
            last, stable = longest, 0
    return last


async def main():
    msgs = sys.argv[1:] or ["Say hello in one short sentence.",
                            "What is 2 + 2? Answer in one short sentence."]
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(_get_profile_dir(), headless=False)
        page = await browser.new_page()
        await page.goto("https://coder.qwen.ai/", wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(8)
        for i, msg in enumerate(msgs):
            known = set(await history_titles(page))
            await send_message(page, msg)
            # give the app a moment to create the chat
            await asyncio.sleep(8)
            title = await wait_for_new_chat(page, known)
            if not title:
                print(f"STEP {i+1}: no new chat appeared")
                continue
            print(f"STEP {i+1}: opened '{title}'")
            answer = await read_answer_in_chat(page, title, msg)
            print(f"ANSWER {i+1}:", (answer[:800] if answer else "(empty)"), flush=True)
            # back to landing for the next message
            await page.goto("https://coder.qwen.ai/", wait_until="domcontentloaded")
            await asyncio.sleep(4)
        print("DONE — closing in 10s (watch the history)")
        await asyncio.sleep(10)
        await browser.close()


asyncio.run(main())
