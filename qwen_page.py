"""Persistent coder.qwen.ai page supervisor (spike).

One browser, one page, kept open across calls. send_and_read() types a
message, clicks send, waits until generation finishes (stop-button gone
AND answer text stable), returns the answer text.

Run: deeperseeker_env/Scripts/python.exe qwen_page.py [message]
"""
import asyncio
import sys

from playwright.async_api import async_playwright

from v2.providers.qwen.cookies import _get_profile_dir

STOP_WORDS = ("stop", "cancel", "pause", "停止", "取消")


async def page_busy(page) -> bool:
    """True while the model is still generating (stop control present)."""
    try:
        return await page.evaluate("""(words) => {
          const btns = [...document.querySelectorAll('button')];
          const t = btns.map(b => ((b.innerText||'') + ' ' + (b.getAttribute('aria-label')||'')).toLowerCase()).join('|');
          if (words.some(w => w && t.includes(w.toLowerCase()))) return true;
          const ta = document.querySelector('textarea');
          if (ta && ta.disabled) return true;
          return [...document.querySelectorAll('*')].some(el =>
            el.children.length === 0 && /generating|thinking|waiting/i.test(el.innerText || ''));
        }""", list(STOP_WORDS))
    except Exception:
        return True


async def read_answer(page, exclude: str) -> str:
    """Longest content-element text that isn't our message or chrome."""
    try:
        texts = await page.evaluate("""() => {
          const all = [];
          document.querySelectorAll('p, li, pre, code, h1, h2, h3, td').forEach(el => {
            const t = (el.innerText || '').trim();
            if (t.length > 20) all.push(t);
          });
          return [...new Set(all)];
        }""")
    except Exception:
        return ""
    cands = [t for t in texts
             if exclude[:20] not in t and "Qwen Coder" not in t
             and "Initializing environment" not in t]
    return max(cands, key=len) if cands else ""


async def send_and_read(page, msg: str, idle_rounds: int = 5) -> str:
    box = page.locator("textarea").first
    await box.click()
    await box.fill(msg)
    await asyncio.sleep(1)
    try:
        send = box.locator("xpath=ancestor::*[self::div or self::form][1]//button").last
        await send.click(timeout=5000)
    except Exception:
        await box.press("Enter")
    print("sent, waiting for generation to finish...", flush=True)

    last, stable = "", 0
    while True:  # no fixed timeout: busy-state + stability decide
        await asyncio.sleep(3)
        busy = await page_busy(page)
        text = await read_answer(page, msg)
        if text == last and text:
            stable += 1
        else:
            stable = 0
            last = text
        if not busy and stable >= idle_rounds and len(last) > 30:
            return last


async def main():
    msg = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(_get_profile_dir(), headless=False)
        page = await browser.new_page()
        await page.goto("https://coder.qwen.ai/", wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(8)
        answer = await send_and_read(page, msg)
        print("ANSWER:", answer[:1500])
        print("--- page stays open; Ctrl+C the process to close ---")
        try:
            while True:
                await asyncio.sleep(60)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        await browser.close()


asyncio.run(main())
