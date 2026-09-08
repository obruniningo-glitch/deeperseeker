"""Harvest a full Qwen session from the persistent profile.

Run headed: user creates/logs in with the NEW account in the opened window,
then signals done (creates .qwen_login_done or says 'done').
Harvests from the same page: JWT (localStorage scan), document.cookie,
baxia tokens. Prints a summary (truncated secrets) and writes the full
bundle to qwen_session.json (git-ignored, delete after seeding).

Run: deeperseeker_env/Scripts/python.exe qwen_harvest.py
"""
import asyncio
import json
import os

from playwright.async_api import async_playwright

from v2.providers.qwen.cookies import _get_profile_dir

HERE = os.path.dirname(os.path.abspath(__file__))
DONE_FILE = os.path.join(HERE, ".qwen_login_done")
OUT_FILE = os.path.join(HERE, "qwen_session.json")


async def main():
    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            _get_profile_dir(), headless=False)
        page = await browser.new_page()
        await page.goto("https://chat.qwen.ai/", wait_until="domcontentloaded")
        print("Create / log in with the NEW account, then signal done.")
        print("Waiting (up to 10 minutes)...")
        for _ in range(600):
            if os.path.exists(DONE_FILE):
                break
            await asyncio.sleep(1)

        bundle = await page.evaluate("""() => {
          const out = {storages: {}, cookie: document.cookie || ''};
          for (const store of [localStorage, sessionStorage]) {
            const hit = {};
            for (let i = 0; i < store.length; i++) {
              const k = store.key(i);
              const v = store.getItem(k) || '';
              const m = v.match(/eyJ[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+\\.[A-Za-z0-9_-]+/);
              if (m) hit[k] = m[0];
            }
            out.storages[store === localStorage ? 'local' : 'session'] = hit;
          }
          let bx = null;
          try {
            const fm = (window.__baxia__ || {}).getFYModule;
            if (fm && fm.fyObj) {
              const u = String(fm.getUidToken()); const f = String(fm.getFYToken());
              bx = {uid: u, fy: f, ver: fm.fyObj.ver || ''};
            }
          } catch (e) { bx = {err: String(e).slice(0, 100)}; }
          out.baxia = bx;
          return out;
        }""")
        await browser.close()

    with open(OUT_FILE, "w") as f:
        json.dump(bundle, f)

    n_jwt = sum(len(v) for v in bundle.get("storages", {}).values())
    bx = bundle.get("baxia") or {}
    print(f"JWTs found: {n_jwt} (full values in {OUT_FILE})")
    print("baxia uid head:", str(bx.get("uid", ""))[:16])
    print("cookie pairs:", len([c for c in bundle.get("cookie", "").split(';') if c.strip()]))
    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)


asyncio.run(main())
