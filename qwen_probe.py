"""Live Qwen probe: fresh chat + one message, raw verdict.

Uses pool token id 2 (new account) and the persistent-profile harvest.
Prints the chat id, event list, and — on non-SSE answers — the raw body
head so blocks (punish/RGV587/HTML) are visible.

Run: DEEPSEEKER_ENCRYPTION_KEY=... deeperseeker_env/Scripts/python.exe qwen_probe.py [message]
"""
import asyncio
import aiosqlite
import sys

from v2.store.repo_tokens import TokenRepo
from v2.providers.qwen import cookies as qc

qc._baxia_token_cache = None
qc._baxia_cache_time = 0.0


async def main():
    msg = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    conn = await aiosqlite.connect("deeperseeker_v2.db")
    conn.row_factory = aiosqlite.Row
    repo = TokenRepo(conn)
    secret = repo.decrypt_secret(await repo.get(2))
    await conn.close()
    from v2.providers.qwen.wire import create_new_chat, send_message
    cid = await create_new_chat(secret, "qwen3.8-max")
    print("new chat:", cid)
    events = []
    async for kind, payload in send_message(secret, cid, msg, model_type="qwen3.8-max"):
        events.append((kind, str(payload)[:300]))
    print("events:", len(events))
    for k, p in events[:12]:
        print(f"[{k}]", p)


asyncio.run(main())
