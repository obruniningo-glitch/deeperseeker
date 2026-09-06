"""Regression tests for the 2026-09-06 local fixes.

Run with the repo venv:  deeperseeker_env/Scripts/python.exe tests/test_local_fixes.py
(also pytest-compatible).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import API_KEY, convert_anthropic_messages, session_hit_ok
from functions import count_tokens, parse_tools
from plugin_helper import (
    DEFAULT_PROMPT_BUDGET,
    build_prompt,
    clip_text,
    compact_history,
    generate_signature_sync,
    project_signature,
)


def test_api_key_never_empty():
    assert API_KEY, "API_KEY must never be empty (fail-open)"
    import importlib
    import unittest.mock as mock
    with mock.patch.dict(os.environ, {"DEEPSEEKER_API_KEY": ""}):
        import app
        importlib.reload(app)
        assert app.API_KEY, "empty DEEPSEEKER_API_KEY must fall back to the default"
    importlib.reload(app)


def test_tool_result_becomes_tool_role():
    msgs = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file.txt"},
        ]},
    ]
    out = convert_anthropic_messages(msgs)
    assert out[1]["tool_calls"][0]["function"]["name"] == "Bash"
    assert out[2]["role"] == "tool", "tool_result must become a role=tool message, not user text"
    assert out[2]["tool_call_id"] == "toolu_1"
    assert out[2]["content"] == "file.txt"


def test_signature_matches_server_reconstruction():
    # What the server reconstructs after parsing model output (as stream_response does)
    model_output = 'Working on it.\n<tool_call>{"name": "Bash", "arguments": {"command": "ls -la"}}</tool_call>'
    parsed_tools, clean_text = parse_tools(model_output)
    assert parsed_tools, "parse_tools should find the tool call"
    server_msgs = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "tool_calls": parsed_tools},
    ]
    # What an Anthropic client echoes back on the next turn
    client_msgs = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Working on it."},
            {"type": "tool_use", "id": "toolu_abc", "name": "Bash", "input": {"command": "ls -la"}},
        ]},
    ]
    converted = convert_anthropic_messages(client_msgs)
    assert generate_signature_sync(server_msgs, "expert") == generate_signature_sync(converted, "expert"), \
        "signature cache must hit when the client echoes the assistant tool turn"


def test_tool_results_reach_build_prompt():
    msgs = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file.txt"},
        ]},
    ]
    converted = convert_anthropic_messages(msgs)
    prompt = asyncio.run(build_prompt(converted, [], "expert", is_first_message=False))
    assert "[TOOL RESULTS]" in prompt, "tool results must appear in the [TOOL RESULTS] section"
    assert "file.txt" in prompt
    assert "[USER]" not in prompt, "the original question must not be re-sent on follow-up turns"


def test_user_text_after_tool_result_preserved():
    msgs = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.txt"},
            {"type": "text", "text": "now list the hidden files"},
        ]},
    ]
    out = convert_anthropic_messages(msgs)
    assert out[0]["role"] == "assistant" and out[0]["tool_calls"]
    assert out[1]["role"] == "tool"
    assert out[2]["role"] == "user"
    assert out[2]["content"] == "now list the hidden files"


def test_next_parent_helper_is_centralized():
    import functions
    import inspect
    import app
    assert functions.next_parent(0) == 2
    assert functions.next_parent(4) == 6
    # every save_session call in app.py must go through next_parent
    assert "parent_message_id + 2" not in inspect.getsource(app), \
        "parent_message_id bookkeeping must use the centralized next_parent() helper"


def test_parse_tools_ignores_tool_markup_in_code_fence():
    text = 'Here is an example:\n```xml\n<tool_call name="Bash">{"command":"rm -rf /"}</tool_call>\n```\nAll done!'
    tools, clean_text = parse_tools(text)
    assert tools == [], "tool-call markup inside a markdown code fence must not be parsed as a real tool call"


def test_parse_tools_still_finds_call_after_fenced_example():
    text = ('Example:\n```\n<tool_call>{"name": "Bash"}</tool_call>\n```\n'
            'Now really do it.\n<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>')
    tools, clean_text = parse_tools(text)
    assert len(tools) == 1, "a real tool call outside the fence must still be parsed"
    assert tools[0]["function"]["name"] == "Bash"


def test_parse_tools_keeps_repeated_identical_calls():
    call = '<tool_call>{"name": "Read", "arguments": {"path": "a.txt"}}</tool_call>'
    tools, clean_text = parse_tools(call + call)
    assert len(tools) == 2, "legitimately repeated identical tool calls must not be deduped away"
    assert tools[0]["id"] != tools[1]["id"]


def test_parse_tools_preserves_companion_text():
    text = 'Let me check that for you.\n<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>'
    tools, clean_text = parse_tools(text)
    assert tools, "the tool call must still be found"
    assert clean_text == "Let me check that for you.", "text accompanying a tool call must be preserved"


def test_stream_flush_discards_partial_tool_call():
    from functions import StreamToolParser
    p = StreamToolParser()
    p.feed("here is the plan: ")
    p.feed('<tool_call>{"name": "Bash", "arguments": {"comm')
    out = p.flush()
    assert out == [], "a stream cut off mid-tool-call must not leak raw XML fragments as text"


def test_fetch_url_bytes_validates_every_redirect_hop():
    import unittest.mock as mock
    from aiohttp import web
    import functions
    import plugin_helper

    async def run():
        async def final(request):
            return web.Response(body=b"image-bytes")

        async def redir(request):
            raise web.HTTPFound("/final")

        async def loop(request):
            raise web.HTTPFound("/loop")

        a = web.Application()
        a.router.add_get("/final", final)
        a.router.add_get("/redir", redir)
        a.router.add_get("/loop", loop)
        runner = web.AppRunner(a)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        base = f"http://127.0.0.1:{port}"
        session = await functions.get_session()
        try:
            with mock.patch.object(plugin_helper, "_assert_public_url", lambda u: None):
                # a redirect chain is followed hop by hop
                data = await plugin_helper._fetch_url_bytes(session, base + "/redir")
                assert data == b"image-bytes"
                # a redirect loop is capped instead of hanging
                try:
                    await plugin_helper._fetch_url_bytes(session, base + "/loop")
                    raise AssertionError("redirect loop must be capped")
                except ValueError as e:
                    assert "redirect" in str(e)
            # without the patch, a non-public address must be rejected
            try:
                await plugin_helper._fetch_url_bytes(session, base + "/redir")
                raise AssertionError("non-public address must be rejected")
            except ValueError:
                pass
        finally:
            await runner.cleanup()
            await session.close()
            functions._session = None

    asyncio.run(run())


def _temp_db(monkeypatched_env):
    import tempfile
    import unittest.mock as mock
    import functions
    tmp = tempfile.mkdtemp(prefix="ds_test_")
    db = os.path.join(tmp, "test.db")
    cm = mock.patch.dict(os.environ, monkeypatched_env)
    cm.start()
    functions._FERNET = None
    old_db = functions._db
    functions._db = db
    return db, old_db, cm


def _restore_db(old_db, cm):
    import functions
    functions._db = old_db
    functions._FERNET = None
    cm.stop()


def test_token_encrypted_at_rest_and_decrypted_on_read():
    import sqlite3
    import functions
    db, old_db, cm = _temp_db({"DEEPSEEKER_ENCRYPTION_KEY": "test-passphrase"})
    try:
        functions.init_db()
        functions.add_token("sk-secret-deepseek-token", "t1")
        conn = sqlite3.connect(db)
        stored = conn.execute("SELECT token FROM tokens WHERE id = 1").fetchone()[0]
        conn.close()
        assert "sk-secret-deepseek-token" not in stored, "token must not be stored in plaintext"
        assert functions.get_token(1)["token"] == "sk-secret-deepseek-token"
        assert functions.get_auth_token() == "sk-secret-deepseek-token"
        assert [t["token"] for t in functions.get_tokens()] == ["sk-secret-deepseek-token"]
    finally:
        _restore_db(old_db, cm)


def test_legacy_plaintext_token_migrates_on_read():
    import sqlite3
    import functions
    db, old_db, cm = _temp_db({"DEEPSEEKER_ENCRYPTION_KEY": "test-passphrase"})
    try:
        functions.init_db()
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO tokens (id, alias, token, status) VALUES (1, 'legacy', 'plain-legacy-token', 'ACTIVE')")
        conn.commit()
        conn.close()
        # read returns the working token and transparently re-encrypts the row
        assert functions.get_token(1)["token"] == "plain-legacy-token"
        conn = sqlite3.connect(db)
        stored = conn.execute("SELECT token FROM tokens WHERE id = 1").fetchone()[0]
        conn.close()
        assert "plain-legacy-token" not in stored, "legacy plaintext row must be re-encrypted on read"
        assert functions.get_token(1)["token"] == "plain-legacy-token"
    finally:
        _restore_db(old_db, cm)


def test_tokens_stay_plaintext_without_encryption_key():
    import sqlite3
    import functions
    db, old_db, cm = _temp_db({})
    try:
        functions.init_db()
        functions.add_token("sk-plain-token", "t1")
        conn = sqlite3.connect(db)
        stored = conn.execute("SELECT token FROM tokens WHERE id = 1").fetchone()[0]
        conn.close()
        assert stored == "sk-plain-token", "without DEEPSEEKER_ENCRYPTION_KEY behavior must stay plaintext"
        assert functions.get_token(1)["token"] == "sk-plain-token"
    finally:
        _restore_db(old_db, cm)


def test_cookie_value_encryption_roundtrip():
    import unittest.mock as mock
    import functions
    with mock.patch.dict(os.environ, {"DEEPSEEKER_ENCRYPTION_KEY": "another-pass"}):
        functions._FERNET = None
        try:
            payload = functions._encode_cookie_value({"aws-waf-token": "abc123"})
            assert isinstance(payload, str) and "abc123" not in payload
            assert functions._decode_cookie_value(payload) == {"aws-waf-token": "abc123"}
            # legacy plaintext dict still decodes
            assert functions._decode_cookie_value({"a": "b"}) == {"a": "b"}
        finally:
            functions._FERNET = None


def test_sessions_have_created_at_and_are_pruned_by_ttl():
    import sqlite3
    import functions
    db, old_db, cm = _temp_db({"DEEPSEEKER_SESSION_TTL_DAYS": "7"})
    try:
        functions.init_db()
        functions.save_session("sig-fresh", 1, "ds1", 2)
        conn = sqlite3.connect(db)
        created = conn.execute("SELECT created_at FROM sessions WHERE signature = 'sig-fresh'").fetchone()
        assert created and created[0], "sessions must record created_at"
        # simulate an old session, then prune
        conn.execute("UPDATE sessions SET created_at = datetime('now', '-8 days') WHERE signature = 'sig-fresh'")
        conn.commit()
        conn.close()
        functions.prune_sessions(7)
        assert functions.find_session("sig-fresh") is None, "sessions older than the TTL must be pruned"
        # save_session also prunes stale rows as a side effect
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO sessions (signature, token_id, deepseek_session_id, parent_message_id, created_at) VALUES ('sig-stale', 1, 'ds2', 0, datetime('now', '-30 days'))")
        conn.commit()
        conn.close()
        functions.save_session("sig-new", 1, "ds3", 4)
        assert functions.find_session("sig-stale") is None
        assert functions.find_session("sig-new") is not None
    finally:
        _restore_db(old_db, cm)


def test_sessions_table_migrates_existing_db():
    import sqlite3
    import tempfile
    import functions
    tmp = tempfile.mkdtemp(prefix="ds_test_")
    db = os.path.join(tmp, "legacy.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE sessions (
            signature TEXT PRIMARY KEY,
            token_id INTEGER,
            deepseek_session_id TEXT,
            parent_message_id INTEGER DEFAULT 0
        );
        INSERT INTO sessions (signature, token_id, deepseek_session_id, parent_message_id)
            VALUES ('legacy-row', 1, 'ds', 0);
    """)
    conn.commit()
    conn.close()
    old_db = functions._db
    functions._db = db
    try:
        functions.init_db()
        conn = sqlite3.connect(db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        created = conn.execute("SELECT created_at FROM sessions WHERE signature = 'legacy-row'").fetchone()
        conn.close()
        assert "created_at" in cols, "init_db must ALTER TABLE the legacy sessions table"
        assert created and created[0], "legacy rows must get a created_at stamp"
    finally:
        functions._db = old_db


def test_sig_lock_removed_after_handle_chat():
    import unittest.mock as mock
    import app
    from plugin_helper import generate_signature_sync

    async def run():
        msgs = [{"role": "user", "content": "hi"}]
        sig = generate_signature_sync(msgs, "instant")
        # No session and no tokens available -> 503, exercising the lock path
        with mock.patch.object(app, "get_auth_token", lambda: "tok"), \
             mock.patch.object(app, "find_session", lambda s: None), \
             mock.patch.object(app, "pick_token", lambda: None):
            resp = await app.handle_chat(msgs, "instant")
            assert resp.status_code == 503
        assert sig not in app._sig_locks, "_sig_locks entries must be removed after use"

    asyncio.run(run())


def test_outbound_calls_have_explicit_timeouts():
    import inspect
    import functions
    src = inspect.getsource(functions.send_message)
    assert "ClientTimeout(total=300)" in src, "send_message completion POST must have a total=300 timeout"
    src = inspect.getsource(functions.get_file_content)
    assert "ClientTimeout(total=120)" in src, "get_file_content signed-URL download must have a total=120 timeout"


def test_get_current_admin_rejects_null_origin():
    import time
    import app
    from starlette.requests import Request

    app.SESSIONS["testsid"] = time.time()

    def make_request(origin_header):
        headers = [(b"cookie", b"session_id=testsid")]
        if origin_header is not None:
            headers.append((b"origin", origin_header))
        headers.append((b"host", b"127.0.0.1:4000"))
        return Request({"type": "http", "headers": headers})

    try:
        for bad_origin in (b"null", b"NULL"):
            try:
                app.get_current_admin(make_request(bad_origin))
                raise AssertionError(f"Origin: {bad_origin.decode()} must be rejected")
            except Exception as e:
                assert getattr(e, "status_code", None) == 403, f"Origin: {bad_origin.decode()} must be 403"
        # cross-origin is still rejected
        try:
            app.get_current_admin(make_request(b"http://evil.example.com"))
            raise AssertionError("cross-origin must be rejected")
        except Exception as e:
            assert getattr(e, "status_code", None) == 403
        # same-origin admin still works
        assert app.get_current_admin(make_request(b"http://127.0.0.1:4000")) == "admin"
    finally:
        app.SESSIONS.pop("testsid", None)


def test_uvicorn_defaults_to_loopback():
    import inspect
    import app
    src = inspect.getsource(app)
    assert 'os.getenv("HOST", "127.0.0.1")' in src, "uvicorn must default to 127.0.0.1, not 0.0.0.0"


def _long_conv():
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug in app.py"},
    ]
    for i in range(10):
        msgs.append({"role": "assistant", "content": f"step {i} done"})
        msgs.append({"role": "user", "content": f"continue {i}"})
    return msgs


def test_clip_text_deterministic_head_tail():
    text = " ".join(f"token{i}" for i in range(2000))
    a = clip_text(text, 100, label="result from Bash")
    b = clip_text(text, 100, label="result from Bash")
    assert a == b, "clip_text must be deterministic"
    assert count_tokens(a) <= 100 + 20, "clipped text must respect the cap (plus marker overhead)"
    assert "token0" in a, "head must be preserved"
    assert "token1999" in a, "tail must be preserved"
    assert "elided" in a, "elision marker must be present"


def test_clip_text_noop_under_cap():
    text = "short output"
    assert clip_text(text, 100) == text


def test_compact_history_tiers_and_elides_old_tools():
    fat = "x" * 8000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"call {i}"})
        msgs.append({"role": "tool", "name": "Bash", "tool_call_id": f"c{i}", "content": fat})
    msgs.append({"role": "user", "content": "RECENT QUESTION verbatim"})
    hist = compact_history(msgs, budget_tokens=100000, recent_turns=2)
    assert "RECENT QUESTION verbatim" in hist, "recent window stays verbatim"
    assert "tool Bash: [~" in hist, "old tool results become stubs naming the tool"
    fat_lines = [l for l in hist.split("\n") if "x" * 100 in l]
    assert len(fat_lines) == 1, "only the recent window's clipped result may carry fat content"


def test_build_prompt_first_message_under_budget():
    fat = "y" * 4000
    msgs = [{"role": "system", "content": "You are an agent."}]
    for i in range(100):
        msgs.append({"role": "user", "content": f"step {i}"})
        msgs.append({"role": "assistant", "content": f"working on {i}"})
        msgs.append({"role": "tool", "name": "Read", "tool_call_id": f"c{i}", "content": fat})
    msgs.append({"role": "user", "content": "final question"})
    prompt = asyncio.run(build_prompt(msgs, [], "expert", is_first_message=True))
    assert count_tokens(prompt) <= DEFAULT_PROMPT_BUDGET, "prompt must fit the default budget"


def test_tolerant_signature_survives_middle_rewrite():
    msgs1 = _long_conv()
    msgs2 = [dict(m) for m in msgs1]
    msgs2[4]["content"] = "REWRITTEN BY COMPACTION"
    assert generate_signature_sync(msgs1, "expert") == generate_signature_sync(msgs2, "expert"), \
        "middle-of-history rewrite must not change the signature"


def test_tolerant_signature_distinguishes_different_bodies():
    msgs1 = _long_conv()
    msgs2 = [dict(m) for m in msgs1]
    # the final user message is post-assistant (excluded from the key), so
    # mutate the last assistant message, which IS inside the tail window
    msgs2[-2]["content"] = "a genuinely different final assistant message"
    assert generate_signature_sync(msgs1, "expert") != generate_signature_sync(msgs2, "expert"), \
        "different tails must produce different signatures"


def test_truncation_at_last_assistant_still_applies():
    msgs = _long_conv()
    extended = msgs + [
        {"role": "user", "content": "new question"},
        {"role": "tool", "name": "Bash", "tool_call_id": "c1", "content": "result"},
    ]
    assert generate_signature_sync(msgs, "expert") == generate_signature_sync(extended, "expert"), \
        "post-assistant messages must stay out of the key"


def test_next_sig_survives_client_side_compaction():
    model_output = '<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>'
    parsed_tools, _ = parse_tools(model_output)
    conv1 = _long_conv()
    server_msgs = conv1 + [{"role": "assistant", "tool_calls": parsed_tools}]
    next_sig = generate_signature_sync(server_msgs, "expert")

    conv2 = [dict(m) for m in conv1]
    conv2[4]["content"] = "REWRITTEN BY COMPACTION"
    client_msgs = conv2 + [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file.txt"},
        ]},
        {"role": "user", "content": "and now?"},
    ]
    converted = convert_anthropic_messages(client_msgs)
    assert generate_signature_sync(converted, "expert") == next_sig, \
        "post-compaction client request must hit the next_sig row"


def test_k_zero_disables_tolerance():
    import unittest.mock as mock
    msgs1 = _long_conv()
    msgs2 = [dict(m) for m in msgs1]
    msgs2[4]["content"] = "REWRITTEN BY COMPACTION"
    with mock.patch.dict(os.environ, {"DEEPSEEKER_SIG_WINDOW_K": "0"}):
        assert generate_signature_sync(msgs1, "expert") != generate_signature_sync(msgs2, "expert"), \
            "K=0 must restore full-history sensitivity"


def test_k_change_invalidates_cache():
    import unittest.mock as mock
    msgs = _long_conv()
    with mock.patch.dict(os.environ, {"DEEPSEEKER_SIG_WINDOW_K": "8"}):
        s8 = generate_signature_sync(msgs, "expert")
    with mock.patch.dict(os.environ, {"DEEPSEEKER_SIG_WINDOW_K": "12"}):
        s12 = generate_signature_sync(msgs, "expert")
    assert s8 != s12, "changing K must invalidate all cached keys"


def test_hist_len_guard_and_storage():
    db, old_db, cm = _temp_db({})
    try:
        import functions
        functions.init_db()
        functions.save_session("sigA", 1, "ds1", 0, hist_len=3)
        sess = functions.find_session("sigA")
        assert sess["hist_len"] == 3
        # one-sided guard: shorter-or-equal continues, longer is a foreign hit
        assert session_hit_ok(sess, 3) and session_hit_ok(sess, 2)
        assert not session_hit_ok(sess, 5)
        legacy = {"hist_len": None}
        assert session_hit_ok(legacy, 999), "legacy rows bypass the guard"
    finally:
        _restore_db(old_db, cm)


def test_save_without_hist_len_stores_null():
    db, old_db, cm = _temp_db({})
    try:
        import functions
        functions.init_db()
        functions.save_session("sigB", 1, "ds2", 0)
        assert functions.find_session("sigB")["hist_len"] is None
    finally:
        _restore_db(old_db, cm)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
