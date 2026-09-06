"""Regression tests for the 2026-09-06 local fixes.

Run with the repo venv:  deeperseeker_env/Scripts/python.exe tests/test_local_fixes.py
(also pytest-compatible).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import API_KEY, convert_anthropic_messages
from functions import parse_tools
from plugin_helper import build_prompt, generate_signature_sync


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
