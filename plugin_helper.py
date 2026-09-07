import asyncio
import base64
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import random
import re
import socket
import string
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
import deepseek_tokenizer
from functions import count_tokens, get_session, upload_file

logger = logging.getLogger("deeperseeker.compaction")


def _env_int(name, default):
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


# --- context-size compaction -------------------------------------------------

# Head+tail split for clip_text: ~70% of the kept tokens come from the head.
_CLIP_HEAD_RATIO_NUM = 7
_CLIP_HEAD_RATIO_DEN = 10


def _clip_text_impl(text, max_tokens, label="output"):
    """Deterministic head+tail clip at the token level.

    Returns (clipped_text, elided_tokens). If text fits the cap it is returned
    unchanged with 0 elided. The elided middle is replaced by a marker naming
    the measured token count so the model knows what is missing.
    """
    text = str(text)
    max_tokens = int(max_tokens)
    tokens = deepseek_tokenizer.ds_token.encode(text)
    n = len(tokens)
    if n <= max_tokens:
        return text, 0
    head_n = max(0, (max_tokens * _CLIP_HEAD_RATIO_NUM) // _CLIP_HEAD_RATIO_DEN)
    tail_n = max(0, max_tokens - head_n)
    if head_n + tail_n > n:
        # Degenerate cap (tiny max_tokens): keep nothing from the middle.
        head_n, tail_n = min(n, max_tokens), 0
    elided = n - head_n - tail_n
    tok = deepseek_tokenizer.ds_token
    head = tok.decode(tokens[:head_n]) if head_n else ""
    tail = tok.decode(tokens[n - tail_n:]) if tail_n else ""
    marker = f"\n[... ~{elided} tokens of {label} elided ...]\n"
    return head + marker + tail, elided


def clip_text(text, max_tokens, label="output"):
    """Public wrapper: clip text to ~max_tokens with a head+tail split."""
    return _clip_text_impl(text, max_tokens, label)[0]


# Config defaults (overridable via env vars, read at call time so changes do
# not require a restart and tests can patch them).
DEFAULT_PROMPT_BUDGET = 24000
DEFAULT_TOOL_RESULT_CLIP = 4000
DEFAULT_HISTORY_RECENT_TURNS = 6

# Older plain user/assistant text kept in the injected history is clipped to
# roughly this many tokens each; the budget degradation loop tightens it.
OLD_TEXT_CLIP_TOKENS = 200
# The verbatim recent window never shrinks below this, even under budget
# pressure.
MIN_RECENT_TURNS = 2
# Floor for the "clip older text harder" degradation phase.
MIN_OLD_TEXT_CLIP = 50
# Floor for the degradation phase that tightens the tool-result clip inside
# the history section (the trailing [TOOL RESULTS] section always uses
# DEEPSEEKER_TOOL_RESULT_CLIP unchanged).
MIN_HIST_TOOL_CLIP = 256

_TOOL_ELIDED_RE = "tokens of result elided — see earlier result"


def _message_text(msg):
    """Plain-text content of a message (list content -> joined text parts)."""
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    return content if isinstance(content, str) else str(content)


def _tool_result_parts(msg):
    """Extract (tool_name_or_id, result_text) pairs from a message."""
    parts = []
    if msg.get("role") == "tool":
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        parts.append((msg.get("name") or "tool", content if isinstance(content, str) else str(content)))
    elif isinstance(msg.get("content"), list):
        for c in msg["content"]:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                res = c.get("content", "")
                if isinstance(res, list):
                    res = " ".join(
                        item.get("text", "") for item in res
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                parts.append((c.get("tool_use_id") or "tool", res if isinstance(res, str) else str(res)))
    return parts


def _assistant_tool_call_names(msg):
    names = []
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            name = fn.get("name") or tc.get("name")
            if name:
                names.append(name)
    return names


def _render_history_message(msg, tool_result_clip, old_text_clip, tier):
    """Render one history message. tier is "recent" or "older".

    Returns (line, tokens_elided). Line may be empty (message skipped).
    """
    role = msg.get("role", "unknown")
    elided_total = 0

    tool_parts = _tool_result_parts(msg)
    if role == "tool" or (tool_parts and role != "assistant"):
        lines = []
        for name, result in tool_parts:
            if tier == "recent":
                clipped, elided = _clip_text_impl(
                    result, tool_result_clip, label=f"result from {name}")
                elided_total += elided
                lines.append(f"TOOL {name}: {clipped}".strip())
            else:
                # One-liner stub: keeps the tool/result pairing readable
                # without re-sending the (earlier) full result.
                n = count_tokens(result)
                elided_total += n
                lines.append(f"tool {name}: [~{n} {_TOOL_ELIDED_RE}]")
        # Plain text alongside the tool results is preserved — except for
        # role=="tool" messages, whose string content IS the result itself.
        text = _message_text(msg)
        if role != "tool" and text.strip():
            lines.append(f"{role.upper()}: {text}")
        return "\n".join(lines), elided_total

    text = _message_text(msg)
    if not text.strip():
        names = _assistant_tool_call_names(msg)
        if names:
            # Keeps the tool-call/tool-result pairing visible in both tiers.
            return f"ASSISTANT: [made tool call(s): {', '.join(names)}]", 0
        return "", 0

    if tier == "older" and old_text_clip is not None:
        text, elided = _clip_text_impl(text, old_text_clip, label="older message")
        elided_total += elided
    return f"{role.upper()}: {text}", elided_total


def _compact_history_impl(messages, budget_tokens, recent_turns,
                          tool_result_clip=None, old_text_clip=None):
    """Build the [PREVIOUS CONVERSATION HISTORY] string under a token budget.

    Tiering, most-verbatim first:
      - the most recent `recent_turns` non-system messages verbatim (tool
        results inside them still clipped to tool_result_clip tokens)
      - older role=="tool" messages (or tool-result content) reduced to
        one-liner stubs that still name the producing tool
      - older plain user/assistant text kept but clipped to old_text_clip
    If the result is still over budget_tokens, oldest entries are dropped
    first; the recent window is never shrunk here.
    """
    if tool_result_clip is None:
        tool_result_clip = _env_int("DEEPSEEKER_TOOL_RESULT_CLIP", DEFAULT_TOOL_RESULT_CLIP)
    if old_text_clip is None:
        old_text_clip = OLD_TEXT_CLIP_TOKENS
    recent_turns = max(MIN_RECENT_TURNS, int(recent_turns))

    non_system = [m for m in messages if m.get("role") != "system"]
    recent = non_system[-recent_turns:]
    older = non_system[:-recent_turns] if len(non_system) > recent_turns else []

    older_lines = []  # (line, tokens)
    elided_total = 0
    for m in older:
        line, elided = _render_history_message(m, tool_result_clip, old_text_clip, "older")
        if line:
            older_lines.append((line, count_tokens(line)))
            elided_total += elided
    recent_lines = []
    for m in recent:
        line, elided = _render_history_message(m, tool_result_clip, old_text_clip, "recent")
        if line:
            recent_lines.append(line)
            elided_total += elided

    # Drop oldest tier entries until the history fits (recent window intact).
    recent_tokens = sum(count_tokens(l) for l in recent_lines) + len(recent_lines)
    if budget_tokens is not None and budget_tokens > 0:
        kept_tokens = sum(t for _, t in older_lines) + len(older_lines) + recent_tokens
        while older_lines and kept_tokens > budget_tokens:
            _, t = older_lines.pop(0)
            kept_tokens -= t
            elided_total += t

    lines = [line for line, _ in older_lines] + recent_lines
    return "\n".join(lines), elided_total


def compact_history(messages, budget_tokens, recent_turns,
                    tool_result_clip=None, old_text_clip=None):
    """History string for the [PREVIOUS CONVERSATION HISTORY] section."""
    return _compact_history_impl(
        messages, budget_tokens, recent_turns, tool_result_clip, old_text_clip)[0]


async def extract_system(messages):
    for i in messages:
        if i.get("role") == "system":
            return i.get("content")
    return None


async def extract_tools(tools):
    if not tools:
        return None
    final_tools = []
    for i in tools:
        if i.get("type") == "function":
            fn = i.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            final_tools.append(f"Tool: {name}\nDescription: {desc}\nParameters: {json.dumps(params)}")
        elif "name" in i:
            name = i.get("name", "")
            desc = i.get("description", "")
            params = i.get("input_schema", i.get("parameters", {}))
            final_tools.append(f"Tool: {name}\nDescription: {desc}\nParameters: {json.dumps(params)}")
        elif i.get("type") in ["computer_use", "text_editor", "bash"]:
            final_tools.append(f"Tool: {i['type']}\nDescription: {json.dumps(i)}")
        else:
            final_tools.append(f"Tool: {json.dumps(i)}")
    return "\n\n".join(final_tools) if final_tools else None


async def extract_tool_results(messages, latest_only=False):
    target_messages = messages
    if latest_only:
        last_ast_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                last_ast_idx = idx
                break
        if last_ast_idx != -1:
            target_messages = messages[last_ast_idx + 1:]
    tools_final = []
    clip = _env_int("DEEPSEEKER_TOOL_RESULT_CLIP", DEFAULT_TOOL_RESULT_CLIP)
    elided_total = 0
    for i in target_messages:
        if i.get("role") == "tool":
            name = i.get("name") or "tool"
            call_id = i.get("tool_call_id", "")
            content = i.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
            content, elided = _clip_text_impl(content, clip, label=f"result from {name}")
            elided_total += elided
            tools_final.append(f"Tool: {name} (Call ID: {call_id})\nResult: {content}")
    if elided_total:
        logger.info(
            "tool result clip: ~%d tokens elided across [TOOL RESULTS] (cap %d tokens per result)",
            elided_total, clip)
    return "\n\n".join(tools_final) if tools_final else None


def _assert_public_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("unsupported url")
    infos = socket.getaddrinfo(parts.hostname, None)
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("url resolves to non-public address")


async def _fetch_url_bytes(session, url, max_bytes=20 * 1024 * 1024 + 1, max_redirects=5):
    """Fetch a URL with redirects followed manually so every hop is re-checked
    against _assert_public_url (a redirect must not bypass the SSRF guard).

    Residual risk: DNS-rebinding TOCTOU — _assert_public_url resolves the host
    for validation, but aiohttp re-resolves independently when connecting, so
    an attacker controlling DNS could still serve a private IP at connect time.
    Full IP-pinning is out of scope here.
    """
    current_url = url
    for _ in range(max_redirects + 1):
        _assert_public_url(current_url)
        async with session.get(
            current_url,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise ValueError("redirect response without Location header")
                current_url = urljoin(current_url, location)
                continue
            return await resp.content.read(max_bytes)
    raise ValueError("too many redirects while fetching url")


def _b64(data):
    data = re.sub(r"[^A-Za-z0-9+/=]", "", data)
    try:
        return base64.b64decode(data + "=" * (-len(data) % 4))
    except Exception:
        return None


async def extract_and_upload_files(messages, auth_token, last_user_only=False):
    result_fileids = []
    scan = messages
    if last_user_only:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                scan = messages[idx:]
                break
    for idx, i in enumerate(scan):
        content = i.get("content")
        if not content:
            continue
        if isinstance(content, str):
            continue
        for j_idx, j in enumerate(content):
            if j["type"] == "text":
                continue
            elif j["type"] == "image_url":
                if j["image_url"]["url"].startswith("http"):
                    _assert_public_url(j["image_url"]["url"])
                    url_path = urlsplit(j["image_url"]["url"]).path
                    filename = Path(url_path).name
                    mime_type, _ = mimetypes.guess_type(filename)
                    session = await get_session()
                    file_bytes = await _fetch_url_bytes(session, j["image_url"]["url"])
                    if len(file_bytes) > 20 * 1024 * 1024:
                        continue

                    async for k in upload_file(file_bytes, filename, mime_type, auth_token):
                        if k[0] == "uploaded":
                            continue
                        elif k[0] == "success":
                            result_fileids.append(k[1]["file_id"])

                else:
                    url_parts = j["image_url"]["url"].split(",", 1)
                    if len(url_parts) != 2:
                        continue
                    mimetype_base, base64_data = url_parts
                    mime_type = mimetype_base.split(":")[1].split(";")[0]
                    filename = (
                        "inline_uploaded_"
                        + str(uuid.uuid4())
                        + (mimetypes.guess_extension(mime_type) or ".bin")
                    )
                    data_bytes = _b64(
                        (base64_data.split("data:")[1] if "data:" in base64_data else base64_data)
                    )
                    if data_bytes is None:
                        continue
                    async for k in upload_file(data_bytes, filename, mime_type, auth_token):
                        if k[0] == "uploaded":
                            continue
                        elif k[0] == "success":
                            result_fileids.append(k[1]["file_id"])
            elif j["type"] == "file":
                if "file_id" in j["file"]:
                    result_fileids.append(j["file"]["file_id"])
                if "file_data" in j["file"]:
                    filename = j["file"]["filename"]
                    data_parts = j["file_data"].split(",", 1)
                    if len(data_parts) != 2:
                        continue
                    mimetype_base, base64_data = data_parts

                    mime_type = mimetype_base.split(":")[1].split(";")[0]
                    data_bytes = _b64(
                        (base64_data.split("data:")[1] if "data:" in base64_data else base64_data)
                    )
                    if data_bytes is None:
                        continue
                    async for k in upload_file(data_bytes, filename, mime_type, auth_token):
                        if k[0] == "uploaded":
                            continue
                        elif k[0] == "success":
                            result_fileids.append(k[1]["file_id"])
            elif j["type"] == "document" or j["type"] == "image":
                if j["source"]["type"] == "base64":
                    base64_data = j["source"]["data"].split(",")[1] if "," in j["source"]["data"] else j["source"]["data"]
                    mime_type = j["source"]["media_type"]
                    filename = (
                        "inline_uploaded_"
                        + str(uuid.uuid4())
                        + (mimetypes.guess_extension(mime_type) or ".bin")
                    )
                    data_bytes = _b64(base64_data)
                    if data_bytes is None:
                        continue
                    async for k in upload_file(data_bytes, filename, mime_type, auth_token):
                        if k[0] == "uploaded":
                            continue
                        elif k[0] == "success":
                            result_fileids.append(k[1]["file_id"])
                elif j["source"]["type"] == "file":
                    result_fileids.append(j["source"]["file_id"])
    return result_fileids


async def extract_user_msg(messages):
    for i in messages[::-1]:
        if i.get("role") == "user":
            content = i.get("content")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                parts = []
                for j in content:
                    if isinstance(j, dict) and j.get("type") == "text":
                        parts.append(j.get("text", ""))
                if parts:
                    return "\n".join(parts)
    return ""


def canonicalize_messages(messages):
    canon = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        tool_calls = m.get("tool_calls")

        if tool_calls and isinstance(tool_calls, list):
            tc_parts = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name") or tc.get("name")
                args = fn.get("arguments") or tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                tc_json = json.dumps({"arguments": args, "name": name}, sort_keys=True)
                tc_parts.append("<tool_call>" + tc_json + "</tool_call>")
            content = "\n".join(tc_parts)
        elif isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "tool_use":
                        args = c.get("input", {})
                        tc_json = json.dumps({"arguments": args, "name": c.get("name")}, sort_keys=True)
                        parts.append("<tool_call>" + tc_json + "</tool_call>")
                    elif c.get("type") == "tool_result":
                        res_content = c.get("content", "")
                        if isinstance(res_content, list):
                            res_content = " ".join(item.get("text", "") for item in res_content if isinstance(item, dict) and item.get("type") == "text")
                        tool_id = c.get("tool_use_id", "tool")
                        parts.append("[Tool Result for " + str(tool_id) + "]: " + str(res_content))
            content = "\n".join(parts)
        elif isinstance(content, str):
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            def repl_tc(match):
                raw_json = match.group(1).strip()
                try:
                    d = json.loads(raw_json)
                    d_name = d.get("name")
                    d_args = d.get("arguments", {})
                    return "<tool_call>" + json.dumps({"arguments": d_args, "name": d_name}, sort_keys=True) + "</tool_call>"
                except Exception:
                    return match.group(0)
            content = re.sub(r"<tool_call>(.*?)</tool_call>", repl_tc, content, flags=re.DOTALL)

        canon.append({"role": role, "content": str(content).strip()})
    return canon


SIG_NAMESPACE = "ds-sig-tolerant-v1"


def sig_window_k():
    """Tolerant-signature tail window (DEEPSEEKER_SIG_WINDOW_K, default 8).

    Read at call time so tests can patch the env. K=0 restores full-history
    sensitivity (every middle-of-history rewrite misses); it does NOT restore
    legacy hash values.
    """
    try:
        return max(0, int(os.getenv("DEEPSEEKER_SIG_WINDOW_K", "8")))
    except (TypeError, ValueError):
        return 8


def project_signature(messages, model, scope=""):
    """Tolerant cache key: (sig, hist_len).

    Hashes a stable projection of the conversation — namespace, window size K,
    canonical system prompt, first non-system message (anchor) and the last K
    canonical non-system messages — instead of the full canonical history, so
    client-side context compaction that rewrites the middle keeps hitting the
    same cached DeepSeek session. The history is still truncated at the last
    assistant message (post-assistant tool results / user text are carried by
    the prompt, not the key). hist_len is the truncated history's non-system
    message count, returned separately for the one-sided length guard and
    never hashed.
    """
    last_ast_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_ast_idx = i
            break

    history = messages if last_ast_idx == -1 else messages[:last_ast_idx + 1]
    canon_history = canonicalize_messages(history)
    non_system = [m for m in canon_history if m.get("role") != "system"]

    system = ""
    for m in canon_history:
        if m.get("role") == "system":
            system = m.get("content", "")
            break
    anchor = non_system[0].get("content", "") if non_system else ""
    k = sig_window_k()
    tail = non_system if k == 0 else non_system[-k:]

    payload = {"v": SIG_NAMESPACE, "k": k, "system": system, "anchor": anchor, "tail": tail}
    dump = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    sig = hashlib.sha256(f"{model}_{scope}_{dump}".encode("utf-8")).hexdigest()
    return sig, len(non_system)


def generate_signature_sync(messages, model, scope=""):
    return project_signature(messages, model, scope)


async def generate_signature(messages, model, scope=""):
    return project_signature(messages, model, scope)


def _pre_assistant_history(messages):
    """Non-system messages up to and including the last assistant message.

    Everything after the last assistant (current-turn tool results / user text)
    is carried by the [TOOL RESULTS] and [USER] sections instead, mirroring the
    non-first-message path, so history and tool-result sections never duplicate
    the whole conversation.
    """
    last_ast_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_ast_idx = idx
            break
    if last_ast_idx == -1:
        return []
    return [m for m in messages[:last_ast_idx + 1] if m.get("role") != "system"]


def _tool_call_example(tools):
    """Concrete <tool_call> example built from the caller's first tool.

    The old static example ({"name": "tool_name", "arguments": {"param":
    "value"}}) was copied verbatim by deepseek-v4-pro, producing a call to a
    nonexistent "tool_name" tool. Building the example from the real schema
    (real tool name, real parameter names, "<value>" placeholders) keeps the
    format instruction while making literal copying obvious and unattractive.
    """
    fn = None
    for t in tools or []:
        if t.get("type") == "function":
            fn = t.get("function", {})
        elif "name" in t:
            fn = t
        if fn and fn.get("name"):
            break
        fn = None
    if not fn:
        return '<tool_call>{"name": "the_tool_name", "arguments": {"argument": "value"}}</tool_call>'
    params = fn.get("parameters") or fn.get("input_schema") or {}
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    args = {}
    for key, spec in list(props.items())[:3]:
        ptype = spec.get("type", "string") if isinstance(spec, dict) else "string"
        if ptype in ("number", "integer"):
            args[key] = 0
        elif ptype == "boolean":
            args[key] = False
        else:
            args[key] = "<value>"
    return '<tool_call>{"name": "%s", "arguments": %s}</tool_call>' % (fn["name"], json.dumps(args))


async def build_prompt(messages, tools, model, is_first_message=False):
    final_prompt = ""
    tools_extract = await extract_tools(tools)
    tool_instructions = (
        "TOOL USE INSTRUCTIONS:\n"
        "You have access to tools. When you need to call a tool, output ONLY the tool call XML block and nothing else:\n"
        f"{_tool_call_example(tools)}\n"
        "This shows the required FORMAT only: use the actual tool's name and real argument values for the task — never copy the placeholder <value> literals. "
        "Never repeat past messages, history, or XML tags. Output exactly one tool call block when invoking a tool."
    )
    if is_first_message:
        head = ""
        if tools_extract:
            head += f"[TOOLS]\n{tools_extract}\n\n"
        system_prompt = await extract_system(messages)
        if system_prompt:
            if tools_extract:
                system_prompt += "\n\n" + tool_instructions
            head += f"[SYSTEM]\n{system_prompt}\n\n"
        elif tools_extract:
            head += f"[SYSTEM]\n{tool_instructions}\n\n"

        # Only the trailing (current-turn) tool results go into [TOOL RESULTS];
        # older results are carried by the compacted history below (stubs for
        # old turns, verbatim-clipped inside the recent window). Sending every
        # historical result here again would make the prompt unbounded.
        tools_result_extract = await extract_tool_results(messages, latest_only=True)
        tail = ""
        if tools_result_extract:
            tail += f"[TOOL RESULTS]\n{tools_result_extract}\n\n"

        user_msg = await extract_user_msg(messages)
        if user_msg:
            tail += f"[USER]\n{user_msg}\n\n"

        final_prompt = head + tail

        if len(messages) > 1:
            history_msgs = _pre_assistant_history(messages)
            if history_msgs:
                budget = _env_int("DEEPSEEKER_PROMPT_BUDGET", DEFAULT_PROMPT_BUDGET)
                recent = _env_int("DEEPSEEKER_HISTORY_RECENT_TURNS", DEFAULT_HISTORY_RECENT_TURNS)
                old_clip = OLD_TEXT_CLIP_TOKENS
                overhead = count_tokens(head) + count_tokens(tail) + 100
                hist_budget = max(0, budget - overhead)
                history_text, elided = _compact_history_impl(
                    history_msgs, hist_budget, recent, old_text_clip=old_clip)
                prompt = head + f"[PREVIOUS CONVERSATION HISTORY]\n{history_text}\n\n" + tail
                total = count_tokens(prompt)
                if total > budget:
                    # Progressive degradation: shrink the verbatim window by 2
                    # (floored at MIN_RECENT_TURNS), then clip older text
                    # harder, re-compacting and re-measuring each round. The
                    # [SYSTEM] and final [USER] sections are never clipped.
                    while recent > MIN_RECENT_TURNS or old_clip > MIN_OLD_TEXT_CLIP:
                        if recent > MIN_RECENT_TURNS:
                            recent = max(MIN_RECENT_TURNS, recent - 2)
                        else:
                            old_clip = max(MIN_OLD_TEXT_CLIP, old_clip // 2)
                        history_text, elided_n = _compact_history_impl(
                            history_msgs, hist_budget, recent, old_text_clip=old_clip)
                        elided += elided_n
                        prompt = head + f"[PREVIOUS CONVERSATION HISTORY]\n{history_text}\n\n" + tail
                        total = count_tokens(prompt)
                        if total <= budget:
                            break
                if elided:
                    logger.info(
                        "prompt compaction: %d prompt tokens (budget %d), ~%d tokens elided",
                        total, budget, elided)
                final_prompt = prompt
    else:
        last_ast_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                last_ast_idx = idx
                break

        trailing_messages = messages[last_ast_idx + 1:] if last_ast_idx != -1 else [messages[-1]]
        tools_result_extract = await extract_tool_results(messages, latest_only=True)
        if tools_result_extract:
            final_prompt += f"[TOOL RESULTS]\n{tools_result_extract}\n\n"

        trailing_user_parts = []
        for m in trailing_messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and c:
                    trailing_user_parts.append(c)
                elif isinstance(c, list):
                    txt = " ".join(part.get("text", "") for part in c if isinstance(part, dict) and part.get("type") == "text")
                    if txt:
                        trailing_user_parts.append(txt)

        if trailing_user_parts:
            final_prompt += f"[USER]\n{chr(10).join(trailing_user_parts)}\n\n"
        elif not tools_result_extract:
            user_msg = await extract_user_msg(messages)
            if user_msg:
                final_prompt += f"[USER]\n{user_msg}\n\n"

        if tools_extract:
            final_prompt += tool_instructions + "\n"

    return final_prompt
