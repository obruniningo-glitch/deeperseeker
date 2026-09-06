# How Our DeeperSeeker Works

**Status:** fork of [AmanCode22/deeperseeker](https://github.com/AmanCode22/deeperseeker) (`upstream`), maintained at [obruniningo-glitch/deeperseeker](https://github.com/obruniningo-glitch/deeperseeker) (`origin`), branch `main`. This document describes the fork as of commit `f6edab9` (2026-09-06) — 13 commits ahead of upstream.

---

## 1. What it is

DeeperSeeker is a local FastAPI gateway that turns DeepSeek's *web* chat into standard LLM APIs:

- `POST /v1/chat/completions` — OpenAI-compatible (chat + tool calling)
- `POST /v1/messages` — Anthropic-compatible (messages + tool_use/tool_result)
- plus `/v1/models`, file endpoints, and an admin dashboard for token management

It exists so tools that speak those APIs (Claude Code, ZCode, Claude Desktop, any OpenAI SDK client) can use DeepSeek's web models (`instant` → V4 Flash, `expert` → V4 Pro) with full **agentic tool calling**, which the web UI does not natively support.

## 2. Request flow (the big picture)

```
client request (OpenAI or Anthropic format)
  │
  ├─ /v1/messages: convert_anthropic_messages()   app.py
  │    tool_result blocks → role="tool" messages
  │    tool_use blocks    → assistant tool_calls
  │
  ├─ signature → session cache                    plugin_helper.py / functions.py
  │    HIT  → reuse DeepSeek chat + parent_message_id (cheap delta prompt)
  │    MISS → new DeepSeek chat, full-history injection (compacted, see §5)
  │
  ├─ token pool (pick_token, rate-limit marking)  functions.py
  ├─ PoW challenge (WASM solver) + cookies        functions.py
  ├─ send_message() → DeepSeek web SSE            functions.py
  │
  └─ response path
       StreamToolParser → extracts tool calls from free text
       re-emitted as OpenAI/Anthropic SSE (tool_calls deltas, reasoning_content)
       on completion: save sig + next_sig rows (parent+2, hist_len)
```

## 3. Tool calling without native support

DeepSeek web knows nothing about tools. The gateway bridges the gap:

1. **Prompt injection** (`build_prompt`): tool schemas are serialized into the first message with instructions to emit `<tool_call>{"name":…,"arguments":{…}}</tool_call>` blocks.
2. **Extraction** (`parse_tools`, `StreamToolParser`): a layered regex/state machine recognizes several tool-call shapes (attribute tags, DSML-style blocks, bare JSON after tags, fenced blocks) — but ignores such markup inside markdown code fences, keeps repeated identical calls, and preserves prose that accompanies a call.
3. **Round-tripping**: results come back as `role:"tool"` (OpenAI) or `tool_result` blocks (Anthropic), are converted to a common form, and fed to the model on the next turn via a `[TOOL RESULTS]` section.

## 4. The session cache (and why signatures are "tolerant")

DeepSeek web keeps conversation state server-side per chat session, so the gateway caches which DeepSeek chat belongs to which conversation instead of re-sending history on every turn.

- **Key:** `project_signature()` hashes a *projection* — namespace + window size K + canonical system prompt + first non-system message (anchor) + the **last K=8 canonical messages** — not the full history. Truncation at the last assistant message excludes the current turn's tool results/user text.
- **Why tolerant:** when the client (Claude Code, etc.) compacts its own context, the middle of the history is rewritten. A full-history hash would then miss on *every* subsequent turn, forcing a full re-injection each time. The tail window keeps matching through middle rewrites.
- **Guard:** a one-sided `hist_len` check rejects hits whose truncated history is *longer* than the stored session ever saw (impossible for a true continuation; characteristic of a foreign conversation). Legacy rows (`hist_len = NULL`) bypass the guard.
- **Bookkeeping:** every successful turn saves two rows — the request's key (pointing at the chat with `parent_message_id + 2`) and the `next_sig` of `messages + [reconstructed assistant turn]`. The reconstruction canonicalizes identically to what clients echo back; a dedicated regression test pins this.
- **Rollback:** `DEEPSEEKER_SIG_WINDOW_K=0` restores full-history sensitivity (not legacy hash values).

## 5. Context compaction

Cache misses re-inject the entire conversation as one first message. Three mechanisms keep that bounded:

| Mechanism | Env var | Default | Effect |
|---|---|---|---|
| Prompt budget | `DEEPSEEKER_PROMPT_BUDGET` | 24000 | total first-message injection cap; progressive degradation (shrink verbatim window → clip older text harder) |
| Tool-result clipping | `DEEPSEEKER_TOOL_RESULT_CLIP` | 4000 | per-result head+tail clip with an elision marker; applies to every tool result, in both prompt paths |
| Verbatim window | `DEEPSEEKER_HISTORY_RECENT_TURNS` | 6 | recent turns stay verbatim; older tool results become one-line stubs naming the tool |

The clipper is **deterministic** by design — identical input yields identical output — so compaction never perturbs signature keys. The system prompt and the final user message are never clipped.

## 6. Security posture

- API key checked with constant-time comparison; an empty `DEEPSEEKER_API_KEY` falls back to the default instead of failing open.
- Optional at-rest encryption of DeepSeek tokens (SQLite) and session cookies via Fernet when `DEEPSEEKER_ENCRYPTION_KEY` is set; plaintext behavior unchanged if unset, with transparent migrate-on-read.
- Admin dashboard: session cookies, origin check, `Origin: null` rejected; login lockout after repeated failures.
- Image fetches (SSRF surface): no auto-redirects — every redirect hop is re-validated against public-IP rules, with a 15 s timeout. *Known residual:* DNS-rebinding TOCTOU (no IP pinning).
- Server binds `127.0.0.1` by default (README-correct); `HOST=0.0.0.0` opts into LAN exposure.

## 7. Reliability

- All outbound calls carry explicit timeouts (chat 300 s, uploads 120 s, URL fetches 15 s).
- Per-signature locks are removed after release (no unbounded dict growth); sessions table is pruned on a 7-day TTL (`DEEPSEEKER_SESSION_TTL_DAYS`).
- The `parent_message_id + 2` invariant is centralized in `next_parent()` with a documented rationale.
- A stream that ends mid-tool-call discards the partial markup instead of emitting XML fragments as text.

## 8. Divergence from upstream

| Theme | Commits (local hashes) |
|---|---|
| Anthropic tool-result/tool_use translation, prefix-fallback removal, API-key fail-open | `0302fde` |
| `parent_message_id` centralization (`next_parent`) | `c0bd902` |
| `parse_tools` hardening (fences, dedupe, companion text) | `bddaac0` |
| StreamToolParser flush, SSRF hardening, encryption-at-rest, TTL pruning, lock cleanup, timeouts, loopback default, Origin:null | the 7 remaining fix commits |
| Compaction + signature tolerance | `50c0959`, `22a0b32`, `f6edab9` |

Upstream remains tracked as the `upstream` remote; relevant fixes there (especially `functions.py` auth/PoW/stream-format churn) are cherry-picked into `main`.

## 9. Testing

`tests/test_local_fixes.py` — 34 standalone tests (also pytest-compatible), run with the repo venv:

```
deeperseeker_env/Scripts/python.exe tests/test_local_fixes.py
```

Covers the Anthropic conversion, signature cache invariants (including the server-reconstruction ↔ client-echo contract), parse_tools edge cases, redirect validation, encryption migration, compaction behavior, and signature tolerance (middle-rewrite survival, K=0 rollback, hist_len guard).

## 10. Operations quick reference

```
start:   start_deeperseeker.bat  (console window; uvicorn on 127.0.0.1:4000)
stop:    netstat -ano | findstr :4000   → taskkill /PID <pid> /F
smoke:   curl -s -H "Authorization: Bearer dseeker" http://127.0.0.1:4000/v1/models
update:  git fetch upstream && git log main..upstream/main --oneline   (cherry-pick what's relevant)
```
