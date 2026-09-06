# deeperseeker v2 — Greenfield Design Document

**Date:** 2026-09-06 · **Status:** proposal · **Grounding:** current fork at `C:/Users/obrun/AI_Tools/deeperseeker` (HEAD `f6edab9`, 13 commits ahead of upstream). All `app.py` / `functions.py` / `plugin_helper.py` line references are to that tree.

**Thesis.** v1 works because a dozen subtle invariants were discovered the hard way and pinned by tests and comments. v2's job is to make those invariants *structural* — enforced by types, state machines, and property tests — while keeping every behavior the fork earned: tolerant session cache, deterministic compaction, hardened tool-call parsing, protocol-correct streaming, fail-fast pools, encrypted secrets.

---

## 0. What v1 taught us (the failure classes v2 must make impossible)

| # | Failure class | Where it lived in v1 | v2 structural answer |
|---|---|---|---|
| F1 | Edge-format translation bugs (Anthropic `tool_result` flattened into user text → model re-answers the original question) | `app.py:838-902` `convert_anthropic_messages`, consumed by `plugin_helper.py:268-295` `extract_tool_results` which only understood OpenAI `role=="tool"` | Canonical IR; all translation in two unit-tested edge adapters; core never sees a client dict |
| F2 | Echo-match invariant fragility (server-reconstructed assistant turn must canonicalize byte-identically to the client's echo or the cache misses forever) | `app.py:347-361, 461-477` (reconstruction in `finally:` blocks), `plugin_helper.py:457-509` `canonicalize_messages`, `plugin_helper.py:528-563` `project_signature` | Store the canonical projection at save time; structured comparison instead of blind hash equality; echo equivalence becomes a typed, property-tested adapter property (§4) |
| F3 | Malformed SSE lifecycle (two open Anthropic content blocks → real clients drop the whole message) | `app.py:383-525` `stream_anthropic_response` — hand-wired index arithmetic, a `tail_events` string assembled in `finally` and replayed | Emitter is a total state machine over typed IR events; malformed sequences are unrepresentable; Hypothesis-verified (§5) |
| F4 | Regex-pile tool parsing (false positives on prose, destructive `clean_text`, fence handling) | `functions.py:517-707` `parse_tools` (~190 lines of layered regex with `[｜\|]` fullwidth-pipe variants), `functions.py:710-794` `StreamToolParser` | Pluggable per-model-family grammar: structured primary parser + free-text fallback; fuzz corpus of real malformed outputs; no-retract streaming contract (§6) |
| F5 | Compaction bolted onto prompt building and coupled to cache keys | `plugin_helper.py:38-236` clip/compact + `592-695` `build_prompt` with a degradation loop inline | First-class `TokenBudget` service applied at exactly one pipeline point; keys computed from *raw* IR, so determinism is a test property, not a correctness dependency (§7) |
| F6 | Provider logic fused into the HTTP layer (~1,200-line `app.py`; `handle_chat` at `app.py:136-254` does session lookup, pool, prompt, stream, save, retry) | whole file | `ProviderAdapter` protocol + thin pipeline; FastAPI routes only parse/authenticate/delegate (§3, §8) |
| F7 | Ops gaps grown by patching (per-call `sqlite3.connect` in `functions.py:112-119`, double `/health` definition at `app.py:968` and `app.py:1174` where the second silently shadows the first, ~10 env vars read at call time) | scattered | Repository layer on one pooled connection, one `/health`, one pydantic-settings object (§9) |

---

## 1. Architecture overview

### 1.1 Module layout (single package, single deployable)

```
ds2/
├── ir/
│   ├── message.py        # Message, ContentBlock, Conversation (the IR)
│   ├── canonical.py      # canonical form + projection payload (§4)
│   └── events.py         # ProviderEvent / SinkEvent stream types (§5)
├── edge/
│   ├── openai.py         # OpenAI wire ⇄ IR (chat/completions + responses)
│   ├── anthropic.py      # Anthropic wire ⇄ IR (messages)
│   └── errors.py         # wire-shaped error bodies per protocol
├── tools/
│   ├── registry.py       # per-model-family grammar selection
│   ├── grammar_dsml.py   # DeepSeek-family grammar (structured primary)
│   ├── grammar_freetext.py
│   ├── inject.py         # tool-schema → prompt section
│   └── corpus/           # recorded malformed outputs (fuzz fixtures)
├── compaction/
│   └── budget.py         # TokenBudget service (§7)
├── pipeline/
│   ├── orchestrator.py   # the ONLY place that composes everything
│   ├── session_cache.py  # projection store lookup/save (§4)
│   └── metering.py
├── streaming/
│   ├── emitter_openai.py
│   ├── emitter_anthropic.py
│   ├── validate.py       # SSE lifecycle invariant checker (used by tests)
│   └── sse.py            # frame serialization (10 lines)
├── providers/
│   ├── base.py           # ProviderAdapter protocol, Capabilities (§8)
│   ├── registry.py
│   ├── deepseek/
│   │   ├── adapter.py    # implements ProviderAdapter
│   │   ├── wire.py       # raw HTTP: create_chat, completion, files  (port of functions.py:860-1071)
│   │   ├── pow.py        # wasmtime solver (port of functions.py:797-864)
│   │   └── cookies.py    # WAF cookie refresh (port of functions.py:227-290)
│   └── fake.py           # scripted in-memory provider for tests
├── store/
│   ├── db.py             # aiosqlite, WAL, migrations
│   ├── repo_sessions.py / repo_tokens.py / repo_usage.py
│   └── crypto.py         # Fernet at-rest (port of functions.py:29-109)
├── ops/
│   ├── config.py         # single pydantic-settings Settings
│   ├── logging.py        # structlog + request-id contextvar
│   ├── health.py
│   └── metrics.py        # prometheus-client, optional
└── api/
    ├── app.py            # FastAPI factory, middleware, ~50 lines per route
    ├── deps.py           # auth (constant-time key), request-id
    └── admin.py          # dashboard, token CRUD
```

Dependency rule: `api → pipeline → {ir, tools, compaction, providers, store}`; `providers` and `edge` never import each other; nothing in `core` imports FastAPI.

### 1.2 Data flow

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                        edge adapters                     │
 client (Claude Code,   │  ┌──────────────┐   IR    ┌───────────────────────────┐  │
 ZCode, OpenAI SDK) ────┼─▶│ /v1/messages │────────▶│      pipeline.orchestrator │  │
 OpenAI or Anthropic    │  │ /v1/chat/... │         │                            │  │
 wire                   │  └──────────────┘         │ 1. IR in (raw, uncompacted)│  │
                        │        ▲                  │ 2. projection = project(IR)│  │
                        │        │ SSE frames       │ 3. session_cache.lookup    │  │
                        │        │ (from emitter    │        │                   │  │
                        │        │  state machine)  │   hit ▼        miss ▼       │  │
                        │  ┌─────┴────────┐         │  resume      pick_token()  │  │
                        │  │ OpenAI emit. │◀──IR────│  session      ensure_chat  │  │
                        │  │ Anthropic    │  events │               (new chat)   │  │
                        │  │ emitter      │         │ 4. TokenBudget.plan(IR)    │  │
                        │  └──────────────┘         │    → RenderedPrompt        │  │
                        │                           │    (ONE compaction site)   │  │
                        │                           │ 5. adapter.send(prompt)    │  │
                        │                           │     → AsyncIterator[       │  │
                        │                           │        ProviderEvent]      │  │
                        │   ┌──────────────┐        │ 6. tool grammar:           │  │
                        │   │ DeepSeek     │◀───────┤    ProviderEvents(text) →  │  │
                        │   │ adapter      │  HTTP  │    IR SinkEvents           │  │
                        │   │ (PoW, cookies│        │ 7. metering + save         │  │
                        │   │ token pool)  │        │    projections             │  │
                        │   └──────────────┘        └───────────────────────────┘  │
                        │                                                          │
                        │   fake provider (tests): same protocol, zero network     │
                        └──────────────────────────────────────────────────────────┘
```

The single most important structural change vs v1: the orchestrator consumes and produces **only IR objects and typed events**. v1's `handle_chat` (`app.py:136-254`) mixed OpenAI dicts, response builders, and DB writes; here each box above is independently testable and the adapters at both ends carry the wire-format risk.

---

## 2. Canonical message IR

### 2.1 Schema

```python
# ir/message.py
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field

class _Node(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")   # unknown client fields
                                                             # must fail loudly in tests,
                                                             # be dropped explicitly in adapters

class TextBlock(_Node):
    type: Literal["text"] = "text"
    text: str

class ThinkingBlock(_Node):
    type: Literal["thinking"] = "thinking"
    text: str                    # provider reasoning; never re-sent cross-provider

class ToolUseBlock(_Node):
    type: Literal["tool_use"] = "tool_use"
    id: str                      # server-assigned "call_..." — v1 functions.py:485
    name: str
    arguments: dict | list | str # str kept only when unparseable; adapters normalize

class ToolResultBlock(_Node):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str                 # flattened text; images hoisted to ImageBlock
    is_error: bool = False       # Anthropic has it, OpenAI doesn't — IR is the superset

class ImageBlock(_Node):
    type: Literal["image"] = "image"
    media_type: str
    source: Literal["url", "base64", "file_ref"]
    data: str                    # url / b64 / file_id

class FileBlock(_Node):
    type: Literal["file"] = "file"
    file_id: str | None = None
    filename: str | None = None
    media_type: str | None = None
    data: str | None = None      # base64 payload

ContentBlock = Annotated[
    Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock, FileBlock],
    Field(discriminator="type"),
]

Role = Literal["system", "user", "assistant", "tool"]

class Message(_Node):
    role: Role
    content: tuple[ContentBlock, ...] = ()     # tuple: hashable, immutable
    # tool-role messages carry exactly one ToolResultBlock; enforced by a model_validator

class Conversation(_Node):
    system: str | None = None                  # exactly one system slot (v1 extracted it ad hoc:
                                               # plugin_helper.py:238-241, app.py:911-927)
    messages: tuple[Message, ...]
    metadata: dict = {}                        # model hints, request id, scope
```

Design notes:

- **Superset, not intersection.** The IR holds everything either wire format can express (`is_error` exists only in Anthropic; `reasoning_content` is a `ThinkingBlock`, not an OpenAI-specific dict field as in v1 `app.py:545-551`). Adapters map IR→wire by *dropping* what the target cannot express, deterministically, with a capability check.
- **Frozen pydantic models** (not dataclasses/msgspec): we already depend on pydantic v2 for settings and request validation; frozen models give us hashing for cache keys and `extra="forbid"` turns "client sent something we didn't model" from silent data loss into a test-visible event. (msgspec would be ~5× faster at parse, but IR volume here is tiny — one conversation per request — and we'd lose discriminated-union ergonomics. Revisit only if profiling ever says otherwise.)
- **`Conversation.system`** kills an entire v1 bug family: v1's system prompt was "first message with role=system" found by three different functions (`plugin_helper.py:238`, `plugin_helper.py:551-555`, `app.py:911`) with subtly different semantics.
- **role `"tool"` is a view, not a storage format**: the Anthropic adapter flattens `assistant(tool_use) + user(tool_result)` pairs into IR `tool` messages on ingest and reconstitutes them on output; the OpenAI adapter maps `role:"tool"` 1:1. This is exactly the seam where v1's critical bug lived (F1) — now it is one function with table-driven tests instead of two functions with an implicit contract.

### 2.2 Adapter contract (the translation-bug quarantine)

```python
# edge/openai.py — one of exactly two places that knows OpenAI wire format
class OpenAIEdge:
    def to_ir(self, body: ChatCompletionsRequest) -> Conversation: ...
    def from_ir(self, msg: AssistantTurn, *, model: str, usage: Usage) -> dict: ...
    def stream_emitter(self) -> OpenAIEmitter: ...        # §5
```

Mandatory property tests (Hypothesis, §10):
- **Round-trip stability:** `to_ir(from_wire(x))` then `from_ir(...)` re-serializes to a wire object semantically equal to `x` for a generated universe of valid and *slightly invalid* requests (duplicate system messages, `tool_result` before its `tool_use`, string-or-list content, `"content": null` assistant with tool_calls — the exact shapes v1's `convert_anthropic_messages` special-cased at `app.py:887-901`).
- **Echo equivalence** (feeds §4): for any IR assistant turn `t`, `canonical(echo_openai(t)) == canonical(t)` and `canonical(echo_anthropic(t)) == canonical(t)`.

---

## 3. Session cache redesign

### 3.1 Recap of the v1 mechanism and its invariant

v1 keys a cached DeepSeek chat on a *tolerant projection*: `sha256(namespace, K, system, anchor, last-K canonical messages truncated at the last assistant)` (`plugin_helper.py:528-563`), plus a one-sided `hist_len` guard (`app.py:123-133`). Because turn N's key cannot contain turn N's assistant message (which doesn't exist yet), v1 saves a second row — `next_sig` — computed from `messages + [reconstructed assistant turn]` in four separate `finally`/post-processing sites (`app.py:180-183, 240-244, 352-361, 467-477`). The reconstruction must canonicalize *byte-identically* to what the client echoes back on turn N+1; a single divergence (ids, `reasoning_content`, companion text, `(no content)` placeholders) is a permanent miss. That invariant is documented, tested once (`test_signature_matches_server_reconstruction`), and fragile by construction.

### 3.2 The three candidate designs, evaluated honestly

**Option A — v1 status quo (hash + server reconstruction).** Simplest diff, keeps all four call sites. The echo-match invariant remains an accident of two canonicalizers agreeing; every future IR change re-opens the risk. Rejected as the v2 design.

**Option B — store the projection, compare projections (the "better idea" from the brief).** At save time, persist the *canonical projection payload itself* (JSON: system, anchor, tail window, hist_len, pending tool-call ids) instead of only its hash. At lookup, compute the incoming request's projection and compare **structurally**:

- `system`, `anchor`: exact string equality (client-supplied, stable — no reconstruction involved);
- tail window: element-wise `Message` equality after canonicalization — **except** the final assistant message, compared via the *echo-equivalence function* from §2.2 (an explicitly typed relation, property-tested for both wire formats) rather than byte equality;
- if the trailing assistant fails echo-equivalence, degrade one level: compare the tail *without* it, and require the previous K−1 elements to match exactly (this is strictly more tolerant than v1, which simply missed);
- `hist_len` one-sided guard unchanged;
- `pending_tool_calls`: the ids *we* issued last turn; if the incoming `tool_result`s reference exactly these ids, the continuation is confirmed regardless of how the client serialized the assistant turn — this alone removes the most common echo divergence (tool-call arguments string-vs-object, id formats) from the hash path entirely.

**What B honestly does and does not buy.** It does *not* literally eliminate the echo problem — the stored projection still contains a server-side description of the assistant turn. What it does is convert a silent byte-match accident into (a) an explicit, typed, property-tested equivalence, (b) a graceful fallback instead of a cliff, (c) full debuggability: on a miss the server can log a structured diff of stored vs incoming projections (`which element diverged`), which v1 could never do because the miss was hidden inside a hash. It also enables offline tooling: dump the projections table and *see* what clients send.

**Option C — reconstruction-free key.** Exclude assistant turns from the key entirely: key = (system, anchor, user-tail up to the last user message, pending-tool-call ids). Every element is then either client-supplied stable text or server-known ids; no echo relation is needed at all. Cost: the tail window loses the assistant turns that v1's tolerance analysis called "the load-bearing discriminator" (`Report/deeperseeker_sig_tolerance_design.md` §0.3) — two *different* agent tasks with the same system prompt and near-suffix user messages collide more easily, and a hit binds a conversation to a DeepSeek chat whose server-side context doesn't match, which is the one error mode that produces *wrong answers* rather than inefficiency. Rejected as default; noted as a possible fallback tier if echo-equivalence proves brittle in production telemetry.

**Decision: Option B**, with `session_cache_outcome{hit, miss_new, miss_guard, miss_fallback}` exported as a metric from day one so the design is validated by data, and Option C's "compare without the trailing assistant" demoted to the built-in fallback tier of B.

### 3.3 Store schema

```sql
CREATE TABLE provider_sessions (
    projection_key  BLOB PRIMARY KEY,      -- sha256 over the canonical payload (fast path)
    projection      TEXT NOT NULL,         -- JSON payload: system, anchor, tail[K], hist_len,
                                           -- pending_tool_calls — Option B's compare input
    provider        TEXT NOT NULL,         -- 'deepseek', 'fake', ...
    provider_chat   TEXT NOT NULL,         -- DeepSeek chat_session_id
    parent_message_id INTEGER NOT NULL,    -- v1's +2 invariant stays centralized (functions.py:425-438),
                                           -- but the count is now tracked from actual appended messages
    token_id        INTEGER NOT NULL REFERENCES tokens(id),
    hist_len        INTEGER NOT NULL,      -- one-sided guard (v1 app.py:123-133)
    created_at      TEXT NOT NULL,
    last_used_at    TEXT NOT NULL
);
CREATE INDEX idx_sessions_last_used ON provider_sessions(last_used_at);

CREATE TABLE tokens (
    id INTEGER PRIMARY KEY,
    alias TEXT, secret_enc BLOB NOT NULL,      -- Fernet (v1 functions.py:29-109, now mandatory,
                                               -- fail at boot if no key: no plaintext fallback)
    provider TEXT NOT NULL DEFAULT 'deepseek',
    status TEXT NOT NULL DEFAULT 'ACTIVE',     -- ACTIVE | RATE_LIMITED | INVALID (v1 had no INVALID)
    rate_limited_until REAL,                   -- time-boxed, not sticky: v1's mark_active() on every
                                               -- success (app.py:330) was the un-stick mechanism; make it explicit
    last_error TEXT,
    requests_ok INTEGER DEFAULT 0, requests_failed INTEGER DEFAULT 0
);

CREATE TABLE usage (
    request_id TEXT PRIMARY KEY, ts TEXT NOT NULL,
    provider TEXT, model TEXT, token_id INTEGER,
    in_tokens INTEGER, out_tokens INTEGER, cost_usd REAL,
    cache_hit INTEGER, stream INTEGER, duration_ms INTEGER
);
```

TTL pruning (v1 `prune_sessions`, `functions.py:175-189`) becomes a periodic asyncio task on `last_used_at` — v1 pruned on every write, an unnecessary cost — plus a startup sweep. Token accounting per `token_id` feeds the admin dashboard and `/health`.

---

## 4. Streaming layer: the emitter state machine

### 4.1 The two event vocabularies

```python
# ir/events.py — what the pipeline produces (provider-agnostic)
class SinkEvent: ...
class MessageStart(SinkEvent): model: str
class ThinkingDelta(SinkEvent): text: str
class TextDelta(SinkEvent): text: str
class ToolCallStart(SinkEvent): index: int; id: str; name: str
class ToolCallArgsDelta(SinkEvent): index: int; partial_json: str
class ToolCallEnd(SinkEvent): index: int
class MessageEnd(SinkEvent): stop_reason: StopReason; usage: Usage
class SinkError(SinkEvent): kind: str; message: str        # never leaks exception text (v1 leaked 300
                                                           # chars: app.py:343, 457 — audit S4)
```

```python
# providers/base.py — what a provider adapter yields (provider-side)
class ProviderEvent: ...
class ProviderText(ProviderEvent): text: str                  # raw incremental text
class ProviderThinking(ProviderEvent): text: str
class ProviderFinished(ProviderEvent): ...
```

The pipeline contains exactly one transformer: `ProviderEvents + grammar → SinkEvents` (§6). Emitters then translate `SinkEvents → SSE frames`.

### 4.2 Emitter design

```python
# streaming/emitter_anthropic.py
class AnthropicEmitter:
    """Total function: any sequence of SinkEvents yields a protocol-valid Anthropic SSE stream.

    The emitter owns the content_block index and the open-block set. Callers CANNOT
    produce a malformed lifecycle because block open/close is not an input — it is
    derived from SinkEvent kinds:
      ThinkingDelta → open('thinking') if closed, delta
      TextDelta     → close thinking block if open; open('text') if closed; delta
      ToolCallStart → close any open block; open('tool_use', index)
      ToolCallEnd   → close tool_use block
      MessageEnd    → close everything, message_delta(stop_reason), message_stop
    """
    def emit(self, ev: SinkEvent) -> bytes: ...
    def close(self) -> bytes: ...        # idempotent; also the cancel/abort path
```

- **Totality.** Every `SinkEvent` is valid in every emitter state; the transition table has no error state. "Two open content blocks" — the bug v1 fixed by assembling `tail_events` strings by hand and re-splitting them (`app.py:479-524`) — cannot be expressed: opening a new block *is* the close of the previous one, in one place.
- **MessageStart** is emitted by the transport wrapper (message id, input usage) before the first frame; **MessageEnd** is the only source of `stop_reason` and output usage. v1's `finish_reason`/`stop_reason` selection was duplicated in four places (`app.py:561, 375-377, 514-517`); here it exists once.
- **Ordering guarantee for late tool calls.** The grammar may discover a tool call only after streaming companion prose. `ToolCallStart` closes the text block first — inside the emitter, mechanically. v1 had to re-derive this in `finally` (the comment at `app.py:488-491` marks the battle scar); here it is the definition of one transition.
- **OpenAI emitter** mirrors it: deltas, `reasoning_content` for thinking, one `finish_reason` at the end, `[DONE]` sentinel, `tool_calls` delta indices owned by the emitter.

### 4.3 Transport

`sse-starlette`'s `EventSourceResponse` wraps the emitter's bytes: it owns client-disconnect semantics and heartbeats, replacing v1's hand-rolled `GeneratorExit` bookkeeping (`app.py:331-334, 445-448, 519-525`). The `sse.py` frame serializer is ~10 lines and unit-tested; protocol correctness lives in the emitter, not the transport.

### 4.4 Verification

1. **Invariant checker** (`streaming/validate.py`): a strict SSE parser implementing the Anthropic block lifecycle (monotonic indices, ≤1 open block, no delta after stop, `message_delta` immediately before `message_stop`, OpenAI: exactly one `finish_reason`, `[DONE]` last). Production does not run it; tests do.
2. **Hypothesis property:** for *any* generated `list[SinkEvent]`, `validate(parse(emitter stream))` passes, and the reconstructed text equals the concatenated `TextDelta`s. Malformed output is not "caught by review" — it is a failing CI property.
3. **Golden fixtures:** recorded real streams (DeepSeek thinking-then-text, text-then-tool, tool-only, mid-stream error, empty) replayed through the whole stack; byte-level snapshots for the frames the protocol pins (lifecycle), semantic snapshots for the rest.

---

## 5. Pipeline orchestrator

```python
# pipeline/orchestrator.py
class Orchestrator:
    async def handle(self, conv: Conversation, req: RequestSpec) -> Response:   # non-streaming
    async def handle_stream(self, conv: Conversation, req: RequestSpec) -> AsyncIterator[bytes]:
```

Responsibilities, in order: resolve model → projection & cache lookup (§3) → token acquisition (fail-fast: no token → `503` immediately, exactly v1's `pick_token` semantics, `functions.py:342-351`) → on miss: `adapter.ensure_chat` under a per-projection lock (v1's `_sig_locks`, `app.py:187-211`, kept but keyed by projection and cleaned up correctly) → `TokenBudget.plan` (§7) → `adapter.send` → grammar transform → emitter → save projections + metering in a `finally`-free explicit completion handler (v1 did save inside generator `finally` blocks, which is where aborted-stream bugs bred; v2 saves from a task fed by the sink event stream, cancelled explicitly on client disconnect).

Retry policy (fixing v1's `app.py:246-254`, audited as C6): one retry, exponential backoff via `tenacity`, only on `401/403/429`; the *session row is deleted before retry* deliberately (v1 deleted it too but only in the non-stream path's `except`, inconsistently); retry re-enters `handle` at the cache-lookup step, never at the HTTP handler.

---

## 6. Tool calling: pluggable grammars

### 6.1 Registry

```python
# tools/registry.py
class Grammar(Protocol):
    def transform(self, stream: AsyncIterator[ProviderEvent]) -> AsyncIterator[SinkEvent]: ...
    def parse_final(self, full_text: str) -> tuple[list[ToolUseBlock], str]: ...  # non-streaming path

def grammar_for(provider: str, model: str, injected: ToolInjection) -> Grammar: ...
```

- **`grammar_dsml`** (DeepSeek family): primary parser is *structured* — it recognizes the exact forms we inject instructions for (`<tool_call>{"name","arguments"}</tool_call>`, the DSML fullwidth-pipe variants that DeepSeek actually emits in the wild, fenced JSON), using a small scanner, not v1's six cascading regex passes (`functions.py:526-707`). The scanner keeps v1's hard-won rules as first-class concepts: code-fence immunity (`_code_fence_spans`), companion-prose preservation, repeated-identical-call preservation, no dedupe (audit C4).
- **`grammar_freetext`**: the permissive fallback (bare JSON after tags, brace-balancing repair — v1's `raw_decode` + `}`-counting heuristic at `functions.py:638-644`), selected per model family or as last resort.
- **Injection** (`tools/inject.py`) is the inverse of parsing and lives next to it — the pair (inject, parse) is the contract, and every corpus entry is tested through *both* directions.

### 6.2 False-positive containment (ties to §4)

The grammar's streaming contract is **no retraction**: it may only hold back text at ambiguity boundaries (generalizing v1's `_hold_think_tags` suffix-buffer, `app.py:275-292`); a `ToolCallStart` is emitted only after a complete, fence-excluded, name+args-valid block. If a candidate fails validation at close, the buffered span flushes as `TextDelta` — text the client already received is never retroactively converted, so a false positive can degrade fidelity but can *never* corrupt the SSE lifecycle (the emitter closes/open blocks regardless of what the grammar believed). The flush-at-end semantics subsume v1's "discard partial tool markup on abort" (`functions.py:785-793`).

### 6.3 Corpus

`tools/corpus/` holds real malformed outputs captured from production (fullwidth-pipe mangling, JSON split across chunks, tool markup quoted in code fences, nested fences, `<think>` echoes) plus Hypothesis-generated mutations of valid calls. Every grammar change runs the whole corpus offline in seconds.

---

## 7. Compaction: the TokenBudget service

```python
# compaction/budget.py
@dataclass(frozen=True)
class BudgetPolicy:
    total: int = 24_000                 # v1 DEEPSEEKER_PROMPT_BUDGET
    tool_result_clip: int = 4_000       # v1 DEEPSEEKER_TOOL_RESULT_CLIP
    recent_turns: int = 6               # v1 DEEPSEEKER_HISTORY_RECENT_TURNS
    min_recent_turns: int = 2
    old_text_clip: int = 200
    head_ratio: float = 0.7             # v1's 7/10 head+tail split, plugin_helper.py:33-35

class TokenBudget:
    def measure(self, blocks: Sequence[ContentBlock]) -> int: ...
    def plan(self, conv: Conversation, policy: BudgetPolicy) -> RenderedPrompt:
        """Deterministic. Tiers: verbatim recent window → tool-result stubs for older
        tool messages → clipped older text → drop-oldest loop. System and final user
        message are never clipped. Pure function of (conv, policy)."""
```

- **One application site:** `orchestrator` step 4, immediately before `adapter.send`. Nothing else in the system calls `plan`.
- **Keys never see it.** Projections (§3) are computed from the *raw* IR. This is the cleanest separation win over v1: v1 needed `clip_text` deterministic *because* compaction output fed `build_prompt` which fed the signature (`ARCHITECTURE.md` §5: "compaction never perturbs signature keys"). v2 keeps determinism — as a tested property, because reproducible prompts are valuable for debugging — but cache correctness no longer depends on it.
- The degradation loop (shrink recent window → halve old-text clip → re-measure, `plugin_helper.py:640-656`) moves inside `plan` unchanged in behavior; it becomes a pure function returning `(prompt, elided_stats)` instead of mutating module-level env reads (`_env_int` at call time, `plugin_helper.py:24-29` — replaced by `BudgetPolicy` from settings).
- Tokenizer stays `deepseek_tokenizer` (v1's `count_tok`); it is the honest measure for what DeepSeek will actually consume, and it's pure-python, cheap to run in-thread.

---

## 8. Multi-provider: the adapter interface and tiers

```python
# providers/base.py
@dataclass(frozen=True)
class Capabilities:
    branching: bool          # server-side conversation tree (DeepSeek: yes, parent_message_id)
    thinking: bool
    tools: bool              # False ⇒ pipeline uses prompt injection + grammar
    attachments: bool
    search: bool
    max_prompt_tokens: int | None

class SessionRef(NamedTuple):
    provider: str
    chat_id: str
    parent_message_id: int | None
    token_id: int

class ProviderAdapter(Protocol):
    name: str
    capabilities: Capabilities
    async def authenticate(self, token: str) -> None: ...          # fail fast at pool-add time
    async def ensure_chat(self, token: str, resume: SessionRef | None) -> SessionRef: ...
    def send(self, session: SessionRef, prompt: RenderedPrompt, *,
             opts: SendOptions) -> AsyncIterator[ProviderEvent]: ...
    async def upload_file(self, token: str, data: bytes, filename: str,
                          media_type: str) -> FileRef: ...          # optional (capabilities)
    async def refresh_credentials(self) -> None: ...                # optional: cookie refresh
    async def health(self) -> ProviderHealth: ...
```

- **Adapter #1 — DeepSeek** (`providers/deepseek/`): `wire.py` is a near-verbatim port of `functions.py:860-1071` (chat completion with its *multiple tolerated wire shapes* — the `p/o/v` fragment handling at `functions.py:944-993` encodes real observed server behavior and should be carried, not redesigned), `pow.py` ports the wasmtime solver, `cookies.py` ports the Playwright refresh. The adapter returns `ProviderText`/`ProviderThinking` events instead of v1's `<think>`-tagged string concatenation (`functions.py:955-993` — the wire already distinguishes THINK/RESPONSE fragments, so v1's tag round-trip through the text channel was pure lossy encoding; v2 types it directly, deleting `_hold_think_tags` and the head-of-message `<think>` heuristics at `app.py:407`).
- **Feasibility tiers for #2/#3:**
  - **Tier 1 — cookie-auth JSON-API provider (Kimi/Qwen class):** same shape as DeepSeek (bearer-or-cookie + JSON chat endpoint + SSE). Estimated 3–5 focused days on top of the interface. Highest value, lowest risk.
  - **Tier 2 — Cloudflare/WAF-guarded provider:** needs `refresh_credentials` with Camoufox (Firefox anti-detect) rather than vanilla Playwright Chromium, plus per-request cookie injection. Feasible but operationally brittle; budget 8–12 days including a private-registry Camoufox image and a manual cookie paste fallback in the admin UI.
  - **Tier 3 — real-API providers (OpenAI/Anthropic APIs):** trivial *once wanted*; deliberately out of scope until someone asks (see LiteLLM note, §11).
- The fake provider (`providers/fake.py`) implements the same protocol with scripted events; it is the backbone of the offline test suite (§10), not a toy.

---

## 9. Ops

- **Fail-fast pools** (kept from v1): `pick_token` returns None → `503` with a typed error body; no doomed-request burning of PoW solves. Added: `INVALID` status (401 twice → dead, needs human), `rate_limited_until` with automatic expiry, per-token success/failure counters.
- **Credentials:** Fernet encryption mandatory (boot fails without a key — v1's silent-plaintext fallback, `functions.py:39-58`, removed); admin dashboard sessions in the DB (not the in-memory `SESSIONS` dict, `app.py:85`), CSRF tokens on POSTs (audit S3), `Secure` cookie behind TLS.
- **One `/health`** returning pool counts, cookie expiry per provider, DB check, cache size — v1 defined it twice and the second definition (`app.py:1174-1190`) silently replaced the richer first one.
- **Logging:** structlog JSON, `request_id` contextvar injected by middleware, echoed in error streams and `usage` rows. **Metrics:** `prometheus-client` behind `metrics_enabled: bool = False`; counters for `session_cache_outcome`, `provider_requests_total{provider,outcome}`, `tool_calls_parsed{grammar,outcome}`, histograms for stream duration.
- **Config:** one pydantic-settings `Settings` with nested sections (`server`, `pool`, `cache`, `budget`, `providers.deepseek`, `admin`, `ops`), loaded from env with the `DS2_` prefix. v1's ~10 `DEEPSEEKER_*` reads scattered across call sites (e.g. `functions.py:167-172`, `plugin_helper.py:24-29,515-523`) collapse into it; `Settings` is frozen and injected, so tests construct it literally.

---

## 10. Testing strategy (entire suite offline, seconds)

| Layer | Technique | What it kills |
|---|---|---|
| Edge adapters | Hypothesis round-trip + malformed-input strategies (§2.2) | F1 |
| Session cache | Unit tests on projection compare (anchor/tail/echo-equiv/fallback/guard) + a Hypothesis "compaction rewrites the middle" strategy driving hit/miss expectations | F2 |
| Emitters | Hypothesis over `SinkEvent` lists vs `validate.py` invariant checker; golden fixtures | F3 |
| Grammars | `tools/corpus` regression + Hypothesis mutations (§6.3); every case through streaming *and* final-parse paths, asserting the two agree | F4 |
| TokenBudget | Determinism property: `plan(x) == plan(x)`; budget-respect property for random conversations; snapshot of tiered output | F5 |
| Orchestrator | `providers/fake.py` end-to-end: full request→SSE bytes over ASGI transport (httpx), including abort mid-stream, pool exhaustion, retry-after-429 | F6 |
| Store | aiosqlite against tmp files; migration tests; crypto round-trips | — |
| DeepSeek wire | Recorded HTTP fixtures (vcr-style) for `wire.py`; `pow.py` tested against known challenge→answer vectors | regression vs live web |

No test touches the network. Target cold-suite time: < 20 s.

---

## 11. Out-of-the-box tool selection (with the honest evaluations)

| Choice | Verdict | Why |
|---|---|---|
| **FastAPI vs Litestar** | **FastAPI** | The hard problems (SSE lifecycle, IR, parsers) are framework-independent; FastAPI's ecosystem (Starlette streaming, `TestClient`/httpx ASGI, community knowledge) outweighs Litestar's marginally better DI/perf. Switching frameworks is risk with no bearing on any F1–F7. Keep v1 team knowledge. |
| **pydantic v2** | **Yes** — settings, wire models, frozen IR | One library covers config, validation at the edge, and immutable IR with discriminated unions; `extra="forbid"` is a test instrument. |
| **orjson** | **Yes** | v1 `json.dumps`-per-delta in hot stream loops (`app.py:318, 328, 442`…); orjson is faster and returns bytes natively for SSE frames; `ORJSONResponse` as default response class. |
| **aiosqlite vs sqlite3+to_thread** | **aiosqlite**, single connection, WAL | Honest: write volume is trivial (2 rows/turn) so either works. The real fix is v1's *connection-per-call* pattern (`functions.py:112-119` opens/closes ~10× per request) — one persistent connection with an async API is the simplest cure; `asyncio.to_thread` + manual pooling saves nothing at this scale. |
| **sse-starlette vs hand-rolled** | **sse-starlette** | It solves client-disconnect and heartbeat — the exact paths where v1's hand-rolled `finally` bookkeeping bred bugs (`app.py:331-334, 445-448, 519-525`). Protocol *lifecycle* correctness stays in our emitter (sse-starlette does nothing for that). |
| **structlog** | **Yes** | Request-ID contextvars + JSON logs; structlog's contextvars binding maps 1:1 onto the pipeline's async task structure. |
| **prometheus-client** | **Optional, default off** | Cheap to include, zero cost when disabled; the metric *names* are fixed in code from day one so dashboards can be built before the flag flips. |
| **tenacity** | **Yes, scoped** | Auth/PoW/upload retries with backoff. *Not* a decorator on chat send: retry there means new-session + full re-inject, which is orchestrator policy (§5), not transport policy. |
| **Hypothesis** | **Yes, load-bearing** | This is the design's main verification instrument (§10). It is the difference between "invariants documented in comments" (v1) and "invariants executable over an infinite input space" (v2). |
| **pytest-asyncio** | **Yes** (over anyio) | Explicit async fixtures for the fake provider and store; ubiquitous. |
| **Playwright vs Camoufox** | **Playwright now, Camoufox for Tier-2 later** | v1's Chromium refresh works today (`functions.py:260-290`); Camoufox is only justified when a Cloudflare-tier provider is actually adopted (§8). Add the admin "paste cookies manually" fallback either way — it de-risks headless refresh failures in Docker. |
| **wasmtime (PoW) vs alternatives** | **Keep wasmtime** | The packaged `.wasm` is the reference algorithm — correctness by construction (`functions.py:797-822`). A pure-Python DeepSeekHashV1 reimplementation is feasible (it's a leading-zero-bits hash search) and drops a native wheel, but risks divergence from the reference for zero user-visible gain. Keep the solver in a thread (`asyncio.to_thread`, as v1 does). |
| **uv / ruff / mypy strict / pre-commit / Docker** | **All yes** | uv for lock+env speed; ruff (lint+format, replaces black/isort/flake8); mypy strict is *enabled by the IR* — typed events and total functions are what make strict mode tractable. Multi-stage Docker, non-root, `LOCALHOST` bind default preserved from v1 (`app.py:1193-1197`). |
| **Build on LiteLLM vs own IR** | **Own IR. Do not adopt LiteLLM as the core.** | What LiteLLM gives: wire-format translation among real API providers, cost tracking, a provider zoo, and (via `litellm.acompletion`) streaming chunks in a normalized shape. What it costs here: (1) *direction mismatch* — LiteLLM is a client for stateless HTTP APIs; our "providers" are stateful web sessions (server-side chat trees, PoW, cookies, prompt-injected tools) that its adapter model simply doesn't represent; the DeepSeek adapter would be 100% ours either way. (2) *The normalized chunk stream is not an IR* — it has no place for tolerant-projection computation, compaction-before-send, or grammar-transformed tool discovery; we'd bolt all three on top, recreating v1's "two concepts spliced together" (audit §1) under someone else's abstractions. (3) *Stream lifecycle* — its chunk normalization is exactly the layer where v1's Anthropic block-lifecycle bugs lived; owning that layer is the point of v2. Recommendation: own IR; if real-API providers are ever wanted, wrap LiteLLM as *one more* `ProviderAdapter` behind our protocol — then it's a cheap win instead of a foundation. |

---

## 12. Build plan (phased, each independently shippable and demoable)

Effort in *focused days* (one engineer, no meetings). Total: **29–35 days**.

**P0 — Skeleton + IR + edge adapters (5–6 d).** uv project, ruff/mypy strict/CI, `Settings`, structlog; `ir/` complete; both edge adapters with round-trip property tests; `/v1/models`.
*Demo: request JSON in either wire format → validated IR dump. Test: adapters + 0 network.*
Ship value: none alone — foundation.

**P1 — Store + fake provider + pipeline, non-streaming (4–5 d).** `store/` with migrations and crypto; projection store with Option-B compare (§3); `providers/fake.py`; `orchestrator.handle` non-streaming end-to-end over ASGI; metering rows.
*Demo: scripted conversation against the fake provider, cache hit/miss visible in logs and `usage`.*
Test: cache invariants (middle-rewrite survival, guard, fallback tier) — the v1 `test_local_fixes.py` signature suite re-expressed as properties.

**P2 — Streaming emitters + grammars (7–8 d).** `SinkEvent` pipeline; both emitters + `validate.py`; Hypothesis emitter properties; grammar registry + DSML scanner + freetext fallback + corpus; `handle_stream` over the fake provider; golden fixtures recorded from fake+real shapes.
*Demo: streaming tool-calling agent loop against the fake provider with protocol-validated SSE.*
Test: this is the heart — emitter properties + corpus + late-tool-call ordering goldens.

**P3 — DeepSeek adapter port (4–5 d).** `wire.py`/`pow.py`/`cookies.py` ports (largely verbatim, carrying the observed wire-shape tolerance of `functions.py:944-993`); adapter returns typed events; file upload/download; token pool with `INVALID`/`rate_limited_until`; live smoke script (network, manual, outside CI).
*Demo: real DeepSeek chat + tools through v2, single provider.*
Test: recorded-HTTP fixtures for wire; challenge vectors for PoW. **Milestone where v2 reaches v1 feature parity.**

**P4 — TokenBudget service (3 d).** `budget.py` + degradation loop as pure function; wire into orchestrator; snapshot + property tests; delete-ad-hoc clipping.
*Demo: over-budget conversation compacted with tiered output shown in admin debug view.*

**P5 — Ops + admin + packaging (4–5 d).** `/health` (one), admin dashboard port (sessions in DB, CSRF), TTL pruning task, Prometheus flag, Docker image, pre-commit, docs.
*Demo: docker compose up, dashboard, health, metrics.*

**P6 — Hardening + parity soak (3–5 d).** Run v1 and v2 side-by-side against the same clients for a few days of real usage; replay recorded production streams; fix divergences; finalize migration script (§13).
*Exit criteria: v2 replaces v1 as the daily driver with zero client-visible behavior change.*

---

## 13. What would be different — the honest list

**Gets easier**
- Translation and streaming bugs become failing property tests instead of production symptoms (F1, F3 — historically the two classes that cost real debugging time).
- The echo-match invariant becomes a typed relation with telemetry and a fallback tier; cache misses become diagnosable from logs instead of inferred from behavior.
- Adding a provider is an adapter + grammar entry, not surgery on a 1,200-line handler; the fake provider means new pipeline features are testable before any provider cooperation.
- Ops posture (single health, real config, metering, structured logs) stops being tribal knowledge.
- mypy strict + frozen IR make large refactors cheap; v1's deepest fear — touching `handle_chat` — disappears.

**Gets harder**
- **Regression risk vs the battle-tested fork.** v1's `wire.py` tolerance, DSML regexes, and signature semantics encode months of observed provider misbehavior. Porting is mostly verbatim, but every one of the 34 existing tests must be re-derived, and the *unrecorded* knowledge (which exact malformed shapes occur at what frequency) only re-validates under real traffic. This is why P6 soak exists and why P3 ports wire code rather than "cleaning it up".
- **Two systems during transition.** ~5 weeks of carrying both; v1 stays the daily driver until P6.
- **Upstream drift.** v1 cherry-picks from upstream (`ARCHITECTURE.md` §8); v2 forks away from that. Auth/PoW/stream-format fixes observed upstream must be re-ported into `wire.py` by hand.
- **Speculative generality.** Multi-provider, grammar registry, and Option-B projection storage are built before provider #2 exists; some of it will be wrong in details and reworked on first real contact.
- **Small honest losses:** orjson/pydantic-strict edge handling may reject exotic client payloads v1 tolerated silently; the first weeks need `extra`-field telemetry to tune the adapters' tolerance.

**Risks specific to the rewrite**
- The projection store adds a JSON payload write per turn (negligible) and a compare path that is more code than a hash lookup — its payoff (diagnosability, fallback) is only realized if the telemetry is actually watched.
- Grammar totality (no-retract contract) constrains future parser cleverness; if a future model family needs genuinely ambiguous lookahead, the buffering contract gets strained.

**Fork-continued vs v2 — when each**
- Choose **fork-continued** if DeepSeek remains the only provider and changes stay incremental: the fork is in good shape post-audit (encryption, SSRF fixes, hardened parsing all landed), and every day spent on v2 is a day the fork isn't improving.
- Choose **v2** when any of these triggers fire: (a) a second provider is concretely wanted, (b) another stream-lifecycle or translation bug escapes to production, (c) the next feature request requires touching both `app.py` stream functions at once.

**Recommendation.** Do not big-bang v2 *now* if multi-provider is still hypothetical. Instead, pull P0–P2 forward as a strangler refactor *inside the fork's replacement path* (the IR, emitters, and grammar corpus are self-contained, ~12–14 days, and immediately de-risk the two worst bug classes even while DeepSeek stays the only adapter) — this captures roughly 80% of v2's value at ~40% of its cost and leaves the fork running. Commit to full v2 (P3–P6, the remaining ~17–21 days) at the first trigger above. If provider #2 is already a near-term commitment, build v2 whole; 29–35 focused days is a sound price for the invariant class it retires.

---

## 14. Migration / interop note

- **`deeperseeker.db` → v2:** *tokens* migrate with a one-time script (decrypt with the v1 `DEEPSEEKER_ENCRYPTION_KEY`, re-encrypt under the v2 key, add `provider`/`rate_limited_until` columns). *sessions* are **not** migrated: keys are recipe-specific (v1's `SIG_NAMESPACE = "ds-sig-tolerant-v1"`), rows expire in 7 days anyway, and the cold-start cost of a miss is one full-history injection on the first turn — by design. *usage* starts empty.
- **`aws_cookies_deepseek.json`:** reusable as-is if v2 derives the same Fernet key; otherwise the admin "paste cookies" fallback re-seeds it in seconds. Either way the v2 store owns it (DB blob, not a side file — v1's file-with-lock dance, `functions.py:227-258`, disappears).
- **Tests:** v1's `test_local_fixes.py` does not port mechanically (different seams), but every invariant it pins has a named v2 counterpart (§10 rows). The genuinely reusable artifacts are v1's *observed malformed outputs* — seed `tools/corpus/` with them — and fresh recordings of live DeepSeek SSE for the golden fixtures.
- **Clients:** zero migration. Same endpoints, same auth header semantics, same model aliases (`instant`/`expert`/`vision` preserved in the model registry).
