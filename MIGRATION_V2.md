# Forward Migration Plan: deeperseeker v1 (this fork) → v2

**Status:** plan (pre-commitment) · **Companion documents:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (how v1 works today) · [`V2_DESIGN.md`](V2_DESIGN.md) (greenfield design, the spec this plan migrates to)

**Date:** 2026-09-06 · Applies to fork state `804c504` (main), 13 commits ahead of upstream.

---

## 1. Philosophy

v1 is not replaced by a cutover event; it is **strangled**. v2 modules land inside an independent package first, the DeepSeek adapter is ported once the v2 core passes its offline protocol suite, and both stacks run side-by-side against the same real clients until v2 has soaked. Every phase ends with the system fully working; there is no point at which the gateway is "down for migration".

Clients (Claude Code, ZCode, Claude Desktop, OpenAI-SDK tools) see **zero migration**: same endpoints, same auth header semantics, same model aliases (`instant` / `expert` / `vision`), same streaming behavior. Any v2 build that requires a client-side change is a defect in v2.

## 2. Triggers — when this plan activates

The plan is written now but executed only when one of these fires:

| # | Trigger | Why it forces v2 |
|---|---------|------------------|
| T1 | A **second provider** is concretely wanted (Kimi/Qwen-class or beyond) | v1's provider logic is fused into the HTTP layer; each provider would re-splinter it |
| T2 | **Another stream-lifecycle or translation bug escapes** to a real client | The v2 emitters make these bug classes unrepresentable; a second escape means the v1 structure can't hold |
| T3 | A planned feature requires **touching both stream functions at once** (`stream_response` + `stream_anthropic_response`) | That is the smell that produced the two worst v1 bugs |

Until a trigger fires, improvements continue to land on v1 (as they have: audit fixes, compaction, signature tolerance, `/health`).

## 3. Phase plan mapped to migration milestones

Effort in focused days (from V2_DESIGN.md §12; total 29–35). Each phase is independently shippable.

| Phase | What lands | Migration meaning |
|-------|-----------|-------------------|
| **P0** (5–6 d) | `v2/` package inside this repo: IR + OpenAI/Anthropic edge adapters + settings/logging; `/v1/models` served from the registry | **No behavior change.** v1 still serves everything; v2 adapters are exercised only by tests |
| **P1** (4–5 d) | Store (aiosqlite, single connection, migrations, crypto) + fake provider + non-streaming orchestrator | Still no client traffic. v1's signature-cache test suite is re-expressed as properties against the projection store |
| **P2** (7–8 d) | SSE emitters as total state machines + grammar registry + malformed-output corpus; streaming over the fake provider | Still no client traffic. This is where the vanishing-reply bug class becomes unrepresentable |
| **P3** (4–5 d) | **DeepSeek adapter port** — `wire.py` / `pow.py` / `cookies.py` moved over verbatim (observed wire-shape tolerance preserved), token pool with `INVALID`/`rate_limited_until` | v2 reaches feature parity. First optional live traffic behind a flag |
| **P4** (3 d) | TokenBudget service (deterministic clipping, tiering, degradation loop) wired at the single pipeline point | v2 now matches v1's compaction behavior |
| **P5** (4–5 d) | `/health`, admin dashboard port (DB sessions, CSRF), TTL pruning, metrics flag, Docker, pre-commit | Operational parity |
| **P6** (3–5 d) | **Side-by-side soak**: v1 and v2 both running; same clients alternated daily; recorded production streams replayed; divergences fixed | Exit: v2 becomes the daily driver |

**Strangler shortcut (recommended if the trigger is T2/T3 rather than T1):** execute P0–P2 only (~12–14 d) and land the IR + emitters + grammars *inside v1's request path* — capturing ~80% of the value at ~40% of the cost, with the fork continuing as the shell. Full P3–P6 then happens only if/when T1 fires.

## 4. Data migration

| Artifact | Migration | Notes |
|----------|-----------|-------|
| **Tokens** (`tokens` table) | One-time script: decrypt with v1 `DEEPSEEKER_ENCRYPTION_KEY`, re-encrypt under the v2 key, add `provider` + `rate_limited_until` columns | Script ships in `v2/tools/`; dry-run mode prints the migration report without writing |
| **Sessions** (`sessions` table) | **Not migrated — intentionally.** | Keys are recipe-specific (`SIG_NAMESPACE = "ds-sig-tolerant-v1"`); rows expire via the 7-day TTL anyway; a cold-start miss costs one full-history (compacted) injection on the first turn — by design |
| **Cookies** (`aws_cookies_deepseek.json`) | Reused as-is if v2 derives the same Fernet key; otherwise re-seeded via the admin "paste cookies" fallback in seconds | v2 stores them as a DB blob — v1's side-file-with-lock dance disappears |
| **Usage/metering** | Starts empty | v1 has no usage history to carry |
| **Tests** | `tests/test_local_fixes.py` does not port mechanically (different seams), but every invariant it pins has a named v2 counterpart (V2_DESIGN.md §10) | The genuinely reusable artifacts are the *observed malformed outputs* — they seed `v2/tools/corpus/` |

## 5. Client compatibility contract (checked at P6 exit)

- [ ] `/v1/chat/completions`: streaming + non-streaming, `tools`/`tool_calls` round-trip, `reasoning_content`, usage block with cost — byte-shape compatible
- [ ] `/v1/messages`: streaming + non-streaming, `tool_use`/`tool_result` blocks, thinking blocks, `stop_reason` semantics — byte-shape compatible
- [ ] `/v1/models`, `/health`, admin dashboard + login — same paths, same auth
- [ ] Model aliases `instant` / `expert` / `vision` resolve identically
- [ ] Long-running conversation survives a v1→v2 swap mid-conversation: the first v2 turn takes one cache-miss cold start (full-history injection), then behaves identically
- [ ] Replaying the recorded real-world SSE corpus (including the malformed shapes that caused the 2026-09-06 vanishing-reply bug) produces valid event sequences in v2

## 6. Cutover checklist (P6 → daily driver)

1. Freeze v1 feature work for the soak window; run both stacks on different ports (`:4000` v1, `:4100` v2).
2. Alternate the daily client between stacks per day; log every divergence in a single file with a replay test per divergence.
3. Replay the full recorded SSE corpus + the malformed-output corpus against v2; all green.
4. Run the token-migration script (dry-run, then real) against a copy of `deeperseeker.db`; verify `/health` pool counts match.
5. Swap ports: v2 takes `:4000`, v1 moves to `:4100` but stays running for one week.
6. After one clean week: v1 demoted to `legacy/` tag, kept for one release cycle, then archived.

## 7. Rollback

At any point before step 6 of the cutover, rollback is trivial by construction: **v1 is still running** on its port with its own DB state. Rolling back after cutover step 5 means swapping the ports back; conversations pay one cold-start miss each way (the tolerant-cache recipe on the v1 side still matches v1-era rows, which are untouched because v2 writes to its own DB). No client configuration changes at any point.

## 8. Upstream drift during migration

While the migration runs, upstream (`AmanCode22/deeperseeker`) keeps moving. Rules:

- `git fetch upstream` weekly; triage `main..upstream/main`.
- Fixes touching the **DeepSeek wire surface** (`functions.py` auth/PoW/stream-shape) must be **re-ported into `v2/providers/deepseek/` by hand** — v1 cherry-picks alone are not enough once the adapter port starts (P3+).
- Fixes touching the API shim are irrelevant post-P2 (v2 owns that layer).
- The three open upstream PRs (#11, #12, #13) continue independently of this plan; if merged upstream, v1's divergence shrinks but the v2 plan is unaffected.

## 9. Risks

| Risk | Mitigation |
|------|-----------|
| Regression vs. battle-tested wire tolerance | P3 ports wire code verbatim; P6 soak is a hard gate |
| Speculative generality (providers/grammars built before provider #2) | Accept: P0–P2 artifacts are immediately de-risking even solo (strangler shortcut); the rest waits for a trigger |
| Two systems during transition (~5 weeks) | Explicit ports, explicit soak window, v1 remains the fallback throughout |
| Strict parsing rejecting exotic client payloads v1 tolerated | `extra`-field telemetry from day one (P0); adapters tuned during P6 |
