"""Non-streaming orchestrator — Conversation -> ChatCompletion with TokenBudget."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from v2.ir import Conversation
from v2.providers.base import ProviderAdapter, RenderedPrompt, SendOptions, SessionRef
from v2.providers.registry import get_registry
from v2.store.repo_sessions import ProjectionStore
from v2.store.repo_tokens import TokenRepo
from v2.store.repo_usage import UsageRepo
from v2.compaction.budget import TokenBudget, BudgetPolicy
from v2.settings import get_settings


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """Incoming request specification."""
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """Non-streaming completion response."""
    id: str
    model: str
    choices: list[dict]
    usage: dict
    created: int


class Orchestrator:
    """Core request pipeline — the ONLY place that composes everything.

    Responsibilities:
      1. Resolve model -> provider + config
      2. Convert wire messages -> IR Conversation
      3. Projection cache lookup (Option B compare)
      4. Token acquisition (fail-fast)
      5. On miss: ensure_chat under per-projection lock
      6. TokenBudget.plan (single compaction site)
      7. Adapter.send -> grammar transform -> response
      8. Save projections + metering
    """

    def __init__(
        self,
        session_store: ProjectionStore,
        token_repo: TokenRepo,
        usage_repo: UsageRepo,
        budget: TokenBudget | None = None,
        budget_policy: BudgetPolicy | None = None,
    ):
        self._session_store = session_store
        self._token_repo = token_repo
        self._usage_repo = usage_repo
        self._budget = budget or TokenBudget(budget_policy)
        self._budget_policy = budget_policy or self._budget.policy
        self._registry = get_registry()
        self._locks: dict[bytes, asyncio.Lock] = {}

    async def handle(self, conv: Conversation, spec: RequestSpec) -> CompletionResponse:
        """Non-streaming request handler."""
        start = time.perf_counter()
        request_id = uuid.uuid4().hex[:16]

        # 1. Resolve model -> provider + config
        provider, model_cfg = self._registry.resolve(spec.model)
        model_name = model_cfg.model_name

        # 2. Projection cache lookup
        hist_len = len(conv.messages)
        session_row = await self._session_store.lookup(conv, hist_len, model_cfg.provider)

        # 3. Token acquisition (fail-fast)
        token = await self._token_repo.pick_available(model_cfg.provider)
        if not token:
            raise RuntimeError("no_available_tokens")  # -> 503

        # 4. On miss: ensure_chat under per-projection lock
        if session_row is None:
            proj = self._session_store._build_projection(conv, hist_len, provider=model_cfg.provider)
            proj_key = self._session_store._hash_projection(proj)
            lock = self._locks.setdefault(proj_key, asyncio.Lock())
            async with lock:
                # Double-check after acquiring lock
                session_row = await self._session_store.lookup(conv, hist_len, model_cfg.provider)
                if session_row is None:
                    provider_chat = await provider.ensure_chat(
                        self._token_repo.decrypt_secret(token), None
                    )
                    session_row = await self._session_store.save(
                        conv=conv,
                        provider=model_cfg.provider,
                        provider_chat=provider_chat.chat_id,
                        token_id=token.id,
                        parent_message_id=0,
                        hist_len=hist_len,
                    )
                parent_msg_id = session_row.parent_message_id
        else:
            parent_msg_id = session_row.parent_message_id

        # 5. TokenBudget.plan (single compaction site)
        rendered = self._budget.plan(conv, self._budget_policy)
        rendered.model = model_name
        rendered.tools = spec.tools
        rendered.max_tokens = spec.max_tokens
        rendered.temperature = spec.temperature
        rendered.metadata = dict(spec.metadata) if spec.metadata else {}

        # The adapter sends under the credential of the token the pool just
        # picked for this provider — never a caller-supplied placeholder.
        rendered.metadata["auth_token"] = self._token_repo.decrypt_secret(token)

        # 6. Send to provider (non-streaming via stream + collect)
        opts: SendOptions = {
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
        }

        # Collect streamed events
        events = []
        async for ev in provider.send(
            SessionRef(
                provider=model_cfg.provider,
                chat_id=session_row.provider_chat,
                parent_message_id=parent_msg_id,
                token_id=token.id,
            ),
            rendered,
            opts=opts,
        ):
            events.append(ev)

        # 7. Transform via grammar (for non-streaming, just use the final text)
        # For the fake provider and simple cases, events contain the text directly
        response_text = ""
        for ev in events:
            if hasattr(ev, "text"):
                response_text += ev.text

        # 8. Build completion response
        duration_ms = int((time.perf_counter() - start) * 1000)
        completion = CompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            model=model_name,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            created=int(time.time()),
        )

        # 9. Save updated projection (parent_message_id advances)
        await self._session_store.save(
            conv=conv,  # Note: this should be the conversation WITH the new assistant message
            provider=model_cfg.provider,
            provider_chat=session_row.provider_chat,
            token_id=token.id,
            parent_message_id=parent_msg_id + 2,  # user + assistant
            hist_len=hist_len + 2,
        )

        # 10. Metering
        await self._usage_repo.record(
            request_id=request_id,
            provider=model_cfg.provider,
            model=model_name,
            token_id=token.id,
            in_tokens=0,  # TODO: calculate from rendered
            out_tokens=0,
            cost_usd=0.0,
            cache_hit=session_row is not None,
            stream=False,
            duration_ms=duration_ms,
        )

        return completion

    async def handle_stream(
        self, conv: Conversation, spec: RequestSpec
    ) -> AsyncIterator[bytes]:
        """Streaming handler (placeholder - full impl in P2)."""
        raise NotImplementedError("Streaming handler requires P2 emitters")