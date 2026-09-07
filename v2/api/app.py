"""v2 FastAPI application factory and routes."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from v2.settings import get_settings
from v2.store import init_db, close_db, connect
from v2.store.repo_sessions import ProjectionStore
from v2.store.repo_tokens import TokenRepo
from v2.store.repo_usage import UsageRepo
from v2.providers.registry import get_registry
from v2.pipeline.orchestrator import Orchestrator, RequestSpec
from v2.compaction.budget import TokenBudget, BudgetPolicy
from v2.ir import Conversation, Message, TextBlock
from v2.adapters import to_ir


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    tools: Optional[list[dict]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    # v2 extensions
    thinking: Optional[bool] = None
    search: Optional[bool] = None
    file_ids: Optional[list[str]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


class HealthResponse(BaseModel):
    tokens: dict
    cached_sessions: int
    cookie_expiry: Optional[int]
    cookie_valid: bool


# Global state (initialized in lifespan)
_orchestrator: Orchestrator | None = None
_session_store: ProjectionStore | None = None
_token_repo: TokenRepo | None = None
_usage_repo: UsageRepo | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB, create orchestrator."""
    global _orchestrator, _session_store, _token_repo, _usage_repo

    # Initialize DB
    await init_db()
    conn = await connect()

    # Create repos
    _session_store = ProjectionStore(conn, sig_namespace="ds-sig-v2", sig_window_k=8)
    _token_repo = TokenRepo(conn)
    _usage_repo = UsageRepo(conn)

    # Create budget policy from settings
    settings = get_settings()
    budget_policy = BudgetPolicy(
        total=settings.PROMPT_BUDGET,
        tool_result_clip=settings.TOOL_RESULT_CLIP,
        recent_turns=settings.HISTORY_RECENT_TURNS,
        min_recent_turns=2,
        old_text_clip=200,
        head_ratio=0.7,
    )

    # Create orchestrator
    _orchestrator = Orchestrator(
        session_store=_session_store,
        token_repo=_token_repo,
        usage_repo=_usage_repo,
        budget=TokenBudget(budget_policy),
        budget_policy=budget_policy,
    )

    # Ensure default token exists
    try:
        await _token_repo.get(1)
    except Exception:
        await _token_repo.add("default", settings.API_KEY, "deepseek")

    yield

    # Shutdown
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="deeperseeker v2",
        description="DeepSeek web-chat gateway (v2 architecture)",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

    # Auth dependency
    async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        api_key = authorization[7:]
        settings = get_settings()
        if api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key

    # ---- Routes ----

    @app.get("/health", response_model=HealthResponse)
    async def health(api_key: str = Header(None, alias="Authorization")):
        """Health check endpoint (API-key protected)."""
        await verify_api_key(api_key)
        assert _token_repo is not None
        assert _session_store is not None

        token_stats = {}
        conn = await connect()
        async with conn.execute("SELECT status, COUNT(*) FROM tokens GROUP BY status") as cursor:
            async for row in cursor:
                token_stats[row[0]] = row[1]

        sessions = await conn.execute("SELECT COUNT(*) FROM provider_sessions")
        row = await sessions.fetchone()
        session_count = row[0] if row else 0

        cookie_expiry = None
        try:
            import os, json
            cookie_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aws_cookies_deepseek.json")
            if os.path.exists(cookie_path):
                with open(cookie_path) as f:
                    cookie_expiry = json.load(f).get("expiry")
        except Exception:
            pass

        return HealthResponse(
            tokens={
                "active": token_stats.get("ACTIVE", 0),
                "rate_limited": token_stats.get("RATE_LIMITED", 0),
                "total": sum(token_stats.values()),
            },
            cached_sessions=session_count or 0,
            cookie_expiry=cookie_expiry,
            cookie_valid=cookie_expiry is not None and cookie_expiry > time.time(),
        )

    @app.get("/v1/models")
    async def list_models(api_key: str = Header(None, alias="Authorization")):
        """List available models (OpenAI-compatible)."""
        await verify_api_key(api_key)
        return {
            "object": "list",
            "data": [
                {"id": "instant", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
                {"id": "expert", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
                {"id": "vision", "object": "model", "created": int(time.time()), "owned_by": "deepseek"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        api_key: str = Header(None, alias="Authorization"),
    ):
        """OpenAI-compatible chat completions (non-streaming + streaming)."""
        await verify_api_key(api_key)

        # Convert wire messages to IR Conversation
        try:
            conv = to_ir(request.model_dump(), "openai")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid message format: {e}")

        # Apply request-level overrides
        metadata = {
            "auth_token": get_settings().API_KEY,  # In v2, we use the pool; this is for adapter metadata
            "thinking": request.thinking,
            "search": request.search,
            "file_ids": request.file_ids or [],
        }

        spec = RequestSpec(
            model=request.model,
            messages=request.messages,
            tools=request.tools,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
            metadata=metadata,
        )

        if request.stream:
            return EventSourceResponse(
                _stream_chat(conv, spec),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming
        assert _orchestrator is not None
        try:
            resp = await _orchestrator.handle(conv, spec)
            return JSONResponse(resp.__dict__)
        except RuntimeError as e:
            if "no_available_tokens" in str(e):
                raise HTTPException(status_code=503, detail="No available tokens")
            raise HTTPException(status_code=500, detail=str(e))

    async def _stream_chat(conv: Conversation, spec: RequestSpec):
        """Stream chat completion as SSE (placeholder for P2 integration)."""
        # TODO: Implement streaming via orchestrator.handle_stream + emitter
        # For now, fall back to non-streaming and simulate
        assert _orchestrator is not None
        resp = await _orchestrator.handle(conv, spec)
        for choice in resp.choices:
            content = choice["message"].get("content", "")
            # Simulate streaming by yielding words
            for word in content.split():
                yield f"data: {json.dumps({'choices': [{'delta': {'content': word + ' '}}]})}\n\n"
                await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"

    # ---- Admin Dashboard (protected) ----

    async def admin_auth(authorization: Optional[str] = Header(None)) -> str:
        await verify_api_key(authorization)
        # In production, check admin role
        return "admin"

    @app.get("/admin/tokens")
    async def admin_list_tokens(user: str = Header(None, alias="Authorization")):
        await admin_auth(user)
        assert _token_repo is not None
        tokens = await _token_repo.list_all()
        return [{"id": t.id, "alias": t.alias, "status": t.status, "requests_ok": t.requests_ok, "requests_failed": t.requests_failed} for t in tokens]

    @app.post("/admin/tokens")
    async def admin_add_token(alias: str, secret: str, provider: str = "deepseek", user: str = Header(None, alias="Authorization")):
        await admin_auth(user)
        assert _token_repo is not None
        token = await _token_repo.add(alias, secret, provider)
        return {"id": token.id, "alias": token.alias, "status": token.status}

    @app.delete("/admin/tokens/{token_id}")
    async def admin_delete_token(token_id: int, user: str = Header(None, alias="Authorization")):
        await admin_auth(user)
        assert _token_repo is not None
        await _token_repo.delete(token_id)
        return {"status": "deleted"}

    @app.post("/admin/prune-sessions")
    async def admin_prune_sessions(ttl_days: float = 7.0, user: str = Header(None, alias="Authorization")):
        await admin_auth(user)
        assert _session_store is not None
        deleted = await _session_store.prune_ttl(ttl_days)
        return {"deleted": deleted}

    @app.get("/admin/sessions")
    async def admin_list_sessions(limit: int = 100, user: str = Header(None, alias="Authorization")):
        await admin_auth(user)
        conn = await connect()
        cur = await conn.execute(
            "SELECT projection_key, provider, provider_chat, token_id, hist_len, created_at, last_used_at FROM provider_sessions ORDER BY last_used_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    @app.get("/admin/usage")
    async def admin_usage(since: Optional[str] = None, user: str = Header(None, alias="Authorization")):
        await admin_auth(user)
        assert _usage_repo is not None
        return await _usage_repo.get_stats(since)

    @app.get("/admin/health")
    async def admin_health(user: str = Header(None, alias="Authorization")):
        await admin_auth(user)
        # Reuse health logic
        return await health(authorization=f"Bearer {get_settings().API_KEY}")

    return app


# Entry point for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)