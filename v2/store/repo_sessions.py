"""ProjectionStore with Option-B compare (cache lookup/save)."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import aiosqlite

from v2.ir import Conversation, Message, ToolResultBlock, ToolUseBlock


@dataclass(frozen=True, slots=True)
class SessionRow:
    projection_key: bytes
    projection: str  # JSON payload
    provider: str
    provider_chat: str
    parent_message_id: int
    token_id: int
    hist_len: int
    created_at: str
    last_used_at: str


@dataclass(frozen=True, slots=True)
class ProjectionPayload:
    """The canonical projection payload stored per session.
    Option B: compare structurally (system, anchor, tail, hist_len, pending_tool_calls).
    """
    system: str | None
    anchor: str  # first user message text
    tail: list[dict]  # canonicalized last K messages (dict form)
    hist_len: int
    pending_tool_calls: list[str] = field(default_factory=list)  # tool_use ids we issued last turn
    provider: str = ""  # provider name for exact-match filtering


class ProjectionStore:
    """Projection cache with Option-B structural compare and fallback tier.

    Hit logic (in order):
      1. Exact projection_key (sha256 of canonical JSON) match
      2. Structural compare: system, anchor, tail (exact), pending_tool_calls (set eq)
      3. Fallback tier: compare without trailing assistant message
      4. hist_len one-sided guard
    """

    def __init__(self, conn: aiosqlite.Connection, sig_namespace: str = "ds-sig-v2", sig_window_k: int = 8):
        self._conn = conn
        self._lock = asyncio.Lock()
        self._sig_namespace = sig_namespace
        self._sig_window_k = sig_window_k

    # ---- public API ----

    async def lookup(self, conv: Conversation, hist_len: int | None = None, provider: str = "deepseek") -> Optional[SessionRow]:
        """Find existing session for this conversation.
        Returns SessionRow on hit, None on miss.
        """
        if not conv.messages:
            return None

        projection = self._build_projection(conv, hist_len, provider=provider)
        proj_key = self._hash_projection(projection)

        # Fast path: exact key match
        cur = await self._conn.execute(
            "SELECT * FROM provider_sessions WHERE projection_key = ?", (proj_key,)
        )
        row = await cur.fetchone()
        if row:
            return self._row_to_session(row)

        # Structural compare (Option B)
        cur = await self._conn.execute(
            "SELECT * FROM provider_sessions WHERE provider = ? AND hist_len = ?",
            (projection.provider, hist_len or 0),
        )
        async for row in cur:
            stored_proj = json.loads(row["projection"])
            if self._structural_match(projection, stored_proj):
                return self._row_to_session(row)

        # Fallback tier: compare without trailing assistant
        if self._has_trailing_assistant(conv):
            fallback_proj = self._build_projection(conv, hist_len, drop_last_assistant=True)
            fb_key = self._hash_projection(fallback_proj)
            cur = await self._conn.execute(
                "SELECT * FROM provider_sessions WHERE projection_key = ?", (fb_key,)
            )
            row = await cur.fetchone()
            if row:
                return self._row_to_session(row)

        return None

    async def save(
        self,
        conv: Conversation,
        provider: str,
        provider_chat: str,
        token_id: int,
        parent_message_id: int,
        hist_len: int,
    ) -> SessionRow:
        """Save or update session row with current projection."""
        projection = self._build_projection(conv, hist_len, provider=provider)
        proj_key = self._hash_projection(projection)
        proj_json = json.dumps(dataclasses.asdict(projection), sort_keys=True, separators=(",", ":"))
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

        async with self._lock:
            # Upsert by projection_key
            await self._conn.execute(
                """
                INSERT INTO provider_sessions
                (projection_key, projection, provider, provider_chat, parent_message_id,
                 token_id, hist_len, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(projection_key) DO UPDATE SET
                    provider_chat=excluded.provider_chat,
                    parent_message_id=excluded.parent_message_id,
                    token_id=excluded.token_id,
                    last_used_at=excluded.last_used_at
                """,
                (proj_key, proj_json, provider, provider_chat, parent_message_id,
                 token_id, hist_len, now, now),
            )
            await self._conn.commit()

        return SessionRow(
            projection_key=proj_key,
            projection=proj_json,
            provider=provider,
            provider_chat=provider_chat,
            parent_message_id=parent_message_id,
            token_id=token_id,
            hist_len=hist_len,
            created_at=now,
            last_used_at=now,
        )

    async def delete(self, projection_key: bytes) -> bool:
        cur = await self._conn.execute(
            "DELETE FROM provider_sessions WHERE projection_key = ?", (projection_key,)
        )
        await self._conn.commit()
        return cur.rowcount > 0

    async def prune_ttl(self, ttl_days: float) -> int:
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.gmtime(time.time() - ttl_days * 86400),
        )
        cur = await self._conn.execute(
            "DELETE FROM provider_sessions WHERE last_used_at < ?", (cutoff,)
        )
        await self._conn.commit()
        return cur.rowcount

    # ---- projection building ----

    def _build_projection(
        self, conv: Conversation, hist_len: int | None, provider: str = "deepseek", drop_last_assistant: bool = False
    ) -> ProjectionPayload:
        # system prompt (single string)
        system = conv.system_text() if conv.system else None

        # anchor = first user message text
        anchor = ""
        for m in conv.messages:
            if m.role == "user":
                anchor = m.text()
                break

        # tail = last K messages (canonicalized)
        msgs = list(conv.messages)
        if drop_last_assistant and msgs and msgs[-1].role == "assistant":
            msgs = msgs[:-1]

        tail_msgs = msgs[-self._sig_window_k :] if self._sig_window_k > 0 else msgs
        tail = [self._canonicalize_message(m) for m in tail_msgs]

        # pending tool calls = tool_use ids from last assistant message
        pending = []
        for m in reversed(conv.messages):
            if m.role == "assistant":
                for b in m.blocks:
                    if isinstance(b, ToolUseBlock) and b.id:
                        pending.append(b.id)
                break

        return ProjectionPayload(
            system=system,
            anchor=anchor,
            tail=tail,
            hist_len=hist_len or 0,
            pending_tool_calls=pending,
            provider=provider,
        )

    def _canonicalize_message(self, msg: Message) -> dict:
        """Return canonical dict form for structural comparison."""
        blocks = []
        for b in msg.blocks:
            if isinstance(b, (str,)):
                continue
            bd = {"type": b.__class__.__name__.replace("Block", "").lower()}
            if isinstance(b, ToolUseBlock):
                bd.update({"id": b.id, "name": b.name, "arguments": b.arguments})
            elif isinstance(b, ToolResultBlock):
                bd.update({"tool_use_id": b.tool_use_id, "content": b.text(), "is_error": b.is_error})
            elif hasattr(b, "text") and isinstance(b.text, str):
                bd["text"] = b.text
            elif hasattr(b, "text") and callable(b.text):
                bd["text"] = b.text()
            blocks.append(bd)
        return {"role": msg.role, "blocks": blocks}

    def _hash_projection(self, projection: ProjectionPayload) -> bytes:
        """SHA256 over canonical JSON with namespace."""
        data = f"{self._sig_namespace}|{json.dumps(dataclasses.asdict(projection), sort_keys=True, separators=(',', ':'))}".encode()
        return hashlib.sha256(data).digest()

    def _structural_match(self, a: ProjectionPayload, b: dict) -> bool:
        """Option B structural compare."""
        # system, anchor, hist_len: exact match
        if a.system != b.get("system"):
            return False
        if a.anchor != b.get("anchor"):
            return False
        if a.hist_len != b.get("hist_len"):
            return False
        # tail: element-wise canonical message equality
        tail_a = a.tail
        tail_b = b.get("tail", [])
        if len(tail_a) != len(tail_b):
            return False
        for ma, mb in zip(tail_a, tail_b):
            if ma != mb:
                return False
        # pending_tool_calls: set equality (order doesn't matter)
        if set(a.pending_tool_calls) != set(b.get("pending_tool_calls", [])):
            return False
        return True

    def _has_trailing_assistant(self, conv: Conversation) -> bool:
        return bool(conv.messages) and conv.messages[-1].role == "assistant"

    def _row_to_session(self, row: aiosqlite.Row) -> SessionRow:
        return SessionRow(
            projection_key=row["projection_key"],
            projection=row["projection"],
            provider=row["provider"],
            provider_chat=row["provider_chat"],
            parent_message_id=row["parent_message_id"],
            token_id=row["token_id"],
            hist_len=row["hist_len"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
        )