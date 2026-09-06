"""Usage/metering repository."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import aiosqlite


@dataclass(frozen=True, slots=True)
class UsageRow:
    request_id: str
    ts: str
    provider: Optional[str]
    model: Optional[str]
    token_id: Optional[int]
    in_tokens: int
    out_tokens: int
    cost_usd: float
    cache_hit: bool
    stream: bool
    duration_ms: int


class UsageRepo:
    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def record(
        self,
        request_id: str,
        provider: str | None,
        model: str | None,
        token_id: int | None,
        in_tokens: int,
        out_tokens: int,
        cost_usd: float,
        cache_hit: bool,
        stream: bool,
        duration_ms: int,
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        await self._conn.execute(
            """
            INSERT INTO usage
            (request_id, ts, provider, model, token_id, in_tokens, out_tokens,
             cost_usd, cache_hit, stream, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, ts, provider, model, token_id,
                in_tokens, out_tokens, cost_usd,
                1 if cache_hit else 0, 1 if stream else 0, duration_ms,
            ),
        )
        await self._conn.commit()

    async def get_stats(self, since_ts: str | None = None) -> dict:
        """Aggregate stats for /health and admin dashboard."""
        where = "WHERE ts >= ?" if since_ts else ""
        params = (since_ts,) if since_ts else ()
        cur = await self._conn.execute(
            f"""
            SELECT
                COUNT(*) as total_requests,
                SUM(in_tokens) as total_in,
                SUM(out_tokens) as total_out,
                SUM(cost_usd) as total_cost,
                SUM(cache_hit) as cache_hits,
                SUM(stream) as stream_requests,
                AVG(duration_ms) as avg_duration_ms
            FROM usage {where}
            """,
            params,
        )
        row = await cur.fetchone()
        return dict(row) if row else {}