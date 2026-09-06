"""Token repository with fail-fast pool semantics."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiosqlite

from v2.store.crypto import fernet_decrypt


@dataclass(frozen=True, slots=True)
class Token:
    id: int
    alias: Optional[str]
    secret_enc: bytes
    provider: str
    status: str  # ACTIVE | RATE_LIMITED | INVALID
    rate_limited_until: Optional[float]
    last_error: Optional[str]
    requests_ok: int
    requests_failed: int

    def is_available(self) -> bool:
        if self.status == "INVALID":
            return False
        if self.status == "RATE_LIMITED":
            return time.time() >= (self.rate_limited_until or 0)
        return self.status == "ACTIVE"


class TokenRepo:
    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn
        self._lock = asyncio.Lock()

    async def add(self, alias: str | None, secret: str, provider: str = "deepseek") -> Token:
        """Add a new token; encrypts secret at rest."""
        from v2.store.crypto import fernet_encrypt
        enc = fernet_encrypt(secret)
        cur = await self._conn.execute(
            "INSERT INTO tokens (alias, secret_enc, provider, status) VALUES (?, ?, ?, 'ACTIVE')",
            (alias, enc, provider),
        )
        await self._conn.commit()
        return Token(
            id=cur.lastrowid,
            alias=alias,
            secret_enc=enc,
            provider=provider,
            status="ACTIVE",
            rate_limited_until=None,
            last_error=None,
            requests_ok=0,
            requests_failed=0,
        )

    async def get(self, token_id: int) -> Optional[Token]:
        cur = await self._conn.execute("SELECT * FROM tokens WHERE id = ?", (token_id,))
        row = await cur.fetchone()
        return self._row_to_token(row) if row else None

    async def get_by_alias(self, alias: str) -> Optional[Token]:
        cur = await self._conn.execute("SELECT * FROM tokens WHERE alias = ?", (alias,))
        row = await cur.fetchone()
        return self._row_to_token(row) if row else None

    async def list_all(self) -> list[Token]:
        cur = await self._conn.execute("SELECT * FROM tokens ORDER BY id")
        return [self._row_to_token(row) for row in await cur.fetchall()]

    async def pick_available(self, provider: str = "deepseek") -> Optional[Token]:
        """Fail-fast pick: returns first ACTIVE (or expired RATE_LIMITED) token.
        Raises if pool exhausted (caller must return 503).
        """
        cur = await self._conn.execute(
            "SELECT * FROM tokens WHERE provider = ? ORDER BY id", (provider,)
        )
        for row in await cur.fetchall():
            t = self._row_to_token(row)
            if t.is_available():
                return t
        return None

    async def mark_rate_limited(self, token_id: int, retry_after: float = 60.0, error: str = "") -> None:
        until = time.time() + retry_after
        await self._conn.execute(
            "UPDATE tokens SET status='RATE_LIMITED', rate_limited_until=?, last_error=?, requests_failed=requests_failed+1 WHERE id=?",
            (until, error, token_id),
        )
        await self._conn.commit()

    async def mark_invalid(self, token_id: int, error: str) -> None:
        await self._conn.execute(
            "UPDATE tokens SET status='INVALID', last_error=?, requests_failed=requests_failed+1 WHERE id=?",
            (error, token_id),
        )
        await self._conn.commit()

    async def mark_ok(self, token_id: int) -> None:
        """Clear rate-limit on success (v1's mark_active behavior)."""
        await self._conn.execute(
            "UPDATE tokens SET status='ACTIVE', rate_limited_until=NULL, last_error=NULL, requests_ok=requests_ok+1 WHERE id=?",
            (token_id,),
        )
        await self._conn.commit()

    async def update_secret(self, token_id: int, secret_enc: bytes) -> None:
        await self._conn.execute("UPDATE tokens SET secret_enc=? WHERE id=?", (secret_enc, token_id))
        await self._conn.commit()

    async def delete(self, token_id: int) -> None:
        await self._conn.execute("DELETE FROM tokens WHERE id=?", (token_id,))
        await self._conn.commit()

    def _row_to_token(self, row: aiosqlite.Row) -> Token:
        return Token(
            id=row["id"],
            alias=row["alias"],
            secret_enc=row["secret_enc"],
            provider=row["provider"],
            status=row["status"],
            rate_limited_until=row["rate_limited_until"],
            last_error=row["last_error"],
            requests_ok=row["requests_ok"],
            requests_failed=row["requests_failed"],
        )

    def decrypt_secret(self, token: Token) -> str:
        return fernet_decrypt(token.secret_enc)