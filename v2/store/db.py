"""Database connection and migrations (aiosqlite, single connection, WAL)."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiosqlite

from v2.settings import get_settings

log = logging.getLogger(__name__)

_DB: aiosqlite.Connection | None = None
_DB_LOCK = asyncio.Lock()


async def connect() -> aiosqlite.Connection:
    """Get or create the single pooled database connection."""
    global _DB
    async with _DB_LOCK:
        if _DB is not None:
            return _DB
        settings = get_settings()
        db_path = Path(settings.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(db_path, isolation_level=None)
        conn.row_factory = aiosqlite.Row
        # WAL mode + busy timeout for concurrent async access
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        await conn.execute("PRAGMA foreign_keys=ON;")
        _DB = conn
        log.info("db_connected path=%s", db_path)
        return conn


async def close_db() -> None:
    """Close the pooled connection (for shutdown)."""
    global _DB
    async with _DB_LOCK:
        if _DB is not None:
            await _DB.close()
            _DB = None
            log.info("db_closed")


async def init_db() -> None:
    """Run migrations; safe to call multiple times."""
    conn = await connect()
    # migrations are versioned by a user_version pragma
    cur = await conn.execute("PRAGMA user_version;")
    row = await cur.fetchone()
    current = row[0] if row else 0

    migrations = [
        (1, """
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY,
                alias TEXT,
                secret_enc BLOB NOT NULL,
                provider TEXT NOT NULL DEFAULT 'deepseek',
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                rate_limited_until REAL,
                last_error TEXT,
                requests_ok INTEGER DEFAULT 0,
                requests_failed INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_tokens_status ON tokens(status);
        """),
        (2, """
            CREATE TABLE IF NOT EXISTS provider_sessions (
                projection_key BLOB PRIMARY KEY,
                projection TEXT NOT NULL,
                provider TEXT NOT NULL,
                provider_chat TEXT NOT NULL,
                parent_message_id INTEGER NOT NULL,
                token_id INTEGER NOT NULL REFERENCES tokens(id),
                hist_len INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_last_used ON provider_sessions(last_used_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON provider_sessions(token_id);
        """),
        (3, """
            CREATE TABLE IF NOT EXISTS usage (
                request_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                token_id INTEGER,
                in_tokens INTEGER,
                out_tokens INTEGER,
                cost_usd REAL,
                cache_hit INTEGER,
                stream INTEGER,
                duration_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
            CREATE INDEX IF NOT EXISTS idx_usage_token ON usage(token_id);
        """),
    ]

    for version, sql in migrations:
        if version > current:
            log.info("db_migration version=%d", version)
            await conn.executescript(sql)
            await conn.execute(f"PRAGMA user_version = {version};")
            current = version

    # Schema-level integrity checks
    await _verify_schema(conn)
    log.info("db_initialized version=%d", current)


async def _verify_schema(conn: aiosqlite.Connection) -> None:
    """Runtime schema sanity checks (dev-time assertions)."""
    required_tables = {"tokens", "provider_sessions", "usage"}
    cur = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    existing = {row[0] for row in await cur.fetchall()}
    missing = required_tables - existing
    if missing:
        raise RuntimeError(f"db_missing_tables: {missing}")


async def _ensure_conn() -> aiosqlite.Connection:
    """Internal helper used by repos; raises if not initialized."""
    conn = _DB
    if conn is None:
        raise RuntimeError("db_not_initialized — call init_db() first")
    return conn