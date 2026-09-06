"""v2 settings (pydantic-settings). Read via get_settings(); nothing at import."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_", env_file=".env", env_file_encoding="utf-8", frozen=True
    )

    API_KEY: str = "dseeker"
    ADMIN_USER: str = "admin"
    ADMIN_PASSWORD: str = "admin"
    HOST: str = "127.0.0.1"
    PORT: int = 4000
    DB_PATH: str = "deeperseeker.db"
    ENCRYPTION_KEY: str | None = None

    # Cache / compaction (v2 owns the same knobs as v1 for behavior parity)
    SIG_WINDOW_K: int = 8
    PROMPT_BUDGET: int = 24000
    TOOL_RESULT_CLIP: int = 4000
    HISTORY_RECENT_TURNS: int = 6
    SESSION_TTL_DAYS: float = 7.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
