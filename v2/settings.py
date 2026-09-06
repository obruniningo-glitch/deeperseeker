"""v2 settings (pydantic-settings). Read via get_settings(); nothing at import."""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEKER_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore", frozen=True,
    )

    API_KEY: str = "dseeker"
    ADMIN_USER: str = "admin"
    ADMIN_PASSWORD: str = "admin"
    # v1's .env ships bare HOST/PORT; accept both spellings.
    HOST: str = Field(default="127.0.0.1",
                      validation_alias=AliasChoices("HOST", "DEEPSEEKER_HOST"))
    PORT: int = Field(default=4000,
                      validation_alias=AliasChoices("PORT", "DEEPSEEKER_PORT"))
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
