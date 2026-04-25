"""Application settings loaded from environment and optional `.env` file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def load_environment() -> None:
    """Load `.env` from project root if present (no override of existing env)."""
    load_dotenv(override=False)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str | None
    model: str
    temperature: float

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        load_environment()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=_get_float("OPENAI_TEMPERATURE", 0.0),
        )


@dataclass(frozen=True)
class AppSettings:
    env: str
    log_level: str
    fast_mode_default: bool

    @classmethod
    def from_env(cls) -> "AppSettings":
        load_environment()
        return cls(
            env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            fast_mode_default=_get_bool("FAST_MODE_DEFAULT", False),
        )
