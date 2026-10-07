from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

_ENV_VARS = (
    "MODE",
    "LIVE_TRADING_CONFIRMED",
    "KRAKEN_API_KEY",
    "KRAKEN_API_SECRET",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "HEALTHCHECK_URL",
    "DATABASE_URL",
    "SETTINGS_PATH",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from the developer's shell environment."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def write_env(tmp_path: Path):
    """Write a .env file from keyword args and return its path."""

    def _write(**values: str) -> Path:
        path = tmp_path / ".env"
        path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
        return path

    return _write
