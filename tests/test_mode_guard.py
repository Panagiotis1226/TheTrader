"""Safety Invariant #1: live mode requires MODE=live AND LIVE_TRADING_CONFIRMED=yes."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ai_trader.config import ConfigError, Mode, load_env
from ai_trader.main import (
    EXIT_CONFIG_ERROR,
    EXIT_LIVE_NOT_CONFIRMED,
    EXIT_OK,
    LiveTradingNotConfirmedError,
    enforce_mode_guard,
    main,
)

from .conftest import REPO_ROOT, SETTINGS_PATH


def test_defaults_to_paper_with_no_env_file(tmp_path: Path) -> None:
    env = load_env(tmp_path / "does-not-exist.env")
    assert enforce_mode_guard(env) is Mode.PAPER


def test_defaults_to_paper_with_empty_mode(write_env) -> None:
    assert enforce_mode_guard(load_env(write_env(MODE=""))) is Mode.PAPER


def test_env_example_is_paper_and_unconfirmed() -> None:
    env = load_env(REPO_ROOT / ".env.example")
    assert env.mode is Mode.PAPER
    assert env.live_trading_confirmed == "no"
    assert enforce_mode_guard(env) is Mode.PAPER


def test_live_without_confirmation_flag_refuses(write_env) -> None:
    with pytest.raises(LiveTradingNotConfirmedError):
        enforce_mode_guard(load_env(write_env(MODE="live")))


@pytest.mark.parametrize("confirmed", ["no", "", "y", "true", "1", "yes please", "on"])
def test_live_with_wrong_confirmation_refuses(write_env, confirmed: str) -> None:
    env = load_env(write_env(MODE="live", LIVE_TRADING_CONFIRMED=confirmed))
    with pytest.raises(LiveTradingNotConfirmedError):
        enforce_mode_guard(env)


@pytest.mark.parametrize("mode", ["LIVE", " Live "])
def test_mode_normalization_does_not_bypass_guard(write_env, mode: str) -> None:
    with pytest.raises(LiveTradingNotConfirmedError):
        enforce_mode_guard(load_env(write_env(MODE=mode)))


def test_confirmation_without_live_mode_stays_paper(write_env) -> None:
    env = load_env(write_env(MODE="paper", LIVE_TRADING_CONFIRMED="yes"))
    assert enforce_mode_guard(env) is Mode.PAPER


def test_confirmation_alone_stays_paper(write_env) -> None:
    env = load_env(write_env(LIVE_TRADING_CONFIRMED="yes"))
    assert enforce_mode_guard(env) is Mode.PAPER


@pytest.mark.parametrize("confirmed", ["yes", "YES", " yes "])
def test_live_with_both_flags_allowed(write_env, confirmed: str) -> None:
    env = load_env(write_env(MODE="live", LIVE_TRADING_CONFIRMED=confirmed))
    assert enforce_mode_guard(env) is Mode.LIVE


def test_process_env_live_without_confirmation_refuses(
    write_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODE", "live")
    with pytest.raises(LiveTradingNotConfirmedError):
        enforce_mode_guard(load_env(write_env(MODE="paper")))


def test_unknown_mode_is_a_config_error(write_env) -> None:
    with pytest.raises(ConfigError, match="MODE"):
        load_env(write_env(MODE="production"))


# --- main() refuses to start ------------------------------------------------------


def _run(env_file: Path) -> int:
    return main(["--env-file", str(env_file), "--settings", str(SETTINGS_PATH)])


def test_main_refuses_live_without_confirmation(
    write_env, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.CRITICAL):
        assert _run(write_env(MODE="live")) == EXIT_LIVE_NOT_CONFIRMED
    assert "refusing to start" in caplog.text
    assert "starting in" not in caplog.text


def test_main_refuses_live_with_confirmation_no(write_env) -> None:
    assert _run(write_env(MODE="live", LIVE_TRADING_CONFIRMED="no")) == EXIT_LIVE_NOT_CONFIRMED


def test_main_starts_in_paper_by_default(write_env, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert _run(write_env()) == EXIT_OK
    assert "starting in paper mode" in caplog.text


def test_main_starts_live_with_both_flags(write_env, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert _run(write_env(MODE="live", LIVE_TRADING_CONFIRMED="yes")) == EXIT_OK
    assert "starting in live mode" in caplog.text


def test_main_refuses_on_invalid_mode(write_env) -> None:
    assert _run(write_env(MODE="prod")) == EXIT_CONFIG_ERROR


def test_main_refuses_on_missing_settings(write_env, tmp_path: Path) -> None:
    code = main(["--env-file", str(write_env()), "--settings", str(tmp_path / "nope.yaml")])
    assert code == EXIT_CONFIG_ERROR
