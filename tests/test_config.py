from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from ai_trader.config import (
    ConfigError,
    TradingSettings,
    load_config,
    load_env,
    load_trading_settings,
)

from .conftest import SETTINGS_PATH

BASE: dict[str, Any] = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))


def _with(**overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE)
    for dotted, value in overrides.items():
        target = data
        *parents, leaf = dotted.split("__")
        for key in parents:
            target = target[key]
        target[leaf] = value
    return data


def _invalid(data: dict[str, Any], tmp_path: Path, match: str) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_trading_settings(path)


def test_repo_settings_load() -> None:
    s = load_trading_settings(SETTINGS_PATH)
    assert s.pairs == ["BTC/CAD", "ETH/CAD"]
    assert s.quote_currency == "CAD"
    assert s.decision_interval_minutes == 240
    assert s.paper.starting_cash_cad == Decimal("10000")
    assert s.risk.max_trade_pct_of_equity == Decimal("10")
    assert s.benchmarks == ["buy_and_hold", "ma_crossover", "do_nothing"]
    [claude] = s.models
    assert (claude.name, claude.provider, claude.model) == (
        "claude",
        "claude_code",
        "claude-opus-5-5",
    )
    assert not claude.is_placeholder
    assert s.llm.max_daily_calls == 12
    assert s.llm.max_daily_cost_usd == Decimal("2")


def test_yaml_floats_become_exact_decimals() -> None:
    s = load_trading_settings(SETTINGS_PATH)
    assert s.paper.taker_fee_pct == Decimal("0.8")
    assert str(s.paper.taker_fee_pct) in {"0.8", "0.80"}
    assert s.risk.min_confidence == Decimal("0.6")


def test_settings_are_immutable() -> None:
    s = load_trading_settings(SETTINGS_PATH)
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen error type varies
        s.risk.max_trade_pct_of_equity = Decimal("100")  # type: ignore[misc]


def test_pairs_are_normalized() -> None:
    s = TradingSettings.model_validate(_with(pairs=[" btc/cad ", "eth/cad"]))
    assert s.pairs == ["BTC/CAD", "ETH/CAD"]


def test_load_config_combines_sources(write_env) -> None:
    cfg = load_config(write_env(), SETTINGS_PATH)
    assert cfg.env.database_url == "sqlite:///data/trader.db"
    assert cfg.trading.pairs == ["BTC/CAD", "ETH/CAD"]


def test_settings_path_from_env(write_env) -> None:
    cfg = load_config(write_env(SETTINGS_PATH=str(SETTINGS_PATH)))
    assert cfg.trading.quote_currency == "CAD"


def test_placeholder_models_detected() -> None:
    s = TradingSettings.model_validate(
        _with(
            models=[
                {"name": "a", "provider": "litellm", "model": "anthropic/<model-id>"},
                {"name": "b", "provider": "litellm", "model": "openai/some-real-model"},
            ]
        )
    )
    assert [m.is_placeholder for m in s.models] == [True, False]


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (_with(risk__max_trade_pct=5), "max_trade_pct"),  # typo'd key
        (_with(unknown_top_level=1), "unknown_top_level"),
        ({k: v for k, v in BASE.items() if k != "risk"}, "risk"),
        (_with(pairs=["BTC/USD:USD"]), "spot pair"),  # derivative symbol
        (_with(pairs=["BTCCAD"]), "spot pair"),
        (_with(pairs=["BTC/USD"]), "not quoted in CAD"),
        (_with(pairs=["BTC/CAD", "btc/cad"]), "unique"),
        (_with(pairs=[]), "pairs"),
        (_with(risk__max_trade_pct_of_equity=40), "max_trade_pct_of_equity must be"),
        (_with(risk__max_position_pct_per_pair=70), "max_position_pct_per_pair must be"),
        (_with(risk__max_total_exposure_pct=150), "max_total_exposure_pct"),
        (_with(risk__min_confidence=1.5), "min_confidence"),
        (_with(risk__daily_loss_limit_pct=0), "daily_loss_limit_pct"),
        (_with(risk__max_trades_per_day=0), "max_trades_per_day"),
        (_with(paper__taker_fee_pct=-0.1), "taker_fee_pct"),
        (_with(paper__starting_cash_cad=0), "starting_cash_cad"),
        (_with(decision_interval_minutes=0), "decision_interval_minutes"),
        (_with(benchmarks=["buy_the_dip"]), "benchmarks"),
        (_with(llm__timeout_seconds=0), "timeout_seconds"),
        (_with(llm__max_daily_cost_usd=0), "max_daily_cost_usd"),
        ({k: v for k, v in BASE.items() if k != "llm"}, "llm"),
        (_with(llm__max_daily_calls=0), "max_daily_calls"),
        (_with(models=[{"name": "x", "provider": "openrouter", "model": "a/b"}]), "provider"),
        (_with(models=[{"name": "x", "model": "a/b"}]), "provider"),
        (
            _with(
                models=[
                    {"name": "x", "provider": "claude_code", "model": "opus", "temperature": 0.2}
                ]
            ),
            "not supported with provider claude_code",
        ),
        (_with(benchmarks=["do_nothing", "do_nothing"]), "benchmarks must be unique"),
        (
            _with(
                models=[
                    {"name": "x", "provider": "litellm", "model": "a/b"},
                    {"name": "x", "provider": "litellm", "model": "c/d"},
                ]
            ),
            "model names must be unique",
        ),
        (_with(models=[{"name": "do_nothing", "provider": "litellm", "model": "a/b"}]), "clash"),
        (_with(models=[{"name": "Bad Name", "provider": "litellm", "model": "a/b"}]), "name"),
    ],
)
def test_invalid_settings_rejected(data: dict[str, Any], match: str, tmp_path: Path) -> None:
    _invalid(data, tmp_path, match)


def test_missing_settings_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_trading_settings(tmp_path / "missing.yaml")


def test_non_mapping_settings_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_trading_settings(path)


# --- secrets (Safety Invariant #7) ----------------------------------------------


def test_secrets_never_appear_in_repr(write_env) -> None:
    secret = "sk-super-secret-value-123"
    env = load_env(
        write_env(
            KRAKEN_API_KEY=secret,
            KRAKEN_API_SECRET=secret,
            ANTHROPIC_API_KEY=secret,
            TELEGRAM_BOT_TOKEN=secret,
            HEALTHCHECK_URL=f"https://hc-ping.com/{secret}",
        )
    )
    assert secret not in repr(env)
    assert secret not in str(env)
    assert secret not in env.model_dump_json()
    assert env.kraken_api_key is not None
    assert env.kraken_api_key.get_secret_value() == secret


def test_empty_secrets_are_none(write_env) -> None:
    env = load_env(write_env(KRAKEN_API_KEY="", TELEGRAM_CHAT_ID=""))
    assert env.kraken_api_key is None
    assert env.telegram_chat_id is None
