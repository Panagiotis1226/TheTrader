"""Entrypoint.

Startup order: load + validate config → mode guard → (Phase 4) scheduler.
The mode guard runs before anything that could touch an exchange.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ai_trader.config import DEFAULT_ENV_FILE, ConfigError, EnvSettings, Mode, load_config

log = logging.getLogger("ai_trader")

LIVE_CONFIRMATION_VALUE = "yes"

EXIT_OK = 0
EXIT_LIVE_NOT_CONFIRMED = 1
EXIT_CONFIG_ERROR = 2


class LiveTradingNotConfirmedError(RuntimeError):
    """MODE=live was requested without LIVE_TRADING_CONFIRMED=yes."""


def enforce_mode_guard(env: EnvSettings) -> Mode:
    """Safety Invariant #1: return the trading mode, or refuse to start.

    Live trading requires BOTH ``MODE=live`` and ``LIVE_TRADING_CONFIRMED=yes``.
    Anything else that asks for live mode raises. The returned ``Mode`` is the only
    value downstream code should use to choose a broker.
    """
    if env.mode is Mode.PAPER:
        return Mode.PAPER
    if env.mode is Mode.LIVE:
        if env.live_trading_confirmed.strip().lower() != LIVE_CONFIRMATION_VALUE:
            raise LiveTradingNotConfirmedError(
                "MODE=live requires LIVE_TRADING_CONFIRMED=yes in .env; refusing to start"
            )
        return Mode.LIVE
    raise LiveTradingNotConfirmedError(f"Unknown mode {env.mode!r}; refusing to start")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai-trader", description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="settings YAML (default: SETTINGS_PATH or config/settings.yaml)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(env_file=args.env_file, settings_path=args.settings)
    except ConfigError as exc:
        log.critical("Refusing to start: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        mode = enforce_mode_guard(config.env)
    except LiveTradingNotConfirmedError as exc:
        log.critical("%s", exc)
        return EXIT_LIVE_NOT_CONFIRMED

    trading = config.trading
    if mode is Mode.LIVE:
        log.warning("LIVE TRADING ENABLED — real orders will be placed")
    log.info(
        "ai-trader starting in %s mode: pairs=%s, models=%s, benchmarks=%s",
        mode.value,
        trading.pairs,
        [m.name for m in trading.models],
        trading.benchmarks,
    )
    for model in trading.models:
        if model.is_placeholder:
            log.warning(
                "Model %r has placeholder ID %r; set a real model in settings.yaml",
                model.name,
                model.litellm_model,
            )

    # The decision loop and scheduler are added in Phase 4.
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
