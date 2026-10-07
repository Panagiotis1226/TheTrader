"""Entrypoint.

Startup order: load + validate config → mode guard → command.
The mode guard runs before anything that could touch an exchange.

Commands:
  (none)  validate config and exit (the scheduler arrives in Phase 4)
  once    run one decision cycle for every configured model's paper account
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Collection
from pathlib import Path

from ai_trader.ai.agent import CompletionFn, DecisionMaker, LLMAgent
from ai_trader.ai.claude_code import ClaudeCodeAgent, Runner, run_subprocess
from ai_trader.ai.prompt import render_system_prompt
from ai_trader.alerts.base import LogAlerter
from ai_trader.brokers.paper import PaperBroker
from ai_trader.config import (
    DEFAULT_ENV_FILE,
    AppConfig,
    ConfigError,
    EnvSettings,
    Mode,
    load_config,
)
from ai_trader.cycle import CycleOutcome, CycleStatus, DecisionCycle, TradingAccount
from ai_trader.data.market import Clock, MarketData, utcnow
from ai_trader.risk.manager import RiskManager
from ai_trader.storage.repo import Repository

log = logging.getLogger("ai_trader")

LIVE_CONFIRMATION_VALUE = "yes"

EXIT_OK = 0
EXIT_LIVE_NOT_CONFIRMED = 1
EXIT_CONFIG_ERROR = 2
EXIT_UNSUPPORTED = 3
EXIT_CYCLE_FAILED = 4


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
    parser = argparse.ArgumentParser(
        prog="ai-trader", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="settings YAML (default: SETTINGS_PATH or config/settings.yaml)",
    )
    sub = parser.add_subparsers(dest="command")
    once = sub.add_parser("once", help="run one decision cycle for each model's paper account")
    once.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="only this model (repeatable); default: all configured models",
    )
    return parser.parse_args(argv)


def build_paper_accounts(
    config: AppConfig,
    repo: Repository,
    market: MarketData,
    only: Collection[str] | None = None,
    completion_fn: CompletionFn | None = None,
    clock: Clock = utcnow,
    runner: Runner = run_subprocess,
) -> list[TradingAccount]:
    """One paper account per configured LLM (``paper-<name>``). Placeholders are skipped."""
    trading = config.trading
    system_prompt = render_system_prompt(trading)
    accounts = []
    for model in trading.models:
        if only and model.name not in only:
            continue
        if model.is_placeholder:
            log.warning("Skipping model %r: placeholder ID %r", model.name, model.model)
            continue
        broker = PaperBroker(
            f"paper-{model.name}",
            repo=repo,
            market=market,
            allowed_pairs=trading.pairs,
            starting_cash=trading.paper.starting_cash_cad,
            taker_fee_pct=trading.paper.taker_fee_pct,
            quote_currency=trading.quote_currency,
            clock=clock,
        )
        agent: DecisionMaker
        if model.provider == "claude_code":
            agent = ClaudeCodeAgent(
                model, trading.llm, config.env, system_prompt, trading.pairs[0], runner
            )
        else:
            agent = LLMAgent(
                model, trading.llm, config.env, system_prompt, trading.pairs[0], completion_fn
            )
        accounts.append(TradingAccount(broker=broker, decider=agent))
    return accounts


async def run_once(
    config: AppConfig,
    mode: Mode,
    only: Collection[str] | None = None,
    market: MarketData | None = None,
    completion_fn: CompletionFn | None = None,
    clock: Clock = utcnow,
    runner: Runner = run_subprocess,
) -> list[CycleOutcome] | None:
    """Run a single cycle. Returns None if refused (live mode / nothing to run)."""
    if mode is not Mode.PAPER:
        log.critical("Live trading is not implemented until Phase 6; refusing to run")
        return None
    trading = config.trading
    repo = Repository.from_url(config.env.database_url)
    market = market or MarketData()
    try:
        accounts = build_paper_accounts(config, repo, market, only, completion_fn, clock, runner)
        if not accounts:
            log.error("No runnable models (check names and model IDs in settings.yaml)")
            return None
        risk = RiskManager(trading.risk, trading.pairs, trading.paper.taker_fee_pct)
        cycle = DecisionCycle(trading, repo, market, risk, LogAlerter(), clock)
        return await cycle.run(accounts)
    finally:
        await market.close()


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
                model.model,
            )

    if args.command == "once":
        outcomes = asyncio.run(run_once(config, mode, args.models))
        if outcomes is None:
            return EXIT_UNSUPPORTED
        for o in outcomes:
            print(f"{o.account_id:24} {o.status.value:12} {o.detail}")
        if all(o.status is CycleStatus.ERROR for o in outcomes):
            return EXIT_CYCLE_FAILED
        return EXIT_OK

    # The scheduler is added in Phase 4.
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
