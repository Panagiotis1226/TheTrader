"""Entrypoint.

Startup order: load + validate config → mode guard → command.
The mode guard runs before anything that could touch an exchange.

Commands:
  (none)    validate config and exit
  run       run unattended: scheduler, Telegram alerts and commands, heartbeat
  once      run one decision cycle for every paper account (models + benchmarks)
  backtest  replay daily Kraken candles through the decision path (benchmarks; --with-llm)
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import signal
import sys
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

from ai_trader.ai.agent import CompletionFn, DecisionMaker, LLMAgent
from ai_trader.ai.claude_code import ClaudeCodeAgent, Runner, run_subprocess
from ai_trader.ai.prompt import render_system_prompt
from ai_trader.alerts.base import Alerter, LogAlerter
from ai_trader.alerts.heartbeat import Heartbeat
from ai_trader.alerts.telegram_bot import CommandRouter, TelegramAlerter, TelegramBot
from ai_trader.backtest.data import (
    DEFAULT_CANDLES_DIR,
    candles_path,
    load_market_infos,
    read_candles_csv,
    refresh_candles,
    save_market_infos,
)
from ai_trader.backtest.engine import BacktestEngine, BacktestReport, format_reports
from ai_trader.brokers.paper import PaperBroker
from ai_trader.config import (
    DEFAULT_ENV_FILE,
    AppConfig,
    ConfigError,
    EnvSettings,
    Mode,
    ModelSettings,
    load_config,
)
from ai_trader.cycle import CycleOutcome, CycleStatus, DecisionCycle, TradingAccount
from ai_trader.data.market import Clock, MarketData, utcnow
from ai_trader.risk.manager import RiskManager
from ai_trader.service import TradingService
from ai_trader.storage.repo import Repository
from ai_trader.strategies import build_benchmark

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
    sub.add_parser("run", help="run unattended (scheduler, Telegram, heartbeat)")
    once = sub.add_parser("once", help="run one decision cycle for each model's paper account")
    once.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help="only this account name (repeatable); default: all models and benchmarks",
    )
    bt = sub.add_parser("backtest", help="replay daily candles through the decision path")
    bt.add_argument("--start", type=_utc_date, help="first decision day, YYYY-MM-DD")
    bt.add_argument("--end", type=_utc_date, help="last decision day, YYYY-MM-DD")
    bt.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        metavar="NAME",
        help="benchmark to run (repeatable); default: all configured benchmarks",
    )
    bt.add_argument(
        "--with-llm",
        action="store_true",
        help="also run the configured models over the last backtest.llm_max_decisions days "
        "(plumbing check only: LLM backtests suffer from look-ahead bias)",
    )
    bt.add_argument(
        "--llm-days",
        type=int,
        default=None,
        metavar="N",
        help="with --with-llm: run the models over the last N days (max llm_max_decisions)",
    )
    bt.add_argument("--refresh-data", action="store_true", help="fetch the latest candles")
    bt.add_argument("--candles-dir", type=Path, default=DEFAULT_CANDLES_DIR)
    bt.add_argument("--out", type=Path, default=Path("data/backtests"))
    return parser.parse_args(argv)


def _utc_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)


def build_agent(
    model: ModelSettings,
    config: AppConfig,
    system_prompt: str,
    completion_fn: CompletionFn | None = None,
    runner: Runner = run_subprocess,
) -> DecisionMaker:
    trading = config.trading
    if model.provider == "claude_code":
        return ClaudeCodeAgent(
            model, trading.llm, config.env, system_prompt, trading.pairs[0], runner
        )
    return LLMAgent(model, trading.llm, config.env, system_prompt, trading.pairs[0], completion_fn)


def build_paper_accounts(
    config: AppConfig,
    repo: Repository,
    market: MarketData,
    only: Collection[str] | None = None,
    completion_fn: CompletionFn | None = None,
    clock: Clock = utcnow,
    runner: Runner = run_subprocess,
) -> list[TradingAccount]:
    """One paper account per configured model and per benchmark (``paper-<name>``).

    Placeholder model IDs are skipped. Benchmarks use the same RiskManager rules but no
    forced stop-loss (see strategies/).
    """
    trading = config.trading
    system_prompt = render_system_prompt(trading)

    def broker(name: str) -> PaperBroker:
        return PaperBroker(
            f"paper-{name}",
            repo=repo,
            market=market,
            allowed_pairs=trading.pairs,
            starting_cash=trading.paper.starting_cash_cad,
            taker_fee_pct=trading.paper.taker_fee_pct,
            quote_currency=trading.quote_currency,
            clock=clock,
        )

    accounts = []
    for model in trading.models:
        if only and model.name not in only:
            continue
        if model.is_placeholder:
            log.warning("Skipping model %r: placeholder ID %r", model.name, model.model)
            continue
        agent = build_agent(model, config, system_prompt, completion_fn, runner)
        accounts.append(TradingAccount(broker=broker(model.name), decider=agent))

    benchmark_risk = RiskManager(
        trading.risk, trading.pairs, trading.paper.taker_fee_pct, stop_loss_required=False
    )
    for name in trading.benchmarks:
        if only and name not in only:
            continue
        accounts.append(
            TradingAccount(
                broker=broker(name), decider=build_benchmark(name, trading), risk=benchmark_risk
            )
        )
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


async def run_service(config: AppConfig, mode: Mode) -> int:
    """Run until SIGTERM/SIGINT. Docker restarts the container if it dies."""
    if mode is not Mode.PAPER:
        log.critical("Live trading is not implemented until Phase 6; refusing to run")
        return EXIT_UNSUPPORTED
    trading, env = config.trading, config.env
    repo = Repository.from_url(env.database_url)
    market = MarketData()
    accounts = build_paper_accounts(config, repo, market)
    if not accounts:
        log.error("No accounts to run")
        await market.close()
        return EXIT_CONFIG_ERROR

    bot: TelegramBot | None = None
    alerter: Alerter = LogAlerter()
    service: TradingService | None = None
    if env.telegram_bot_token and env.telegram_chat_id:
        chat_id = int(env.telegram_chat_id)

        # The bot is built before the service (the service alerts through it), so
        # commands look the service up when they run.
        async def dispatch(name: str) -> str:
            assert service is not None
            return await service.commands()[name]()

        router = CommandRouter(
            chat_id, {n: functools.partial(dispatch, n) for n in TradingService.COMMANDS}
        )
        bot = TelegramBot(env.telegram_bot_token.get_secret_value(), router)
        alerter = TelegramAlerter(bot.bot, chat_id)
    else:
        log.warning("Telegram not configured: alerts go to the log only and /stop is unavailable")
    heartbeat = Heartbeat(env.healthcheck_url.get_secret_value() if env.healthcheck_url else None)
    if not heartbeat.enabled:
        log.warning("HEALTHCHECK_URL not set: no external heartbeat")

    risk = RiskManager(trading.risk, trading.pairs, trading.paper.taker_fee_pct)
    cycle = DecisionCycle(trading, repo, market, risk, alerter)
    service = TradingService(trading, repo, cycle, accounts, alerter, heartbeat)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        if bot is not None:
            await bot.start()
        service.schedule(run_cycle_now=True)
        service.scheduler.start()
        halts = repo.active_halts(None, utcnow())
        await alerter.send(
            f"ai-trader started (paper): {', '.join(a.account_id for a in accounts)}. "
            f"Cycle every {trading.decision_interval_minutes} min"
            + (f". Active halts: {', '.join(h.kind.value for h in halts)}" if halts else "")
        )
        await stop_event.wait()
        log.info("Shutting down")
    finally:
        if service.scheduler.running:
            service.scheduler.shutdown(wait=False)
        if bot is not None:
            await bot.stop()
        await market.close()
    return EXIT_OK


async def load_backtest_data(config: AppConfig, candles_dir: Path, refresh: bool):
    """Daily candles + market metadata, from the local cache (fetched when missing)."""
    pairs = config.trading.pairs
    need_fetch = refresh or not (candles_dir / "markets.json").exists()
    need_fetch = need_fetch or any(not candles_path(candles_dir, p).exists() for p in pairs)
    if need_fetch:
        log.info("Fetching daily candles and market info from Kraken into %s", candles_dir)
        async with MarketData() as market:
            save_market_infos(candles_dir, {p: await market.market_info(p) for p in pairs})
            for pair in pairs:
                await refresh_candles(market, pair, candles_dir, utcnow())
    candles = {p: read_candles_csv(candles_path(candles_dir, p)) for p in pairs}
    return candles, load_market_infos(candles_dir)


async def run_backtest(
    config: AppConfig,
    args: argparse.Namespace,
    completion_fn: CompletionFn | None = None,
    runner: Runner = run_subprocess,
) -> list[BacktestReport]:
    trading = config.trading
    candles, infos = await load_backtest_data(config, args.candles_dir, args.refresh_data)
    for pair, series in candles.items():
        log.info(
            "%s: %d daily candles, %s -> %s",
            pair,
            len(series),
            series[0].opened_at.date(),
            series[-1].opened_at.date(),
        )
    engine = BacktestEngine(trading, candles, infos, start=args.start, end=args.end)

    results = []
    for name in args.strategies or trading.benchmarks:
        log.info("Backtesting benchmark %s over %d days", name, len(engine.steps))
        results.append(await engine.run(build_benchmark(name, trading), stop_loss_required=False))
    if args.with_llm:
        cap = trading.backtest.llm_max_decisions
        llm_days = min(args.llm_days or cap, cap)
        log.warning(
            "LLM backtest = plumbing check only: the model may have seen these prices in "
            "training (look-ahead bias). Running %d decisions per model.",
            llm_days,
        )
        system_prompt = render_system_prompt(trading)
        for model in trading.models:
            if model.is_placeholder:
                continue
            agent = build_agent(model, config, system_prompt, completion_fn, runner)
            results.append(
                await engine.run(
                    agent,
                    stop_loss_required=True,
                    max_decisions=llm_days,
                )
            )

    reports = [r.report for r in results]
    print(format_reports(reports))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"backtest-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(
        json.dumps(
            [
                {
                    **r.report.as_dict(),
                    "equity_curve": [[t.isoformat(), str(e)] for t, e in r.equity_curve],
                }
                for r in results
            ],
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {path}")
    return reports


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

    if args.command == "run":
        return asyncio.run(run_service(config, mode))

    if args.command == "backtest":
        reports = asyncio.run(run_backtest(config, args))
        return EXIT_OK if reports else EXIT_CYCLE_FAILED

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
