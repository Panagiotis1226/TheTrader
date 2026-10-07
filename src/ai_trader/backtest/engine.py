"""Backtest engine: replays daily candles through the *live* decision path.

Each simulated day, at the daily close (00:00 UTC):

1. Stop-losses are checked against that day's candle: a gap below the trigger fills at
   the open, otherwise touching the low fills at the trigger (both minus slippage).
2. ``DecisionCycle.run`` runs unchanged: snapshot -> decider -> RiskManager ->
   PaperBroker -> log. Only the market (``ReplayMarket``) and the clock are simulated.
3. Equity (positions at the bid) is recorded.

Fills use the candle price +/- ``slippage_pct`` plus the configured taker fee.

Caveat: an LLM backtest is not trustworthy (the model may have seen these prices in
training). Use it to check plumbing only; the real test is forward paper trading.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.ai.agent import DecisionMaker
from ai_trader.alerts.base import AlertLevel
from ai_trader.backtest import metrics
from ai_trader.backtest.replay import ReplayClock, ReplayMarket
from ai_trader.brokers.base import Fill, OrderType
from ai_trader.brokers.paper import PaperBroker
from ai_trader.config import TradingSettings
from ai_trader.cycle import CycleOutcome, CycleStatus, DecisionCycle, TradingAccount
from ai_trader.data.market import Candle, MarketInfo
from ai_trader.risk.manager import RiskManager
from ai_trader.storage.repo import Repository

log = logging.getLogger(__name__)
DAY = timedelta(days=1)


class CollectingAlerter:
    def __init__(self, clock: ReplayClock) -> None:
        self._clock = clock
        self.alerts: list[tuple[AlertLevel, datetime, str]] = []

    async def send(self, text: str, level: AlertLevel = AlertLevel.INFO) -> None:
        self.alerts.append((level, self._clock(), text))


@dataclass(frozen=True)
class BacktestReport:
    strategy: str
    start: datetime
    end: datetime
    days: int
    decisions: int
    start_equity: Decimal
    end_equity: Decimal
    total_return_pct: float
    cagr_pct: float | None
    max_drawdown_pct: float
    sharpe: float | None
    trades: int
    stop_loss_fills: int
    fees_paid: Decimal
    sells: int
    win_rate_pct: float | None
    outcomes: dict[str, int]
    halts: list[str]
    market_return_pct: dict[str, float]  # 100% buy-and-hold of each pair, no fees

    def as_dict(self) -> dict:
        d = dict(vars(self))
        for k in ("start", "end"):
            d[k] = d[k].isoformat()
        for k in ("start_equity", "end_equity", "fees_paid"):
            d[k] = str(d[k])
        return d


@dataclass
class BacktestResult:
    report: BacktestReport
    equity_curve: list[tuple[datetime, Decimal]]
    fills: list[Fill]
    outcomes: list[CycleOutcome] = field(default_factory=list)


class BacktestEngine:
    def __init__(
        self,
        settings: TradingSettings,
        candles: Mapping[str, Sequence[Candle]],
        infos: Mapping[str, MarketInfo],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        missing = [p for p in settings.pairs if not candles.get(p)]
        if missing:
            raise ValueError(f"no candle history for {missing}")
        self._settings = settings
        self._candles = {p: list(candles[p]) for p in settings.pairs}
        self._infos = infos
        self._steps = self._decision_times(start, end)
        if not self._steps:
            raise ValueError("no decision days in the requested window (check warm-up/range)")

    @property
    def steps(self) -> list[datetime]:
        return list(self._steps)

    def _decision_times(self, start: datetime | None, end: datetime | None) -> list[datetime]:
        """Daily closes where every pair has a candle and the warm-up has passed."""
        warmup = self._settings.backtest.warmup_days
        per_pair = []
        for series in self._candles.values():
            closes = [c.opened_at + DAY for c in series]
            per_pair.append(set(closes[warmup:]))
        times = sorted(set.intersection(*per_pair))
        return [t for t in times if (start is None or t >= start) and (end is None or t <= end)]

    def _day_candle(self, pair: str, close_time: datetime) -> Candle | None:
        opened = close_time - DAY
        for c in reversed(self._candles[pair]):
            if c.opened_at == opened:
                return c
            if c.opened_at < opened:
                return None
        return None

    async def run(
        self,
        decider: DecisionMaker,
        *,
        stop_loss_required: bool,
        max_decisions: int | None = None,
    ) -> BacktestResult:
        s = self._settings
        steps = self._steps[-max_decisions:] if max_decisions else self._steps
        clock = ReplayClock(steps[0])
        market = ReplayMarket(self._candles, self._infos, s.backtest.slippage_pct, clock)
        repo = Repository.from_url("sqlite://")
        broker = PaperBroker(
            f"backtest-{decider.name.replace('/', '-')}",
            repo=repo,
            market=market,
            allowed_pairs=s.pairs,
            starting_cash=s.paper.starting_cash_cad,
            taker_fee_pct=s.paper.taker_fee_pct,
            quote_currency=s.quote_currency,
            clock=clock,
        )
        risk = RiskManager(
            s.risk, s.pairs, s.paper.taker_fee_pct, stop_loss_required=stop_loss_required
        )
        alerter = CollectingAlerter(clock)
        cycle = DecisionCycle(s, repo, market, risk, alerter, clock, require_intraday=False)
        account = TradingAccount(broker=broker, decider=decider)

        curve: list[tuple[datetime, Decimal]] = []
        outcomes: list[CycleOutcome] = []
        for t in steps:
            clock.now = t
            await self._simulate_stops(broker, market, t)
            [outcome] = await cycle.run([account])
            if outcome.status is CycleStatus.ERROR:
                log.warning("%s @ %s: cycle error: %s", decider.name, t, outcome.detail)
            outcomes.append(outcome)
            curve.append((t, await broker.get_equity(s.quote_currency)))

        fills = repo.list_fills(broker.account_id)
        halts = [
            f"{at:%Y-%m-%d} {text.split(': ', 1)[-1]}"
            for level, at, text in alerter.alerts
            if level is AlertLevel.CRITICAL
        ]
        report = self._report(decider.name, steps, curve, fills, outcomes, halts)
        return BacktestResult(report, curve, fills, outcomes)

    async def _simulate_stops(
        self, broker: PaperBroker, market: ReplayMarket, close_time: datetime
    ) -> None:
        """Fill stops that the day's candle would have hit, at a realistic price."""
        stops = [o for o in await broker.get_open_orders() if o.type is OrderType.STOP_LOSS]
        prices: dict[str, Decimal] = {}
        for stop in stops:
            candle = self._day_candle(stop.pair, close_time)
            if candle is None or stop.trigger_price is None:
                continue
            if candle.open <= stop.trigger_price:
                prices[stop.pair] = candle.open  # gapped through the stop
            elif candle.low <= stop.trigger_price:
                prices[stop.pair] = stop.trigger_price
        if not prices:
            return
        # Pairs whose stop was not hit are parked at the day's high so they can't trigger.
        for stop in stops:
            if stop.pair not in prices:
                candle = self._day_candle(stop.pair, close_time)
                if candle is not None:
                    prices[stop.pair] = candle.high
        market.set_prices(prices)
        try:
            await broker.check_stops()
        finally:
            market.clear_prices()

    def _report(
        self,
        name: str,
        steps: list[datetime],
        curve: list[tuple[datetime, Decimal]],
        fills: list[Fill],
        outcomes: list[CycleOutcome],
        halts: list[str],
    ) -> BacktestReport:
        equity = [self._settings.paper.starting_cash_cad] + [e for _, e in curve]
        start = steps[0] - DAY  # equity starts at starting cash the day before
        days = (steps[-1] - start).days
        sells, win_rate = metrics.win_rate_pct(fills)
        counts: dict[str, int] = {}
        for o in outcomes:
            counts[o.status.value] = counts.get(o.status.value, 0) + 1
        market_return = {}
        for pair in self._settings.pairs:
            first = self._day_candle(pair, steps[0])
            last = self._day_candle(pair, steps[-1])
            if first and last:
                market_return[pair] = round((float(last.close) / float(first.open) - 1) * 100, 2)
        return BacktestReport(
            strategy=name,
            start=start,
            end=steps[-1],
            days=days,
            decisions=len(outcomes),
            start_equity=equity[0],
            end_equity=equity[-1].quantize(Decimal("0.01")),
            total_return_pct=round(metrics.total_return_pct(equity), 2),
            cagr_pct=_r(metrics.cagr_pct(equity, days)),
            max_drawdown_pct=round(metrics.max_drawdown_pct(equity), 2),
            sharpe=_r(metrics.sharpe(equity)),
            trades=len(fills),
            stop_loss_fills=sum(1 for f in fills if f.order_type is OrderType.STOP_LOSS),
            fees_paid=sum((f.fee for f in fills), Decimal(0)).quantize(Decimal("0.01")),
            sells=sells,
            win_rate_pct=_r(win_rate),
            outcomes=counts,
            halts=halts,
            market_return_pct=market_return,
        )


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def format_reports(reports: Sequence[BacktestReport]) -> str:
    header = (
        f"{'strategy':28} {'days':>5} {'return%':>8} {'CAGR%':>7} {'maxDD%':>7} {'Sharpe':>7} "
        f"{'trades':>6} {'stops':>5} {'fees':>9} {'win%':>6} {'end equity':>12}"
    )
    lines = [header, "-" * len(header)]

    def f(v: float | None, width: int) -> str:
        return f"{v:>{width}.2f}" if v is not None else f"{'-':>{width}}"

    for r in reports:
        lines.append(
            f"{r.strategy:28} {r.days:>5} {f(r.total_return_pct, 8)} {f(r.cagr_pct, 7)} "
            f"{f(r.max_drawdown_pct, 7)} {f(r.sharpe, 7)} {r.trades:>6} "
            f"{r.stop_loss_fills:>5} {r.fees_paid:>9} {f(r.win_rate_pct, 6)} {r.end_equity:>12}"
        )
    if reports:
        r = reports[0]
        lines.append("")
        lines.append(
            f"Window {r.start:%Y-%m-%d} -> {r.end:%Y-%m-%d} ({r.days} days). "
            "Market (100% hold, no fees): "
            + ", ".join(f"{p} {v:+.2f}%" for p, v in r.market_return_pct.items())
        )
        lines.append(
            "Halts: daily-loss halts lift at the next UTC midnight; a drawdown halt needs a "
            "manual /resume, which never happens in a backtest (sells only from then on)."
        )
        for rep in reports:
            for halt in rep.halts:
                lines.append(f"  {rep.strategy}: {halt}")
    return "\n".join(lines)
