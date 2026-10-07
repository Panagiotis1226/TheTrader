"""One decision cycle: snapshot -> decision maker -> risk -> broker -> log -> alert.

Order of operations for a run over several accounts:

1. Check every account's stop-losses (so snapshots see post-stop positions).
2. Fetch market data once for all whitelisted pairs. Stale or missing data skips
   the whole cycle.
3. Per account, concurrently: skip if halted or over the daily call or cost limit (no
   LLM call), build the snapshot, ask the decision maker, re-fetch order books (the LLM
   may have taken a while), evaluate risk, place the order if approved.

Every decision is logged with the snapshot hash, model, prompt hash, raw response,
parsed proposal, risk outcome and reason, LLM telemetry, and the order ID (whose fill
lives in ``fills``) — Safety Invariant #9. A failure in one account never stops the
others.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from ai_trader.ai.agent import AgentResult, DecisionMaker
from ai_trader.alerts.base import Alerter, AlertLevel
from ai_trader.brokers.paper import PaperBroker
from ai_trader.config import TradingSettings
from ai_trader.data.market import Clock, MarketData, utcnow
from ai_trader.data.snapshot import (
    PairMarketData,
    StaleDataError,
    build_snapshot,
    fetch_pair_data,
)
from ai_trader.risk.manager import (
    AccountState,
    RiskDecision,
    RiskManager,
    RiskOutcome,
    load_account_state,
    record_halt,
    utc_day_start,
)
from ai_trader.storage.repo import REDUCE_ONLY_HALTS, Repository

log = logging.getLogger(__name__)


MARKET_DATA_SKIP = "market data unusable"


class CycleStatus(StrEnum):
    TRADED = "traded"
    HELD = "held"  # hold proposal (including fallback holds on bad output)
    REJECTED = "rejected"  # by the risk manager
    ORDER_FAILED = "order_failed"  # approved, but the broker rejected the order
    SKIPPED = "skipped"  # halted, over budget, or no usable market data
    ERROR = "error"


@dataclass(frozen=True)
class TradingAccount:
    """A paper account and the strategy (LLM or benchmark) that drives it."""

    broker: PaperBroker
    decider: DecisionMaker
    risk: RiskManager | None = None  # overrides the cycle's default (e.g. benchmarks)

    @property
    def account_id(self) -> str:
        return self.broker.account_id


@dataclass(frozen=True)
class CycleOutcome:
    account_id: str
    status: CycleStatus
    detail: str
    decision_id: int | None = None


class DecisionCycle:
    def __init__(
        self,
        settings: TradingSettings,
        repo: Repository,
        market: MarketData,
        risk: RiskManager,
        alerter: Alerter,
        clock: Clock = utcnow,
        require_intraday: bool = True,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._market = market
        self._risk = risk
        self._alerter = alerter
        self._clock = clock
        self._require_intraday = require_intraday

    async def run(self, accounts: Sequence[TradingAccount]) -> list[CycleOutcome]:
        for account in accounts:
            await self.check_stops(account)

        try:
            data = {
                pair: await fetch_pair_data(self._market, pair) for pair in self._settings.pairs
            }
        except Exception as exc:  # ccxt network errors, stale/malformed data, ...
            log.exception("market data fetch failed")
            await self._alerter.send(
                f"Cycle skipped, market data unusable: {type(exc).__name__}: {exc}",
                AlertLevel.WARNING,
            )
            return [
                CycleOutcome(a.account_id, CycleStatus.SKIPPED, f"{MARKET_DATA_SKIP}: {exc}")
                for a in accounts
            ]

        return list(await asyncio.gather(*(self.run_account(a, data) for a in accounts)))

    async def run_account(
        self, account: TradingAccount, data: dict[str, PairMarketData]
    ) -> CycleOutcome:
        try:
            return await self._run_account(account, data)
        except Exception as exc:
            log.exception("%s: cycle failed", account.account_id)
            await self._alerter.send(
                f"{account.account_id}: cycle error: {type(exc).__name__}: {exc}",
                AlertLevel.WARNING,
            )
            return CycleOutcome(account.account_id, CycleStatus.ERROR, f"{type(exc).__name__}")

    # ------------------------------------------------------------------ internals

    async def check_stops(self, account: TradingAccount) -> None:
        """Trigger due stop-losses for one account and alert on fills. Never raises."""
        try:
            for result in await account.broker.check_stops():
                if result.filled:
                    await self._alerter.send(
                        f"{account.account_id}: STOP-LOSS sold {result.filled_amount} "
                        f"@ {result.avg_price:,.2f} (fee {result.fee:.2f})",
                        AlertLevel.WARNING,
                    )
        except Exception:
            log.exception("%s: stop check failed", account.account_id)

    async def _run_account(
        self, account: TradingAccount, data: dict[str, PairMarketData]
    ) -> CycleOutcome:
        acct = account.account_id
        broker = account.broker
        now = self._clock()

        halts = self._repo.active_halts(acct, now)
        if any(h.kind not in REDUCE_ONLY_HALTS for h in halts):
            kinds = ", ".join(sorted({h.kind.value for h in halts}))
            return CycleOutcome(acct, CycleStatus.SKIPPED, f"halted ({kinds})")
        # Reduce-only halts (daily loss, drawdown) still run: the RiskManager allows sells.

        day_start = utc_day_start(now)
        calls = self._repo.count_decisions_since(acct, day_start)
        if calls >= self._settings.llm.max_daily_calls:
            await self._alerter.send(
                f"{acct}: daily call limit reached ({calls}); holding", AlertLevel.WARNING
            )
            return CycleOutcome(acct, CycleStatus.SKIPPED, "daily call limit reached")

        spent = self._repo.llm_cost_since(acct, day_start)
        budget = self._settings.llm.max_daily_cost_usd
        if spent >= budget:
            await self._alerter.send(
                f"{acct}: daily LLM budget used (${spent:.2f} of ${budget}); holding",
                AlertLevel.WARNING,
            )
            return CycleOutcome(acct, CycleStatus.SKIPPED, "daily LLM budget reached")

        balances = await broker.get_balances()
        try:
            snapshot = build_snapshot(
                data,
                cash=balances.get(self._settings.quote_currency, Decimal(0)),
                positions=await broker.get_positions(),
                recent_decisions=self._repo.recent_decisions(acct, 5),
                now=now,
                max_data_age_seconds=self._settings.max_data_age_seconds,
                quote_currency=self._settings.quote_currency,
                require_intraday=self._require_intraday,
            )
        except StaleDataError as exc:
            await self._alerter.send(f"{acct}: skipped, {exc}", AlertLevel.WARNING)
            return CycleOutcome(acct, CycleStatus.SKIPPED, str(exc))

        result = await account.decider.decide(snapshot)

        # Fresh books: the decision may have taken a while.
        books = {pair: await self._market.fetch_order_book(pair) for pair in data}
        decided_at = self._clock()
        state = await load_account_state(
            broker, self._repo, books, decided_at, self._settings.quote_currency
        )
        self._repo.record_equity(acct, decided_at, state.equity, state.cash)

        decision_id = self._record_decision(acct, decided_at, snapshot.content_hash(), result)
        if result.error:
            await self._alerter.send(
                f"{acct}: unusable decision from {result.model}, holding: {result.error}",
                AlertLevel.WARNING,
            )

        risk = (account.risk or self._risk).evaluate(result.proposal, state, decision_id)
        self._repo.update_decision(
            decision_id, risk_outcome=risk.outcome.value, risk_reason=risk.reason
        )
        if risk.halt is not None:
            await self._halt(acct, risk, state)

        if not risk.tradable:
            if risk.outcome is RiskOutcome.REJECT:
                await self._alerter.send(
                    f"{acct}: risk rejected {result.proposal.action} {result.proposal.pair}: "
                    f"{risk.reason}"
                )
                return CycleOutcome(acct, CycleStatus.REJECTED, risk.reason, decision_id)
            return CycleOutcome(acct, CycleStatus.HELD, result.error or "hold", decision_id)

        assert risk.order is not None
        order = await broker.place_order(risk.order)
        self._repo.update_decision(decision_id, order_id=order.order_id)
        if not order.filled:
            await self._alerter.send(
                f"{acct}: order {risk.order.side.value} {risk.order.pair} not filled: "
                f"{order.reason}",
                AlertLevel.WARNING,
            )
            return CycleOutcome(
                acct, CycleStatus.ORDER_FAILED, order.reason or "rejected", decision_id
            )

        try:  # keep equity history current after a trade (fee and spread just paid)
            after = await broker.get_equity(self._settings.quote_currency)
            self._repo.record_equity(acct, self._clock(), after, broker.cash)
        except Exception:
            log.exception("%s: post-trade equity snapshot failed", acct)
        await self._alerter.send(
            f"{acct}: {risk.order.side.value.upper()} {order.filled_amount} {risk.order.pair} "
            f"@ {order.avg_price:,.2f} = {order.cost:,.2f} + fee {order.fee:.2f} "
            f"({risk.outcome.value}: {risk.reason})"
        )
        return CycleOutcome(acct, CycleStatus.TRADED, risk.reason, decision_id)

    def _record_decision(
        self, acct: str, at: datetime, snapshot_hash: str, result: AgentResult
    ) -> int:
        p = result.proposal
        return self._repo.record_decision(
            acct,
            at,
            snapshot_hash=snapshot_hash,
            model=result.model,
            prompt_hash=result.prompt_hash,
            raw_response=result.raw_response,
            proposal=p.model_dump(mode="json"),
            action=p.action,
            pair=p.pair,
            size_pct=p.size_pct,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            error=result.error,
        )

    async def _halt(self, acct: str, risk: RiskDecision, state: AccountState) -> None:
        assert risk.halt is not None
        reason = (
            f"{risk.halt.value} limit breached: equity {state.equity:.2f}, "
            f"day start {state.day_start_equity:.2f}, peak {state.peak_equity:.2f}"
        )
        record_halt(self._repo, acct, risk.halt, reason, state.now)
        await self._alerter.send(f"{acct}: TRADING HALTED: {reason}", AlertLevel.CRITICAL)
