"""Deterministic risk layer (Safety Invariants #2, #3, #5).

``RiskManager.evaluate(proposal, state)`` is a pure function of the proposal, the
account state, and the config. It returns approve / resize / reject, plus a halt to
record if the account has just breached the daily-loss or drawdown limit.

Interpretation choices:

* Buy ``size_pct`` is a % of total equity (same unit as the risk caps), then capped by
  the per-trade, per-pair, total-exposure, and cash limits. Cash is reserved for the fee.
* Sell ``size_pct`` is a % of the current position. Sells never add exposure, so they
  are not size-capped, but they still obey halts, confidence, and trade-frequency rules.
* The LLM may tighten the stop-loss but never loosen it beyond ``default_stop_loss_pct``.
* The trading day is the UTC day.
* Halts: manual (/stop) and error halts block every trade. The automatic daily-loss and
  drawdown halts are reduce-only: buys are rejected, sells still allowed, so a halt never
  traps the account in a falling position.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from enum import StrEnum

from ai_trader.ai.schema import TradeProposal
from ai_trader.brokers.base import Broker, OrderRequest, Side
from ai_trader.config import RiskSettings
from ai_trader.data.market import OrderBook
from ai_trader.storage.repo import REDUCE_ONLY_HALTS, HaltKind, Repository

HUNDRED = Decimal(100)


class RiskOutcome(StrEnum):
    APPROVE = "approve"
    RESIZE = "resize"
    REJECT = "reject"


@dataclass(frozen=True)
class AccountState:
    account_id: str
    now: datetime
    equity: Decimal
    cash: Decimal
    position_amounts: Mapping[str, Decimal]  # base quantity per pair
    position_values: Mapping[str, Decimal]  # quote value per pair (marked at best bid)
    buy_prices: Mapping[str, Decimal]  # current best ask per pair (buys need a market)
    day_start_equity: Decimal
    peak_equity: Decimal
    trades_today: int
    last_trade_at: datetime | None
    active_halts: tuple[HaltKind, ...] = field(default=())

    @property
    def total_exposure(self) -> Decimal:
        return sum(self.position_values.values(), Decimal(0))


@dataclass(frozen=True)
class RiskDecision:
    outcome: RiskOutcome
    reason: str
    order: OrderRequest | None = None
    requested_notional: Decimal | None = None  # buys: quote value asked for
    approved_notional: Decimal | None = None  # buys: quote value approved
    halt: HaltKind | None = None  # newly breached limit the caller must record

    @property
    def tradable(self) -> bool:
        return self.order is not None and self.outcome is not RiskOutcome.REJECT


class RiskManager:
    def __init__(
        self,
        settings: RiskSettings,
        allowed_pairs: Collection[str],
        taker_fee_pct: Decimal,
        *,
        stop_loss_required: bool = True,
    ) -> None:
        """``stop_loss_required=False`` is only for rule-based benchmarks (a buy-and-hold
        with a forced 5% stop would not be buy-and-hold). LLM accounts always get stops."""
        self._s = settings
        self._pairs = frozenset(allowed_pairs)
        self._fee_rate = taker_fee_pct / HUNDRED
        self._stop_loss_required = stop_loss_required

    def detect_halt(self, state: AccountState) -> HaltKind | None:
        """Return a newly breached limit (drawdown takes precedence), else None."""
        if state.peak_equity > 0:
            drawdown_pct = (state.peak_equity - state.equity) / state.peak_equity * HUNDRED
            if drawdown_pct >= self._s.max_drawdown_halt_pct:
                return HaltKind.DRAWDOWN
        if state.day_start_equity > 0:
            day_loss_pct = (
                (state.day_start_equity - state.equity) / state.day_start_equity * HUNDRED
            )
            if day_loss_pct >= self._s.daily_loss_limit_pct:
                return HaltKind.DAILY_LOSS
        return None

    def evaluate(
        self, proposal: TradeProposal, state: AccountState, decision_id: int | None = None
    ) -> RiskDecision:
        new_halt = self.detect_halt(state)
        if new_halt in state.active_halts:
            new_halt = None  # already recorded

        def reject(reason: str) -> RiskDecision:
            return RiskDecision(RiskOutcome.REJECT, reason, halt=new_halt)

        if proposal.action == "hold":
            return RiskDecision(RiskOutcome.APPROVE, "hold: no order", halt=new_halt)

        halts = [*state.active_halts, *([new_halt] if new_halt else [])]
        if halts:
            names = ", ".join(h.value for h in halts)
            if not all(h in REDUCE_ONLY_HALTS for h in halts):
                return reject(f"trading halted ({names})")
            if proposal.action == "buy":
                return reject(f"trading halted ({names}): sells only")
        if proposal.pair not in self._pairs:
            return reject(f"{proposal.pair} is not a whitelisted pair")
        if proposal.confidence < self._s.min_confidence:
            return reject(
                f"confidence {proposal.confidence} below minimum {self._s.min_confidence}"
            )
        if state.trades_today >= self._s.max_trades_per_day:
            return reject(f"max trades per day reached ({self._s.max_trades_per_day})")
        if state.last_trade_at is not None:
            wait = timedelta(minutes=self._s.min_minutes_between_trades)
            if state.now - state.last_trade_at < wait:
                return reject(
                    f"last trade was less than {self._s.min_minutes_between_trades} minutes ago"
                )
        if proposal.size_pct <= 0:
            return reject("size_pct must be greater than 0")
        if state.equity <= 0:
            return reject("account equity is not positive")

        if proposal.action == "sell":
            return self._evaluate_sell(proposal, state, decision_id, new_halt)
        return self._evaluate_buy(proposal, state, decision_id, new_halt)

    def _evaluate_sell(
        self,
        proposal: TradeProposal,
        state: AccountState,
        decision_id: int | None,
        new_halt: HaltKind | None,
    ) -> RiskDecision:
        held = state.position_amounts.get(proposal.pair, Decimal(0))
        if held <= 0:
            return RiskDecision(
                RiskOutcome.REJECT, f"no {proposal.pair} position to sell", halt=new_halt
            )
        amount = held * proposal.size_pct / HUNDRED
        order = OrderRequest(
            pair=proposal.pair, side=Side.SELL, amount=amount, decision_id=decision_id
        )
        return RiskDecision(
            RiskOutcome.APPROVE, f"sell {proposal.size_pct}% of position", order, halt=new_halt
        )

    def _evaluate_buy(
        self,
        proposal: TradeProposal,
        state: AccountState,
        decision_id: int | None,
        new_halt: HaltKind | None,
    ) -> RiskDecision:
        price = state.buy_prices.get(proposal.pair)
        if price is None or price <= 0:
            return RiskDecision(
                RiskOutcome.REJECT, f"no current price for {proposal.pair}", halt=new_halt
            )

        s = self._s
        equity = state.equity
        requested = equity * proposal.size_pct / HUNDRED
        pair_value = state.position_values.get(proposal.pair, Decimal(0))
        caps = {
            f"max_trade_pct_of_equity ({s.max_trade_pct_of_equity}%)": (
                equity * s.max_trade_pct_of_equity / HUNDRED
            ),
            f"max_position_pct_per_pair ({s.max_position_pct_per_pair}%)": (
                equity * s.max_position_pct_per_pair / HUNDRED - pair_value
            ),
            f"max_total_exposure_pct ({s.max_total_exposure_pct}%)": (
                equity * s.max_total_exposure_pct / HUNDRED - state.total_exposure
            ),
            "available cash (after fees)": state.cash / (Decimal(1) + self._fee_rate),
        }
        binding_name, binding_cap = min(caps.items(), key=lambda kv: kv[1])
        approved = min(requested, binding_cap)

        if approved <= 0:
            return RiskDecision(
                RiskOutcome.REJECT,
                f"no room under {binding_name}",
                requested_notional=requested,
                approved_notional=Decimal(0),
                halt=new_halt,
            )

        stop_pct: Decimal | None = s.default_stop_loss_pct if self._stop_loss_required else None
        if proposal.stop_loss_pct is not None:
            stop_pct = min(proposal.stop_loss_pct, s.default_stop_loss_pct)

        order = OrderRequest(
            pair=proposal.pair,
            side=Side.BUY,
            quote_amount=approved,  # spend at most this; slippage reduces the amount bought
            stop_loss_pct=stop_pct,
            decision_id=decision_id,
        )
        if approved < requested:
            outcome = RiskOutcome.RESIZE
            reason = f"resized {requested:.2f} -> {approved:.2f} by {binding_name}"
        else:
            outcome = RiskOutcome.APPROVE
            reason = f"buy {approved:.2f} within limits"
        return RiskDecision(
            outcome,
            reason,
            order,
            requested_notional=requested,
            approved_notional=approved,
            halt=new_halt,
        )


# --------------------------------------------------------------------------- helpers


def utc_day_start(now: datetime) -> datetime:
    return datetime.combine(now.astimezone(UTC).date(), time(0), tzinfo=UTC)


def halt_until(kind: HaltKind, now: datetime) -> datetime | None:
    """Daily-loss halts expire at the next UTC midnight; all others need /resume."""
    if kind is HaltKind.DAILY_LOSS:
        return utc_day_start(now) + timedelta(days=1)
    return None


def record_halt(
    repo: Repository, account_id: str, kind: HaltKind, reason: str, now: datetime
) -> int:
    return repo.add_halt(kind, reason, now, account_id=account_id, until=halt_until(kind, now))


async def load_account_state(
    broker: Broker,
    repo: Repository,
    books: Mapping[str, OrderBook],
    now: datetime,
    quote: str = "CAD",
) -> AccountState:
    """Assemble ``AccountState`` from the broker, the DB, and freshly fetched books.

    ``books`` must cover every pair the account holds or might buy.
    """
    balances = await broker.get_balances()
    cash = balances.get(quote, Decimal(0))
    positions = await broker.get_positions()
    amounts = {p.pair: p.amount for p in positions}
    values = {p.pair: p.amount * books[p.pair].best_bid for p in positions}
    equity = cash + sum(values.values(), Decimal(0))

    day_start = utc_day_start(now)
    day_start_equity = (
        repo.equity_at_or_before(broker.account_id, day_start)
        or repo.first_equity_since(broker.account_id, day_start)
        or equity
    )
    since_resume = repo.last_resume(broker.account_id, HaltKind.DRAWDOWN)
    peak = max(
        repo.peak_equity_since(broker.account_id, since_resume) or equity,
        equity,
    )
    trades_today, last_trade_at = repo.trade_stats(broker.account_id, day_start)
    halts = tuple(dict.fromkeys(h.kind for h in repo.active_halts(broker.account_id, now)))

    return AccountState(
        account_id=broker.account_id,
        now=now,
        equity=equity,
        cash=cash,
        position_amounts=amounts,
        position_values=values,
        buy_prices={pair: book.best_ask for pair, book in books.items()},
        day_start_equity=day_start_equity,
        peak_equity=peak,
        trades_today=trades_today,
        last_trade_at=last_trade_at,
        active_halts=halts,
    )
