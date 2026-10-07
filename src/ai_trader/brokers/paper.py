"""PaperBroker: simulated fills against the live Kraken order book.

* Market orders walk the book (asks for buys, bids for sells) to get a realistic
  average price including slippage. If the fetched book is too thin, the order is
  rejected rather than partially filled.
* Every fill pays the taker fee, charged in the quote currency (as Kraken does by default).
* Amounts are rounded down to Kraken's step size; min amount and min cost are enforced.
* Stop-losses are resting orders inside the broker, checked by ``check_stops()`` and
  executed as market sells against the bids.
* State is persisted after every change and rebuilt from the fill log on startup, so
  balances can never drift from the recorded fills.

Each instance is one independent account (``paper-claude``, ``paper-buy_and_hold``...).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_UP, Decimal

from ai_trader.brokers.base import (
    Fill,
    Order,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from ai_trader.data.market import BookLevel, Clock, MarketDataError, OrderBookSource, utcnow
from ai_trader.storage.repo import Repository

log = logging.getLogger(__name__)

FEE_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True)
class BookWalk:
    filled: Decimal
    cost: Decimal

    @property
    def avg_price(self) -> Decimal:
        return self.cost / self.filled


def walk_book(levels: Sequence[BookLevel], amount: Decimal) -> BookWalk:
    """Consume ``amount`` from ``levels`` (best first). ``filled`` < amount if too thin."""
    remaining = amount
    cost = Decimal(0)
    for level in levels:
        if remaining <= 0:
            break
        take = min(remaining, level.amount)
        cost += take * level.price
        remaining -= take
    return BookWalk(filled=amount - remaining, cost=cost)


@dataclass
class _PositionState:
    amount: Decimal = Decimal(0)
    cost_basis: Decimal = Decimal(0)  # sum of buy cost for the current amount, excl. fees

    @property
    def avg_entry(self) -> Decimal:
        return self.cost_basis / self.amount if self.amount > 0 else Decimal(0)

    def apply(self, side: Side, amount: Decimal, cost: Decimal) -> None:
        if side is Side.BUY:
            self.amount += amount
            self.cost_basis += cost
        else:
            # Selling leaves the average entry unchanged.
            avg = self.avg_entry
            self.amount -= amount
            self.cost_basis = avg * self.amount if self.amount > 0 else Decimal(0)


class PaperBroker:
    def __init__(
        self,
        account_id: str,
        *,
        repo: Repository,
        market: OrderBookSource,
        allowed_pairs: Collection[str],
        starting_cash: Decimal,
        taker_fee_pct: Decimal,
        quote_currency: str = "CAD",
        clock: Clock = utcnow,
    ) -> None:
        self.account_id = account_id
        self._repo = repo
        self._market = market
        self._allowed_pairs = frozenset(allowed_pairs)
        self._fee_rate = taker_fee_pct / Decimal(100)
        self._quote = quote_currency
        self._clock = clock
        self._lock = asyncio.Lock()

        account = repo.ensure_account(
            account_id,
            kind="paper",
            quote_currency=quote_currency,
            starting_cash=starting_cash,
            now=clock(),
        )
        self._cash = account.starting_cash
        self._positions: dict[str, _PositionState] = {}
        for fill in repo.list_fills(account_id):
            self._apply_fill(fill)

    # ------------------------------------------------------------ Broker protocol

    async def get_balances(self) -> dict[str, Decimal]:
        balances = {self._quote: self._cash}
        for pair, pos in self._positions.items():
            if pos.amount > 0:
                base = pair.split("/")[0]
                balances[base] = balances.get(base, Decimal(0)) + pos.amount
        return balances

    async def get_positions(self) -> list[Position]:
        return [
            Position(pair=pair, amount=pos.amount, avg_entry_price=pos.avg_entry)
            for pair, pos in sorted(self._positions.items())
            if pos.amount > 0
        ]

    async def get_open_orders(self) -> list[Order]:
        return self._repo.open_orders(self.account_id)

    async def get_equity(self, quote: str = "CAD") -> Decimal:
        """Cash plus positions marked at the best bid (what a sale would fetch)."""
        if quote != self._quote:
            raise ValueError(f"paper account {self.account_id} is denominated in {self._quote}")
        equity = self._cash
        for pair, pos in self._positions.items():
            if pos.amount > 0:
                book = await self._market.fetch_order_book(pair)
                equity += pos.amount * book.best_bid
        return equity

    @property
    def cash(self) -> Decimal:
        return self._cash

    async def place_order(self, order: OrderRequest) -> OrderResult:
        async with self._lock:
            return await self._execute_market(order, order_type=OrderType.MARKET)

    async def cancel_all(self, *, keep_stop_losses: bool = True) -> None:
        """Cancel open orders. Protective stop-losses stay unless explicitly included."""
        async with self._lock:
            ids = [
                o.id
                for o in self._repo.open_orders(self.account_id)
                if not (keep_stop_losses and o.type is OrderType.STOP_LOSS)
            ]
            self._repo.set_order_status(ids, OrderStatus.CANCELLED, self._clock(), "cancel_all")
            if ids:
                log.warning("%s: cancelled %d open order(s)", self.account_id, len(ids))

    # --------------------------------------------------------------- stop-losses

    async def check_stops(self) -> list[OrderResult]:
        """Trigger any stop whose price has been reached (best bid <= trigger)."""
        results: list[OrderResult] = []
        async with self._lock:
            for stop in self._repo.open_orders(self.account_id):
                if stop.type is not OrderType.STOP_LOSS or stop.trigger_price is None:
                    continue
                try:
                    book = await self._market.fetch_order_book(stop.pair)
                except MarketDataError as exc:
                    log.error("%s: stop check for %s failed: %s", self.account_id, stop.pair, exc)
                    continue
                if book.best_bid > stop.trigger_price:
                    continue
                log.warning(
                    "%s: stop-loss triggered on %s (bid %s <= trigger %s)",
                    self.account_id,
                    stop.pair,
                    book.best_bid,
                    stop.trigger_price,
                )
                held = self._position(stop.pair).amount
                amount = min(stop.amount, held)
                if amount <= 0:
                    self._repo.set_order_status(
                        [stop.id], OrderStatus.CANCELLED, self._clock(), "no position"
                    )
                    continue
                result = await self._execute_market(
                    OrderRequest(pair=stop.pair, side=Side.SELL, amount=amount),
                    order_type=OrderType.STOP_LOSS,
                    existing=stop,
                )
                results.append(result)
        return results

    # ------------------------------------------------------------------ internals

    def _position(self, pair: str) -> _PositionState:
        return self._positions.setdefault(pair, _PositionState())

    def _apply_fill(self, fill: Fill) -> None:
        if fill.side is Side.BUY:
            self._cash -= fill.cost + fill.fee
        else:
            self._cash += fill.cost - fill.fee
        self._position(fill.pair).apply(fill.side, fill.amount, fill.cost)

    def _new_order(self, request: OrderRequest, order_type: OrderType, amount: Decimal) -> Order:
        return Order(
            id=uuid.uuid4().hex,
            account_id=self.account_id,
            pair=request.pair,
            side=request.side,
            type=order_type,
            amount=amount,
            status=OrderStatus.OPEN,
            created_at=self._clock(),
            decision_id=request.decision_id,
        )

    def _reject(self, order: Order, reason: str, *, transient: bool = False) -> OrderResult:
        """Record a rejection. ``transient`` = might succeed later (data/liquidity issue)."""
        log.info(
            "%s: order rejected (%s %s %s): %s",
            self.account_id,
            order.side.value,
            order.amount,
            order.pair,
            reason,
        )
        if order.type is OrderType.STOP_LOSS:
            if transient:
                # Leave the stop resting so the next check retries it.
                return OrderResult(order_id=order.id, status=OrderStatus.REJECTED, reason=reason)
            # Can never fill (e.g. dust below Kraken minimum): cancel instead of retrying forever.
            self._repo.save_order(
                replace(order, status=OrderStatus.CANCELLED, reason=reason), self._clock()
            )
            return OrderResult(order_id=order.id, status=OrderStatus.CANCELLED, reason=reason)
        self._repo.save_order(
            replace(order, status=OrderStatus.REJECTED, reason=reason), self._clock()
        )
        return OrderResult(order_id=order.id, status=OrderStatus.REJECTED, reason=reason)

    async def _execute_market(
        self, request: OrderRequest, *, order_type: OrderType, existing: Order | None = None
    ) -> OrderResult:
        order = existing or self._new_order(request, order_type, request.amount)

        if request.pair not in self._allowed_pairs:
            return self._reject(order, f"{request.pair} is not a whitelisted pair")
        try:
            info = await self._market.market_info(request.pair)
        except MarketDataError as exc:
            return self._reject(order, f"market info unavailable: {exc}", transient=True)

        amount = info.round_amount(request.amount)
        order = replace(order, amount=amount)
        if amount <= 0 or (info.min_amount is not None and amount < info.min_amount):
            return self._reject(order, f"amount {amount} below Kraken minimum {info.min_amount}")
        if request.side is Side.SELL and amount > self._position(request.pair).amount:
            return self._reject(
                order,
                f"insufficient {info.base}: have {self._position(request.pair).amount}, "
                f"need {amount}",
            )

        try:
            book = await self._market.fetch_order_book(request.pair)
        except MarketDataError as exc:
            return self._reject(order, f"order book unavailable: {exc}", transient=True)
        levels = book.asks if request.side is Side.BUY else book.bids
        walk = walk_book(levels, amount)
        if walk.filled < amount:
            return self._reject(
                order, f"order book too thin: only {walk.filled} available", transient=True
            )

        cost = walk.cost
        if info.min_cost is not None and cost < info.min_cost:
            return self._reject(order, f"order value {cost} below Kraken minimum {info.min_cost}")
        fee = (cost * self._fee_rate).quantize(FEE_QUANTUM, rounding=ROUND_UP)
        if request.side is Side.BUY and cost + fee > self._cash:
            return self._reject(
                order, f"insufficient {self._quote}: need {cost + fee}, have {self._cash}"
            )

        now = self._clock()
        filled_order = replace(order, status=OrderStatus.FILLED, reason=None)
        fill = Fill(
            order_id=order.id,
            account_id=self.account_id,
            pair=request.pair,
            side=request.side,
            amount=amount,
            price=walk.avg_price,
            cost=cost,
            fee=fee,
            fee_currency=self._quote,
            created_at=now,
            order_type=order_type,
        )
        self._repo.record_fill(filled_order, fill, now)
        self._apply_fill(fill)
        log.info(
            "%s: filled %s %s %s @ %s (fee %s %s)",
            self.account_id,
            request.side.value,
            amount,
            request.pair,
            walk.avg_price,
            fee,
            self._quote,
        )

        self._sync_stop(request, info.round_price_down)
        return OrderResult(
            order_id=order.id,
            status=OrderStatus.FILLED,
            filled_amount=amount,
            avg_price=walk.avg_price,
            cost=cost,
            fee=fee,
        )

    def _sync_stop(self, request: OrderRequest, round_price: Callable[[Decimal], Decimal]) -> None:
        """Keep at most one stop per pair, covering the whole position."""
        pos = self._position(request.pair)
        stops = [
            o
            for o in self._repo.open_orders(self.account_id)
            if o.type is OrderType.STOP_LOSS and o.pair == request.pair
        ]
        now = self._clock()
        if pos.amount <= 0:
            self._repo.set_order_status(
                [s.id for s in stops], OrderStatus.CANCELLED, now, "position closed"
            )
            return

        if request.side is Side.BUY and request.stop_loss_pct is not None:
            trigger = round_price(
                pos.avg_entry * (Decimal(1) - request.stop_loss_pct / Decimal(100))
            )
        elif stops:
            trigger = stops[-1].trigger_price
        else:
            return  # no stop requested and none existing

        self._repo.set_order_status([s.id for s in stops], OrderStatus.CANCELLED, now, "replaced")
        stop = Order(
            id=uuid.uuid4().hex,
            account_id=self.account_id,
            pair=request.pair,
            side=Side.SELL,
            type=OrderType.STOP_LOSS,
            amount=pos.amount,
            status=OrderStatus.OPEN,
            created_at=now,
            trigger_price=trigger,
            decision_id=request.decision_id,
        )
        self._repo.save_order(stop, now)
