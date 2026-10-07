"""Broker protocol and the domain types shared by paper and live brokers.

All money and quantity fields are ``Decimal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Protocol


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    STOP_LOSS = "stop_loss"  # resting sell, triggers as a market sell


class OrderStatus(StrEnum):
    OPEN = "open"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class OrderRequest:
    """A market order approved by the RiskManager.

    Give exactly one of ``amount`` (base quantity) or ``quote_amount`` (quote currency to
    spend, buys only, before fees). Risk-sized buys use ``quote_amount`` so slippage can
    never push the cost above what the RiskManager approved.
    """

    pair: str
    side: Side
    amount: Decimal | None = None  # base quantity; the broker rounds down to the step
    stop_loss_pct: Decimal | None = None  # buys only: protective stop this % below avg entry
    decision_id: int | None = None
    quote_amount: Decimal | None = None

    def __post_init__(self) -> None:
        if (self.amount is None) == (self.quote_amount is None):
            raise ValueError("give exactly one of amount or quote_amount")
        size = self.amount if self.amount is not None else self.quote_amount
        if size is None or size <= 0:
            raise ValueError("order size must be positive")
        if self.quote_amount is not None and self.side is not Side.BUY:
            raise ValueError("quote_amount is only valid for buys")
        if self.stop_loss_pct is not None and not 0 < self.stop_loss_pct < 100:
            raise ValueError("stop_loss_pct must be between 0 and 100")


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    status: OrderStatus
    reason: str | None = None
    filled_amount: Decimal = Decimal(0)
    avg_price: Decimal | None = None
    cost: Decimal = Decimal(0)  # gross quote value of the fill, before fees
    fee: Decimal = Decimal(0)

    @property
    def filled(self) -> bool:
        return self.status is OrderStatus.FILLED


@dataclass(frozen=True)
class Order:
    id: str
    account_id: str
    pair: str
    side: Side
    type: OrderType
    amount: Decimal
    status: OrderStatus
    created_at: datetime
    trigger_price: Decimal | None = None
    reason: str | None = None
    decision_id: int | None = None


@dataclass(frozen=True)
class Fill:
    order_id: str
    account_id: str
    pair: str
    side: Side
    amount: Decimal
    price: Decimal  # average fill price
    cost: Decimal  # amount * price, gross
    fee: Decimal
    fee_currency: str
    created_at: datetime
    order_type: OrderType = OrderType.MARKET


@dataclass(frozen=True)
class Position:
    pair: str
    amount: Decimal
    avg_entry_price: Decimal  # volume-weighted buy price, excluding fees


class Broker(Protocol):
    account_id: str

    async def get_balances(self) -> dict[str, Decimal]: ...
    async def get_positions(self) -> list[Position]: ...
    async def place_order(self, order: OrderRequest) -> OrderResult: ...
    async def cancel_all(self, *, keep_stop_losses: bool = True) -> None:
        """Cancel open orders; protective stop-losses survive unless told otherwise."""
        ...

    async def get_open_orders(self) -> list[Order]: ...
    async def get_equity(self, quote: str = "CAD") -> Decimal: ...
