"""Deterministic stand-ins for the exchange, market data, and clock."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_trader.data.market import BookLevel, MarketDataError, MarketInfo, OrderBook

FIXTURES = Path(__file__).resolve().parent / "fixtures"
T0 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def load_kraken_fixture() -> dict[str, Any]:
    return json.loads((FIXTURES / "kraken" / "snapshot_inputs.json").read_text())


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


class FakeExchange:
    """Replays recorded raw ccxt responses."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or load_kraken_fixture()
        self.closed = False

    async def load_markets(self, reload: bool = False) -> dict[str, Any]:
        return self.data["markets"]

    def _key(self, pair: str) -> str:
        return pair.replace("/", "_")

    async def fetch_ticker(self, pair: str) -> dict[str, Any]:
        return self.data[f"ticker_{self._key(pair)}"]

    async def fetch_order_book(self, pair: str, limit: int | None = None) -> dict[str, Any]:
        return self.data[f"order_book_{self._key(pair)}"]

    async def fetch_ohlcv(self, pair: str, timeframe: str, limit: int | None = None) -> list:
        return self.data[f"ohlcv_{timeframe}_{self._key(pair)}"]

    async def close(self) -> None:
        self.closed = True


BTC_INFO = MarketInfo(
    pair="BTC/CAD",
    base="BTC",
    quote="CAD",
    amount_step=Decimal("0.00000001"),
    price_step=Decimal("0.1"),
    min_amount=Decimal("0.00005"),
    min_cost=Decimal("1"),
)
ETH_INFO = MarketInfo(
    pair="ETH/CAD",
    base="ETH",
    quote="CAD",
    amount_step=Decimal("0.00000001"),
    price_step=Decimal("0.01"),
    min_amount=Decimal("0.001"),
    min_cost=Decimal("1"),
)


def book(
    pair: str,
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    received_at: datetime = T0,
) -> OrderBook:
    return OrderBook(
        pair=pair,
        bids=tuple(BookLevel(Decimal(p), Decimal(a)) for p, a in bids),
        asks=tuple(BookLevel(Decimal(p), Decimal(a)) for p, a in asks),
        received_at=received_at,
    )


@dataclass
class FakeBookSource:
    """Static, settable order books per pair (implements OrderBookSource)."""

    books: dict[str, OrderBook] = field(default_factory=dict)
    infos: dict[str, MarketInfo] = field(
        default_factory=lambda: {"BTC/CAD": BTC_INFO, "ETH/CAD": ETH_INFO}
    )
    fail: bool = False

    async def fetch_order_book(self, pair: str) -> OrderBook:
        if self.fail:
            raise MarketDataError("simulated outage")
        if pair not in self.books:
            raise MarketDataError(f"no book for {pair}")
        return self.books[pair]

    async def market_info(self, pair: str) -> MarketInfo:
        if pair not in self.infos:
            raise MarketDataError(f"{pair} unknown")
        return self.infos[pair]
