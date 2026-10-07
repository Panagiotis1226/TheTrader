"""Kraken public market data via ccxt (no API key).

Everything crossing this boundary is converted from ccxt floats to ``Decimal``.
Kraken's ticker and order-book responses carry no server timestamp, so freshness is
measured from ``received_at`` (local receipt time, UTC).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any, Protocol

import ccxt.async_support as ccxt_async

Clock = Callable[[], datetime]

TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_decimal(value: Any) -> Decimal:
    """Convert a ccxt number to Decimal via ``str`` so 0.1 stays 0.1."""
    if value is None:
        raise ValueError("expected a number, got None")
    return Decimal(str(value))


def _optional_decimal(value: Any) -> Decimal | None:
    return None if value is None else to_decimal(value)


class MarketDataError(Exception):
    """Market data is unavailable or unusable."""


@dataclass(frozen=True)
class MarketInfo:
    pair: str
    base: str
    quote: str
    amount_step: Decimal  # smallest amount increment (Kraken uses tick-size precision)
    price_step: Decimal
    min_amount: Decimal | None
    min_cost: Decimal | None

    def round_amount(self, amount: Decimal) -> Decimal:
        """Round an amount *down* to the exchange's step size."""
        return (amount / self.amount_step).to_integral_value(ROUND_DOWN) * self.amount_step

    def round_price_down(self, price: Decimal) -> Decimal:
        return (price / self.price_step).to_integral_value(ROUND_DOWN) * self.price_step


@dataclass(frozen=True)
class BookLevel:
    price: Decimal
    amount: Decimal


@dataclass(frozen=True)
class OrderBook:
    pair: str
    bids: tuple[BookLevel, ...]  # best (highest) first
    asks: tuple[BookLevel, ...]  # best (lowest) first
    received_at: datetime

    @property
    def best_bid(self) -> Decimal:
        if not self.bids:
            raise MarketDataError(f"{self.pair}: order book has no bids")
        return self.bids[0].price

    @property
    def best_ask(self) -> Decimal:
        if not self.asks:
            raise MarketDataError(f"{self.pair}: order book has no asks")
        return self.asks[0].price


@dataclass(frozen=True)
class Ticker:
    pair: str
    last: Decimal
    bid: Decimal
    ask: Decimal
    received_at: datetime


@dataclass(frozen=True)
class Candle:
    opened_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class OrderBookSource(Protocol):
    async def fetch_order_book(self, pair: str) -> OrderBook: ...
    async def market_info(self, pair: str) -> MarketInfo: ...


def make_kraken_exchange() -> ccxt_async.Exchange:
    """Public (unauthenticated) Kraken client. Honors HTTPS_PROXY / SSL_CERT_FILE."""
    config: dict[str, Any] = {"enableRateLimit": True, "aiohttp_trust_env": True}
    cafile = os.environ.get("SSL_CERT_FILE")
    if cafile:
        config["cafile"] = cafile
    return ccxt_async.kraken(config)


class MarketData:
    """Thin async wrapper around a ccxt exchange. Pass a fake exchange in tests."""

    def __init__(self, exchange: Any | None = None, clock: Clock = utcnow) -> None:
        self._exchange = exchange if exchange is not None else make_kraken_exchange()
        self._clock = clock
        self._markets: dict[str, MarketInfo] | None = None

    async def close(self) -> None:
        await self._exchange.close()

    async def __aenter__(self) -> MarketData:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def load_markets(self, reload: bool = False) -> dict[str, MarketInfo]:
        if self._markets is None or reload:
            raw = await self._exchange.load_markets(reload)
            self._markets = {
                symbol: self._parse_market(m)
                for symbol, m in raw.items()
                if m.get("spot") and m.get("active", True) is not False
            }
        return self._markets

    async def market_info(self, pair: str) -> MarketInfo:
        markets = await self.load_markets()
        if pair not in markets:
            raise MarketDataError(f"{pair} is not an active Kraken spot market")
        return markets[pair]

    async def fetch_ticker(self, pair: str) -> Ticker:
        raw = await self._exchange.fetch_ticker(pair)
        try:
            return Ticker(
                pair=pair,
                last=to_decimal(raw["last"]),
                bid=to_decimal(raw["bid"]),
                ask=to_decimal(raw["ask"]),
                received_at=self._clock(),
            )
        except (KeyError, ValueError) as exc:
            raise MarketDataError(f"{pair}: malformed ticker: {exc}") from exc

    async def fetch_order_book(self, pair: str, limit: int = 100) -> OrderBook:
        raw = await self._exchange.fetch_order_book(pair, limit)
        book = OrderBook(
            pair=pair,
            bids=tuple(BookLevel(to_decimal(lv[0]), to_decimal(lv[1])) for lv in raw["bids"]),
            asks=tuple(BookLevel(to_decimal(lv[0]), to_decimal(lv[1])) for lv in raw["asks"]),
            received_at=self._clock(),
        )
        if not book.bids or not book.asks:
            raise MarketDataError(f"{pair}: empty order book")
        if book.best_bid >= book.best_ask:
            raise MarketDataError(f"{pair}: crossed order book")
        return book

    async def fetch_ohlcv(
        self, pair: str, timeframe: str, limit: int | None = None
    ) -> list[Candle]:
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        raw = await self._exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        return [
            Candle(
                opened_at=datetime.fromtimestamp(row[0] / 1000, tz=UTC),
                open=to_decimal(row[1]),
                high=to_decimal(row[2]),
                low=to_decimal(row[3]),
                close=to_decimal(row[4]),
                volume=to_decimal(row[5]),
            )
            for row in raw
        ]

    @staticmethod
    def _parse_market(m: dict[str, Any]) -> MarketInfo:
        limits = m.get("limits") or {}
        return MarketInfo(
            pair=m["symbol"],
            base=m["base"],
            quote=m["quote"],
            amount_step=to_decimal(m["precision"]["amount"]),
            price_step=to_decimal(m["precision"]["price"]),
            min_amount=_optional_decimal((limits.get("amount") or {}).get("min")),
            min_cost=_optional_decimal((limits.get("cost") or {}).get("min")),
        )
