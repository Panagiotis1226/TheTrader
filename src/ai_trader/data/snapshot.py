"""Compact, pre-computed market snapshot for the LLM (PLAN.md §5.3).

The LLM never sees raw candles: code calculates, the LLM judges.

Data sources per pair: ticker (last price), order book (spread, depth, marking),
closed 1h candles (24h/7d change, realized volatility), closed 1d candles
(30d change, SMA 20/50/200, RSI 14). The in-progress candle is always dropped.

Kraken's ticker and book have no server timestamp, so data age is measured from
receipt. The snapshot is refused (``StaleDataError``) if any ticker/book is older than
``max_data_age_seconds`` or the candle feed has stopped updating.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from ai_trader.brokers.base import Position
from ai_trader.data import indicators as ind
from ai_trader.data.market import (
    TIMEFRAME_SECONDS,
    Candle,
    MarketData,
    MarketDataError,
    OrderBook,
    Ticker,
)
from ai_trader.storage.repo import DecisionRecord

DEPTH_BAND_PCT = Decimal(1)  # report book depth within 1% of the best price
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


class StaleDataError(MarketDataError):
    """Market data is too old to act on. Skip the cycle."""


@dataclass(frozen=True)
class PairMarketData:
    pair: str
    ticker: Ticker
    book: OrderBook
    hourly: Sequence[Candle]
    daily: Sequence[Candle]


async def fetch_pair_data(market: MarketData, pair: str) -> PairMarketData:
    # Candles first, then the time-sensitive ticker/book, so their age stays minimal.
    hourly = await market.fetch_ohlcv(pair, "1h")
    daily = await market.fetch_ohlcv(pair, "1d")
    ticker = await market.fetch_ticker(pair)
    book = await market.fetch_order_book(pair)
    return PairMarketData(pair, ticker, book, hourly, daily)


# ----------------------------------------------------------------------- models


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PairSnapshot(_Model):
    pair: str
    last_price: Decimal
    change_24h_pct: float | None
    change_7d_pct: float | None
    change_30d_pct: float | None
    volatility_24h_annualized_pct: float | None
    volatility_7d_annualized_pct: float | None
    sma_20d: float | None
    sma_50d: float | None
    sma_200d: float | None
    price_vs_sma_20d_pct: float | None
    price_vs_sma_50d_pct: float | None
    price_vs_sma_200d_pct: float | None
    rsi_14d: float | None
    bid: Decimal
    ask: Decimal
    spread_pct: float
    bid_depth_1pct_quote: Decimal
    ask_depth_1pct_quote: Decimal
    position_amount: Decimal
    position_value: Decimal
    avg_entry_price: Decimal | None
    unrealized_pnl_pct: float | None


class DecisionSummary(_Model):
    at: datetime
    action: str | None
    pair: str | None
    size_pct: Decimal | None
    risk_outcome: str | None
    fill_price: Decimal | None
    # Signed so positive = the decision has worked out so far (price rose after a buy,
    # fell after a sell). Measured against the current best bid.
    outcome_since_pct: float | None


class MarketSnapshot(_Model):
    timestamp: datetime
    data_age_seconds: float
    quote_currency: str
    cash_available: Decimal
    total_equity: Decimal
    pairs: list[PairSnapshot]
    recent_decisions: list[DecisionSummary]

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON, logged with every decision."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


# --------------------------------------------------------------------- building


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def closed_candles(candles: Sequence[Candle], timeframe: str, now: datetime) -> list[Candle]:
    span = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    return [c for c in candles if c.opened_at + span <= now]


def close_at_or_before(candles: Sequence[Candle], timeframe: str, when: datetime) -> Decimal | None:
    """Close of the latest candle that had closed by ``when``."""
    span = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    for candle in reversed(candles):
        if candle.opened_at + span <= when:
            return candle.close
    return None


def _check_fresh(
    data: PairMarketData, now: datetime, max_age_s: int, require_intraday: bool
) -> float:
    age = max(
        (now - data.ticker.received_at).total_seconds(),
        (now - data.book.received_at).total_seconds(),
    )
    if age > max_age_s:
        raise StaleDataError(f"{data.pair}: data is {age:.0f}s old (max {max_age_s}s)")
    if require_intraday and (not data.hourly or data.hourly[-1].opened_at < now - 2 * HOUR):
        raise StaleDataError(f"{data.pair}: hourly candles are not current")
    if not data.daily or data.daily[-1].opened_at < now - 2 * DAY:
        raise StaleDataError(f"{data.pair}: daily candles are not current")
    return age


def _pair_snapshot(data: PairMarketData, position: Position | None, now: datetime) -> PairSnapshot:
    last = data.ticker.last
    hourly = closed_candles(data.hourly, "1h", now)
    daily = closed_candles(data.daily, "1d", now)
    daily_closes = [c.close for c in daily]
    hourly_closes = [c.close for c in hourly]

    def change(ref: Decimal | None) -> float | None:
        return _round(ind.pct_change(ref, last)) if ref is not None else None

    def vs(avg: float | None) -> float | None:
        return _round(ind.pct_change(avg, last)) if avg else None

    if hourly:
        fast: tuple[Sequence[Candle], str] = (hourly, "1h")
        vol_24h = _round(ind.realized_volatility(hourly_closes[-25:]))
        vol_7d = _round(ind.realized_volatility(hourly_closes[-169:]))
    else:
        # Daily-only data (backtests): changes from daily closes, no 24h volatility.
        fast = (daily, "1d")
        vol_24h = None
        vol_7d = _round(ind.realized_volatility(daily_closes[-8:], periods_per_year=365))

    sma20, sma50, sma200 = (ind.sma(daily_closes, n) for n in (20, 50, 200))
    bid, ask = data.book.best_bid, data.book.best_ask

    amount = position.amount if position else Decimal(0)
    avg_entry = position.avg_entry_price if position and position.amount > 0 else None
    pnl = _round(ind.pct_change(avg_entry, bid)) if avg_entry else None

    return PairSnapshot(
        pair=data.pair,
        last_price=last,
        change_24h_pct=change(close_at_or_before(*fast, now - DAY)),
        change_7d_pct=change(close_at_or_before(*fast, now - 7 * DAY)),
        change_30d_pct=change(close_at_or_before(daily, "1d", now - 30 * DAY)),
        volatility_24h_annualized_pct=vol_24h,
        volatility_7d_annualized_pct=vol_7d,
        sma_20d=_round(sma20),
        sma_50d=_round(sma50),
        sma_200d=_round(sma200),
        price_vs_sma_20d_pct=vs(sma20),
        price_vs_sma_50d_pct=vs(sma50),
        price_vs_sma_200d_pct=vs(sma200),
        rsi_14d=_round(ind.rsi(daily_closes, 14), 1),
        bid=bid,
        ask=ask,
        spread_pct=round(ind.spread_pct(bid, ask), 4),
        bid_depth_1pct_quote=ind.depth_within(data.book.bids, bid, DEPTH_BAND_PCT).quantize(
            Decimal(1)
        ),
        ask_depth_1pct_quote=ind.depth_within(data.book.asks, ask, DEPTH_BAND_PCT).quantize(
            Decimal(1)
        ),
        position_amount=amount,
        position_value=(amount * bid).quantize(Decimal("0.01")),
        avg_entry_price=avg_entry,
        unrealized_pnl_pct=pnl,
    )


def _decision_summary(d: DecisionRecord, books: Mapping[str, OrderBook]) -> DecisionSummary:
    outcome = None
    if d.fill_price and d.pair in books and d.action in ("buy", "sell"):
        move = ind.pct_change(d.fill_price, books[d.pair].best_bid)
        if move is not None:
            outcome = round(move if d.action == "buy" else -move, 2)
    return DecisionSummary(
        at=d.created_at,
        action=d.action,
        pair=d.pair,
        size_pct=d.size_pct,
        risk_outcome=d.risk_outcome,
        fill_price=d.fill_price,
        outcome_since_pct=outcome,
    )


def build_snapshot(
    data: Mapping[str, PairMarketData],
    *,
    cash: Decimal,
    positions: Sequence[Position],
    recent_decisions: Sequence[DecisionRecord],
    now: datetime,
    max_data_age_seconds: int,
    quote_currency: str = "CAD",
    require_intraday: bool = True,
) -> MarketSnapshot:
    """Pure function: validate freshness, compute indicators, assemble the snapshot.

    ``require_intraday=False`` is for backtests, which only have daily candles.
    """
    if not data:
        raise MarketDataError("no market data")
    ages = [_check_fresh(d, now, max_data_age_seconds, require_intraday) for d in data.values()]

    by_pair = {p.pair: p for p in positions}
    missing = set(by_pair) - set(data)
    if missing:
        raise MarketDataError(f"no market data for held pairs {sorted(missing)}")

    pairs = [_pair_snapshot(d, by_pair.get(pair), now) for pair, d in data.items()]
    equity = cash + sum((p.position_amount * p.bid for p in pairs), Decimal(0))
    books = {pair: d.book for pair, d in data.items()}

    return MarketSnapshot(
        timestamp=now,
        data_age_seconds=round(max(ages), 1),
        quote_currency=quote_currency,
        cash_available=cash.quantize(Decimal("0.01")),
        total_equity=equity.quantize(Decimal("0.01")),
        pairs=pairs,
        recent_decisions=[_decision_summary(d, books) for d in recent_decisions],
    )
