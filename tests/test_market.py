from __future__ import annotations

import copy
from datetime import UTC, timedelta
from decimal import Decimal

import pytest

from ai_trader.data.market import MarketData, MarketDataError, to_decimal

from .fakes import FakeClock, FakeExchange, load_kraken_fixture

D = Decimal


async def test_load_markets_parses_exact_decimals() -> None:
    md = MarketData(FakeExchange())
    info = await md.market_info("BTC/CAD")
    assert info.base == "BTC" and info.quote == "CAD"
    assert info.amount_step == D("0.00000001")
    assert info.price_step == D("0.1")
    assert info.min_amount == D("0.00005")
    assert info.min_cost == D("1")


async def test_non_spot_markets_are_excluded() -> None:
    data = load_kraken_fixture()
    fut = copy.deepcopy(data["markets"]["BTC/CAD"])
    fut.update(symbol="BTC/CAD:CAD", spot=False, type="swap")
    data["markets"]["BTC/CAD:CAD"] = fut
    md = MarketData(FakeExchange(data))
    with pytest.raises(MarketDataError, match="not an active Kraken spot market"):
        await md.market_info("BTC/CAD:CAD")
    with pytest.raises(MarketDataError):
        await md.market_info("DOGE/CAD")


async def test_ticker_and_book_conversion() -> None:
    clock = FakeClock()
    md = MarketData(FakeExchange(), clock=clock)
    ticker = await md.fetch_ticker("BTC/CAD")
    assert isinstance(ticker.last, Decimal)
    assert ticker.received_at == clock.now
    book = await md.fetch_order_book("BTC/CAD")
    assert len(book.asks) == 100
    assert book.best_bid < book.best_ask
    assert all(a.price <= b.price for a, b in zip(book.asks, book.asks[1:], strict=False))
    assert all(a.price >= b.price for a, b in zip(book.bids, book.bids[1:], strict=False))


async def test_crossed_or_empty_book_rejected() -> None:
    data = load_kraken_fixture()
    data["order_book_BTC_CAD"] = {"bids": [[101, 1]], "asks": [[100, 1]]}
    data["order_book_ETH_CAD"] = {"bids": [], "asks": [[100, 1]]}
    md = MarketData(FakeExchange(data))
    with pytest.raises(MarketDataError, match="crossed"):
        await md.fetch_order_book("BTC/CAD")
    with pytest.raises(MarketDataError, match="empty"):
        await md.fetch_order_book("ETH/CAD")


async def test_ohlcv_is_utc_and_decimal() -> None:
    md = MarketData(FakeExchange())
    candles = await md.fetch_ohlcv("BTC/CAD", "1h")
    assert len(candles) == 721
    assert candles[0].opened_at.tzinfo is UTC
    assert isinstance(candles[-1].close, Decimal)
    assert candles[1].opened_at - candles[0].opened_at == timedelta(hours=1)
    with pytest.raises(ValueError):
        await md.fetch_ohlcv("BTC/CAD", "3h")


async def test_close_closes_exchange() -> None:
    ex = FakeExchange()
    async with MarketData(ex):
        pass
    assert ex.closed


def test_to_decimal_avoids_float_artifacts() -> None:
    assert to_decimal(0.1) == D("0.1")
    assert to_decimal(1e-08) == D("0.00000001")
    with pytest.raises(ValueError):
        to_decimal(None)


async def test_round_amount_down() -> None:
    info = await MarketData(FakeExchange()).market_info("BTC/CAD")
    assert info.round_amount(D("0.123456789")) == D("0.12345678")
    assert info.round_price_down(D("95000.09")) == D("95000.0")
