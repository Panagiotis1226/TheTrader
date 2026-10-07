from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.brokers.base import Position
from ai_trader.data import indicators as ind
from ai_trader.data.market import Candle, MarketData, MarketDataError
from ai_trader.data.snapshot import (
    PairMarketData,
    StaleDataError,
    build_snapshot,
    close_at_or_before,
    closed_candles,
    fetch_pair_data,
)
from ai_trader.storage.repo import DecisionRecord

from .fakes import FakeClock, FakeExchange, load_kraken_fixture

D = Decimal
FIXTURE = load_kraken_fixture()
RECORDED_AT = datetime.fromtimestamp(FIXTURE["recorded_at_ms"] / 1000, tz=UTC)


async def _data(clock: FakeClock) -> dict[str, PairMarketData]:
    md = MarketData(FakeExchange(), clock=clock)
    return {p: await fetch_pair_data(md, p) for p in ("BTC/CAD", "ETH/CAD")}


def _build(data, now, **kw):
    args = dict(cash=D(10000), positions=[], recent_decisions=[], now=now, max_data_age_seconds=60)
    args.update(kw)
    return build_snapshot(data, **args)


async def test_snapshot_from_recorded_kraken_data() -> None:
    clock = FakeClock(RECORDED_AT)
    data = await _data(clock)
    snap = _build(data, RECORDED_AT)

    assert [p.pair for p in snap.pairs] == ["BTC/CAD", "ETH/CAD"]
    assert snap.total_equity == D("10000.00")
    assert snap.data_age_seconds == 0
    for p in snap.pairs:
        # 720 closed daily candles: every indicator is available.
        for field in (
            "change_24h_pct",
            "change_7d_pct",
            "change_30d_pct",
            "sma_200d",
            "rsi_14d",
            "volatility_24h_annualized_pct",
            "volatility_7d_annualized_pct",
        ):
            assert getattr(p, field) is not None, field
        assert 0 <= p.rsi_14d <= 100
        assert p.spread_pct > 0
        assert p.bid < p.ask
        assert p.position_amount == 0 and p.unrealized_pnl_pct is None


async def test_snapshot_values_match_manual_computation() -> None:
    data = await _data(FakeClock(RECORDED_AT))
    btc = _build(data, RECORDED_AT).pairs[0]
    d = data["BTC/CAD"]
    daily = closed_candles(d.daily, "1d", RECORDED_AT)
    assert daily[-1].opened_at + timedelta(days=1) <= RECORDED_AT
    assert len(daily) == len(d.daily) - 1  # in-progress candle dropped
    assert btc.sma_20d == round(ind.sma([c.close for c in daily], 20), 2)
    ref = close_at_or_before(d.hourly, "1h", RECORDED_AT - timedelta(days=1))
    assert btc.change_24h_pct == round(ind.pct_change(ref, d.ticker.last), 2)


async def test_snapshot_hash_is_deterministic() -> None:
    data = await _data(FakeClock(RECORDED_AT))
    a, b = _build(data, RECORDED_AT), _build(data, RECORDED_AT)
    assert a.content_hash() == b.content_hash()
    assert len(a.content_hash()) == 64
    assert _build(data, RECORDED_AT, cash=D(1)).content_hash() != a.content_hash()


async def test_stale_ticker_rejected() -> None:
    data = await _data(FakeClock(RECORDED_AT))
    with pytest.raises(StaleDataError, match="61s old"):
        _build(data, RECORDED_AT + timedelta(seconds=61))
    _build(data, RECORDED_AT + timedelta(seconds=60))  # exactly at the limit is fine


async def test_stale_candle_feed_rejected() -> None:
    later = RECORDED_AT + timedelta(hours=3)
    data = await _data(FakeClock(later))  # fresh ticker/book, but candles 3h old
    with pytest.raises(StaleDataError, match="hourly candles"):
        _build(data, later)


async def test_snapshot_with_position_and_decisions() -> None:
    data = await _data(FakeClock(RECORDED_AT))
    bid = data["BTC/CAD"].book.best_bid
    entry = bid / D("1.1")  # position is up 10% at the bid
    decisions = [
        DecisionRecord(
            1, RECORDED_AT - timedelta(hours=4), "buy", "BTC/CAD", D(5), "approve", None, entry
        ),
        DecisionRecord(
            2, RECORDED_AT - timedelta(hours=8), "sell", "BTC/CAD", D(50), "approve", None, entry
        ),
        DecisionRecord(
            3, RECORDED_AT - timedelta(hours=12), "hold", "BTC/CAD", D(0), "approve", None, None
        ),
    ]
    snap = _build(
        data,
        RECORDED_AT,
        cash=D(5000),
        positions=[Position("BTC/CAD", D("0.01"), entry)],
        recent_decisions=decisions,
    )
    btc = snap.pairs[0]
    assert btc.position_amount == D("0.01")
    assert btc.unrealized_pnl_pct == pytest.approx(10.0, abs=0.01)
    assert snap.total_equity == (D(5000) + D("0.01") * bid).quantize(D("0.01"))
    outcomes = [d.outcome_since_pct for d in snap.recent_decisions]
    assert outcomes[0] == pytest.approx(10.0, abs=0.01)
    assert outcomes[1] == pytest.approx(-10.0, abs=0.01)
    assert outcomes[2] is None


async def test_held_pair_without_data_rejected() -> None:
    data = await _data(FakeClock(RECORDED_AT))
    only_eth = {"ETH/CAD": data["ETH/CAD"]}
    with pytest.raises(MarketDataError, match="held pairs"):
        _build(only_eth, RECORDED_AT, positions=[Position("BTC/CAD", D(1), D(1))])
    with pytest.raises(MarketDataError):
        _build({}, RECORDED_AT)


async def test_short_history_gives_none_not_garbage() -> None:
    data = await _data(FakeClock(RECORDED_AT))
    btc = data["BTC/CAD"]
    short = replace(btc, daily=btc.daily[-30:], hourly=btc.hourly[-30:])
    p = _build({"BTC/CAD": short}, RECORDED_AT).pairs[0]
    assert p.sma_200d is None and p.price_vs_sma_200d_pct is None
    assert p.change_7d_pct is None and p.change_30d_pct is None
    assert p.sma_20d is not None and p.change_24h_pct is not None


def test_close_at_or_before() -> None:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [Candle(t + timedelta(hours=i), D(1), D(1), D(1), D(i), D(1)) for i in range(5)]
    assert close_at_or_before(candles, "1h", t + timedelta(hours=3)) == D(2)
    assert close_at_or_before(candles, "1h", t + timedelta(minutes=30)) is None
