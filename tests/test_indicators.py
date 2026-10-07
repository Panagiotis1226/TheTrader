from __future__ import annotations

import math
from decimal import Decimal

import pytest

from ai_trader.data import indicators as ind
from ai_trader.data.market import BookLevel

# StockCharts' published Wilder RSI(14) worked example.
RSI_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
    45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
]  # fmt: skip
RSI_EXPECTED = [70.53, 66.32, 66.55, 69.41, 66.36, 57.97]


def test_rsi_first_value_exact() -> None:
    # First 14 deltas: gains sum to 3.34, losses to 1.40.
    assert ind.rsi(RSI_CLOSES[:15], 14) == pytest.approx(100 - 100 / (1 + 3.34 / 1.40))


def test_rsi_matches_wilder_reference() -> None:
    # StockCharts rounds intermediate averages, so allow a small tolerance.
    got = [ind.rsi(RSI_CLOSES[: 15 + i], 14) for i in range(len(RSI_EXPECTED))]
    assert got == pytest.approx(RSI_EXPECTED, abs=0.1)


def test_rsi_edge_cases() -> None:
    assert ind.rsi(list(range(1, 20))) == 100.0
    assert ind.rsi(list(range(20, 1, -1))) == 0.0
    assert ind.rsi([5] * 20) == 50.0
    assert ind.rsi([1] * 14) is None  # needs period + 1 closes


def test_sma() -> None:
    assert ind.sma([Decimal(1), Decimal(2), Decimal(3), Decimal(4)], 2) == 3.5
    assert ind.sma([1, 2], 3) is None
    with pytest.raises(ValueError):
        ind.sma([1], 0)


def test_pct_change() -> None:
    assert ind.pct_change(Decimal(100), Decimal(110)) == pytest.approx(10.0)
    assert ind.pct_change(200, 150) == pytest.approx(-25.0)
    assert ind.pct_change(0, 5) is None


def test_realized_volatility() -> None:
    # Alternating +1%/-1% log moves: stdev of returns is ~0.01 (sample).
    closes = [100 * math.exp(0.01 * (i % 2)) for i in range(25)]
    vol = ind.realized_volatility(closes, periods_per_year=1)
    returns = ind.log_returns(closes)
    assert vol == pytest.approx(float(returns.std(ddof=1)) * 100)
    assert ind.realized_volatility(closes) == pytest.approx(vol * math.sqrt(24 * 365))
    assert ind.realized_volatility([100, 100, 100]) == 0.0
    assert ind.realized_volatility([100, 101]) is None


def test_spread_pct() -> None:
    assert ind.spread_pct(Decimal("99"), Decimal("101")) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        ind.spread_pct(0, 0)


def test_depth_within_band() -> None:
    asks = [
        BookLevel(Decimal("100"), Decimal("1")),
        BookLevel(Decimal("100.5"), Decimal("2")),
        BookLevel(Decimal("101"), Decimal("3")),  # exactly 1%: included
        BookLevel(Decimal("101.5"), Decimal("4")),  # outside
    ]
    assert ind.depth_within(asks, Decimal("100"), Decimal("1")) == Decimal("604")
    bids = [BookLevel(Decimal("100"), Decimal("1")), BookLevel(Decimal("98"), Decimal("5"))]
    assert ind.depth_within(bids, Decimal("100"), Decimal("1")) == Decimal("100")
