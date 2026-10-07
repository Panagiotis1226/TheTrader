"""Pure indicator functions.

Indicators are statistics shown to the LLM, so they return ``float``. Money and
quantities elsewhere stay ``Decimal``. Functions return ``None`` when there is not
enough data rather than inventing a value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal

import numpy as np

from ai_trader.data.market import BookLevel

Number = Decimal | float | int

HOURS_PER_YEAR = 24 * 365  # crypto trades every hour of the year


def _floats(values: Sequence[Number]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=float)


def pct_change(old: Number, new: Number) -> float | None:
    """Percent change from ``old`` to ``new``; ``None`` if ``old`` is zero."""
    if old == 0:
        return None
    return (float(new) - float(old)) / float(old) * 100.0


def log_returns(closes: Sequence[Number]) -> np.ndarray:
    arr = _floats(closes)
    if len(arr) < 2 or np.any(arr <= 0):
        return np.asarray([], dtype=float)
    return np.diff(np.log(arr))


def realized_volatility(
    closes: Sequence[Number], periods_per_year: int = HOURS_PER_YEAR
) -> float | None:
    """Annualized realized volatility in percent (sample stdev of log returns)."""
    returns = log_returns(closes)
    if len(returns) < 2:
        return None
    return float(np.std(returns, ddof=1)) * math.sqrt(periods_per_year) * 100.0


def sma(closes: Sequence[Number], window: int) -> float | None:
    """Simple moving average of the last ``window`` closes."""
    if window <= 0:
        raise ValueError("window must be positive")
    if len(closes) < window:
        return None
    return float(np.mean(_floats(closes[-window:])))


def rsi(closes: Sequence[Number], period: int = 14) -> float | None:
    """Wilder's RSI of the final close. Needs ``period + 1`` closes.

    Seeds with the simple average of the first ``period`` gains/losses, then applies
    Wilder smoothing. A flat series returns 50.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    arr = _floats(closes)
    if len(arr) < period + 1:
        return None
    deltas = np.diff(arr)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for gain, loss in zip(gains[period:], losses[period:], strict=True):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def spread_pct(bid: Number, ask: Number) -> float:
    """Bid/ask spread as a percent of the mid price."""
    bid_f, ask_f = float(bid), float(ask)
    mid = (bid_f + ask_f) / 2
    if mid <= 0:
        raise ValueError("bid/ask must be positive")
    return (ask_f - bid_f) / mid * 100.0


def depth_within(levels: Sequence[BookLevel], reference: Decimal, pct: Decimal) -> Decimal:
    """Quote-currency value resting within ``pct`` percent of ``reference``.

    Works for either side: levels are included while their price is within the band.
    """
    band = reference * pct / Decimal(100)
    total = Decimal(0)
    for level in levels:
        if abs(level.price - reference) > band:
            break
        total += level.price * level.amount
    return total
