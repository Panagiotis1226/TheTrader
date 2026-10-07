"""Performance metrics. Equity curves are daily."""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal
from itertools import pairwise

from ai_trader.brokers.base import Fill, Side

DAYS_PER_YEAR = 365  # crypto trades every day


def total_return_pct(equity: Sequence[Decimal]) -> float:
    return (float(equity[-1]) / float(equity[0]) - 1) * 100


def cagr_pct(equity: Sequence[Decimal], days: float) -> float | None:
    if days <= 0 or equity[0] <= 0 or equity[-1] <= 0:
        return None
    return ((float(equity[-1]) / float(equity[0])) ** (DAYS_PER_YEAR / days) - 1) * 100


def max_drawdown_pct(equity: Sequence[Decimal]) -> float:
    peak = float(equity[0])
    worst = 0.0
    for value in equity:
        v = float(value)
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst * 100


def sharpe(equity: Sequence[Decimal]) -> float | None:
    """Annualized Sharpe of daily returns, risk-free rate 0. None if undefined."""
    values = [float(v) for v in equity]
    returns = [b / a - 1 for a, b in pairwise(values) if a > 0]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var) * math.sqrt(DAYS_PER_YEAR)


def win_rate_pct(fills: Sequence[Fill]) -> tuple[int, float | None]:
    """(sells, % of sells above the running average entry price after fees)."""
    amounts: dict[str, Decimal] = {}
    costs: dict[str, Decimal] = {}  # including buy fees
    sells = wins = 0
    for f in fills:
        if f.side is Side.BUY:
            amounts[f.pair] = amounts.get(f.pair, Decimal(0)) + f.amount
            costs[f.pair] = costs.get(f.pair, Decimal(0)) + f.cost + f.fee
            continue
        held = amounts.get(f.pair, Decimal(0))
        if held <= 0:
            continue
        avg_cost = costs[f.pair] / held
        sells += 1
        if f.cost - f.fee > avg_cost * f.amount:
            wins += 1
        amounts[f.pair] = held - f.amount
        costs[f.pair] = avg_cost * amounts[f.pair]
    return sells, (wins / sells * 100 if sells else None)
