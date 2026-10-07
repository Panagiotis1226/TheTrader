"""Benchmark: daily SMA 20/50 crossover, per pair.

* SMA20 > SMA50 (uptrend): be invested, up to ``max_position_pct_per_pair`` at cost,
  buying ``max_trade_pct_of_equity`` per decision.
* SMA20 < SMA50 (downtrend): hold no position; sell it all.

One proposal per cycle: exits first (they reduce risk), then entries, in pair order.
Runs without a forced stop-loss: the crossover is its exit rule.
"""

from __future__ import annotations

from decimal import Decimal

from ai_trader.ai.agent import AgentResult
from ai_trader.ai.schema import TradeProposal
from ai_trader.config import RiskSettings
from ai_trader.data.snapshot import MarketSnapshot
from ai_trader.strategies.base import buy, invested_at_cost_pct, result, sell_all

TOLERANCE_PCT = Decimal("0.5")


class MACrossover:
    name = "ma_crossover"

    def __init__(self, risk: RiskSettings, fallback_pair: str) -> None:
        self._target = risk.max_position_pct_per_pair
        self._step = risk.max_trade_pct_of_equity
        self._fallback_pair = fallback_pair

    async def decide(self, snapshot: MarketSnapshot) -> AgentResult:
        ready = [p for p in snapshot.pairs if p.sma_20d is not None and p.sma_50d is not None]

        for p in ready:
            if p.sma_20d < p.sma_50d and p.position_amount > 0:
                return result(
                    self.name,
                    sell_all(p.pair, f"SMA20 {p.sma_20d} < SMA50 {p.sma_50d}: exit"),
                )
        for p in ready:
            if p.sma_20d > p.sma_50d:
                missing = self._target - invested_at_cost_pct(snapshot, p.pair)
                if missing > TOLERANCE_PCT:
                    return result(
                        self.name,
                        buy(
                            p.pair,
                            min(self._step, missing),
                            f"SMA20 {p.sma_20d} > SMA50 {p.sma_50d}: uptrend",
                        ),
                    )

        reason = "no signal" if ready else "not enough history for SMA 20/50"
        return result(self.name, TradeProposal.hold(self._fallback_pair, reason))
