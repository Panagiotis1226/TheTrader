"""Benchmark: buy one pair up to the per-pair cap, then hold forever.

Buys ``max_trade_pct_of_equity`` per decision until the capital invested (at cost) reaches
``max_position_pct_per_pair``, then never trades again. It never sells and never
rebalances. Runs without a forced stop-loss.
"""

from __future__ import annotations

from decimal import Decimal

from ai_trader.ai.agent import AgentResult
from ai_trader.ai.schema import TradeProposal
from ai_trader.config import RiskSettings
from ai_trader.data.snapshot import MarketSnapshot
from ai_trader.strategies.base import buy, invested_at_cost_pct, result

# Stop topping up within half a percent of the target (fees/rounding leave a sliver).
TOLERANCE_PCT = Decimal("0.5")


class BuyAndHold:
    name = "buy_and_hold"

    def __init__(self, pair: str, risk: RiskSettings) -> None:
        self._pair = pair
        self._target = risk.max_position_pct_per_pair
        self._step = risk.max_trade_pct_of_equity

    async def decide(self, snapshot: MarketSnapshot) -> AgentResult:
        invested = invested_at_cost_pct(snapshot, self._pair)
        missing = self._target - invested
        if missing <= TOLERANCE_PCT:
            return result(self.name, TradeProposal.hold(self._pair, "fully invested; holding"))
        size = min(self._step, missing)
        return result(
            self.name,
            buy(self._pair, size, f"accumulating: {invested:.1f}% of {self._target}% invested"),
        )
