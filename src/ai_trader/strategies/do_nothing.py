"""Benchmark: never trade. Equity stays at starting cash."""

from __future__ import annotations

from ai_trader.ai.agent import AgentResult
from ai_trader.ai.schema import TradeProposal
from ai_trader.data.snapshot import MarketSnapshot
from ai_trader.strategies.base import result


class DoNothing:
    name = "do_nothing"

    def __init__(self, pair: str) -> None:
        self._pair = pair

    async def decide(self, snapshot: MarketSnapshot) -> AgentResult:
        return result(self.name, TradeProposal.hold(self._pair, "benchmark: never trades"))
