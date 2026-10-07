"""Benchmark strategies sharing the agent interface."""

from __future__ import annotations

from ai_trader.ai.agent import DecisionMaker
from ai_trader.config import TradingSettings
from ai_trader.strategies.buy_and_hold import BuyAndHold
from ai_trader.strategies.do_nothing import DoNothing
from ai_trader.strategies.ma_crossover import MACrossover


def build_benchmark(name: str, settings: TradingSettings) -> DecisionMaker:
    first = settings.pairs[0]
    if name == "buy_and_hold":
        return BuyAndHold(first, settings.risk)
    if name == "ma_crossover":
        return MACrossover(settings.risk, first)
    if name == "do_nothing":
        return DoNothing(first)
    raise ValueError(f"unknown benchmark {name!r}")


__all__ = ["BuyAndHold", "DoNothing", "MACrossover", "build_benchmark"]
