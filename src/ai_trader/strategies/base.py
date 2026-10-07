"""Shared helpers for rule-based benchmarks.

Benchmarks implement the same ``DecisionMaker`` interface as the LLM agents: snapshot in,
``TradeProposal`` out, through the same RiskManager and broker. They are stateless, so a
restart can't change their behavior: everything is derived from the snapshot.
"""

from __future__ import annotations

from decimal import Decimal

from ai_trader.ai.agent import AgentResult
from ai_trader.ai.schema import TradeProposal
from ai_trader.data.snapshot import MarketSnapshot, PairSnapshot

HUNDRED = Decimal(100)
ONE = Decimal(1)


def cost_basis(p: PairSnapshot) -> Decimal:
    return p.position_amount * (p.avg_entry_price or Decimal(0))


def invested_at_cost_pct(snapshot: MarketSnapshot, pair: str) -> Decimal:
    """Capital put into ``pair`` at cost, as % of (cash + all positions at cost).

    Unlike market value this doesn't move with price, so "top up to 30%" never turns into
    buying dips or trimming rallies.
    """
    total = snapshot.cash_available + sum((cost_basis(p) for p in snapshot.pairs), Decimal(0))
    if total <= 0:
        return Decimal(0)
    target = next((p for p in snapshot.pairs if p.pair == pair), None)
    return cost_basis(target) / total * HUNDRED if target else Decimal(0)


def result(name: str, proposal: TradeProposal) -> AgentResult:
    return AgentResult(proposal=proposal, model=name)


def buy(pair: str, size_pct: Decimal, reason: str) -> TradeProposal:
    return TradeProposal(action="buy", pair=pair, size_pct=size_pct, confidence=ONE, reason=reason)


def sell_all(pair: str, reason: str) -> TradeProposal:
    return TradeProposal(action="sell", pair=pair, size_pct=HUNDRED, confidence=ONE, reason=reason)
