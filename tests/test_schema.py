from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from ai_trader.ai.schema import TradeProposal

VALID = dict(action="buy", pair="btc/cad", size_pct="5", confidence="0.7", reason="trend")


def test_valid_proposal_normalizes_pair() -> None:
    p = TradeProposal.model_validate(VALID)
    assert p.pair == "BTC/CAD"
    assert p.size_pct == Decimal(5)
    assert p.stop_loss_pct is None


@pytest.mark.parametrize(
    "override",
    [
        {"action": "short"},
        {"size_pct": "101"},
        {"size_pct": "-1"},
        {"confidence": "1.01"},
        {"confidence": "-0.1"},
        {"stop_loss_pct": "0"},
        {"stop_loss_pct": "100"},
        {"reason": "x" * 501},
        {"leverage": 5},  # unknown fields are rejected
    ],
)
def test_invalid_proposals_rejected(override) -> None:
    with pytest.raises(ValidationError):
        TradeProposal.model_validate({**VALID, **override})


def test_hold_factory() -> None:
    p = TradeProposal.hold("BTC/CAD", "x" * 600)
    assert p.action == "hold"
    assert p.size_pct == 0
    assert len(p.reason) == 500
