"""The only thing an LLM (or benchmark strategy) may produce: a trade *proposal*.

A proposal never places an order. It goes through the RiskManager first
(Safety Invariant #2). Parsing raw LLM text into this model is added in Phase 2.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_REASON_CHARS = 500


class TradeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["buy", "sell", "hold"]
    pair: Annotated[str, Field(min_length=3, max_length=20)]
    # % of total equity (buy) or of the current position (sell).
    size_pct: Annotated[Decimal, Field(ge=0, le=100)]
    confidence: Annotated[Decimal, Field(ge=0, le=1)]
    reason: Annotated[str, Field(max_length=MAX_REASON_CHARS)]
    # % below entry. The RiskManager only lets this tighten the configured default.
    stop_loss_pct: Annotated[Decimal, Field(gt=0, lt=100)] | None = None

    @field_validator("pair", mode="before")
    @classmethod
    def _normalize_pair(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @classmethod
    def hold(cls, pair: str, reason: str) -> TradeProposal:
        return cls(
            action="hold",
            pair=pair,
            size_pct=Decimal(0),
            confidence=Decimal(0),
            reason=reason[:MAX_REASON_CHARS],
        )
