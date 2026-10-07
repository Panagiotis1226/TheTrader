"""SQLAlchemy tables. SQLite now, Postgres-ready.

Decimals are stored as strings: SQLite has no exact numeric type and SQLAlchemy's
``Numeric`` would round-trip through float. Datetimes are stored as UTC and always
come back timezone-aware.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class DecimalString(TypeDecorator[Decimal]):
    impl = String(64)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if not isinstance(value, Decimal):
            raise TypeError(f"expected Decimal, got {type(value).__name__}")
        if not value.is_finite():
            raise ValueError(f"refusing to store non-finite Decimal {value}")
        return format(value, "f")

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        return None if value is None else Decimal(value)


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; use timezone-aware UTC")
        value = value.astimezone(UTC)
        # SQLite drops tzinfo anyway; store naive UTC so string comparisons line up.
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class AccountRow(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # paper | live
    quote_currency: Mapped[str] = mapped_column(String(8))
    starting_cash: Mapped[Decimal] = mapped_column(DecimalString)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class DecisionRow(Base):
    __tablename__ = "decisions"
    __table_args__ = (Index("ix_decisions_account_created", "account_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    snapshot_hash: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128))  # litellm model or strategy name
    raw_response: Mapped[str | None] = mapped_column(Text)
    proposal: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    action: Mapped[str | None] = mapped_column(String(8))
    pair: Mapped[str | None] = mapped_column(String(20))
    size_pct: Mapped[Decimal | None] = mapped_column(DecimalString)
    risk_outcome: Mapped[str | None] = mapped_column(String(16))
    risk_reason: Mapped[str | None] = mapped_column(Text)
    order_id: Mapped[str | None] = mapped_column(String(32))
    # LLM telemetry (populated in Phase 2)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(DecimalString)
    error: Mapped[str | None] = mapped_column(Text)


class OrderRow(Base):
    __tablename__ = "orders"
    __table_args__ = (Index("ix_orders_account_status", "account_id", "status"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    decision_id: Mapped[int | None] = mapped_column(Integer)
    pair: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(4))
    type: Mapped[str] = mapped_column(String(16))
    amount: Mapped[Decimal] = mapped_column(DecimalString)
    trigger_price: Mapped[Decimal | None] = mapped_column(DecimalString)
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)


class FillRow(Base):
    __tablename__ = "fills"
    __table_args__ = (Index("ix_fills_account_created", "account_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    pair: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(16))
    amount: Mapped[Decimal] = mapped_column(DecimalString)
    price: Mapped[Decimal] = mapped_column(DecimalString)
    cost: Mapped[Decimal] = mapped_column(DecimalString)
    fee: Mapped[Decimal] = mapped_column(DecimalString)
    fee_currency: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class EquitySnapshotRow(Base):
    __tablename__ = "equity_snapshots"
    __table_args__ = (Index("ix_equity_account_ts", "account_id", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    ts: Mapped[datetime] = mapped_column(UTCDateTime)
    equity: Mapped[Decimal] = mapped_column(DecimalString)
    cash: Mapped[Decimal] = mapped_column(DecimalString)


class HaltRow(Base):
    __tablename__ = "halts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str | None] = mapped_column(String(64))  # NULL = all accounts
    kind: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    until: Mapped[datetime | None] = mapped_column(UTCDateTime)  # NULL = until /resume
    resumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
