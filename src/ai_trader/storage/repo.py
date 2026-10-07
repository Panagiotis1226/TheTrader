"""Read/write helpers. One short transaction per call."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

from sqlalchemy import Engine, create_engine, event, func, or_, select, update
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

from ai_trader.brokers.base import Fill, Order, OrderStatus, OrderType, Side
from ai_trader.storage.models import (
    AccountRow,
    Base,
    DecisionRow,
    EquitySnapshotRow,
    FillRow,
    HaltRow,
    OrderRow,
)


class HaltKind(StrEnum):
    MANUAL = "manual"  # kill switch / Telegram /stop — until /resume
    DRAWDOWN = "drawdown"  # max drawdown breached — until /resume
    DAILY_LOSS = "daily_loss"  # daily loss limit — until next UTC midnight
    ERRORS = "errors"  # repeated cycle failures — until /resume


# Automatic risk-limit halts block new buys but still allow sells (reducing exposure).
# Manual (/stop) and error halts stop all trading.
REDUCE_ONLY_HALTS = frozenset({HaltKind.DRAWDOWN, HaltKind.DAILY_LOSS})


@dataclass(frozen=True)
class AccountRecord:
    id: str
    kind: str
    quote_currency: str
    starting_cash: Decimal
    created_at: datetime


@dataclass(frozen=True)
class HaltRecord:
    id: int
    account_id: str | None
    kind: HaltKind
    reason: str
    created_at: datetime
    until: datetime | None


@dataclass(frozen=True)
class DecisionRecord:
    id: int
    created_at: datetime
    action: str | None
    pair: str | None
    size_pct: Decimal | None
    risk_outcome: str | None
    risk_reason: str | None
    fill_price: Decimal | None


def make_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database not in (None, "", ":memory:"):
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url)
    if url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def _order_from_row(row: OrderRow) -> Order:
    return Order(
        id=row.id,
        account_id=row.account_id,
        pair=row.pair,
        side=Side(row.side),
        type=OrderType(row.type),
        amount=row.amount,
        status=OrderStatus(row.status),
        created_at=row.created_at,
        trigger_price=row.trigger_price,
        reason=row.reason,
        decision_id=row.decision_id,
    )


def _fill_from_row(row: FillRow) -> Fill:
    return Fill(
        order_id=row.order_id,
        account_id=row.account_id,
        pair=row.pair,
        side=Side(row.side),
        amount=row.amount,
        price=row.price,
        cost=row.cost,
        fee=row.fee,
        fee_currency=row.fee_currency,
        created_at=row.created_at,
        order_type=OrderType(row.order_type),
    )


def _halt_from_row(row: HaltRow) -> HaltRecord:
    return HaltRecord(
        id=row.id,
        account_id=row.account_id,
        kind=HaltKind(row.kind),
        reason=row.reason,
        created_at=row.created_at,
        until=row.until,
    )


class Repository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, database_url: str) -> Repository:
        engine = make_engine(database_url)
        init_db(engine)
        return cls(engine)

    # ------------------------------------------------------------------ accounts

    def ensure_account(
        self,
        account_id: str,
        *,
        kind: str,
        quote_currency: str,
        starting_cash: Decimal,
        now: datetime,
    ) -> AccountRecord:
        """Create the account if missing. An existing account keeps its original values."""
        with self._sessions.begin() as s:
            row = s.get(AccountRow, account_id)
            if row is None:
                row = AccountRow(
                    id=account_id,
                    kind=kind,
                    quote_currency=quote_currency,
                    starting_cash=starting_cash,
                    created_at=now,
                )
                s.add(row)
            return AccountRecord(
                row.id, row.kind, row.quote_currency, row.starting_cash, row.created_at
            )

    def list_accounts(self) -> list[AccountRecord]:
        with self._sessions() as s:
            rows = s.scalars(select(AccountRow).order_by(AccountRow.id))
            return [
                AccountRecord(r.id, r.kind, r.quote_currency, r.starting_cash, r.created_at)
                for r in rows
            ]

    # -------------------------------------------------------------- orders/fills

    def save_order(self, order: Order, now: datetime) -> None:
        with self._sessions.begin() as s:
            s.merge(self._order_row(order, now))

    def record_fill(self, order: Order, fill: Fill, now: datetime) -> None:
        """Persist the (filled) order and its fill in one transaction."""
        with self._sessions.begin() as s:
            s.merge(self._order_row(order, now))
            s.flush()
            s.add(
                FillRow(
                    order_id=fill.order_id,
                    account_id=fill.account_id,
                    pair=fill.pair,
                    side=fill.side.value,
                    order_type=fill.order_type.value,
                    amount=fill.amount,
                    price=fill.price,
                    cost=fill.cost,
                    fee=fill.fee,
                    fee_currency=fill.fee_currency,
                    created_at=fill.created_at,
                )
            )

    def set_order_status(
        self, order_ids: list[str], status: OrderStatus, now: datetime, reason: str | None = None
    ) -> None:
        if not order_ids:
            return
        with self._sessions.begin() as s:
            s.execute(
                update(OrderRow)
                .where(OrderRow.id.in_(order_ids))
                .values(status=status.value, reason=reason, updated_at=now)
            )

    def open_orders(self, account_id: str) -> list[Order]:
        with self._sessions() as s:
            rows = s.scalars(
                select(OrderRow)
                .where(OrderRow.account_id == account_id, OrderRow.status == OrderStatus.OPEN)
                .order_by(OrderRow.created_at)
            )
            return [_order_from_row(r) for r in rows]

    def list_fills(self, account_id: str | None = None) -> list[Fill]:
        with self._sessions() as s:
            q = select(FillRow).order_by(FillRow.created_at, FillRow.id)
            if account_id is not None:
                q = q.where(FillRow.account_id == account_id)
            return [_fill_from_row(r) for r in s.scalars(q)]

    def trade_stats(self, account_id: str, since: datetime) -> tuple[int, datetime | None]:
        """(decision-driven fills since ``since``, time of the latest one ever).

        Stop-loss fills are excluded: they are protective, not trading decisions.
        """
        market = OrderType.MARKET.value
        with self._sessions() as s:
            count = s.scalar(
                select(func.count(FillRow.id)).where(
                    FillRow.account_id == account_id,
                    FillRow.order_type == market,
                    FillRow.created_at >= since,
                )
            )
            last = s.scalar(
                select(FillRow.created_at)
                .where(FillRow.account_id == account_id, FillRow.order_type == market)
                .order_by(FillRow.created_at.desc())
                .limit(1)
            )
            return int(count or 0), last

    @staticmethod
    def _order_row(order: Order, now: datetime) -> OrderRow:
        return OrderRow(
            id=order.id,
            account_id=order.account_id,
            decision_id=order.decision_id,
            pair=order.pair,
            side=order.side.value,
            type=order.type.value,
            amount=order.amount,
            trigger_price=order.trigger_price,
            status=order.status.value,
            reason=order.reason,
            created_at=order.created_at,
            updated_at=now,
        )

    # ----------------------------------------------------------------- decisions

    def record_decision(self, account_id: str, created_at: datetime, **fields: Any) -> int:
        with self._sessions.begin() as s:
            row = DecisionRow(account_id=account_id, created_at=created_at, **fields)
            s.add(row)
            s.flush()
            return row.id

    def update_decision(self, decision_id: int, **fields: Any) -> None:
        with self._sessions.begin() as s:
            s.execute(update(DecisionRow).where(DecisionRow.id == decision_id).values(**fields))

    def list_decisions(self, limit: int = 500) -> list[dict[str, Any]]:
        """Most recent decisions (all accounts), every logged column."""
        with self._sessions() as s:
            rows = s.scalars(
                select(DecisionRow).order_by(DecisionRow.created_at.desc()).limit(limit)
            ).all()
            return [{c.key: getattr(r, c.key) for c in DecisionRow.__table__.columns} for r in rows]

    def decision_details(self, decision_id: int) -> dict[str, Any] | None:
        """Every logged column of one decision (for audits and the dashboard)."""
        with self._sessions() as s:
            row = s.get(DecisionRow, decision_id)
            if row is None:
                return None
            return {c.key: getattr(row, c.key) for c in DecisionRow.__table__.columns}

    def decision_stats(self, account_id: str) -> dict[str, Any]:
        """Decision counts: trade proposals by risk outcome, unusable outputs, LLM cost."""
        with self._sessions() as s:
            rows = s.execute(
                select(
                    DecisionRow.action,
                    DecisionRow.risk_outcome,
                    DecisionRow.cost_usd,
                    DecisionRow.error,
                ).where(DecisionRow.account_id == account_id)
            ).all()
        stats: dict[str, Any] = {
            "decisions": len(rows),
            "trade_proposals": 0,
            "rejected": 0,
            "resized": 0,
            "errors": 0,
            "llm_cost_usd": Decimal(0),
        }
        for action, outcome, cost, error in rows:
            if error:
                stats["errors"] += 1
            if action in ("buy", "sell"):
                stats["trade_proposals"] += 1
                if outcome == "reject":
                    stats["rejected"] += 1
                elif outcome == "resize":
                    stats["resized"] += 1
            if cost is not None:
                stats["llm_cost_usd"] += cost
        return stats

    def equity_series(self, account_id: str) -> list[tuple[datetime, Decimal]]:
        with self._sessions() as s:
            rows = s.execute(
                select(EquitySnapshotRow.ts, EquitySnapshotRow.equity)
                .where(EquitySnapshotRow.account_id == account_id)
                .order_by(EquitySnapshotRow.ts, EquitySnapshotRow.id)
            ).all()
            return [(ts, eq) for ts, eq in rows]

    def count_decisions_since(self, account_id: str, since: datetime) -> int:
        with self._sessions() as s:
            count = s.scalar(
                select(func.count(DecisionRow.id)).where(
                    DecisionRow.account_id == account_id, DecisionRow.created_at >= since
                )
            )
            return int(count or 0)

    def llm_cost_since(self, account_id: str, since: datetime) -> Decimal:
        """Total LLM cost (USD) of this account's decisions since ``since``."""
        with self._sessions() as s:
            costs = s.scalars(
                select(DecisionRow.cost_usd).where(
                    DecisionRow.account_id == account_id,
                    DecisionRow.created_at >= since,
                    DecisionRow.cost_usd.is_not(None),
                )
            )
            return sum(costs, Decimal(0))

    def recent_decisions(self, account_id: str, limit: int = 5) -> list[DecisionRecord]:
        """Most recent first, with the average fill price when the decision traded."""
        with self._sessions() as s:
            rows = s.execute(
                select(DecisionRow, FillRow.price)
                .outerjoin(FillRow, FillRow.order_id == DecisionRow.order_id)
                .where(DecisionRow.account_id == account_id)
                .order_by(DecisionRow.created_at.desc(), DecisionRow.id.desc())
                .limit(limit)
            ).all()
            return [
                DecisionRecord(
                    id=d.id,
                    created_at=d.created_at,
                    action=d.action,
                    pair=d.pair,
                    size_pct=d.size_pct,
                    risk_outcome=d.risk_outcome,
                    risk_reason=d.risk_reason,
                    fill_price=price,
                )
                for d, price in rows
            ]

    # -------------------------------------------------------------------- equity

    def record_equity(self, account_id: str, ts: datetime, equity: Decimal, cash: Decimal) -> None:
        with self._sessions.begin() as s:
            s.add(EquitySnapshotRow(account_id=account_id, ts=ts, equity=equity, cash=cash))

    def equity_at_or_before(self, account_id: str, ts: datetime) -> Decimal | None:
        with self._sessions() as s:
            return s.scalar(
                select(EquitySnapshotRow.equity)
                .where(EquitySnapshotRow.account_id == account_id, EquitySnapshotRow.ts <= ts)
                .order_by(EquitySnapshotRow.ts.desc(), EquitySnapshotRow.id.desc())
                .limit(1)
            )

    def first_equity_since(self, account_id: str, ts: datetime) -> Decimal | None:
        with self._sessions() as s:
            return s.scalar(
                select(EquitySnapshotRow.equity)
                .where(EquitySnapshotRow.account_id == account_id, EquitySnapshotRow.ts >= ts)
                .order_by(EquitySnapshotRow.ts, EquitySnapshotRow.id)
                .limit(1)
            )

    def peak_equity_since(self, account_id: str, since: datetime | None) -> Decimal | None:
        # Equity is stored as text, so take the max in Python rather than in SQL.
        with self._sessions() as s:
            q = select(EquitySnapshotRow.equity).where(EquitySnapshotRow.account_id == account_id)
            if since is not None:
                q = q.where(EquitySnapshotRow.ts >= since)
            values = list(s.scalars(q))
            return max(values) if values else None

    # --------------------------------------------------------------------- halts

    def add_halt(
        self,
        kind: HaltKind,
        reason: str,
        now: datetime,
        *,
        account_id: str | None = None,
        until: datetime | None = None,
    ) -> int:
        with self._sessions.begin() as s:
            row = HaltRow(
                account_id=account_id, kind=kind.value, reason=reason, created_at=now, until=until
            )
            s.add(row)
            s.flush()
            return row.id

    def active_halts(self, account_id: str | None, now: datetime) -> list[HaltRecord]:
        """Halts in force for ``account_id`` (including global ones)."""
        with self._sessions() as s:
            scope = (
                HaltRow.account_id.is_(None)
                if account_id is None
                else or_(HaltRow.account_id.is_(None), HaltRow.account_id == account_id)
            )
            rows = s.scalars(
                select(HaltRow)
                .where(
                    scope,
                    HaltRow.resumed_at.is_(None),
                    or_(HaltRow.until.is_(None), HaltRow.until > now),
                )
                .order_by(HaltRow.created_at)
            )
            return [_halt_from_row(r) for r in rows]

    def resume(self, now: datetime, account_id: str | None = None) -> int:
        """Lift active halts: all of them, or only those scoped to ``account_id``."""
        with self._sessions.begin() as s:
            q = update(HaltRow).where(HaltRow.resumed_at.is_(None))
            if account_id is not None:
                q = q.where(HaltRow.account_id == account_id)
            result = s.execute(q.values(resumed_at=now))
            return int(result.rowcount or 0)

    def last_resume(self, account_id: str, kind: HaltKind) -> datetime | None:
        with self._sessions() as s:
            return s.scalar(
                select(func.max(HaltRow.resumed_at)).where(
                    HaltRow.kind == kind.value,
                    or_(HaltRow.account_id.is_(None), HaltRow.account_id == account_id),
                )
            )

    # ----------------------------------------------------------------------- tax

    def export_fills_csv(self, out: IO[str], account_id: str | None = None) -> int:
        """Write fills as CSV for CRA / Revenu Québec records. Returns the row count."""
        writer = csv.writer(out)
        writer.writerow(
            [
                "datetime_utc",
                "account",
                "pair",
                "side",
                "quantity",
                "price",
                "fee",
                "fee_currency",
                "gross_value",
                "net_value",
                "order_type",
            ]
        )
        fills = self.list_fills(account_id)
        for f in fills:
            net = f.cost + f.fee if f.side is Side.BUY else f.cost - f.fee
            writer.writerow(
                [
                    f.created_at.isoformat(),
                    f.account_id,
                    f.pair,
                    f.side.value,
                    format(f.amount, "f"),
                    format(f.price, "f"),
                    format(f.fee, "f"),
                    f.fee_currency,
                    format(f.cost, "f"),
                    format(net, "f"),
                    f.order_type.value,
                ]
            )
        return len(fills)
