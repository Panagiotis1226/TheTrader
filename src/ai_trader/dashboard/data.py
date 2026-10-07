"""Read-only data access for the dashboard. Returns pandas DataFrames."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from ai_trader.storage.repo import Repository


def account_order(repo: Repository, preferred: list[str]) -> list[str]:
    """Stable account order (configured order first) so colors follow the account."""
    existing = [a.id for a in repo.list_accounts()]
    ordered = [a for a in preferred if a in existing]
    return ordered + sorted(a for a in existing if a not in ordered)


def equity_frame(repo: Repository) -> pd.DataFrame:
    """Equity snapshots with return % since each account's starting cash."""
    rows = []
    for account in repo.list_accounts():
        start = account.starting_cash
        rows.append((account.id, account.created_at, float(start), 0.0))
        for ts, equity in repo.equity_series(account.id):
            rows.append((account.id, ts, float(equity), float((equity / start - 1) * 100)))
    return pd.DataFrame(rows, columns=["account", "time", "equity", "return_pct"])


def latest_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return equity
    last = equity.sort_values("time").groupby("account", as_index=False).last()
    return last[["account", "equity", "return_pct", "time"]]


def fills_frame(repo: Repository) -> pd.DataFrame:
    rows = [
        {
            "time": f.created_at,
            "account": f.account_id,
            "pair": f.pair,
            "side": f.side.value,
            "type": f.order_type.value,
            "amount": float(f.amount),
            "price": float(f.price),
            "value": float(f.cost),
            "fee": float(f.fee),
        }
        for f in repo.list_fills()
    ]
    columns = ["time", "account", "pair", "side", "type", "amount", "price", "value", "fee"]
    return pd.DataFrame(rows, columns=columns).sort_values("time", ascending=False)


def decisions_frame(repo: Repository, limit: int = 500) -> pd.DataFrame:
    records = []
    for d in repo.list_decisions(limit):
        proposal = d["proposal"] or {}
        records.append(
            {
                "time": d["created_at"],
                "account": d["account_id"],
                "model": d["model"],
                "action": d["action"],
                "pair": d["pair"],
                "size_pct": _float(d["size_pct"]),
                "confidence": proposal.get("confidence"),
                "reason": proposal.get("reason"),
                "risk": d["risk_outcome"],
                "risk_reason": d["risk_reason"],
                "error": d["error"],
                "cost_usd": _float(d["cost_usd"]),
                "snapshot_hash": (d["snapshot_hash"] or "")[:12],
                "prompt_hash": d["prompt_hash"],
            }
        )
    columns = [
        "time", "account", "model", "action", "pair", "size_pct", "confidence", "reason",
        "risk", "risk_reason", "error", "cost_usd", "snapshot_hash", "prompt_hash",
    ]  # fmt: skip
    return pd.DataFrame(records, columns=columns)


def rejections_frame(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions[decisions["risk"] == "reject"]


def llm_cost_by_day(decisions: pd.DataFrame) -> pd.DataFrame:
    costs = decisions.dropna(subset=["cost_usd"])
    if costs.empty:
        return pd.DataFrame(columns=["day", "account", "cost_usd"])
    costs = costs.assign(day=pd.to_datetime(costs["time"]).dt.strftime("%Y-%m-%d"))
    return costs.groupby(["day", "account"], as_index=False)["cost_usd"].sum()


def _float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)
