"""Performance reports per account: daily summary, weekly report, dashboard.

Metrics follow PLAN.md Phase 5: return after fees and slippage, max drawdown, Sharpe,
trades and fees, win rate, risk rejection rate, LLM cost vs P&L. Sharpe and drawdown use
the equity snapshots resampled to one value per UTC day (the last of each day).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from ai_trader.backtest import metrics
from ai_trader.brokers.base import OrderType, Position
from ai_trader.storage.repo import Repository


@dataclass(frozen=True)
class AccountReport:
    account_id: str
    starting_cash: Decimal
    equity: Decimal
    return_pct: float
    max_drawdown_pct: float
    sharpe: float | None
    trades: int
    stop_loss_fills: int
    fees: Decimal
    sells: int
    win_rate_pct: float | None
    decisions: int
    trade_proposals: int
    rejection_rate_pct: float | None
    unusable_outputs: int
    llm_cost_usd: Decimal
    positions: list[Position]
    halts: list[str]


def daily_closes(
    series: Sequence[tuple[datetime, Decimal]],
) -> list[Decimal]:
    """Last equity value of each UTC day."""
    by_day: dict[str, Decimal] = {}
    for ts, equity in series:
        by_day[ts.date().isoformat()] = equity
    return [by_day[d] for d in sorted(by_day)]


def account_report(
    repo: Repository,
    account_id: str,
    equity: Decimal,
    positions: list[Position],
    now: datetime,
) -> AccountReport:
    account = next(a for a in repo.list_accounts() if a.id == account_id)
    start = account.starting_cash
    curve = [start, *daily_closes(repo.equity_series(account_id)), equity]
    fills = repo.list_fills(account_id)
    sells, win_rate = metrics.win_rate_pct(fills)
    stats = repo.decision_stats(account_id)
    proposals = stats["trade_proposals"]
    return AccountReport(
        account_id=account_id,
        starting_cash=start,
        equity=equity,
        return_pct=round(metrics.total_return_pct([start, equity]), 2),
        max_drawdown_pct=round(metrics.max_drawdown_pct(curve), 2),
        sharpe=None if (s := metrics.sharpe(curve)) is None else round(s, 2),
        trades=len(fills),
        stop_loss_fills=sum(1 for f in fills if f.order_type is OrderType.STOP_LOSS),
        fees=sum((f.fee for f in fills), Decimal(0)),
        sells=sells,
        win_rate_pct=None if win_rate is None else round(win_rate, 1),
        decisions=stats["decisions"],
        trade_proposals=proposals,
        rejection_rate_pct=round(stats["rejected"] / proposals * 100, 1) if proposals else None,
        unusable_outputs=stats["errors"],
        llm_cost_usd=stats["llm_cost_usd"],
        positions=positions,
        halts=[f"{h.kind.value}: {h.reason}" for h in repo.active_halts(account_id, now)],
    )


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:+.2f}%"


def format_daily_summary(reports: Sequence[AccountReport], now: datetime) -> str:
    lines = [f"Daily summary {now:%Y-%m-%d} (paper)"]
    for r in sorted(reports, key=lambda r: r.equity, reverse=True):
        pos = ", ".join(f"{p.amount.normalize()} {p.pair.split('/')[0]}" for p in r.positions)
        lines.append(
            f"{r.account_id}: {r.equity:,.2f} ({_pct(r.return_pct)}), "
            f"DD {r.max_drawdown_pct:.1f}%, {r.trades} trades"
            + (f", holding {pos}" if pos else "")
            + (f" [HALTED: {'; '.join(r.halts)}]" if r.halts else "")
        )
    return "\n".join(lines)


def format_weekly_report(reports: Sequence[AccountReport], now: datetime) -> str:
    lines = [f"Weekly report {now:%Y-%m-%d} (paper, after fees and slippage)"]
    for r in sorted(reports, key=lambda r: r.equity, reverse=True):
        lines.append(
            f"\n{r.account_id}\n"
            f"  equity {r.equity:,.2f} ({_pct(r.return_pct)}), max DD {r.max_drawdown_pct:.1f}%, "
            f"Sharpe {r.sharpe if r.sharpe is not None else '-'}\n"
            f"  trades {r.trades} (stops {r.stop_loss_fills}), fees {r.fees:,.2f}, "
            f"win rate {r.win_rate_pct if r.win_rate_pct is not None else '-'}%\n"
            f"  proposals {r.trade_proposals}, rejected "
            f"{r.rejection_rate_pct if r.rejection_rate_pct is not None else '-'}%, "
            f"unusable outputs {r.unusable_outputs}\n"
            f"  LLM cost ${r.llm_cost_usd:.2f} vs P&L {r.equity - r.starting_cash:+,.2f} CAD"
        )
    return "\n".join(lines)
