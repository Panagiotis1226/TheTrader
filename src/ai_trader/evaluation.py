"""Phase 5: the paper-trading evaluation and the go-live scorecard (PLAN.md sections 7-8).

**Evaluation window.** Every decision stores ``prompt_hash``, a fingerprint of the rendered
system prompt, which includes the risk limits, fees, pairs and interval. The window starts
at the first decision of the evaluated account in the current unbroken run of the current
fingerprint. Changing the prompt or any risk/fee setting therefore restarts the clock
automatically, as the plan requires. All accounts are measured over the same window.

**Scorecard.** Each criterion is PASS, FAIL, PENDING (not enough time/trades yet) or CHECK
(needs your judgement). Going live is never automatic: Phase 6 needs your decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from ai_trader.ai.prompt import prompt_hash, render_system_prompt
from ai_trader.backtest import metrics
from ai_trader.brokers.base import OrderType, Side
from ai_trader.config import TradingSettings
from ai_trader.reports import daily_closes
from ai_trader.storage.repo import HaltKind, Repository

EPS = Decimal("0.00000001")


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    CHECK = "CHECK"


@dataclass(frozen=True)
class Criterion:
    name: str
    verdict: Verdict
    detail: str


@dataclass(frozen=True)
class AccountResult:
    account_id: str
    start_equity: Decimal
    equity: Decimal
    return_pct: float
    max_drawdown_pct: float
    trades: int


@dataclass
class Scorecard:
    fingerprint: str
    start: datetime | None
    now: datetime
    weeks: float
    accounts: dict[str, AccountResult] = field(default_factory=dict)
    criteria: list[Criterion] = field(default_factory=list)
    integrity_issues: list[str] = field(default_factory=list)
    llm_cost_usd: Decimal = Decimal(0)
    models_seen: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """All automatic criteria pass (manual CHECK items still need you)."""
        return bool(self.criteria) and all(
            c.verdict in (Verdict.PASS, Verdict.CHECK) for c in self.criteria
        )


def current_fingerprint(settings: TradingSettings) -> str:
    return prompt_hash(render_system_prompt(settings))


def config_change_notice(settings: TradingSettings, repo: Repository) -> str | None:
    """A warning if the configuration differs from the one used by the last decision."""
    llm_id = f"paper-{settings.evaluation.llm_account}"
    decisions = repo.account_decisions(llm_id)
    last = next((d["prompt_hash"] for d in reversed(decisions) if d["prompt_hash"]), None)
    current = current_fingerprint(settings)
    if last is None or last == current:
        return None
    return (
        f"Configuration changed (prompt, risk or fee settings: {last} -> {current}). "
        "The Phase 5 evaluation clock restarts with the next decision."
    )


def reached_model(decision: dict) -> bool:
    """The model actually answered (vs. auth/CLI/network failures that never reached it)."""
    return (decision.get("completion_tokens") or 0) > 0


def window_start(repo: Repository, account_id: str, fingerprint: str) -> datetime | None:
    """Start of the evaluation: the first real model answer in the latest unbroken run of
    ``fingerprint``. None if not started (e.g. the model was never reachable)."""
    run = []
    for d in reversed(repo.account_decisions(account_id)):
        if d["prompt_hash"] != fingerprint:
            break
        run.append(d)
    answered = [d for d in reversed(run) if reached_model(d)]
    return answered[0]["created_at"] if answered else None


def integrity_check(repo: Repository) -> list[str]:
    """State consistency checks: an empty list means no problems found."""
    issues: list[str] = []
    for account in repo.list_accounts():
        acct = account.id
        cash = account.starting_cash
        held: dict[str, Decimal] = {}
        for f in repo.list_fills(acct):
            if f.side is Side.BUY:
                cash -= f.cost + f.fee
                held[f.pair] = held.get(f.pair, Decimal(0)) + f.amount
            else:
                cash += f.cost - f.fee
                held[f.pair] = held.get(f.pair, Decimal(0)) - f.amount
            if cash < -EPS:
                issues.append(
                    f"{acct}: cash went negative ({cash}) at {f.created_at:%Y-%m-%d %H:%M}"
                )
            if held[f.pair] < -EPS:
                issues.append(f"{acct}: {f.pair} position went negative at {f.created_at:%Y-%m-%d}")
        for order in repo.open_orders(acct):
            if order.type is OrderType.STOP_LOSS and order.amount > held.get(order.pair, 0) + EPS:
                issues.append(
                    f"{acct}: open stop on {order.pair} for {order.amount} exceeds the position"
                )
        order_ids = repo.order_ids(acct)
        for d in repo.account_decisions(acct):
            if d["order_id"] and d["order_id"] not in order_ids:
                issues.append(f"{acct}: decision {d['id']} points to a missing order")
    return issues


def _account_result(
    repo: Repository, account_id: str, start: datetime, equity: Decimal
) -> AccountResult:
    account = next(a for a in repo.list_accounts() if a.id == account_id)
    start_equity = repo.equity_at_or_before(account_id, start) or account.starting_cash
    series = [(ts, e) for ts, e in repo.equity_series(account_id) if ts >= start]
    curve = [start_equity, *daily_closes(series), equity]
    trades = sum(1 for f in repo.list_fills(account_id) if f.created_at >= start)
    return AccountResult(
        account_id=account_id,
        start_equity=start_equity,
        equity=equity,
        return_pct=round(metrics.total_return_pct([start_equity, equity]), 2),
        max_drawdown_pct=round(metrics.max_drawdown_pct(curve), 2),
        trades=trades,
    )


def scorecard(
    settings: TradingSettings,
    repo: Repository,
    equity: Mapping[str, Decimal],
    now: datetime,
) -> Scorecard:
    """``equity``: current equity per account ID (live, or the latest snapshot)."""
    ev = settings.evaluation
    fingerprint = current_fingerprint(settings)
    llm_id = f"paper-{ev.llm_account}"
    start = window_start(repo, llm_id, fingerprint)
    card = Scorecard(fingerprint=fingerprint, start=start, now=now, weeks=0.0)
    card.integrity_issues = integrity_check(repo)
    if start is None:
        card.criteria.append(
            Criterion(
                "Evaluation started",
                Verdict.PENDING,
                f"no answer from the model for {llm_id} with the current configuration yet "
                "(see `ai-trader status` for errors such as a missing token)",
            )
        )
        return card

    card.weeks = round((now - start).total_seconds() / (7 * 86400), 1)
    for account_id, eq in equity.items():
        card.accounts[account_id] = _account_result(repo, account_id, start, eq)
    decisions = [d for d in repo.account_decisions(llm_id) if d["created_at"] >= start]
    card.llm_cost_usd = sum((d["cost_usd"] or Decimal(0) for d in decisions), Decimal(0))
    card.models_seen = sorted({d["model"] for d in decisions if d["model"]})

    llm = card.accounts.get(llm_id)
    bh = card.accounts.get("paper-buy_and_hold")
    ma = card.accounts.get("paper-ma_crossover")
    add = card.criteria.append

    add(Criterion(
        "Duration",
        Verdict.PASS if card.weeks >= ev.min_weeks else Verdict.PENDING,
        f"{card.weeks} of {ev.min_weeks} weeks (since {start:%Y-%m-%d})",
    ))  # fmt: skip
    trades = llm.trades if llm else 0
    add(Criterion(
        "Trades",
        Verdict.PASS if trades >= ev.min_trades else Verdict.PENDING,
        f"{trades} of {ev.min_trades} paper trades",
    ))  # fmt: skip

    if llm is None or bh is None:
        add(Criterion("1. vs buy-and-hold", Verdict.CHECK, "buy_and_hold account not running"))
    else:
        tol = float(ev.match_tolerance_pct)
        dd_target = bh.max_drawdown_pct * (1 - float(ev.drawdown_improvement_pct) / 100)
        beats = llm.return_pct > bh.return_pct
        matches = abs(llm.return_pct - bh.return_pct) <= tol and llm.max_drawdown_pct <= dd_target
        add(Criterion(
            "1. vs buy-and-hold",
            Verdict.PASS if beats or matches else Verdict.FAIL,
            f"return {llm.return_pct:+.2f}% vs {bh.return_pct:+.2f}%; max DD "
            f"{llm.max_drawdown_pct:.2f}% vs {bh.max_drawdown_pct:.2f}% "
            + ("(beats it)" if beats else "(matches it with clearly lower drawdown)" if matches
               else f"(needs a higher return, or within {tol} pt with DD <= {dd_target:.2f}%)"),
        ))  # fmt: skip

    if llm is None or ma is None:
        add(Criterion("2. vs MA crossover", Verdict.CHECK, "ma_crossover account not running"))
    else:
        add(Criterion(
            "2. vs MA crossover",
            Verdict.PASS if llm.return_pct > ma.return_pct else Verdict.FAIL,
            f"return {llm.return_pct:+.2f}% vs {ma.return_pct:+.2f}% (if a simple rule does "
            "as well, the LLM adds no value)",
        ))  # fmt: skip

    error_halts = [h for h in repo.halt_history(start) if h.kind is HaltKind.ERRORS]
    if card.integrity_issues or error_halts:
        problems = card.integrity_issues + [f"error halt: {h.reason}" for h in error_halts]
        add(Criterion("3. No bugs or state mismatches", Verdict.FAIL, "; ".join(problems[:5])))
    else:
        add(Criterion(
            "3. No bugs or state mismatches",
            Verdict.CHECK,
            "automatic checks clean (state replay, stops, error halts); confirm nothing "
            "unexplained happened",
        ))  # fmt: skip

    missed = [d for d in decisions if not reached_model(d)]
    add(Criterion(
        "Model reached every cycle",
        Verdict.PASS if not missed else Verdict.CHECK,
        "yes" if not missed else
        f"{len(missed)} of {len(decisions)} cycles never reached the model (holds by default); "
        f"last: {missed[-1]['error'] or 'unknown'}",
    ))  # fmt: skip

    pnl = (llm.equity - llm.start_equity) if llm else Decimal(0)
    add(Criterion(
        "4. LLM cost vs profit",
        Verdict.CHECK,
        f"${card.llm_cost_usd:.2f} (API-equivalent; on a Claude seat the real cost is the "
        f"subscription) vs P&L {pnl:+,.2f} {settings.quote_currency}",
    ))  # fmt: skip
    if len(card.models_seen) > 1:
        add(
            Criterion(
                "Same model throughout",
                Verdict.FAIL,
                f"the model changed during the window: {', '.join(card.models_seen)}",
            )
        )
    return card


def format_scorecard(card: Scorecard) -> str:
    lines = [f"Go-live scorecard ({card.now:%Y-%m-%d}), configuration {card.fingerprint}"]
    if card.accounts:
        lines.append("")
        for r in sorted(card.accounts.values(), key=lambda r: r.return_pct, reverse=True):
            lines.append(
                f"  {r.account_id:22} {r.return_pct:+7.2f}%  max DD {r.max_drawdown_pct:5.2f}%  "
                f"{r.trades} trades"
            )
    lines.append("")
    for c in card.criteria:
        lines.append(f"  [{c.verdict.value:7}] {c.name}: {c.detail}")
    lines.append("")
    if card.ready:
        lines.append(
            "All automatic criteria pass. Review the CHECK items; going live (Phase 6) is "
            "your decision."
        )
    elif any(c.verdict is Verdict.FAIL for c in card.criteria):
        lines.append("NOT READY: at least one criterion fails.")
    else:
        lines.append("IN PROGRESS: keep running; don't change the prompt or risk settings.")
    return "\n".join(lines)
