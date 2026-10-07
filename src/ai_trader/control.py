"""Local control from the machine itself: status, stop, resume, report, evaluate.

These commands work alongside a running bot (``ai-trader run`` or Docker): they share the
database, so a ``stop`` from here is seen by the bot's next cycle, and also blocks an order
from a decision already in progress. No Telegram needed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal

from ai_trader.alerts.base import AlertLevel
from ai_trader.config import AppConfig
from ai_trader.cycle import TradingAccount
from ai_trader.evaluation import format_scorecard, scorecard
from ai_trader.reports import AccountReport, account_report, format_weekly_report
from ai_trader.risk import killswitch
from ai_trader.storage.repo import Repository

log = logging.getLogger(__name__)


def _ago(then: datetime, now: datetime) -> str:
    minutes = int((now - then).total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return f"{minutes // 1440} days ago"


async def account_equity(
    accounts: list[TradingAccount], repo: Repository, quote: str
) -> tuple[dict[str, Decimal], bool]:
    """Live equity per account; falls back to the latest snapshot if Kraken is unreachable.

    Returns (equity by account, True if live).
    """
    equity: dict[str, Decimal] = {}
    live = True
    for account in accounts:
        try:
            equity[account.account_id] = await account.broker.get_equity(quote)
        except Exception as exc:
            log.warning("live equity unavailable for %s: %s", account.account_id, exc)
            live = False
            series = repo.equity_series(account.account_id)
            equity[account.account_id] = series[-1][1] if series else account.broker.cash
    return equity, live


async def reports(
    config: AppConfig, repo: Repository, accounts: list[TradingAccount], now: datetime
) -> list[AccountReport]:
    equity, _ = await account_equity(accounts, repo, config.trading.quote_currency)
    return [
        account_report(
            repo, a.account_id, equity[a.account_id], await a.broker.get_positions(), now
        )
        for a in accounts
    ]


async def status_text(
    config: AppConfig, repo: Repository, accounts: list[TradingAccount], now: datetime
) -> str:
    trading = config.trading
    equity, live = await account_equity(accounts, repo, trading.quote_currency)
    lines = [f"ai-trader (paper) — {now:%Y-%m-%d %H:%M} UTC"]

    halts = repo.active_halts(None, now)
    if halts:
        for h in halts:
            lines.append(
                f"TRADING HALTED ({h.kind.value}) since {h.created_at:%Y-%m-%d %H:%M}: {h.reason}"
            )
        lines.append("  -> `ai-trader resume` to lift")
    else:
        lines.append("Trading: active")

    decisions = repo.list_decisions(limit=1)
    if decisions:
        last = decisions[0]["created_at"]
        stale = now - last > timedelta(minutes=2 * trading.decision_interval_minutes + 30)
        lines.append(
            f"Last decision: {last:%Y-%m-%d %H:%M} UTC ({_ago(last, now)})"
            + ("  <- older than expected: is the bot running?" if stale else "")
        )
    else:
        lines.append("Last decision: none yet")

    card = scorecard(trading, repo, equity, now)
    llm_id = f"paper-{trading.evaluation.llm_account}"
    if card.start is None:
        lines.append("Evaluation: not started")
    else:
        llm_trades = card.accounts[llm_id].trades if llm_id in card.accounts else 0
        lines.append(
            f"Evaluation: week {card.weeks} of {trading.evaluation.min_weeks}, "
            f"{llm_trades} of {trading.evaluation.min_trades} trades "
            f"(`ai-trader evaluate` for the scorecard)"
        )

    lines.append("")
    lines.append("Accounts" + ("" if live else " (Kraken unreachable: last snapshot)") + ":")
    for account in accounts:
        acct = account.account_id
        eq = equity[acct]
        start = next(a.starting_cash for a in repo.list_accounts() if a.id == acct)
        positions = await account.broker.get_positions()
        held = ", ".join(f"{p.amount.normalize()} {p.pair.split('/')[0]}" for p in positions)
        own = [h.kind.value for h in repo.active_halts(acct, now) if h.account_id]
        last = repo.recent_decisions(acct, 1)
        last_txt = (
            f"last: {last[0].action} ({last[0].risk_outcome}) {_ago(last[0].created_at, now)}"
            if last
            else "no decisions yet"
        )
        lines.append(
            f"  {acct:22} {eq:>11,.2f} ({(eq / start - 1) * 100:+.2f}%)  "
            f"{held or 'cash'}; {last_txt}" + (f"  HALTED ({', '.join(own)})" if own else "")
        )

    alerts = repo.recent_alerts(8)
    if alerts:
        lines.append("")
        lines.append("Recent alerts:")
        for at, level, text in alerts:
            mark = {"warning": "!", "critical": "!!"}.get(level, " ")
            lines.append(f"  {at:%m-%d %H:%M} {mark:2} {text[:150]}")
    return "\n".join(lines)


async def stop(repo: Repository, accounts: list[TradingAccount], reason: str, now: datetime) -> str:
    failed = await killswitch.engage(repo, [a.broker for a in accounts], reason, now)
    repo.add_alert(AlertLevel.CRITICAL.value, f"KILL SWITCH: {reason}", now)
    msg = (
        "Trading halted for all accounts (stop-losses stay active).\n"
        "The running bot skips its next cycle; `ai-trader resume` to restart."
    )
    if failed:
        msg += f"\nCould not cancel orders for: {', '.join(failed)}"
    return msg


def resume(repo: Repository, now: datetime) -> str:
    count = killswitch.resume(repo, now)
    repo.add_alert(AlertLevel.WARNING.value, f"Trading resumed locally ({count} halt(s))", now)
    return f"Resumed: {count} halt(s) lifted."


async def report_text(
    config: AppConfig, repo: Repository, accounts: list[TradingAccount], now: datetime
) -> str:
    return format_weekly_report(await reports(config, repo, accounts, now), now)


async def evaluate_text(
    config: AppConfig, repo: Repository, accounts: list[TradingAccount], now: datetime
) -> str:
    equity, _ = await account_equity(accounts, repo, config.trading.quote_currency)
    return format_scorecard(scorecard(config.trading, repo, equity, now))
