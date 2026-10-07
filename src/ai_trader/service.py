"""Unattended operation: scheduler, error budget, reports, kill switch.

Jobs (APScheduler, one asyncio loop, each job never overlaps itself):

* decision cycle every ``decision_interval_minutes`` (first one at startup)
* stop-loss check every ``stop_check_seconds``
* equity snapshot every ``equity_snapshot_minutes``
* daily summary at ``daily_summary_time`` and weekly report on ``weekly_report_day``
  (``monitoring.timezone``)

Errors: any exception is logged and alerted and the loop continues. An account whose
cycle errors ``max_consecutive_errors`` times in a row is halted (needs /resume); so is
everything if the cycle as a whole fails that many times in a row. Market data outages
skip the cycle and are not counted (the heartbeat stops instead).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ai_trader.alerts.base import Alerter, AlertLevel
from ai_trader.alerts.heartbeat import Heartbeat
from ai_trader.alerts.telegram_bot import CommandFn
from ai_trader.config import TradingSettings
from ai_trader.cycle import (
    MARKET_DATA_SKIP,
    CycleOutcome,
    CycleStatus,
    DecisionCycle,
    TradingAccount,
)
from ai_trader.data.market import Clock, utcnow
from ai_trader.evaluation import format_scorecard, scorecard
from ai_trader.reports import (
    AccountReport,
    account_report,
    format_daily_summary,
    format_weekly_report,
)
from ai_trader.risk import killswitch
from ai_trader.risk.manager import record_halt
from ai_trader.storage.repo import HaltKind, Repository

log = logging.getLogger(__name__)

JOB_CYCLE = "decision_cycle"
JOB_STOPS = "stop_check"
JOB_EQUITY = "equity_snapshot"
JOB_DAILY = "daily_summary"
JOB_WEEKLY = "weekly_report"


class TradingService:
    def __init__(
        self,
        settings: TradingSettings,
        repo: Repository,
        cycle: DecisionCycle,
        accounts: Sequence[TradingAccount],
        alerter: Alerter,
        heartbeat: Heartbeat,
        clock: Clock = utcnow,
    ) -> None:
        self._s = settings
        self._repo = repo
        self._cycle = cycle
        self.accounts = list(accounts)
        self._alerter = alerter
        self._heartbeat = heartbeat
        self._clock = clock
        self._account_errors: dict[str, int] = {}
        self._cycle_failures = 0
        self.last_cycle_at: datetime | None = None
        self.last_outcomes: list[CycleOutcome] = []
        self.scheduler = AsyncIOScheduler(timezone=ZoneInfo(settings.monitoring.timezone))

    # --------------------------------------------------------------- scheduling

    def schedule(self, run_cycle_now: bool = True) -> None:
        m = self._s.monitoring
        hour, minute = (int(x) for x in m.daily_summary_time.split(":"))
        tz = ZoneInfo(m.timezone)
        common = {"max_instances": 1, "coalesce": True, "misfire_grace_time": 300}
        self.scheduler.add_job(
            self.run_cycle,
            IntervalTrigger(minutes=self._s.decision_interval_minutes, timezone=tz),
            id=JOB_CYCLE,
            next_run_time=self._clock() if run_cycle_now else None,
            **common,
        )
        self.scheduler.add_job(
            self.check_stops,
            IntervalTrigger(seconds=m.stop_check_seconds, timezone=tz),
            id=JOB_STOPS,
            **common,
        )
        self.scheduler.add_job(
            self.snapshot_equity,
            IntervalTrigger(minutes=m.equity_snapshot_minutes, timezone=tz),
            id=JOB_EQUITY,
            **common,
        )
        self.scheduler.add_job(
            self.daily_summary,
            CronTrigger(hour=hour, minute=minute, timezone=tz),
            id=JOB_DAILY,
            **common,
        )
        self.scheduler.add_job(
            self.weekly_report,
            CronTrigger(day_of_week=m.weekly_report_day, hour=hour, minute=minute, timezone=tz),
            id=JOB_WEEKLY,
            **common,
        )

    def next_cycle_at(self) -> datetime | None:
        job = self.scheduler.get_job(JOB_CYCLE)
        return job.next_run_time if job else None

    # --------------------------------------------------------------------- jobs

    async def run_cycle(self) -> list[CycleOutcome]:
        self.last_cycle_at = self._clock()
        try:
            outcomes = await self._cycle.run(self.accounts)
        except Exception as exc:
            log.exception("decision cycle failed")
            self._cycle_failures += 1
            await self._alerter.send(
                f"Decision cycle failed ({self._cycle_failures} in a row): "
                f"{type(exc).__name__}: {exc}",
                AlertLevel.WARNING,
            )
            await self._heartbeat.fail(f"cycle failed: {type(exc).__name__}")
            if self._cycle_failures >= self._s.monitoring.max_consecutive_errors:
                await self._halt_all_on_errors()
            return []

        self._cycle_failures = 0
        self.last_outcomes = outcomes
        await self._count_account_errors(outcomes)

        market_down = bool(outcomes) and all(
            o.status is CycleStatus.SKIPPED and o.detail.startswith(MARKET_DATA_SKIP)
            for o in outcomes
        )
        errors = [o for o in outcomes if o.status is CycleStatus.ERROR]
        if market_down:
            await self._heartbeat.fail("market data unusable")
        elif errors:
            await self._heartbeat.fail(f"{len(errors)} account(s) errored")
        else:
            summary = ", ".join(f"{o.account_id}={o.status.value}" for o in outcomes)
            await self._heartbeat.success(summary)
        return outcomes

    async def check_stops(self) -> None:
        for account in self.accounts:
            await self._cycle.check_stops(account)

    async def snapshot_equity(self) -> None:
        now = self._clock()
        for account in self.accounts:
            try:
                equity = await account.broker.get_equity(self._s.quote_currency)
                self._repo.record_equity(account.account_id, now, equity, account.broker.cash)
            except Exception:
                log.exception("%s: equity snapshot failed", account.account_id)

    async def daily_summary(self) -> None:
        reports = await self.reports()
        await self._alerter.send(format_daily_summary(reports, self._clock()))

    async def weekly_report(self) -> None:
        reports = await self.reports()
        now = self._clock()
        card = scorecard(self._s, self._repo, {r.account_id: r.equity for r in reports}, now)
        await self._alerter.send(
            format_weekly_report(reports, now) + "\n\n" + format_scorecard(card)
        )

    async def reports(self) -> list[AccountReport]:
        now = self._clock()
        out = []
        for account in self.accounts:
            try:
                equity = await account.broker.get_equity(self._s.quote_currency)
                positions = await account.broker.get_positions()
                out.append(account_report(self._repo, account.account_id, equity, positions, now))
            except Exception:
                log.exception("%s: report failed", account.account_id)
        return out

    # -------------------------------------------------------------- kill switch

    async def kill_switch(self, reason: str) -> list[str]:
        """Halt everything now; cancel open orders (stop-losses stay)."""
        failed = await killswitch.engage(
            self._repo, [a.broker for a in self.accounts], reason, self._clock()
        )
        await self._alerter.send(f"KILL SWITCH: {reason}", AlertLevel.CRITICAL)
        return failed

    async def resume(self) -> int:
        count = killswitch.resume(self._repo, self._clock())
        self._account_errors.clear()
        self._cycle_failures = 0
        await self._alerter.send(f"Trading resumed ({count} halt(s) lifted)", AlertLevel.WARNING)
        return count

    # ---------------------------------------------------------------- internals

    async def _count_account_errors(self, outcomes: Sequence[CycleOutcome]) -> None:
        limit = self._s.monitoring.max_consecutive_errors
        for o in outcomes:
            if o.status is not CycleStatus.ERROR:
                self._account_errors[o.account_id] = 0
                continue
            n = self._account_errors.get(o.account_id, 0) + 1
            self._account_errors[o.account_id] = n
            already = any(
                h.kind is HaltKind.ERRORS and h.account_id == o.account_id
                for h in self._repo.active_halts(o.account_id, self._clock())
            )
            # >= (not ==): after a /resume from another process the counter may already
            # be past the limit, and the account must still be halted again.
            if n >= limit and not already:
                reason = f"{n} failed cycles in a row (last: {o.detail})"
                record_halt(self._repo, o.account_id, HaltKind.ERRORS, reason, self._clock())
                await self._alerter.send(
                    f"{o.account_id}: TRADING HALTED: {reason}. Send /resume after fixing.",
                    AlertLevel.CRITICAL,
                )

    async def _halt_all_on_errors(self) -> None:
        reason = f"decision cycle failed {self._cycle_failures} times in a row"
        self._repo.add_halt(HaltKind.ERRORS, reason, self._clock())
        await self._alerter.send(
            f"ALL TRADING HALTED: {reason}. Send /resume after fixing.", AlertLevel.CRITICAL
        )

    # ----------------------------------------------------------- bot commands

    COMMANDS = ("status", "equity", "stop", "resume", "help")

    def commands(self) -> dict[str, CommandFn]:
        return {name: getattr(self, f"cmd_{name}") for name in self.COMMANDS}

    async def cmd_help(self) -> str:
        return (
            "/status  halts, positions, last and next cycle\n"
            "/equity  equity per account vs. start\n"
            "/stop    KILL SWITCH: halt all trading, cancel open orders (stop-losses stay)\n"
            "/resume  lift all halts"
        )

    async def cmd_stop(self) -> str:
        failed = await self.kill_switch("/stop from Telegram")
        msg = "Trading halted for all accounts. Stop-losses stay active. /resume to restart."
        if failed:
            msg += f"\nCould not cancel orders for: {', '.join(failed)}"
        return msg

    async def cmd_resume(self) -> str:
        count = await self.resume()
        return f"Resumed: {count} halt(s) lifted."

    async def cmd_equity(self) -> str:
        lines = []
        for r in await self.reports():
            lines.append(f"{r.account_id}: {r.equity:,.2f} ({r.return_pct:+.2f}%)")
        return "\n".join(lines) or "No accounts."

    async def cmd_status(self) -> str:
        now = self._clock()
        global_halts = self._repo.active_halts(None, now)
        lines = ["Mode: paper"]
        lines.append(
            "Halts: " + ("; ".join(f"{h.kind.value} ({h.reason})" for h in global_halts) or "none")
        )
        last = f"{self.last_cycle_at:%Y-%m-%d %H:%M} UTC" if self.last_cycle_at else "not yet"
        nxt = self.next_cycle_at()
        lines.append(f"Last cycle: {last}; next: {f'{nxt:%Y-%m-%d %H:%M %Z}' if nxt else '-'}")
        by_account = {o.account_id: o for o in self.last_outcomes}
        for account in self.accounts:
            acct = account.account_id
            positions = await account.broker.get_positions()
            pos = ", ".join(f"{p.amount.normalize()} {p.pair}" for p in positions) or "cash"
            own = [h.kind.value for h in self._repo.active_halts(acct, now) if h.account_id]
            last_o = by_account.get(acct)
            lines.append(
                f"{acct}: {pos}"
                + (f"; last: {last_o.status.value}" if last_o else "")
                + (f"; HALTED ({', '.join(own)})" if own else "")
            )
        return "\n".join(lines)
