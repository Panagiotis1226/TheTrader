from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from ai_trader.ai.agent import AgentResult
from ai_trader.brokers.base import OrderRequest, OrderType, Side
from ai_trader.cycle import CycleStatus, TradingAccount
from ai_trader.service import (
    JOB_CYCLE,
    JOB_DAILY,
    JOB_EQUITY,
    JOB_STOPS,
    JOB_WEEKLY,
    TradingService,
)
from ai_trader.storage.repo import HaltKind

from .test_cycle import NOW, SETTINGS, Harness, proposal_json

D = Decimal


class FakeHeartbeat:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def success(self, message: str = "") -> None:
        self.calls.append(("success", message))

    async def fail(self, message: str = "") -> None:
        self.calls.append(("fail", message))


class Exploding:
    name = "boom"

    async def decide(self, snapshot) -> AgentResult:
        raise RuntimeError("bug in strategy")


@pytest.fixture
def h(repo, write_env) -> Harness:
    return Harness(repo, write_env)


def service(h: Harness, accounts) -> tuple[TradingService, FakeHeartbeat]:
    hb = FakeHeartbeat()
    svc = TradingService(SETTINGS, h.repo, h.cycle, accounts, h.alerter, hb, h.clock)
    return svc, hb


# --------------------------------------------------------------------- scheduling


def test_schedule_registers_all_jobs(h) -> None:
    svc, _ = service(h, [])
    svc.schedule(run_cycle_now=True)
    jobs = {j.id: j for j in svc.scheduler.get_jobs()}
    assert set(jobs) == {JOB_CYCLE, JOB_STOPS, JOB_EQUITY, JOB_DAILY, JOB_WEEKLY}
    assert isinstance(jobs[JOB_CYCLE].trigger, IntervalTrigger)
    assert jobs[JOB_CYCLE].trigger.interval == timedelta(minutes=240)
    assert jobs[JOB_STOPS].trigger.interval == timedelta(seconds=60)
    assert isinstance(jobs[JOB_DAILY].trigger, CronTrigger)
    assert "hour='8'" in str(jobs[JOB_DAILY].trigger)
    assert "day_of_week='mon'" in str(jobs[JOB_WEEKLY].trigger)
    assert all(j.max_instances == 1 and j.coalesce for j in jobs.values())


# ------------------------------------------------------------------------ cycles


async def test_successful_cycle_pings_heartbeat(h) -> None:
    svc, hb = service(h, [h.account("claude", proposal_json())])
    [outcome] = await svc.run_cycle()
    assert outcome.status is CycleStatus.TRADED
    assert hb.calls == [("success", "paper-claude=traded")]
    assert svc.last_cycle_at == NOW


async def test_market_outage_fails_heartbeat_without_halting(repo, write_env) -> None:
    import ccxt

    from .fakes import FakeExchange

    class Down(FakeExchange):
        async def fetch_ohlcv(self, *a, **kw):
            raise ccxt.NetworkError("down")

    h = Harness(repo, write_env, Down())
    svc, hb = service(h, [h.account("claude", proposal_json())])
    for _ in range(SETTINGS.monitoring.max_consecutive_errors + 1):
        await svc.run_cycle()
    assert hb.calls[-1] == ("fail", "market data unusable")
    assert repo.active_halts("paper-claude", NOW) == []


async def test_repeated_account_errors_halt_that_account(h) -> None:
    bad = TradingAccount(broker=h.broker("bad"), decider=Exploding())
    good = h.account("good", proposal_json(action="hold", size_pct=0))
    svc, hb = service(h, [bad, good])
    for _ in range(3):
        await svc.run_cycle()
    assert [x.kind for x in h.repo.active_halts("paper-bad", NOW)] == [HaltKind.ERRORS]
    assert h.repo.active_halts("paper-good", NOW) == []
    assert any("paper-bad: TRADING HALTED" in t for t in h.alerter.texts("critical"))
    assert hb.calls[-1][0] == "fail"

    outcomes = {o.account_id: o.status for o in await svc.run_cycle()}
    assert outcomes == {"paper-bad": CycleStatus.SKIPPED, "paper-good": CycleStatus.HELD}

    await svc.resume()
    assert h.repo.active_halts("paper-bad", NOW) == []


async def test_errors_must_be_consecutive(h) -> None:
    flaky = TradingAccount(broker=h.broker("flaky"), decider=Exploding())
    svc, _ = service(h, [flaky])
    await svc.run_cycle()
    await svc.run_cycle()
    svc.accounts = [TradingAccount(broker=flaky.broker, decider=h.account("x", "{}").decider)]
    await svc.run_cycle()  # not an ERROR (garbage output is a hold): counter resets
    svc.accounts = [flaky]
    await svc.run_cycle()
    assert h.repo.active_halts("paper-flaky", NOW) == []


async def test_repeated_cycle_crashes_halt_everything(h, monkeypatch) -> None:
    svc, hb = service(h, [h.account("claude", proposal_json())])

    async def crash(accounts):
        raise RuntimeError("database locked")

    monkeypatch.setattr(h.cycle, "run", crash)
    for _ in range(3):
        assert await svc.run_cycle() == []
    assert [x.kind for x in h.repo.active_halts(None, NOW)] == [HaltKind.ERRORS]
    assert hb.calls == [("fail", "cycle failed: RuntimeError")] * 3
    assert any("ALL TRADING HALTED" in t for t in h.alerter.texts("critical"))


# ------------------------------------------------------------------- kill switch


async def test_stop_halts_next_cycle_and_keeps_stops(h) -> None:
    acct = h.account("claude", proposal_json())
    svc, _ = service(h, [acct])
    h.clock.now = NOW - timedelta(hours=2)
    await acct.broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("0.001"), D(5)))
    h.clock.now = NOW

    reply = await svc.cmd_stop()
    assert "halted" in reply
    [stop] = await acct.broker.get_open_orders()
    assert stop.type is OrderType.STOP_LOSS  # protection stays

    [outcome] = await svc.run_cycle()
    assert outcome.status is CycleStatus.SKIPPED
    assert acct.decider.calls == []  # no model call while stopped


async def test_stop_during_a_decision_blocks_the_order(h) -> None:
    """/stop arriving while the model is thinking still prevents the trade."""
    svc_holder = {}

    class StopsMidDecision:
        name = "slow"

        async def decide(self, snapshot) -> AgentResult:
            await svc_holder["svc"].cmd_stop()  # user hits /stop now
            from ai_trader.ai.schema import TradeProposal

            buy = TradeProposal(
                action="buy", pair="BTC/CAD", size_pct=D(5), confidence=D(1), reason="late"
            )
            return AgentResult(buy, model=self.name)

    acct = TradingAccount(broker=h.broker("slow"), decider=StopsMidDecision())
    svc, _ = service(h, [acct])
    svc_holder["svc"] = svc
    [outcome] = await svc.run_cycle()
    assert outcome.status is CycleStatus.REJECTED
    assert "halted" in outcome.detail
    assert h.repo.list_fills("paper-slow") == []


# ---------------------------------------------------------------- other jobs


async def test_stop_check_job_fills_and_alerts(h) -> None:
    from .fakes import load_kraken_fixture

    acct = h.account("claude", proposal_json())
    svc, _ = service(h, [acct])
    bid = D(str(load_kraken_fixture()["order_book_BTC_CAD"]["bids"][0][0]))
    pricey = load_kraken_fixture()
    pricey["order_book_BTC_CAD"]["asks"] = [[float(bid * 2), 1.0]]
    h.exchange.data = pricey
    await acct.broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("0.001"), D(5)))
    h.exchange.data = load_kraken_fixture()

    await svc.check_stops()
    assert await acct.broker.get_positions() == []
    assert any("STOP-LOSS" in t for t in h.alerter.texts("warning"))


async def test_equity_snapshot_job(h) -> None:
    svc, _ = service(h, [h.account("a", "{}"), h.account("b", "{}")])
    await svc.snapshot_equity()
    assert h.repo.equity_at_or_before("paper-a", NOW) == D(10000)
    assert h.repo.equity_at_or_before("paper-b", NOW) == D(10000)


async def test_daily_and_weekly_reports(h) -> None:
    acct = h.account("claude", proposal_json())
    svc, _ = service(h, [acct, h.account("idle", "{}")])
    await svc.run_cycle()
    await svc.daily_summary()
    await svc.weekly_report()
    daily, weekly = h.alerter.texts()[-2:]
    assert daily.startswith("Daily summary") and "paper-claude" in daily and "paper-idle" in daily
    assert "Weekly report" in weekly and "LLM cost $" in weekly and "rejected" in weekly


async def test_status_equity_help(h) -> None:
    svc, _ = service(h, [h.account("claude", proposal_json())])
    await svc.run_cycle()
    status = await svc.cmd_status()
    assert "Mode: paper" in status and "Halts: none" in status
    claude_line = next(line for line in status.splitlines() if line.startswith("paper-claude:"))
    assert "BTC/CAD" in claude_line and "last: traded" in claude_line
    assert "paper-claude:" in await svc.cmd_equity()
    assert "/stop" in await svc.cmd_help()
    assert set(svc.commands()) == set(TradingService.COMMANDS)


async def test_error_halt_repeats_after_external_resume(h) -> None:
    bad = TradingAccount(broker=h.broker("bad"), decider=Exploding())
    svc, _ = service(h, [bad])
    for _ in range(3):
        await svc.run_cycle()
    assert len(h.repo.active_halts("paper-bad", NOW)) == 1

    # `ai-trader resume` from another process: the service's error counter stays at 3,
    # so the next failure is the 4th in a row and must halt again.
    h.repo.resume(NOW)
    await svc.run_cycle()
    assert [x.kind for x in h.repo.active_halts("paper-bad", NOW)] == [HaltKind.ERRORS]
