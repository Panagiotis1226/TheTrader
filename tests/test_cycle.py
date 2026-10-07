"""Integration: full decision cycles with real PaperBroker, RiskManager, storage, and
litellm (mocked responses), on recorded Kraken data."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import ccxt
import pytest

from ai_trader.ai.agent import AgentResult, LLMAgent
from ai_trader.ai.prompt import render_system_prompt
from ai_trader.ai.schema import TradeProposal
from ai_trader.brokers.base import OrderRequest, OrderType, Side
from ai_trader.brokers.paper import PaperBroker
from ai_trader.config import Mode, ModelSettings, load_config, load_env, load_trading_settings
from ai_trader.cycle import CycleStatus, DecisionCycle, TradingAccount
from ai_trader.data.market import MarketData
from ai_trader.main import EXIT_UNSUPPORTED, main, run_once
from ai_trader.risk.manager import RiskManager
from ai_trader.storage.repo import HaltKind

from .conftest import SETTINGS_PATH
from .fakes import FakeClock, FakeExchange, RecordingAlerter, load_kraken_fixture, mock_llm

D = Decimal
SETTINGS = load_trading_settings(SETTINGS_PATH)
FIXTURE = load_kraken_fixture()
NOW = datetime.fromtimestamp(FIXTURE["recorded_at_ms"] / 1000, tz=UTC)
SECRET = "sk-test-key"


def proposal_json(**kw) -> str:
    base = {
        "action": "buy",
        "pair": "BTC/CAD",
        "size_pct": 5,
        "confidence": 0.8,
        "reason": "test",
        "stop_loss_pct": None,
    }
    return json.dumps({**base, **kw})


class Harness:
    def __init__(self, repo, write_env, exchange=None) -> None:
        self.repo = repo
        self.clock = FakeClock(NOW)
        self.exchange = exchange or FakeExchange()
        self.market = MarketData(self.exchange, clock=self.clock)
        self.alerter = RecordingAlerter()
        self.env = load_env(write_env(ANTHROPIC_API_KEY=SECRET))
        self.cycle = DecisionCycle(
            SETTINGS,
            repo,
            self.market,
            RiskManager(SETTINGS.risk, SETTINGS.pairs, SETTINGS.paper.taker_fee_pct),
            self.alerter,
            self.clock,
        )

    def broker(self, name: str) -> PaperBroker:
        return PaperBroker(
            f"paper-{name}",
            repo=self.repo,
            market=self.market,
            allowed_pairs=SETTINGS.pairs,
            starting_cash=D(10000),
            taker_fee_pct=SETTINGS.paper.taker_fee_pct,
            clock=self.clock,
        )

    def account(self, name: str, response) -> TradingAccount:
        completion = mock_llm(response)
        agent = LLMAgent(
            ModelSettings(name=name, provider="litellm", model="anthropic/claude-opus-5-5"),
            SETTINGS.llm,
            self.env,
            render_system_prompt(SETTINGS),
            "BTC/CAD",
            completion,
        )
        agent.calls = completion.calls  # type: ignore[attr-defined]
        return TradingAccount(broker=self.broker(name), decider=agent)


@pytest.fixture
def h(repo, write_env) -> Harness:
    return Harness(repo, write_env)


# ------------------------------------------- the three cases required by PLAN.md §9


async def test_a_valid_buy_is_traded_and_fully_logged(h) -> None:
    acct = h.account("claude", proposal_json(size_pct=5))
    [outcome] = await h.cycle.run([acct])

    assert outcome.status is CycleStatus.TRADED
    d = h.repo.decision_details(outcome.decision_id)
    assert len(d["snapshot_hash"]) == 64
    assert d["model"] == "anthropic/claude-opus-5-5"
    assert len(d["prompt_hash"]) == 16
    assert d["raw_response"] == proposal_json(size_pct=5)
    assert d["proposal"]["action"] == "buy"
    assert d["risk_outcome"] == "approve"
    assert d["cost_usd"] > 0 and d["prompt_tokens"] > 0
    assert d["error"] is None

    [fill] = h.repo.list_fills("paper-claude")
    assert fill.order_id == d["order_id"]
    assert D(495) < fill.cost <= D(500)  # 5% of 10,000 sized at the ask
    [stop] = await acct.broker.get_open_orders()
    assert stop.type is OrderType.STOP_LOSS
    assert stop.trigger_price < fill.price * D("0.951")  # default 5% stop
    assert any("BUY" in t for t in h.alerter.texts())
    # Snapshots before and after the trade: the second one includes fee and spread.
    series = h.repo.equity_series("paper-claude")
    assert series[0][1] == D(10000)
    assert series[-1][1] < D(10000) - fill.fee + D(1)


async def test_b_garbage_is_held_logged_and_alerted(h) -> None:
    acct = h.account("claude", "I'd go long here, maybe 20%?")
    [outcome] = await h.cycle.run([acct])

    assert outcome.status is CycleStatus.HELD
    d = h.repo.decision_details(outcome.decision_id)
    assert d["action"] == "hold"
    assert d["raw_response"] == "I'd go long here, maybe 20%?"
    assert "not a single JSON object" in d["error"]
    assert d["order_id"] is None
    assert h.repo.list_fills("paper-claude") == []
    assert len(acct.decider.calls) == 1  # no retry
    assert any("unusable decision" in t for t in h.alerter.texts("warning"))


async def test_c_oversized_buy_is_resized(h) -> None:
    acct = h.account("claude", proposal_json(size_pct=50, stop_loss_pct=20))
    [outcome] = await h.cycle.run([acct])

    assert outcome.status is CycleStatus.TRADED
    d = h.repo.decision_details(outcome.decision_id)
    assert d["risk_outcome"] == "resize"
    assert "max_trade_pct_of_equity" in d["risk_reason"]
    [fill] = h.repo.list_fills("paper-claude")
    assert D(990) < fill.cost <= D(1000)  # capped at 10% of equity
    [stop] = await acct.broker.get_open_orders()
    assert stop.trigger_price >= (fill.price * D("0.95")).quantize(D("0.1")) - D("0.1")


# ---------------------------------------------------------------- other paths


async def test_non_whitelisted_pair_rejected_and_alerted(h) -> None:
    [outcome] = await h.cycle.run([h.account("claude", proposal_json(pair="DOGE/CAD"))])
    assert outcome.status is CycleStatus.REJECTED
    assert "whitelisted" in outcome.detail
    assert any("risk rejected" in t for t in h.alerter.texts())


async def test_hold_is_quiet(h) -> None:
    resp = proposal_json(action="hold", size_pct=0, confidence=0.3)
    [outcome] = await h.cycle.run([h.account("claude", resp)])
    assert outcome.status is CycleStatus.HELD
    assert h.alerter.sent == []


async def test_halted_account_skips_without_calling_llm(h) -> None:
    acct = h.account("claude", proposal_json())
    h.repo.add_halt(HaltKind.MANUAL, "/stop", NOW - timedelta(minutes=1))
    [outcome] = await h.cycle.run([acct])
    assert outcome.status is CycleStatus.SKIPPED
    assert "manual" in outcome.detail
    assert acct.decider.calls == []


async def test_daily_llm_budget_skips_without_calling_llm(h) -> None:
    acct = h.account("claude", proposal_json())
    h.repo.record_decision("paper-claude", NOW - timedelta(hours=1), cost_usd=D("2.00"))
    [outcome] = await h.cycle.run([acct])
    assert outcome.status is CycleStatus.SKIPPED
    assert acct.decider.calls == []


async def test_daily_loss_records_halt(h) -> None:
    acct = h.account("claude", proposal_json())
    h.repo.record_equity("paper-claude", NOW - timedelta(days=1), D(10400), D(10400))
    [outcome] = await h.cycle.run([acct])  # equity 10,000 is -3.8% on the day

    assert outcome.status is CycleStatus.REJECTED
    assert [x.kind for x in h.repo.active_halts("paper-claude", NOW)] == [HaltKind.DAILY_LOSS]
    assert any("TRADING HALTED" in t for t in h.alerter.texts("critical"))
    # Reduce-only: the cycle still runs, buys are rejected.
    [again] = await h.cycle.run([acct])
    assert again.status is CycleStatus.REJECTED
    assert "sells only" in again.detail


async def test_sell_allowed_during_drawdown_halt(h) -> None:
    acct = h.account("claude", proposal_json(action="sell", size_pct=100))
    h.clock.now = NOW - timedelta(hours=2)  # bought earlier (trade-spacing rule)
    await acct.broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("0.001")))
    h.clock.now = NOW
    h.repo.add_halt(HaltKind.DRAWDOWN, "dd", NOW - timedelta(hours=1), account_id="paper-claude")
    [outcome] = await h.cycle.run([acct])
    assert outcome.status is CycleStatus.TRADED
    assert await acct.broker.get_positions() == []


async def test_exchange_outage_skips_whole_cycle(repo, write_env) -> None:
    class DownExchange(FakeExchange):
        async def fetch_ohlcv(self, *a, **kw):
            raise ccxt.NetworkError("kraken unreachable")

    h = Harness(repo, write_env, DownExchange())
    accounts = [h.account("a", proposal_json()), h.account("b", proposal_json())]
    outcomes = await h.cycle.run(accounts)
    assert [o.status for o in outcomes] == [CycleStatus.SKIPPED] * 2
    assert all(a.decider.calls == [] for a in accounts)


async def test_stale_data_skips(h) -> None:
    acct = h.account("claude", proposal_json())
    data_clock_start = h.clock.now  # market data is stamped at NOW...
    original_run_account = h.cycle._run_account

    async def late(account, data):
        h.clock.now = data_clock_start + timedelta(minutes=2)  # ...the account runs later
        return await original_run_account(account, data)

    h.cycle._run_account = late
    [outcome] = await h.cycle.run([acct])
    assert outcome.status is CycleStatus.SKIPPED
    assert "old" in outcome.detail
    assert acct.decider.calls == []


async def test_one_failing_account_does_not_stop_others(h) -> None:
    class Exploding:
        name = "boom"

        async def decide(self, snapshot) -> AgentResult:
            raise RuntimeError("bug")

    bad = TradingAccount(broker=h.broker("bad"), decider=Exploding())
    good = h.account("good", proposal_json())
    outcomes = await h.cycle.run([bad, good])
    assert [o.status for o in outcomes] == [CycleStatus.ERROR, CycleStatus.TRADED]


async def test_benchmark_style_decider_uses_same_path(h) -> None:
    class AlwaysHold:
        name = "do_nothing"

        async def decide(self, snapshot) -> AgentResult:
            return AgentResult(TradeProposal.hold("BTC/CAD", "benchmark"), model=self.name)

    [outcome] = await h.cycle.run([TradingAccount(h.broker("bench"), AlwaysHold())])
    assert outcome.status is CycleStatus.HELD
    assert h.repo.decision_details(outcome.decision_id)["model"] == "do_nothing"


async def test_stop_loss_checked_at_cycle_start(h) -> None:
    acct = h.account("claude", proposal_json(action="hold", size_pct=0))
    btc_bid = D(str(FIXTURE["order_book_BTC_CAD"]["bids"][0][0]))
    # Simulate an earlier entry far above the current price, with a tight stop.
    entry_book = load_kraken_fixture()
    entry_book["order_book_BTC_CAD"]["asks"] = [[float(btc_bid * 2), 1.0]]
    h.exchange.data = entry_book
    await acct.broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("0.001"), D(5)))
    h.exchange.data = load_kraken_fixture()

    await h.cycle.run([acct])
    assert await acct.broker.get_positions() == []
    assert any("STOP-LOSS" in t for t in h.alerter.texts("warning"))


# ----------------------------------------------------------------- one-off command


async def test_run_once_runs_each_configured_model(tmp_path, write_env) -> None:
    from .test_claude_code import FakeRunner, cli_output

    env_file = write_env(CLAUDE_CODE_OAUTH_TOKEN="tok", DATABASE_URL=f"sqlite:///{tmp_path}/t.db")
    config = load_config(env_file, SETTINGS_PATH)
    clock = FakeClock(NOW)
    market = MarketData(FakeExchange(), clock=clock)
    runner = FakeRunner(cli_output(proposal_json(size_pct=5)))

    outcomes = await run_once(config, Mode.PAPER, None, market, None, clock, runner)
    assert outcomes is not None
    by_account = {o.account_id: o for o in outcomes}
    # settings.yaml: one model (claude_code) plus the three benchmarks
    assert set(by_account) == {
        "paper-claude",
        "paper-buy_and_hold",
        "paper-ma_crossover",
        "paper-do_nothing",
    }
    assert by_account["paper-claude"].status is CycleStatus.TRADED
    assert by_account["paper-buy_and_hold"].status is CycleStatus.TRADED
    assert by_account["paper-do_nothing"].status is CycleStatus.HELD
    assert len(runner.calls) == 1  # benchmarks never call the model


async def test_daily_call_limit_skips(h) -> None:
    acct = h.account("claude", proposal_json())
    for i in range(SETTINGS.llm.max_daily_calls):
        h.repo.record_decision("paper-claude", NOW - timedelta(minutes=i + 1))
    [outcome] = await h.cycle.run([acct])
    assert outcome.status is CycleStatus.SKIPPED
    assert "call limit" in outcome.detail
    assert acct.decider.calls == []


def test_once_refuses_live_mode(write_env) -> None:
    env = write_env(MODE="live", LIVE_TRADING_CONFIRMED="yes")
    assert main(["--env-file", str(env), "--settings", str(SETTINGS_PATH), "once"]) == (
        EXIT_UNSUPPORTED
    )
