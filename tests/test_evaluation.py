from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ai_trader.brokers.base import Fill, Order, OrderStatus, OrderType, Side
from ai_trader.config import load_trading_settings
from ai_trader.evaluation import (
    Verdict,
    config_change_notice,
    current_fingerprint,
    format_scorecard,
    integrity_check,
    scorecard,
    window_start,
)
from ai_trader.storage.repo import HaltKind, Repository

from .conftest import SETTINGS_PATH

D = Decimal
SETTINGS = load_trading_settings(SETTINGS_PATH)  # min 8 weeks, 50 trades, tol 1pt, DD -25%
FP = current_fingerprint(SETTINGS)
T0 = datetime(2026, 1, 1, tzinfo=UTC)
ACCOUNTS = ["paper-claude", "paper-buy_and_hold", "paper-ma_crossover", "paper-do_nothing"]


def setup(repo: Repository, weeks: float = 9, trades: int = 60) -> datetime:
    for a in ACCOUNTS:
        repo.ensure_account(a, kind="paper", quote_currency="CAD", starting_cash=D(10000), now=T0)
    for i in range(int(weeks * 7)):
        repo.record_decision(
            "paper-claude",
            T0 + timedelta(days=i),
            prompt_hash=FP,
            model="claude_code/claude-opus-5-5",
            action="hold",
            completion_tokens=120,
        )
    for i in range(trades):
        oid = f"o{i}"
        ts = T0 + timedelta(hours=i)
        side = Side.BUY if i % 2 == 0 else Side.SELL
        order = Order(
            oid,
            "paper-claude",
            "BTC/CAD",
            side,
            OrderType.MARKET,
            D("0.001"),
            OrderStatus.FILLED,
            ts,
        )
        repo.record_fill(
            order,
            Fill(
                oid, "paper-claude", "BTC/CAD", side, D("0.001"), D(100), D("0.1"), D(0), "CAD", ts
            ),
            ts,
        )
    return T0 + timedelta(weeks=weeks)


def equity(claude="11000", bh="10500", ma="10200", nothing="10000") -> dict[str, Decimal]:
    return dict(zip(ACCOUNTS, map(D, (claude, bh, ma, nothing)), strict=True))


def verdicts(card) -> dict[str, Verdict]:
    return {c.name: c.verdict for c in card.criteria}


def test_not_started(repo) -> None:
    card = scorecard(SETTINGS, repo, {}, T0)
    assert card.start is None
    assert verdicts(card) == {"Evaluation started": Verdict.PENDING}
    assert "IN PROGRESS" in format_scorecard(card)


def test_all_automatic_criteria_pass(repo) -> None:
    now = setup(repo)
    card = scorecard(SETTINGS, repo, equity(), now)
    v = verdicts(card)
    assert v["Duration"] is Verdict.PASS
    assert v["Trades"] is Verdict.PASS
    assert v["1. vs buy-and-hold"] is Verdict.PASS
    assert v["2. vs MA crossover"] is Verdict.PASS
    assert v["3. No bugs or state mismatches"] is Verdict.CHECK  # needs your judgement
    assert v["4. LLM cost vs profit"] is Verdict.CHECK
    assert card.ready
    assert "going live (Phase 6) is your decision" in format_scorecard(card)


def test_pending_until_enough_time_and_trades(repo) -> None:
    now = setup(repo, weeks=3, trades=10)
    card = scorecard(SETTINGS, repo, equity(), now)
    assert verdicts(card)["Duration"] is Verdict.PENDING
    assert verdicts(card)["Trades"] is Verdict.PENDING
    assert not card.ready


def test_losing_to_benchmarks_fails(repo) -> None:
    now = setup(repo)
    card = scorecard(SETTINGS, repo, equity(claude="10100"), now)
    assert verdicts(card)["1. vs buy-and-hold"] is Verdict.FAIL
    assert verdicts(card)["2. vs MA crossover"] is Verdict.FAIL
    assert "NOT READY" in format_scorecard(card)


def test_matching_buy_and_hold_needs_clearly_lower_drawdown(repo) -> None:
    now = setup(repo)
    # B&H dipped 10% mid-window; Claude barely moved. Same final return within 1 point.
    repo.record_equity("paper-buy_and_hold", T0 + timedelta(days=20), D(9000), D(0))
    repo.record_equity("paper-claude", T0 + timedelta(days=20), D(9900), D(0))
    card = scorecard(SETTINGS, repo, equity(claude="10450", bh="10500"), now)
    assert verdicts(card)["1. vs buy-and-hold"] is Verdict.PASS
    assert "lower drawdown" in next(c.detail for c in card.criteria if c.name.startswith("1."))

    repo.record_equity("paper-claude", T0 + timedelta(days=21), D(9200), D(0))  # DD 8%
    card = scorecard(SETTINGS, repo, equity(claude="10450", bh="10500"), now)
    assert verdicts(card)["1. vs buy-and-hold"] is Verdict.FAIL


def test_config_change_restarts_the_clock(repo) -> None:
    setup(repo)
    later = T0 + timedelta(weeks=10)
    repo.record_decision(
        "paper-claude", later, prompt_hash="0ldc0nf1g0000000", completion_tokens=100
    )
    assert window_start(repo, "paper-claude", FP) is None
    repo.record_decision(
        "paper-claude", later + timedelta(hours=4), prompt_hash=FP, completion_tokens=100
    )
    assert window_start(repo, "paper-claude", FP) == later + timedelta(hours=4)


def test_clock_waits_for_a_real_model_answer(repo) -> None:
    repo.ensure_account(
        "paper-claude", kind="paper", quote_currency="CAD", starting_cash=D(1), now=T0
    )
    for i in range(3):  # e.g. "Not logged in": the model was never reached
        repo.record_decision(
            "paper-claude",
            T0 + timedelta(hours=4 * i),
            prompt_hash=FP,
            error="Claude Code error: Not logged in",
        )
    assert window_start(repo, "paper-claude", FP) is None
    assert "no answer from the model" in format_scorecard(scorecard(SETTINGS, repo, {}, T0))
    repo.record_decision(
        "paper-claude", T0 + timedelta(hours=12), prompt_hash=FP, completion_tokens=150
    )
    assert window_start(repo, "paper-claude", FP) == T0 + timedelta(hours=12)


def test_cycles_that_never_reach_the_model_are_flagged(repo) -> None:
    now = setup(repo)
    repo.record_decision(
        "paper-claude",
        now - timedelta(hours=1),
        prompt_hash=FP,
        error="Claude Code error: usage limit reached",
    )
    card = scorecard(SETTINGS, repo, equity(), now)
    check = next(c for c in card.criteria if c.name == "Model reached every cycle")
    assert check.verdict is Verdict.CHECK and "usage limit" in check.detail
    assert card.ready  # CHECK items need your judgement but don't block


def test_config_change_notice(repo) -> None:
    assert config_change_notice(SETTINGS, repo) is None  # nothing recorded yet
    repo.ensure_account(
        "paper-claude", kind="paper", quote_currency="CAD", starting_cash=D(1), now=T0
    )
    repo.record_decision("paper-claude", T0, prompt_hash=FP)
    assert config_change_notice(SETTINGS, repo) is None
    repo.record_decision("paper-claude", T0 + timedelta(hours=1), prompt_hash="0ldc0nf1g0000000")
    notice = config_change_notice(SETTINGS, repo)
    assert notice is not None and "clock restarts" in notice


def test_error_halts_and_model_changes_fail(repo) -> None:
    now = setup(repo)
    repo.add_halt(
        HaltKind.ERRORS, "3 failed cycles", T0 + timedelta(days=5), account_id="paper-claude"
    )
    repo.record_decision(
        "paper-claude",
        now - timedelta(hours=1),
        prompt_hash=FP,
        model="claude_code/claude-sonnet-5-5",
        completion_tokens=90,
    )
    v = verdicts(scorecard(SETTINGS, repo, equity(), now))
    assert v["3. No bugs or state mismatches"] is Verdict.FAIL
    assert v["Same model throughout"] is Verdict.FAIL


def test_integrity_check_detects_problems(repo) -> None:
    setup(repo, trades=0)
    assert integrity_check(repo) == []
    # A stop larger than the position, and a decision pointing to a missing order.
    repo.save_order(
        Order(
            "s1",
            "paper-claude",
            "BTC/CAD",
            Side.SELL,
            OrderType.STOP_LOSS,
            D(1),
            OrderStatus.OPEN,
            T0,
            trigger_price=D(90),
        ),
        T0,
    )
    repo.record_decision("paper-claude", T0, order_id="ghost")
    issues = integrity_check(repo)
    assert any("exceeds the position" in i for i in issues)
    assert any("missing order" in i for i in issues)


async def test_integrity_clean_after_real_trading(repo) -> None:
    from ai_trader.brokers.base import OrderRequest
    from ai_trader.brokers.paper import PaperBroker

    from .fakes import FakeBookSource, FakeClock, book

    market = FakeBookSource(books={"BTC/CAD": book("BTC/CAD", [("99", "9")], [("100", "9")])})
    broker = PaperBroker(
        "paper-x",
        repo=repo,
        market=market,
        allowed_pairs=["BTC/CAD"],
        starting_cash=D(1000),
        taker_fee_pct=D("0.8"),
        clock=FakeClock(),
    )
    await broker.place_order(
        OrderRequest("BTC/CAD", Side.BUY, quote_amount=D(500), stop_loss_pct=D(5))
    )
    await broker.place_order(OrderRequest("BTC/CAD", Side.SELL, D("2")))
    await broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("100")))  # rejected: no cash
    assert integrity_check(repo) == []
