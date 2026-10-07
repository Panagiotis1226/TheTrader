from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.ai.schema import TradeProposal
from ai_trader.brokers.base import OrderRequest, Side
from ai_trader.brokers.paper import PaperBroker
from ai_trader.config import load_trading_settings
from ai_trader.risk.manager import (
    AccountState,
    RiskManager,
    RiskOutcome,
    halt_until,
    load_account_state,
    record_halt,
)
from ai_trader.storage.repo import HaltKind

from .conftest import SETTINGS_PATH
from .fakes import T0, FakeBookSource, FakeClock, book

D = Decimal
SETTINGS = load_trading_settings(SETTINGS_PATH)  # 10/30/60 caps, 3% day, 15% DD, 0.6 conf
BTC_ASK = D("100000")


@pytest.fixture
def rm() -> RiskManager:
    return RiskManager(SETTINGS.risk, SETTINGS.pairs, taker_fee_pct=D("0.40"))


def state(**overrides) -> AccountState:
    base = AccountState(
        account_id="paper-test",
        now=T0,
        equity=D(10000),
        cash=D(10000),
        position_amounts={},
        position_values={},
        buy_prices={"BTC/CAD": BTC_ASK, "ETH/CAD": D(4000)},
        day_start_equity=D(10000),
        peak_equity=D(10000),
        trades_today=0,
        last_trade_at=None,
    )
    return replace(base, **overrides)


def with_positions(cash: str, **values: str) -> dict:
    """State overrides for positions given as CAD values, e.g. BTC='2500'."""
    pv = {f"{k}/CAD": D(v) for k, v in values.items()}
    amounts = {p: v / (BTC_ASK if p == "BTC/CAD" else D(4000)) for p, v in pv.items()}
    return dict(
        cash=D(cash),
        position_values=pv,
        position_amounts=amounts,
        equity=D(cash) + sum(pv.values()),
    )


def proposal(action="buy", pair="BTC/CAD", size="5", confidence="0.8", stop=None):
    return TradeProposal(
        action=action,
        pair=pair,
        size_pct=D(size),
        confidence=D(confidence),
        reason="test",
        stop_loss_pct=D(stop) if stop is not None else None,
    )


# ----------------------------------------------------------------------- approve


def test_hold_is_approved_without_order(rm) -> None:
    d = rm.evaluate(proposal(action="hold", size="0", confidence="0"), state())
    assert d.outcome is RiskOutcome.APPROVE
    assert d.order is None
    assert not d.tradable


def test_buy_within_limits_approved(rm) -> None:
    d = rm.evaluate(proposal(size="5"), state(), decision_id=42)
    assert d.outcome is RiskOutcome.APPROVE
    assert d.approved_notional == D(500)
    assert d.order == OrderRequest(
        pair="BTC/CAD", side=Side.BUY, quote_amount=D(500), stop_loss_pct=D(5), decision_id=42
    )
    assert d.tradable


def test_buy_exactly_at_min_confidence_approved(rm) -> None:
    assert rm.evaluate(proposal(confidence="0.6"), state()).outcome is RiskOutcome.APPROVE


# ------------------------------------------------------------------------ resize


def test_resize_to_max_trade_pct(rm) -> None:
    d = rm.evaluate(proposal(size="20"), state())
    assert d.outcome is RiskOutcome.RESIZE
    assert d.requested_notional == D(2000)
    assert d.approved_notional == D(1000)
    assert "max_trade_pct_of_equity" in d.reason
    assert d.order.quote_amount == D(1000)


def test_resize_to_max_position_per_pair(rm) -> None:
    s = state(**with_positions("7500", BTC="2500"))  # BTC at 25%
    d = rm.evaluate(proposal(size="10"), s)
    assert d.outcome is RiskOutcome.RESIZE
    assert d.approved_notional == D(500)
    assert "max_position_pct_per_pair" in d.reason


def test_resize_to_max_total_exposure(rm) -> None:
    s = state(**with_positions("4500", BTC="2000", ETH="3500"))  # 55% exposed
    d = rm.evaluate(proposal(size="10"), s)
    assert d.outcome is RiskOutcome.RESIZE
    assert d.approved_notional == D(500)
    assert "max_total_exposure_pct" in d.reason


def test_resize_to_available_cash_after_fee(rm) -> None:
    s = state(cash=D(300), equity=D(10000))
    d = rm.evaluate(proposal(size="10"), s)
    assert d.outcome is RiskOutcome.RESIZE
    assert "available cash" in d.reason
    assert d.approved_notional * D("1.004") == pytest.approx(D(300))


# ------------------------------------------------------------------------ reject


def test_reject_when_pair_position_full(rm) -> None:
    d = rm.evaluate(proposal(size="5"), state(**with_positions("7000", BTC="3000")))
    assert d.outcome is RiskOutcome.REJECT
    assert "max_position_pct_per_pair" in d.reason
    assert d.order is None


def test_reject_non_whitelisted_pair(rm) -> None:
    d = rm.evaluate(proposal(pair="DOGE/CAD"), state())
    assert d.outcome is RiskOutcome.REJECT
    assert "whitelisted" in d.reason


def test_reject_low_confidence(rm) -> None:
    d = rm.evaluate(proposal(confidence="0.59"), state())
    assert d.outcome is RiskOutcome.REJECT
    assert "confidence" in d.reason


def test_reject_max_trades_per_day(rm) -> None:
    d = rm.evaluate(proposal(), state(trades_today=6))
    assert d.outcome is RiskOutcome.REJECT
    assert "max trades per day" in d.reason
    assert rm.evaluate(proposal(), state(trades_today=5)).outcome is RiskOutcome.APPROVE


def test_reject_min_minutes_between_trades(rm) -> None:
    recent = state(last_trade_at=T0 - timedelta(minutes=59))
    assert rm.evaluate(proposal(), recent).outcome is RiskOutcome.REJECT
    ok = state(last_trade_at=T0 - timedelta(minutes=60))
    assert rm.evaluate(proposal(), ok).outcome is RiskOutcome.APPROVE


def test_reject_zero_size(rm) -> None:
    assert rm.evaluate(proposal(size="0"), state()).outcome is RiskOutcome.REJECT


def test_reject_without_price(rm) -> None:
    d = rm.evaluate(proposal(), state(buy_prices={}))
    assert d.outcome is RiskOutcome.REJECT
    assert "no current price" in d.reason


def test_reject_non_positive_equity(rm) -> None:
    d = rm.evaluate(
        proposal(), state(equity=D(0), cash=D(0), day_start_equity=D(0), peak_equity=D(0))
    )
    assert d.outcome is RiskOutcome.REJECT


@pytest.mark.parametrize("kind", list(HaltKind))
def test_buys_rejected_under_any_halt(rm, kind) -> None:
    d = rm.evaluate(proposal(), state(active_halts=(kind,)))
    assert d.outcome is RiskOutcome.REJECT
    assert "halted" in d.reason
    assert d.halt is None  # already recorded


@pytest.mark.parametrize("kind", [HaltKind.MANUAL, HaltKind.ERRORS])
def test_full_stop_halts_reject_sells(rm, kind) -> None:
    s = state(**with_positions("7000", BTC="3000"), active_halts=(kind,))
    assert rm.evaluate(proposal(action="sell", size="100"), s).outcome is RiskOutcome.REJECT


@pytest.mark.parametrize("kind", [HaltKind.DAILY_LOSS, HaltKind.DRAWDOWN])
def test_risk_limit_halts_still_allow_sells(rm, kind) -> None:
    s = state(**with_positions("7000", BTC="3000"), active_halts=(kind,))
    d = rm.evaluate(proposal(action="sell", size="100"), s)
    assert d.outcome is RiskOutcome.APPROVE
    assert d.order.amount == D("0.03")


def test_mixed_halts_are_a_full_stop(rm) -> None:
    s = state(
        **with_positions("7000", BTC="3000"), active_halts=(HaltKind.DRAWDOWN, HaltKind.MANUAL)
    )
    assert rm.evaluate(proposal(action="sell", size="100"), s).outcome is RiskOutcome.REJECT


def test_newly_breached_limit_still_allows_sells(rm) -> None:
    s = state(
        equity=D(8500),
        cash=D(5500),
        peak_equity=D(10000),
        day_start_equity=D(8500),
        position_amounts={"BTC/CAD": D("0.03")},
        position_values={"BTC/CAD": D(3000)},
    )
    d = rm.evaluate(proposal(action="sell", size="100"), s)
    assert d.halt is HaltKind.DRAWDOWN
    assert d.outcome is RiskOutcome.APPROVE


def test_hold_allowed_while_halted(rm) -> None:
    d = rm.evaluate(proposal(action="hold"), state(active_halts=(HaltKind.MANUAL,)))
    assert d.outcome is RiskOutcome.APPROVE
    assert d.order is None


# --------------------------------------------------------------------------- halts


def test_daily_loss_limit_triggers_halt(rm) -> None:
    s = state(equity=D(9700), cash=D(9700))  # exactly -3% on the day
    d = rm.evaluate(proposal(), s)
    assert d.outcome is RiskOutcome.REJECT
    assert d.halt is HaltKind.DAILY_LOSS
    # Detected even when the model holds, so the cycle records the halt.
    assert rm.evaluate(proposal(action="hold"), s).halt is HaltKind.DAILY_LOSS


def test_daily_loss_just_under_limit_is_fine(rm) -> None:
    d = rm.evaluate(proposal(), state(equity=D(9701), cash=D(9701)))
    assert d.outcome is RiskOutcome.APPROVE
    assert d.halt is None


def test_drawdown_halt_takes_precedence(rm) -> None:
    s = state(equity=D(8500), cash=D(8500), peak_equity=D(10000), day_start_equity=D(9000))
    d = rm.evaluate(proposal(), s)
    assert d.outcome is RiskOutcome.REJECT
    assert d.halt is HaltKind.DRAWDOWN


def test_drawdown_not_resignalled_when_already_halted(rm) -> None:
    s = state(
        equity=D(8000),
        cash=D(8000),
        peak_equity=D(10000),
        day_start_equity=D(8000),
        active_halts=(HaltKind.DRAWDOWN,),
    )
    assert rm.evaluate(proposal(), s).halt is None


# ---------------------------------------------------------------------------- sells


def test_sell_percent_of_position(rm) -> None:
    s = state(**with_positions("7000", BTC="3000"))
    d = rm.evaluate(proposal(action="sell", size="50"), s)
    assert d.outcome is RiskOutcome.APPROVE
    assert d.order.side is Side.SELL
    assert d.order.amount == D("0.015")
    assert d.order.stop_loss_pct is None


def test_sell_without_position_rejected(rm) -> None:
    d = rm.evaluate(proposal(action="sell", size="50"), state())
    assert d.outcome is RiskOutcome.REJECT
    assert "no BTC/CAD position" in d.reason


def test_sell_blocked_by_manual_halt_and_confidence(rm) -> None:
    s = state(**with_positions("7000", BTC="3000"))
    halted = replace(s, active_halts=(HaltKind.MANUAL,))
    assert rm.evaluate(proposal(action="sell", size="100"), halted).outcome is RiskOutcome.REJECT
    low = proposal(action="sell", size="100", confidence="0.1")
    assert rm.evaluate(low, s).outcome is RiskOutcome.REJECT


# ------------------------------------------------------------------------ stop-loss


@pytest.mark.parametrize(("requested", "expected"), [(None, "5"), ("3", "3"), ("10", "5")])
def test_stop_loss_can_only_tighten(rm, requested, expected) -> None:
    d = rm.evaluate(proposal(stop=requested), state())
    assert d.order.stop_loss_pct == D(expected)


# ------------------------------------------------------------- state from the DB


def test_halt_until() -> None:
    now = datetime(2026, 3, 1, 22, 30, tzinfo=UTC)
    assert halt_until(HaltKind.DAILY_LOSS, now) == datetime(2026, 3, 2, tzinfo=UTC)
    assert halt_until(HaltKind.DRAWDOWN, now) is None
    assert halt_until(HaltKind.MANUAL, now) is None


async def test_load_account_state(repo) -> None:
    clock = FakeClock(datetime(2026, 3, 2, 10, 0, tzinfo=UTC))
    btc = book("BTC/CAD", bids=[("99900", "5")], asks=[("100000", "5")])
    eth = book("ETH/CAD", bids=[("3999", "50")], asks=[("4000", "50")])
    market = FakeBookSource(books={"BTC/CAD": btc, "ETH/CAD": eth})
    broker = PaperBroker(
        "paper-x",
        repo=repo,
        market=market,
        allowed_pairs=SETTINGS.pairs,
        starting_cash=D(10000),
        taker_fee_pct=D("0.40"),
        clock=clock,
    )
    acct = "paper-x"
    repo.record_equity(acct, datetime(2026, 3, 1, 12, tzinfo=UTC), D(11000), D(11000))  # peak
    repo.record_equity(acct, datetime(2026, 3, 1, 23, tzinfo=UTC), D(10500), D(10500))
    repo.record_equity(acct, datetime(2026, 3, 2, 1, tzinfo=UTC), D(10200), D(10200))

    await broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("0.01"), D(5)))
    clock.advance(minutes=5)

    s = await load_account_state(broker, repo, market.books, clock.now)
    assert s.cash == D(10000) - D(1000) - D(4)
    assert s.position_amounts == {"BTC/CAD": D("0.01")}
    assert s.position_values == {"BTC/CAD": D("999.000")}
    assert s.equity == s.cash + D("999")
    assert s.day_start_equity == D(10500)  # last snapshot before UTC midnight
    assert s.peak_equity == D(11000)
    assert s.trades_today == 1
    assert s.last_trade_at == datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    assert s.buy_prices == {"BTC/CAD": D(100000), "ETH/CAD": D(4000)}
    assert s.active_halts == ()


async def test_stop_fills_do_not_count_as_trades(repo) -> None:
    clock = FakeClock()
    market = FakeBookSource(
        books={"BTC/CAD": book("BTC/CAD", bids=[("99900", "5")], asks=[("100000", "5")])}
    )
    broker = PaperBroker(
        "paper-y",
        repo=repo,
        market=market,
        allowed_pairs=["BTC/CAD"],
        starting_cash=D(10000),
        taker_fee_pct=D("0.4"),
        clock=clock,
    )
    await broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("0.01"), D(5)))
    market.books["BTC/CAD"] = book("BTC/CAD", bids=[("90000", "5")], asks=[("90001", "5")])
    clock.advance(minutes=1)
    assert len(await broker.check_stops()) == 1
    count, last = repo.trade_stats("paper-y", T0 - timedelta(hours=1))
    assert count == 1
    assert last == T0


def test_daily_halt_expires_at_midnight_and_drawdown_needs_resume(repo) -> None:
    now = datetime(2026, 3, 1, 15, tzinfo=UTC)
    repo.ensure_account("paper-z", kind="paper", quote_currency="CAD", starting_cash=D(1), now=now)
    record_halt(repo, "paper-z", HaltKind.DAILY_LOSS, "day loss", now)
    record_halt(repo, "paper-z", HaltKind.DRAWDOWN, "dd", now)
    assert {h.kind for h in repo.active_halts("paper-z", now)} == {
        HaltKind.DAILY_LOSS,
        HaltKind.DRAWDOWN,
    }
    next_day = datetime(2026, 3, 2, 0, 0, 1, tzinfo=UTC)
    assert [h.kind for h in repo.active_halts("paper-z", next_day)] == [HaltKind.DRAWDOWN]
    assert repo.resume(next_day) >= 1
    assert repo.active_halts("paper-z", next_day) == []
    assert repo.last_resume("paper-z", HaltKind.DRAWDOWN) == next_day


async def test_peak_resets_after_drawdown_resume(repo) -> None:
    clock = FakeClock(datetime(2026, 3, 5, 12, tzinfo=UTC))
    market = FakeBookSource(books={})
    broker = PaperBroker(
        "paper-p",
        repo=repo,
        market=market,
        allowed_pairs=["BTC/CAD"],
        starting_cash=D(8000),
        taker_fee_pct=D("0.4"),
        clock=clock,
    )
    repo.record_equity("paper-p", datetime(2026, 3, 1, tzinfo=UTC), D(10000), D(10000))
    record_halt(repo, "paper-p", HaltKind.DRAWDOWN, "dd", datetime(2026, 3, 2, tzinfo=UTC))
    repo.resume(datetime(2026, 3, 3, tzinfo=UTC))
    repo.record_equity("paper-p", datetime(2026, 3, 4, tzinfo=UTC), D(8100), D(8100))

    s = await load_account_state(broker, repo, {}, clock.now)
    assert s.peak_equity == D(8100)
    assert RiskManager(SETTINGS.risk, SETTINGS.pairs, D("0.4")).detect_halt(s) is None


def test_global_halt_applies_to_every_account(repo) -> None:
    repo.add_halt(HaltKind.MANUAL, "/stop", T0)
    assert [h.kind for h in repo.active_halts("paper-anything", T0)] == [HaltKind.MANUAL]
