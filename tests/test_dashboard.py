from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from streamlit.testing.v1 import AppTest

from ai_trader.brokers.base import Fill, Order, OrderStatus, OrderType, Side
from ai_trader.dashboard.data import (
    account_order,
    decisions_frame,
    equity_frame,
    fills_frame,
    latest_equity,
    llm_cost_by_day,
    rejections_frame,
)
from ai_trader.reports import account_report, daily_closes, format_daily_summary
from ai_trader.storage.repo import Repository

from .conftest import REPO_ROOT

D = Decimal
T0 = datetime(2026, 9, 1, tzinfo=UTC)
APP = str(REPO_ROOT / "src" / "ai_trader" / "dashboard" / "app.py")


def populate(repo: Repository) -> None:
    for acct, end in (("paper-claude", "10500"), ("paper-do_nothing", "10000")):
        repo.ensure_account(
            acct, kind="paper", quote_currency="CAD", starting_cash=D(10000), now=T0
        )
        repo.record_equity(
            acct, T0 + timedelta(days=1), D("9500") if "claude" in acct else D(10000), D(0)
        )
        repo.record_equity(acct, T0 + timedelta(days=2), D(end), D(0))
    order = Order(
        "o1",
        "paper-claude",
        "BTC/CAD",
        Side.BUY,
        OrderType.MARKET,
        D("0.01"),
        OrderStatus.FILLED,
        T0,
    )
    repo.record_fill(
        order,
        Fill(
            "o1",
            "paper-claude",
            "BTC/CAD",
            Side.BUY,
            D("0.01"),
            D(100000),
            D(1000),
            D(8),
            "CAD",
            T0,
        ),
        T0,
    )
    repo.record_decision(
        "paper-claude",
        T0,
        action="buy",
        pair="BTC/CAD",
        size_pct=D(10),
        risk_outcome="approve",
        proposal={"reason": "trend", "confidence": "0.7"},
        cost_usd=D("0.03"),
        order_id="o1",
    )
    repo.record_decision(
        "paper-claude",
        T0 + timedelta(days=1),
        action="buy",
        pair="ETH/CAD",
        size_pct=D(10),
        risk_outcome="reject",
        risk_reason="max trades",
        cost_usd=D("0.02"),
    )


def test_data_frames(repo) -> None:
    populate(repo)
    eq = equity_frame(repo)
    latest = latest_equity(eq).set_index("account")
    assert latest.loc["paper-claude", "return_pct"] == pytest.approx(5.0)
    assert latest.loc["paper-do_nothing", "equity"] == 10000
    assert len(fills_frame(repo)) == 1
    decisions = decisions_frame(repo)
    assert list(rejections_frame(decisions)["risk_reason"]) == ["max trades"]
    cost = llm_cost_by_day(decisions)
    assert cost["cost_usd"].sum() == pytest.approx(0.05)
    assert account_order(repo, ["paper-do_nothing"]) == ["paper-do_nothing", "paper-claude"]


def test_account_report_metrics(repo) -> None:
    populate(repo)
    r = account_report(repo, "paper-claude", D(10500), [], T0 + timedelta(days=2))
    assert r.return_pct == 5.0
    assert r.max_drawdown_pct == 5.0  # 10000 -> 9500
    assert (r.trades, r.fees) == (1, D(8))
    assert (r.trade_proposals, r.rejection_rate_pct) == (2, 50.0)
    assert r.llm_cost_usd == D("0.05")
    text = format_daily_summary([r], T0)
    assert "paper-claude: 10,500.00 (+5.00%)" in text


def test_daily_closes_takes_last_value_per_day() -> None:
    series = [(T0, D(1)), (T0 + timedelta(hours=5), D(2)), (T0 + timedelta(days=1), D(3))]
    assert daily_closes(series) == [D(2), D(3)]


def test_dashboard_renders(tmp_path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'dash.db'}"
    populate(Repository.from_url(url))
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.chdir(REPO_ROOT)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["paper-claude", "paper-do_nothing"]
    assert any(s.value == "Decision log" for s in at.subheader)


def test_dashboard_empty_database(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}")
    monkeypatch.chdir(REPO_ROOT)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "No accounts yet" in at.info[0].value
