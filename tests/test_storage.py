from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ai_trader.storage.repo import Repository

D = Decimal
T = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _account(repo: Repository) -> None:
    repo.ensure_account("a", kind="paper", quote_currency="CAD", starting_cash=D(1), now=T)


def test_decimals_round_trip_exactly(repo) -> None:
    _account(repo)
    value = D("0.1") + D("0.2")
    repo.record_equity("a", T, value, D("123456789.123456789"))
    assert repo.equity_at_or_before("a", T) == D("0.3")


def test_peak_is_numeric_not_lexicographic(repo) -> None:
    _account(repo)
    repo.record_equity("a", T, D("9999"), D(0))
    repo.record_equity("a", T + timedelta(hours=1), D("10000"), D(0))
    assert repo.peak_equity_since("a", None) == D("10000")


def test_datetimes_are_aware_utc(repo) -> None:
    eastern = timezone(timedelta(hours=-5))
    repo.ensure_account(
        "b",
        kind="paper",
        quote_currency="CAD",
        starting_cash=D(1),
        now=datetime(2026, 1, 1, 7, tzinfo=eastern),
    )
    [acct] = [a for a in repo.list_accounts() if a.id == "b"]
    assert acct.created_at == T
    assert acct.created_at.tzinfo is UTC


def test_naive_datetime_and_float_rejected(repo) -> None:
    _account(repo)
    with pytest.raises(Exception, match="naive"):
        repo.record_equity("a", datetime(2026, 1, 1), D(1), D(1))  # noqa: DTZ001
    with pytest.raises(Exception, match="expected Decimal"):
        repo.record_equity("a", T, 1.5, D(1))  # type: ignore[arg-type]


def test_existing_account_keeps_original_values(repo) -> None:
    _account(repo)
    again = repo.ensure_account("a", kind="live", quote_currency="USD", starting_cash=D(999), now=T)
    assert (again.kind, again.starting_cash) == ("paper", D(1))


def test_decision_round_trip(repo) -> None:
    _account(repo)
    did = repo.record_decision(
        "a",
        T,
        action="buy",
        pair="BTC/CAD",
        size_pct=D(5),
        snapshot_hash="h",
        proposal={"action": "buy"},
    )
    repo.update_decision(did, risk_outcome="approve", risk_reason="ok")
    [rec] = repo.recent_decisions("a")
    assert (rec.id, rec.action, rec.risk_outcome, rec.fill_price) == (did, "buy", "approve", None)


def test_sqlite_parent_dir_created(tmp_path) -> None:
    Repository.from_url(f"sqlite:///{tmp_path / 'nested' / 'dir' / 'x.db'}")
    assert (tmp_path / "nested" / "dir" / "x.db").exists()
