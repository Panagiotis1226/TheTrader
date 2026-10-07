from __future__ import annotations

import io
from decimal import Decimal

import pytest

from ai_trader.brokers.base import OrderRequest, OrderStatus, OrderType, Side
from ai_trader.brokers.paper import PaperBroker, walk_book
from ai_trader.data.market import BookLevel
from ai_trader.storage.repo import Repository

from .fakes import FakeBookSource, FakeClock, book

D = Decimal
FEE = D("0.40")


def btc_book():
    return book(
        "BTC/CAD",
        bids=[("99900", "0.01"), ("99800", "0.05"), ("99000", "1")],
        asks=[("100000", "0.01"), ("100100", "0.02"), ("100200", "1")],
    )


@pytest.fixture
def market() -> FakeBookSource:
    return FakeBookSource(
        books={
            "BTC/CAD": btc_book(),
            "ETH/CAD": book("ETH/CAD", bids=[("4000", "10")], asks=[("4001", "10")]),
        }
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_broker(repo: Repository, market, clock, account_id: str = "paper-test", cash="10000"):
    return PaperBroker(
        account_id,
        repo=repo,
        market=market,
        allowed_pairs=["BTC/CAD", "ETH/CAD"],
        starting_cash=D(cash),
        taker_fee_pct=FEE,
        clock=clock,
    )


def buy(amount: str, pair: str = "BTC/CAD", stop: str | None = None) -> OrderRequest:
    return OrderRequest(
        pair=pair, side=Side.BUY, amount=D(amount), stop_loss_pct=D(stop) if stop else None
    )


def sell(amount: str, pair: str = "BTC/CAD") -> OrderRequest:
    return OrderRequest(pair=pair, side=Side.SELL, amount=D(amount))


# ----------------------------------------------------------------------- fill math


def test_walk_book_across_levels() -> None:
    levels = [BookLevel(D("10"), D("1")), BookLevel(D("11"), D("2"))]
    walk = walk_book(levels, D("2.5"))
    assert walk.filled == D("2.5")
    assert walk.cost == D("26.5")
    assert walk.avg_price == D("10.6")
    assert walk_book(levels, D("5")).filled == D("3")


async def test_buy_walks_asks_and_pays_fee(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    result = await broker.place_order(buy("0.025"))

    assert result.status is OrderStatus.FILLED
    assert result.filled_amount == D("0.025")
    assert result.cost == D("2501.5")  # 0.01 @ 100000 + 0.015 @ 100100
    assert result.avg_price == D("100060")
    assert result.fee == D("10.006")  # 0.40% of cost
    balances = await broker.get_balances()
    assert balances["CAD"] == D("10000") - D("2501.5") - D("10.006")
    assert balances["BTC"] == D("0.025")
    [pos] = await broker.get_positions()
    assert pos.avg_entry_price == D("100060")


async def test_sell_walks_bids_and_pays_fee(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.05"))
    cash_before = broker.cash
    result = await broker.place_order(sell("0.03"))

    assert result.status is OrderStatus.FILLED
    assert result.cost == D("0.01") * D("99900") + D("0.02") * D("99800")
    assert result.fee == (result.cost * D("0.004")).quantize(D("0.00000001"))
    assert broker.cash == cash_before + result.cost - result.fee
    [pos] = await broker.get_positions()
    assert pos.amount == D("0.02")
    # Selling does not change the average entry price.
    expected_avg = (D("0.01") * 100000 + D("0.02") * 100100 + D("0.02") * 100200) / D("0.05")
    assert pos.avg_entry_price == expected_avg


async def test_fee_rounds_up_to_8_decimals(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    result = await broker.place_order(buy("0.00012345"))
    assert result.cost == D("12.345")
    assert result.fee == D("0.04938")


# ----------------------------------------------------------------------- rejections


async def test_insufficient_cash_rejected(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock, cash="1000")
    result = await broker.place_order(buy("0.01"))  # 1000 + 4 fee > 1000
    assert result.status is OrderStatus.REJECTED
    assert "insufficient CAD" in result.reason
    assert broker.cash == D("1000")
    assert await broker.get_positions() == []


async def test_fee_is_included_in_balance_check(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock, cash="1004")
    assert (await broker.place_order(buy("0.01"))).status is OrderStatus.FILLED
    assert broker.cash == D("0")


async def test_sell_more_than_held_rejected(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.001"))
    result = await broker.place_order(sell("0.002"))
    assert result.status is OrderStatus.REJECTED
    assert "insufficient BTC" in result.reason


async def test_sell_without_position_rejected(repo, market, clock) -> None:
    result = await make_broker(repo, market, clock).place_order(sell("0.001"))
    assert result.status is OrderStatus.REJECTED


async def test_below_min_amount_rejected(repo, market, clock) -> None:
    result = await make_broker(repo, market, clock).place_order(buy("0.00004"))
    assert result.status is OrderStatus.REJECTED
    assert "below Kraken minimum" in result.reason


async def test_below_min_cost_rejected(repo, market, clock) -> None:
    market.books["ETH/CAD"] = book("ETH/CAD", bids=[("0.5", "10")], asks=[("0.6", "10")])
    result = await make_broker(repo, market, clock).place_order(buy("0.001", pair="ETH/CAD"))
    assert result.status is OrderStatus.REJECTED
    assert "order value" in result.reason


async def test_amount_rounded_down_to_step(repo, market, clock) -> None:
    result = await make_broker(repo, market, clock).place_order(buy("0.000123456789"))
    assert result.filled_amount == D("0.00012345")


async def test_thin_book_rejected_not_partially_filled(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock, cash="1000000")
    result = await broker.place_order(buy("5"))  # book only has 1.03 BTC
    assert result.status is OrderStatus.REJECTED
    assert "too thin" in result.reason
    assert await broker.get_positions() == []


async def test_non_whitelisted_pair_rejected(repo, market, clock) -> None:
    result = await make_broker(repo, market, clock).place_order(buy("1", pair="DOGE/CAD"))
    assert result.status is OrderStatus.REJECTED
    assert "whitelisted" in result.reason


async def test_market_outage_rejects(repo, market, clock) -> None:
    market.fail = True
    result = await make_broker(repo, market, clock).place_order(buy("0.001"))
    assert result.status is OrderStatus.REJECTED
    assert "unavailable" in result.reason


async def test_rejections_are_persisted(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock, cash="10")
    await broker.place_order(buy("0.01"))
    assert repo.list_fills("paper-test") == []
    assert await broker.get_open_orders() == []  # rejected, not open


def test_order_request_validation() -> None:
    with pytest.raises(ValueError):
        OrderRequest(pair="BTC/CAD", side=Side.BUY, amount=D(0))
    with pytest.raises(ValueError):
        OrderRequest(pair="BTC/CAD", side=Side.BUY, amount=D(1), stop_loss_pct=D(100))


# ----------------------------------------------------------------------- stop-losses


async def test_buy_with_stop_creates_resting_stop(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    [stop] = await broker.get_open_orders()
    assert stop.type is OrderType.STOP_LOSS
    assert stop.side is Side.SELL
    assert stop.amount == D("0.01")
    assert stop.trigger_price == D("95000.0")  # 100000 * 0.95, rounded down to 0.1


async def test_stop_not_triggered_above_trigger(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    assert await broker.check_stops() == []
    assert len(await broker.get_open_orders()) == 1


async def test_stop_triggers_as_market_sell(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    market.books["BTC/CAD"] = book(
        "BTC/CAD", bids=[("94900", "0.004"), ("94000", "1")], asks=[("95000", "1")]
    )
    [result] = await broker.check_stops()

    assert result.status is OrderStatus.FILLED
    assert result.cost == D("0.004") * 94900 + D("0.006") * 94000  # gap slippage
    assert await broker.get_positions() == []
    assert await broker.get_open_orders() == []
    fills = repo.list_fills("paper-test")
    assert [f.order_type for f in fills] == [OrderType.MARKET, OrderType.STOP_LOSS]


async def test_stop_triggers_exactly_at_trigger(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    market.books["BTC/CAD"] = book("BTC/CAD", bids=[("95000", "1")], asks=[("95001", "1")])
    [result] = await broker.check_stops()
    assert result.filled


async def test_second_buy_replaces_stop_for_whole_position(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    market.books["BTC/CAD"] = book("BTC/CAD", bids=[("109900", "5")], asks=[("110000", "5")])
    await broker.place_order(buy("0.01", stop="5"))
    [stop] = await broker.get_open_orders()
    assert stop.amount == D("0.02")
    assert stop.trigger_price == D("99750.0")  # avg entry 105000 * 0.95


async def test_partial_sell_shrinks_stop_and_full_sell_cancels_it(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    await broker.place_order(sell("0.004"))
    [stop] = await broker.get_open_orders()
    assert stop.amount == D("0.006")
    assert stop.trigger_price == D("95000.0")
    await broker.place_order(sell("0.006"))
    assert await broker.get_open_orders() == []


async def test_stop_on_dust_position_is_cancelled_not_retried(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.0001", stop="5"))
    await broker.place_order(sell("0.00006"))  # leaves 0.00004 BTC, below the 0.00005 minimum
    market.books["BTC/CAD"] = book("BTC/CAD", bids=[("90000", "1")], asks=[("90001", "1")])
    [result] = await broker.check_stops()
    assert result.status is OrderStatus.CANCELLED
    assert await broker.get_open_orders() == []


async def test_stop_stays_open_during_outage(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    market.fail = True
    assert await broker.check_stops() == []
    assert len(await broker.get_open_orders()) == 1


async def test_cancel_all_cancels_open_orders(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01", stop="5"))
    await broker.place_order(buy("0.01", pair="ETH/CAD", stop="5"))
    assert len(await broker.get_open_orders()) == 2
    await broker.cancel_all()
    assert await broker.get_open_orders() == []
    assert len(await broker.get_positions()) == 2  # positions untouched


# --------------------------------------------------------------- persistence/accounts


async def test_state_survives_restart(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.025", stop="5"))
    await broker.place_order(sell("0.005"))
    balances, positions = await broker.get_balances(), await broker.get_positions()

    restarted = make_broker(repo, market, clock, cash="999999")  # ignored for existing account
    assert await restarted.get_balances() == balances
    assert await restarted.get_positions() == positions
    assert len(await restarted.get_open_orders()) == 1


async def test_accounts_are_independent(repo, market, clock) -> None:
    a = make_broker(repo, market, clock, account_id="paper-a")
    b = make_broker(repo, market, clock, account_id="paper-b", cash="500")
    await a.place_order(buy("0.01", stop="5"))
    assert (await b.get_balances()) == {"CAD": D("500")}
    assert await b.get_open_orders() == []
    await b.cancel_all()
    assert len(await a.get_open_orders()) == 1


async def test_equity_marks_positions_at_best_bid(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01"))
    assert await broker.get_equity() == broker.cash + D("0.01") * D("99900")
    with pytest.raises(ValueError):
        await broker.get_equity("USD")


async def test_fills_csv_export(repo, market, clock) -> None:
    broker = make_broker(repo, market, clock)
    await broker.place_order(buy("0.01"))
    await broker.place_order(sell("0.01"))
    out = io.StringIO()
    assert repo.export_fills_csv(out) == 2
    lines = out.getvalue().strip().splitlines()
    assert lines[0].startswith("datetime_utc,account,pair,side,quantity,price,fee")
    assert lines[1].split(",")[2:6] == ["BTC/CAD", "buy", "0.01", "100000"]
