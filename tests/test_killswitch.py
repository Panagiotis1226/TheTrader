from __future__ import annotations

from decimal import Decimal

from ai_trader.brokers.base import OrderRequest, OrderType, Side
from ai_trader.brokers.paper import PaperBroker
from ai_trader.risk import killswitch
from ai_trader.storage.repo import HaltKind

from .fakes import T0, FakeBookSource, FakeClock, book

D = Decimal


class ExplodingBroker:
    account_id = "broken"

    async def cancel_all(self, *, keep_stop_losses: bool = True) -> None:
        raise RuntimeError("exchange down")


async def test_engage_halts_everything_and_keeps_stops(repo) -> None:
    clock = FakeClock()
    market = FakeBookSource(
        books={"BTC/CAD": book("BTC/CAD", bids=[("99900", "5")], asks=[("100000", "5")])}
    )
    broker = PaperBroker(
        "paper-a",
        repo=repo,
        market=market,
        allowed_pairs=["BTC/CAD"],
        starting_cash=D(10000),
        taker_fee_pct=D("0.4"),
        clock=clock,
    )
    await broker.place_order(OrderRequest("BTC/CAD", Side.BUY, D("0.01"), D(5)))

    failed = await killswitch.engage(repo, [ExplodingBroker(), broker], "/stop", T0)

    assert failed == ["broken"]  # one failure does not stop the others or the halt
    assert [h.kind for h in repo.active_halts("paper-a", T0)] == [HaltKind.MANUAL]
    assert [h.kind for h in repo.active_halts("any-other-account", T0)] == [HaltKind.MANUAL]
    [stop] = await broker.get_open_orders()
    assert stop.type is OrderType.STOP_LOSS

    assert killswitch.resume(repo, T0) == 1
    assert repo.active_halts("paper-a", T0) == []
