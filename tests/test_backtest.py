from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ai_trader.ai.agent import AgentResult
from ai_trader.ai.schema import TradeProposal
from ai_trader.backtest import metrics
from ai_trader.backtest.data import (
    candles_path,
    merge_candles,
    read_candles_csv,
    save_market_infos,
    write_candles_csv,
)
from ai_trader.backtest.engine import BacktestEngine, format_reports
from ai_trader.backtest.replay import ReplayClock, ReplayMarket
from ai_trader.brokers.base import Fill, OrderType, Side
from ai_trader.config import load_config, load_trading_settings
from ai_trader.data.market import Candle
from ai_trader.data.snapshot import MarketSnapshot
from ai_trader.main import run_backtest
from ai_trader.strategies import BuyAndHold, DoNothing, MACrossover, build_benchmark

from .conftest import SETTINGS_PATH
from .fakes import BTC_INFO, ETH_INFO

D = Decimal
DAY = timedelta(days=1)
DAY0 = datetime(2025, 1, 1, tzinfo=UTC)
BASE = load_trading_settings(SETTINGS_PATH)
# One pair, 2 warm-up days, no slippage, 0.8% fee: every number below is checkable by hand.
SETTINGS = BASE.model_copy(
    update={
        "pairs": ["BTC/CAD"],
        "backtest": BASE.backtest.model_copy(update={"warmup_days": 2, "slippage_pct": D(0)}),
    }
)
INFOS = {"BTC/CAD": BTC_INFO, "ETH/CAD": ETH_INFO}


def series(closes, start=DAY0, lows=None, opens=None) -> list[Candle]:
    out = []
    for i, c in enumerate(closes):
        c = D(str(c))
        o = D(str(opens[i])) if opens else (D(str(closes[i - 1])) if i else c)
        low = D(str(lows[i])) if lows else min(o, c)
        out.append(Candle(start + i * DAY, o, max(o, c), low, c, D(1)))
    return out


class Scripted:
    """Decider returning proposals by step index; records what it saw."""

    name = "scripted"

    def __init__(self, proposals: dict[int, TradeProposal]) -> None:
        self.proposals = proposals
        self.seen: list[MarketSnapshot] = []

    async def decide(self, snapshot: MarketSnapshot) -> AgentResult:
        step = len(self.seen)
        self.seen.append(snapshot)
        p = self.proposals.get(step, TradeProposal.hold("BTC/CAD", "idle"))
        return AgentResult(p, model=self.name)


def buy(size="10", stop=None) -> TradeProposal:
    return TradeProposal(
        action="buy",
        pair="BTC/CAD",
        size_pct=D(size),
        confidence=D(1),
        reason="t",
        stop_loss_pct=D(stop) if stop else None,
    )


def engine(candles, settings=SETTINGS, **kw) -> BacktestEngine:
    return BacktestEngine(settings, {"BTC/CAD": candles}, INFOS, **kw)


# ------------------------------------------------------------------------- metrics


def test_metrics() -> None:
    eq = [D(100), D(110), D(99), D(121)]
    assert metrics.total_return_pct(eq) == pytest.approx(21.0)
    assert metrics.max_drawdown_pct(eq) == pytest.approx(10.0)
    assert metrics.cagr_pct([D(100), D(121)], 730) == pytest.approx(10.0)
    assert metrics.sharpe([D(100), D(100), D(100)]) is None
    assert metrics.sharpe([D(100), D(101), D(103), D(102)]) > 0


def test_win_rate_uses_average_cost_after_fees() -> None:
    def fill(side, amount, price, fee="0"):
        return Fill(
            "o",
            "a",
            "BTC/CAD",
            side,
            D(amount),
            D(price),
            D(amount) * D(price),
            D(fee),
            "CAD",
            DAY0,
        )

    fills = [
        fill(Side.BUY, "1", "100", "1"),
        fill(Side.SELL, "0.5", "101", "1"),  # 50.5 - 1 = 49.5 < 50.5 cost: loss
        fill(Side.SELL, "0.5", "110", "0"),  # 55 > 50.5: win
    ]
    assert metrics.win_rate_pct(fills) == (2, 50.0)


# ---------------------------------------------------------------------------- data


def test_candle_csv_round_trip_and_kraken_format(tmp_path) -> None:
    candles = series([100, 101, 102])
    path = tmp_path / "x.csv"
    write_candles_csv(path, candles)
    assert read_candles_csv(path) == candles

    kraken = tmp_path / "kraken.csv"  # Kraken OHLCVT export: no header, has trades column
    kraken.write_text(f"{int(DAY0.timestamp())},1,2,0.5,1.5,10,42\n")
    [c] = read_candles_csv(kraken)
    assert (c.opened_at, c.close, c.volume) == (DAY0, D("1.5"), D(10))


def test_merge_candles_dedupes_and_sorts() -> None:
    old = series([1, 2, 3])
    new = series([9, 10], start=DAY0 + 2 * DAY)
    merged = merge_candles(old, new)
    assert [c.close for c in merged] == [D(1), D(2), D(9), D(10)]


# -------------------------------------------------------------------------- replay


async def test_replay_never_shows_unclosed_candles() -> None:
    candles = series([100, 110, 120])
    clock = ReplayClock(DAY0 + 2 * DAY)  # day 1's candle just closed; day 2's is open
    market = ReplayMarket({"BTC/CAD": candles}, INFOS, D("1"), clock)
    assert [c.close for c in await market.fetch_ohlcv("BTC/CAD", "1d")] == [D(100), D(110)]
    assert await market.fetch_ohlcv("BTC/CAD", "1h") == []
    ticker = await market.fetch_ticker("BTC/CAD")
    assert ticker.last == D(110)
    book = await market.fetch_order_book("BTC/CAD")
    assert (book.best_bid, book.best_ask) == (D("108.90"), D("111.10"))


# -------------------------------------------------------------------------- engine


async def test_window_respects_warmup_and_dates() -> None:
    candles = series(range(100, 110))
    assert len(engine(candles).steps) == 8  # 10 days - 2 warm-up
    e = engine(candles, start=DAY0 + 5 * DAY, end=DAY0 + 7 * DAY)
    assert e.steps == [DAY0 + 5 * DAY, DAY0 + 6 * DAY, DAY0 + 7 * DAY]
    with pytest.raises(ValueError):
        engine(series([1, 2]))


async def test_no_look_ahead() -> None:
    candles = series([100, 101, 102, 103, 104, 105])
    decider = Scripted({})
    await engine(candles).run(decider, stop_loss_required=True)
    for snap in decider.seen:
        # The latest price the decider sees is the close of the candle that just closed.
        just_closed = next(c for c in candles if c.opened_at + DAY == snap.timestamp)
        assert snap.pairs[0].last_price == just_closed.close


async def test_do_nothing_is_flat() -> None:
    result = await engine(series(range(100, 130))).run(
        DoNothing("BTC/CAD"), stop_loss_required=False
    )
    r = result.report
    assert r.total_return_pct == 0 and r.max_drawdown_pct == 0 and r.trades == 0
    assert all(e == SETTINGS.paper.starting_cash_cad for _, e in result.equity_curve)


async def test_buy_and_hold_accumulates_then_holds() -> None:
    closes = [100] * 3 + [100, 100, 100] + [200] * 5 + [50] * 5  # never sells into the crash
    result = await engine(series(closes)).run(
        BuyAndHold("BTC/CAD", SETTINGS.risk), stop_loss_required=False
    )
    buys = [f for f in result.fills if f.side is Side.BUY]
    assert len(buys) == 3 and not [f for f in result.fills if f.side is Side.SELL]
    assert result.report.fees_paid == (sum(f.cost for f in buys) * D("0.008")).quantize(D("0.01"))


async def test_trade_pays_fee_and_marks_equity() -> None:
    closes = [100, 100, 100, 110]
    result = await engine(series(closes)).run(Scripted({0: buy("10")}), stop_loss_required=False)
    [fill] = result.fills
    assert fill.cost == D(1000) and fill.fee == D(8)
    # Day after the buy: 9000 - 8 cash + 10 units * 110.
    assert result.equity_curve[-1][1] == D(9000) - D(8) + D(10) * D(110)


async def test_stop_loss_fills_at_trigger_when_low_touches() -> None:
    # Buy at 100 with a 5% stop (trigger 95); next day trades down to 90 intraday, closes 99.
    candles = series([100, 100, 100, 99], lows=[100, 100, 100, 90], opens=[100, 100, 100, 100])
    result = await engine(candles).run(Scripted({0: buy("10", stop="5")}), stop_loss_required=True)
    stop_fill = [f for f in result.fills if f.order_type is OrderType.STOP_LOSS]
    assert [f.price for f in stop_fill] == [D("95.0")]
    assert result.report.stop_loss_fills == 1


async def test_stop_loss_gap_down_fills_at_open() -> None:
    candles = series([100, 100, 100, 85], lows=[100, 100, 100, 80], opens=[100, 100, 100, 88])
    result = await engine(candles).run(Scripted({0: buy("10", stop="5")}), stop_loss_required=True)
    [stop_fill] = [f for f in result.fills if f.order_type is OrderType.STOP_LOSS]
    assert stop_fill.price == D(88)


async def test_benchmarks_run_without_forced_stops() -> None:
    candles = series([100, 100, 100, 100], lows=[100, 100, 100, 50])
    result = await engine(candles).run(Scripted({0: buy("10")}), stop_loss_required=False)
    assert all(f.order_type is OrderType.MARKET for f in result.fills)


async def test_max_decisions_uses_the_most_recent_days() -> None:
    decider = Scripted({})
    result = await engine(series(range(100, 120))).run(
        decider, stop_loss_required=True, max_decisions=3
    )
    assert result.report.decisions == 3
    assert decider.seen[-1].timestamp == DAY0 + 20 * DAY


async def test_report_formatting() -> None:
    result = await engine(series(range(100, 120))).run(
        DoNothing("BTC/CAD"), stop_loss_required=False
    )
    text = format_reports([result.report])
    assert "do_nothing" in text and "Market (100% hold, no fees): BTC/CAD" in text
    json.dumps(result.report.as_dict())  # serializable


# ---------------------------------------------------------------------- strategies


def snap(sma20=None, sma50=None, amount="0", avg=None, cash="10000") -> MarketSnapshot:
    from ai_trader.data.snapshot import PairSnapshot

    price = D(100)
    pair = PairSnapshot(
        pair="BTC/CAD",
        last_price=price,
        change_24h_pct=None,
        change_7d_pct=None,
        change_30d_pct=None,
        volatility_24h_annualized_pct=None,
        volatility_7d_annualized_pct=None,
        sma_20d=sma20,
        sma_50d=sma50,
        sma_200d=None,
        price_vs_sma_20d_pct=None,
        price_vs_sma_50d_pct=None,
        price_vs_sma_200d_pct=None,
        rsi_14d=None,
        bid=price,
        ask=price,
        spread_pct=0,
        bid_depth_1pct_quote=D(0),
        ask_depth_1pct_quote=D(0),
        position_amount=D(amount),
        position_value=D(amount) * price,
        avg_entry_price=D(avg) if avg else None,
        unrealized_pnl_pct=None,
    )
    return MarketSnapshot(
        timestamp=DAY0,
        data_age_seconds=0,
        quote_currency="CAD",
        cash_available=D(cash),
        total_equity=D(cash),
        pairs=[pair],
        recent_decisions=[],
    )


async def test_ma_crossover_signals() -> None:
    ma = MACrossover(SETTINGS.risk, "BTC/CAD")
    assert (await ma.decide(snap(sma20=110, sma50=100))).proposal.action == "buy"
    exit_ = (await ma.decide(snap(sma20=90, sma50=100, amount="10", avg="100"))).proposal
    assert (exit_.action, exit_.size_pct) == ("sell", D(100))
    assert (await ma.decide(snap(sma20=90, sma50=100))).proposal.action == "hold"
    assert (await ma.decide(snap())).proposal.action == "hold"  # no history yet
    full = snap(sma20=110, sma50=100, amount="30", avg="100", cash="7000")  # 30% at cost
    assert (await ma.decide(full)).proposal.action == "hold"


async def test_buy_and_hold_ignores_price_moves() -> None:
    bh = BuyAndHold("BTC/CAD", SETTINGS.risk)
    # 30% invested at cost; the price has since halved. Still "full": no dip buying.
    crashed = snap(amount="30", avg="100", cash="7000")
    assert (await bh.decide(crashed)).proposal.action == "hold"
    partial = (await bh.decide(snap(amount="25", avg="100", cash="7500"))).proposal
    assert (partial.action, partial.size_pct) == ("buy", D(5))


def test_build_benchmark() -> None:
    assert build_benchmark("ma_crossover", BASE).name == "ma_crossover"
    with pytest.raises(ValueError):
        build_benchmark("nope", BASE)


# ------------------------------------------------------------------------------ CLI


async def test_run_backtest_offline_with_llm(tmp_path, write_env, capsys) -> None:
    from .test_claude_code import FakeRunner, cli_output

    candles_dir = tmp_path / "candles"
    save_market_infos(candles_dir, INFOS)
    for pair in BASE.pairs:
        write_candles_csv(candles_path(candles_dir, pair), series(range(100, 180)))
    config = load_config(write_env(), SETTINGS_PATH)
    args = argparse.Namespace(
        start=None,
        end=None,
        strategies=None,
        with_llm=True,
        llm_days=2,
        refresh_data=False,
        candles_dir=candles_dir,
        out=tmp_path / "out",
    )
    runner = FakeRunner(
        cli_output(
            json.dumps(
                {
                    "action": "hold",
                    "pair": "BTC/CAD",
                    "size_pct": 0,
                    "confidence": 0.5,
                    "reason": "x",
                }
            )
        )
    )
    reports = await run_backtest(config, args, runner=runner)

    assert [r.strategy for r in reports] == [
        "buy_and_hold",
        "ma_crossover",
        "do_nothing",
        "claude_code/claude-opus-5-5",
    ]
    assert reports[-1].decisions == 2 and len(runner.calls) == 2
    assert len(reports[0].market_return_pct) == 2
    [saved] = list((tmp_path / "out").glob("backtest-*.json"))
    assert len(json.loads(saved.read_text())[0]["equity_curve"]) == reports[0].decisions
    assert "buy_and_hold" in capsys.readouterr().out
