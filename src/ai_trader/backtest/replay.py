"""A market that replays daily candles, with the same interface as ``MarketData``.

At replay time ``now`` it only exposes candles that had *closed* by ``now``, so a
decision can never see the future. Historical order books don't exist, so the book is
synthetic: one deep level at the price +/- ``slippage_pct``. ``set_prices`` lets the
engine simulate intraday stop-loss fills.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ai_trader.data.market import BookLevel, Candle, MarketDataError, MarketInfo, OrderBook, Ticker

DAY = timedelta(days=1)
DEPTH = Decimal("1000000000")


class ReplayClock:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(1970, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class ReplayMarket:
    def __init__(
        self,
        candles: Mapping[str, Sequence[Candle]],
        infos: Mapping[str, MarketInfo],
        slippage_pct: Decimal,
        clock: ReplayClock,
    ) -> None:
        self._candles = {pair: list(series) for pair, series in candles.items()}
        self._close_times = {
            pair: [c.opened_at + DAY for c in series] for pair, series in self._candles.items()
        }
        self._infos = dict(infos)
        self._slip = slippage_pct / Decimal(100)
        self._clock = clock
        self._overrides: dict[str, Decimal] = {}

    # --------------------------------------------------------------- engine hooks

    def visible(self, pair: str) -> list[Candle]:
        """Daily candles closed by now."""
        if pair not in self._candles:
            raise MarketDataError(f"no history for {pair}")
        n = bisect_right(self._close_times[pair], self._clock())
        return self._candles[pair][:n]

    def set_prices(self, prices: Mapping[str, Decimal]) -> None:
        self._overrides = dict(prices)

    def clear_prices(self) -> None:
        self._overrides = {}

    def _price(self, pair: str) -> Decimal:
        if pair in self._overrides:
            return self._overrides[pair]
        visible = self.visible(pair)
        if not visible:
            raise MarketDataError(f"{pair}: no closed candle yet at {self._clock()}")
        return visible[-1].close

    # ------------------------------------------------------ MarketData interface

    async def load_markets(self, reload: bool = False) -> dict[str, MarketInfo]:
        return dict(self._infos)

    async def market_info(self, pair: str) -> MarketInfo:
        if pair not in self._infos:
            raise MarketDataError(f"{pair} is not a known market")
        return self._infos[pair]

    async def fetch_ohlcv(self, pair: str, timeframe: str, limit: int | None = None):
        if timeframe == "1h":
            return []  # daily history only
        if timeframe != "1d":
            raise ValueError(f"replay only has daily candles, not {timeframe!r}")
        visible = self.visible(pair)
        return visible[-limit:] if limit else visible

    async def fetch_ticker(self, pair: str) -> Ticker:
        price = self._price(pair)
        return Ticker(
            pair=pair,
            last=price,
            bid=price * (1 - self._slip),
            ask=price * (1 + self._slip),
            received_at=self._clock(),
        )

    async def fetch_order_book(self, pair: str, limit: int = 100) -> OrderBook:
        price = self._price(pair)
        return OrderBook(
            pair=pair,
            bids=(BookLevel(price * (1 - self._slip), DEPTH),),
            asks=(BookLevel(price * (1 + self._slip), DEPTH),),
            received_at=self._clock(),
        )

    async def close(self) -> None:
        return None
