"""Daily candle history for backtests.

Kraken's OHLC API only returns the most recent 720 candles (~2 years of daily data). The
local cache (one CSV per pair) is merged on every refresh, so history keeps growing. For
longer history, drop in Kraken's downloadable OHLCVT file for the pair, daily interval
(headerless ``timestamp,open,high,low,close,volume,trades``), at the same path.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from ai_trader.data.market import Candle, MarketData, MarketInfo

DAY = timedelta(days=1)
DEFAULT_CANDLES_DIR = Path("data/candles")


def candles_path(directory: Path, pair: str) -> Path:
    base, quote = pair.split("/")
    return directory / f"{base}_{quote}_1d.csv"


def _parse_ts(value: str) -> datetime:
    ts = float(value)
    if ts > 1e11:  # milliseconds
        ts /= 1000
    return datetime.fromtimestamp(ts, tz=UTC)


def read_candles_csv(path: Path) -> list[Candle]:
    """Read our cache format or Kraken's headerless OHLCVT export."""
    candles: list[Candle] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            try:
                opened_at = _parse_ts(row[0])
            except ValueError:
                continue  # header row
            o, h, low, c, v = (Decimal(x) for x in row[1:6])
            candles.append(Candle(opened_at, o, h, low, c, v))
    return merge_candles(candles)


def write_candles_csv(path: Path, candles: Iterable[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([int(c.opened_at.timestamp()), c.open, c.high, c.low, c.close, c.volume])


def merge_candles(*series: Iterable[Candle]) -> list[Candle]:
    """Union by open time (later series win), sorted."""
    by_time: dict[datetime, Candle] = {}
    for candles in series:
        for c in candles:
            by_time[c.opened_at] = c
    return [by_time[t] for t in sorted(by_time)]


async def refresh_candles(
    market: MarketData, pair: str, directory: Path, now: datetime
) -> list[Candle]:
    """Fetch the latest daily candles, merge them into the cache, return the full history.

    The in-progress candle is never stored.
    """
    path = candles_path(directory, pair)
    existing = read_candles_csv(path) if path.exists() else []
    fetched = [c for c in await market.fetch_ohlcv(pair, "1d") if c.opened_at + DAY <= now]
    merged = merge_candles(existing, fetched)
    write_candles_csv(path, merged)
    return merged


def save_market_infos(directory: Path, infos: dict[str, MarketInfo]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        pair: {k: (str(v) if isinstance(v, Decimal) else v) for k, v in vars(info).items()}
        for pair, info in infos.items()
    }
    (directory / "markets.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_market_infos(directory: Path) -> dict[str, MarketInfo]:
    raw = json.loads((directory / "markets.json").read_text(encoding="utf-8"))
    out = {}
    for pair, d in raw.items():
        out[pair] = MarketInfo(
            pair=d["pair"],
            base=d["base"],
            quote=d["quote"],
            amount_step=Decimal(d["amount_step"]),
            price_step=Decimal(d["price_step"]),
            min_amount=Decimal(d["min_amount"]) if d["min_amount"] is not None else None,
            min_cost=Decimal(d["min_cost"]) if d["min_cost"] is not None else None,
        )
    return out
