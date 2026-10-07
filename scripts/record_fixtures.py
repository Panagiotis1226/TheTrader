"""Record raw Kraken public responses into tests/fixtures/kraken/ for deterministic tests.

Usage: python scripts/record_fixtures.py
Re-recording changes test inputs; tests assert on structure and math, not on live prices.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from ai_trader.data.market import make_kraken_exchange

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "kraken"
PAIRS = ["BTC/CAD", "ETH/CAD"]
MARKET_KEYS = ["symbol", "id", "base", "quote", "spot", "type", "active", "precision", "limits"]


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex = make_kraken_exchange()
    try:
        markets = await ex.load_markets()
        recorded_at_ms = int(time.time() * 1000)
        data = {
            "recorded_at_ms": recorded_at_ms,
            "markets": {p: {k: markets[p][k] for k in MARKET_KEYS} for p in PAIRS},
        }
        for pair in PAIRS:
            key = pair.replace("/", "_")
            ticker = await ex.fetch_ticker(pair)
            data[f"ticker_{key}"] = {k: ticker[k] for k in ("symbol", "last", "bid", "ask")}
            book = await ex.fetch_order_book(pair, 100)
            data[f"order_book_{key}"] = {"bids": book["bids"], "asks": book["asks"]}
            data[f"ohlcv_1h_{key}"] = await ex.fetch_ohlcv(pair, "1h")
            data[f"ohlcv_1d_{key}"] = await ex.fetch_ohlcv(pair, "1d")
    finally:
        await ex.close()
    path = OUT / "snapshot_inputs.json"
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
