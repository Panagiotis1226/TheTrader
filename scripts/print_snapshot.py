"""Print a live MarketSnapshot from Kraken public data.

Usage: python scripts/print_snapshot.py [PAIR ...]   (default: BTC/CAD)

Uses a throwaway in-memory paper account (starting cash from settings.yaml), so the
position fields are empty. No API keys needed; nothing is written to disk.
"""

from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal

from ai_trader.brokers.paper import PaperBroker
from ai_trader.config import load_trading_settings
from ai_trader.data.market import MarketData, utcnow
from ai_trader.data.snapshot import build_snapshot, fetch_pair_data
from ai_trader.storage.repo import Repository


async def main(pairs: list[str], settings_path: str) -> None:
    settings = load_trading_settings(settings_path)
    repo = Repository.from_url("sqlite://")
    async with MarketData() as market:
        broker = PaperBroker(
            "paper-snapshot-demo",
            repo=repo,
            market=market,
            allowed_pairs=settings.pairs,
            starting_cash=settings.paper.starting_cash_cad,
            taker_fee_pct=settings.paper.taker_fee_pct,
            quote_currency=settings.quote_currency,
        )
        data = {pair: await fetch_pair_data(market, pair) for pair in pairs}
        balances = await broker.get_balances()
        snapshot = build_snapshot(
            data,
            cash=balances.get(settings.quote_currency, Decimal(0)),
            positions=await broker.get_positions(),
            recent_decisions=repo.recent_decisions(broker.account_id),
            now=utcnow(),
            max_data_age_seconds=settings.max_data_age_seconds,
            quote_currency=settings.quote_currency,
        )
    print(snapshot.model_dump_json(indent=2))
    print(f"snapshot hash: {snapshot.content_hash()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="*", default=["BTC/CAD"])
    parser.add_argument("--settings", default="config/settings.yaml")
    args = parser.parse_args()
    asyncio.run(main([p.upper() for p in args.pairs], args.settings))
