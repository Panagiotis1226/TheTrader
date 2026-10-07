"""Kill switch (Safety Invariant #8).

``engage`` records a global manual halt *first*, so trading stops even if cancelling
orders fails, then cancels each account's open orders. Protective stop-losses are
kept: halting new trades should never leave existing positions unprotected.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

from ai_trader.brokers.base import Broker
from ai_trader.storage.repo import HaltKind, Repository

log = logging.getLogger(__name__)


async def engage(
    repo: Repository, brokers: Iterable[Broker], reason: str, now: datetime
) -> list[str]:
    """Halt all trading and cancel non-protective open orders.

    Returns the account IDs whose cancellation failed (the halt still holds).
    """
    repo.add_halt(HaltKind.MANUAL, reason, now)
    log.critical("KILL SWITCH engaged: %s", reason)
    failed: list[str] = []
    for broker in brokers:
        try:
            await broker.cancel_all(keep_stop_losses=True)
        except Exception:
            log.exception("kill switch: cancelling orders failed for %s", broker.account_id)
            failed.append(broker.account_id)
    return failed


def resume(repo: Repository, now: datetime) -> int:
    """Lift every active halt (manual, drawdown, daily loss, errors)."""
    count = repo.resume(now)
    log.warning("Trading resumed; %d halt(s) lifted", count)
    return count
