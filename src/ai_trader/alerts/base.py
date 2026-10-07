"""Alert sink interface. Phase 4 adds Telegram; until then alerts go to the log."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

log = logging.getLogger("ai_trader.alerts")


class AlertLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alerter(Protocol):
    async def send(self, text: str, level: AlertLevel = AlertLevel.INFO) -> None: ...


_LOG_LEVELS = {
    AlertLevel.INFO: logging.INFO,
    AlertLevel.WARNING: logging.WARNING,
    AlertLevel.CRITICAL: logging.CRITICAL,
}


class LogAlerter:
    async def send(self, text: str, level: AlertLevel = AlertLevel.INFO) -> None:
        log.log(_LOG_LEVELS[level], "ALERT: %s", text)


class DbAlerter:
    """Stores alerts in the database: readable with `ai-trader status` and the dashboard."""

    def __init__(self, repo, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._repo = repo
        self._clock = clock

    async def send(self, text: str, level: AlertLevel = AlertLevel.INFO) -> None:
        try:
            self._repo.add_alert(level.value, text, self._clock())
        except Exception:
            log.exception("could not store alert")


class MultiAlerter:
    """Sends every alert to each sink; one failing sink never blocks the others."""

    def __init__(self, sinks: Sequence[Alerter]) -> None:
        self._sinks = list(sinks)

    async def send(self, text: str, level: AlertLevel = AlertLevel.INFO) -> None:
        for sink in self._sinks:
            try:
                await sink.send(text, level)
            except Exception:
                log.exception("alert sink %s failed", type(sink).__name__)
