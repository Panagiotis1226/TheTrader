"""Alert sink interface. Phase 4 adds Telegram; until then alerts go to the log."""

from __future__ import annotations

import logging
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
