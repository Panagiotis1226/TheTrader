"""healthchecks.io heartbeat.

``success()`` after each successful cycle; ``fail()`` when a cycle fails (healthchecks.io
alerts immediately on ``/fail``). If pings stop arriving, healthchecks.io alerts after the
check's period + grace. Ping failures are logged and never break the bot. The URL embeds a
secret UUID, so it is never logged.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class Heartbeat:
    def __init__(self, url: str | None, client: httpx.AsyncClient | None = None) -> None:
        self._url = url.rstrip("/") if url else None
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._url is not None

    async def success(self, message: str = "") -> None:
        await self._ping("", message)

    async def fail(self, message: str = "") -> None:
        await self._ping("/fail", message)

    async def _ping(self, suffix: str, message: str) -> None:
        if not self._url:
            return
        try:
            if self._client is not None:
                await self._client.post(self._url + suffix, content=message[:10000], timeout=10)
            else:
                async with httpx.AsyncClient(trust_env=True) as client:
                    await client.post(self._url + suffix, content=message[:10000], timeout=10)
        except httpx.HTTPError as exc:
            log.warning("heartbeat ping failed: %s", type(exc).__name__)
