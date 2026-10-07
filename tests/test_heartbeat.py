from __future__ import annotations

import logging

import httpx

from ai_trader.alerts.heartbeat import Heartbeat

URL = "https://hc-ping.com/secret-uuid-1234"


def client(seen: list, fail: bool = False) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("no route")
        seen.append((str(request.url), request.content.decode()))
        return httpx.Response(200)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_success_and_fail_pings() -> None:
    seen: list = []
    hb = Heartbeat(URL + "/", client(seen))
    await hb.success("all good")
    await hb.fail("cycle failed")
    assert seen == [(URL, "all good"), (URL + "/fail", "cycle failed")]


async def test_disabled_is_a_no_op() -> None:
    hb = Heartbeat(None)
    assert not hb.enabled
    await hb.success()


async def test_network_errors_are_swallowed_and_url_not_logged(caplog) -> None:
    hb = Heartbeat(URL, client([], fail=True))
    with caplog.at_level(logging.DEBUG):
        await hb.success()
    assert "heartbeat ping failed: ConnectError" in caplog.text
    assert "secret-uuid" not in caplog.text
