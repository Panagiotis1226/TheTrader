from __future__ import annotations

import logging
from types import SimpleNamespace

from ai_trader.alerts.base import AlertLevel
from ai_trader.alerts.telegram_bot import (
    MAX_MESSAGE_CHARS,
    CommandRouter,
    TelegramAlerter,
    TelegramBot,
)

OWNER = 424242


class FakeBot:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[dict] = []
        self.fail = fail

    async def send_message(self, chat_id, text, **kwargs):
        if self.fail:
            raise ConnectionError("telegram down")
        self.sent.append({"chat_id": chat_id, "text": text, **kwargs})


def router(calls: list[str]) -> CommandRouter:
    async def stop() -> str:
        calls.append("stop")
        return "halted"

    async def broken() -> str:
        raise RuntimeError("nope")

    return CommandRouter(OWNER, {"stop": stop, "status": broken})


async def test_only_owner_chat_can_command(caplog) -> None:
    calls: list[str] = []
    r = router(calls)
    with caplog.at_level(logging.WARNING):
        assert await r.handle(999, "/stop") is None
    assert calls == []
    assert "unauthorized chat 999" in caplog.text
    assert await r.handle(None, "/stop") is None
    assert await r.handle(OWNER, "/stop") == "halted"
    assert calls == ["stop"]


async def test_command_parsing() -> None:
    calls: list[str] = []
    r = router(calls)
    assert await r.handle(OWNER, "/stop@MyTraderBot extra words") == "halted"
    assert "Commands: /status /stop" in await r.handle(OWNER, "/sell everything")
    assert "Commands:" in await r.handle(OWNER, "   ")
    assert await r.handle(OWNER, "/status") == "/status failed: RuntimeError"


async def test_alerter_sends_plain_text_and_truncates() -> None:
    bot = FakeBot()
    alerter = TelegramAlerter(bot, OWNER)
    await alerter.send("x" * 5000, AlertLevel.CRITICAL)
    [msg] = bot.sent
    assert msg["chat_id"] == OWNER
    assert len(msg["text"]) == MAX_MESSAGE_CHARS
    assert msg["text"].startswith("🛑 ")
    assert "parse_mode" not in msg  # no Markdown/HTML injection from model text


async def test_alerter_never_raises(caplog) -> None:
    alerter = TelegramAlerter(FakeBot(fail=True), OWNER)
    with caplog.at_level(logging.INFO):
        await alerter.send("trade happened")
    assert "Telegram send failed" in caplog.text


async def test_bot_wiring_filters_chat_and_replies() -> None:
    calls: list[str] = []
    bot = TelegramBot("123456:TEST-TOKEN", router(calls))
    [handler] = bot.app.handlers[0]
    assert handler.commands == frozenset({"stop", "status"})
    assert OWNER in handler.filters.chat_ids

    replies: list[str] = []

    async def reply_text(text):
        replies.append(text)

    def update(chat_id):
        return SimpleNamespace(
            effective_chat=SimpleNamespace(id=chat_id),
            effective_message=SimpleNamespace(text="/stop", reply_text=reply_text),
        )

    await bot._on_command(update(1), None)  # defense in depth behind the filter
    assert replies == [] and calls == []
    await bot._on_command(update(OWNER), None)
    assert replies == ["halted"] and calls == ["stop"]
