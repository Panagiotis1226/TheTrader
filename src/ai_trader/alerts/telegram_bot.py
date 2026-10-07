"""Telegram: alerts out, commands in (kill switch).

* Commands are accepted only from ``TELEGRAM_CHAT_ID``; anything else is ignored and
  logged. Use a private chat with the bot: in a group, every member could send /stop.
* Messages are sent as plain text (no Markdown/HTML parsing), so text from the model or
  the market can't inject formatting or links.
* Sending never raises: a Telegram outage must not break a trading cycle. Telegram is
  optional; alerts always go to the log and the database too (``MultiAlerter``).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ai_trader.alerts.base import AlertLevel

log = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000  # Telegram's limit is 4096
_PREFIX = {AlertLevel.INFO: "", AlertLevel.WARNING: "⚠️ ", AlertLevel.CRITICAL: "🛑 "}


class _Bot(Protocol):
    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> Any: ...


class TelegramAlerter:
    def __init__(self, bot: _Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id

    async def send(self, text: str, level: AlertLevel = AlertLevel.INFO) -> None:
        message = (_PREFIX[level] + text)[:MAX_MESSAGE_CHARS]
        try:
            await self._bot.send_message(chat_id=self._chat_id, text=message)
        except Exception as exc:
            log.error("Telegram send failed (%s); alert kept in log/database", type(exc).__name__)


CommandFn = Callable[[], Awaitable[str]]


class CommandRouter:
    """Maps command names to handlers. Independent of the Telegram library (testable)."""

    def __init__(self, allowed_chat_id: int, commands: dict[str, CommandFn]) -> None:
        self.allowed_chat_id = allowed_chat_id
        self._commands = commands

    @property
    def names(self) -> list[str]:
        return sorted(self._commands)

    async def handle(self, chat_id: int | None, text: str) -> str | None:
        """Reply text, or None when the message must be ignored (unauthorized chat)."""
        if chat_id != self.allowed_chat_id:
            log.warning("Ignoring Telegram message from unauthorized chat %s", chat_id)
            return None
        name = text.strip().split()[0].lstrip("/").split("@")[0].lower() if text.strip() else ""
        command = self._commands.get(name)
        if command is None:
            return "Unknown command. " + "Commands: " + " ".join(f"/{n}" for n in self.names)
        try:
            return await command()
        except Exception as exc:
            log.exception("Telegram command /%s failed", name)
            return f"/{name} failed: {type(exc).__name__}"


class TelegramBot:
    """Long-polling bot wired to a ``CommandRouter``."""

    def __init__(self, token: str, router: CommandRouter) -> None:
        from telegram.ext import Application, CommandHandler, filters

        self._router = router
        self.app = Application.builder().token(token).build()
        only_owner = filters.Chat(chat_id=router.allowed_chat_id)
        self.app.add_handler(CommandHandler(router.names, self._on_command, filters=only_owner))

    @property
    def bot(self) -> _Bot:
        return self.app.bot

    async def _on_command(self, update: Any, context: Any) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if chat is None or message is None:
            return
        reply = await self._router.handle(chat.id, message.text or "")
        if reply is not None:
            await message.reply_text(reply[:MAX_MESSAGE_CHARS])

    async def start(self) -> None:
        await self.app.initialize()
        await self.app.start()
        assert self.app.updater is not None
        # Keep commands sent while the bot was down: a queued /stop must still apply.
        await self.app.updater.start_polling(drop_pending_updates=False)

    async def stop(self) -> None:
        if self.app.updater is not None and self.app.updater.running:
            await self.app.updater.stop()
        if self.app.running:
            await self.app.stop()
        await self.app.shutdown()
