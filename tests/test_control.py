"""Local control (no Telegram): status, stop, resume, report, evaluate."""

from __future__ import annotations

from ai_trader import control
from ai_trader.alerts.base import AlertLevel, DbAlerter, LogAlerter, MultiAlerter
from ai_trader.brokers.base import OrderType
from ai_trader.config import load_config
from ai_trader.cycle import CycleStatus
from ai_trader.main import EXIT_OK, main

from .conftest import SETTINGS_PATH
from .test_cycle import NOW, Harness, proposal_json


async def test_local_stop_halts_the_running_bot(repo, write_env) -> None:
    h = Harness(repo, write_env)
    acct = h.account("claude", proposal_json())
    [first] = await h.cycle.run([acct])
    assert first.status is CycleStatus.TRADED

    msg = await control.stop(repo, [acct], "testing", NOW)
    assert "halted" in msg
    [stop] = await acct.broker.get_open_orders()
    assert stop.type is OrderType.STOP_LOSS  # protection stays

    [after] = await h.cycle.run([acct])  # the bot process sees the halt via the database
    assert after.status is CycleStatus.SKIPPED
    assert "KILL SWITCH: testing" in repo.recent_alerts(1)[0][2]

    assert control.resume(repo, NOW) == "Resumed: 1 halt(s) lifted."
    assert repo.active_halts(None, NOW) == []


async def test_status_text(repo, write_env) -> None:
    h = Harness(repo, write_env)
    acct = h.account("claude", proposal_json())
    await h.cycle.run([acct])
    await DbAlerter(repo, lambda: NOW).send("something happened", AlertLevel.WARNING)
    config = load_config(write_env(), SETTINGS_PATH)

    text = await control.status_text(config, repo, [acct], NOW)
    assert "Trading: active" in text
    assert "Last decision:" in text and "0 min ago" in text
    assert "Evaluation: week 0.0 of 8, 1 of 50 trades" in text
    assert "paper-claude" in text and "BTC" in text and "last: buy (approve)" in text
    assert "!  something happened" in text

    await control.stop(repo, [acct], "maintenance", NOW)
    text = await control.status_text(config, repo, [acct], NOW)
    assert "TRADING HALTED (manual)" in text and "ai-trader resume" in text


async def test_report_and_evaluate_text(repo, write_env) -> None:
    h = Harness(repo, write_env)
    acct = h.account("claude", proposal_json())
    await h.cycle.run([acct])
    config = load_config(write_env(), SETTINGS_PATH)
    assert "Weekly report" in await control.report_text(config, repo, [acct], NOW)
    card = await control.evaluate_text(config, repo, [acct], NOW)
    assert "Go-live scorecard" in card and "[PENDING] Duration" in card


async def test_equity_falls_back_to_snapshot_when_offline(repo, write_env) -> None:
    from decimal import Decimal

    h = Harness(repo, write_env)
    acct = h.account("claude", proposal_json())
    await h.cycle.run([acct])  # records an equity snapshot
    h.exchange.data = {}  # Kraken unreachable from now on
    equity, live = await control.account_equity([acct], repo, "CAD")
    assert not live
    assert equity["paper-claude"] == repo.equity_series("paper-claude")[-1][1]
    assert isinstance(equity["paper-claude"], Decimal)


def test_resume_command_without_network(tmp_path, write_env, capsys) -> None:
    env = write_env(DATABASE_URL=f"sqlite:///{tmp_path}/c.db")
    from ai_trader.storage.repo import HaltKind, Repository

    Repository.from_url(f"sqlite:///{tmp_path}/c.db").add_halt(HaltKind.MANUAL, "x", NOW)
    assert main(["--env-file", str(env), "--settings", str(SETTINGS_PATH), "resume"]) == EXIT_OK
    assert "1 halt(s) lifted" in capsys.readouterr().out


async def test_alerts_reach_every_sink_even_if_one_fails(repo, caplog) -> None:
    class Broken:
        async def send(self, text, level=AlertLevel.INFO):
            raise RuntimeError("telegram down")

    alerter = MultiAlerter([Broken(), LogAlerter(), DbAlerter(repo, lambda: NOW)])
    await alerter.send("stop-loss hit", AlertLevel.WARNING)
    assert repo.recent_alerts(1)[0][1:] == ("warning", "stop-loss hit")
    assert "ALERT: stop-loss hit" in caplog.text
