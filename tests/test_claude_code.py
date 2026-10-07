from __future__ import annotations

import json
import sys
from decimal import Decimal

import pytest

from ai_trader.ai.claude_code import ClaudeCodeAgent, ProcessResult, build_command
from ai_trader.ai.prompt import render_system_prompt
from ai_trader.config import ModelSettings, load_env, load_trading_settings
from ai_trader.data.snapshot import MarketSnapshot

from .conftest import SETTINGS_PATH
from .fakes import FIXTURES, T0

SETTINGS = load_trading_settings(SETTINGS_PATH)
TOKEN = "sk-ant-oat01-TOKEN-abcdef123456"
BUY = {
    "action": "buy",
    "pair": "ETH/CAD",
    "size_pct": 4,
    "confidence": 0.7,
    "reason": "breakout",
    "stop_loss_pct": None,
}
SNAPSHOT = MarketSnapshot(
    timestamp=T0,
    data_age_seconds=1.0,
    quote_currency="CAD",
    cash_available=Decimal(10000),
    total_equity=Decimal(10000),
    pairs=[],
    recent_decisions=[],
)


def cli_output(result: str, **overrides) -> str:
    data = json.loads((FIXTURES / "claude_code" / "result_success.json").read_text())
    data["result"] = result
    data.update(overrides)
    return json.dumps(data)


class FakeRunner:
    def __init__(
        self,
        stdout: str = "",
        returncode: int = 0,
        stderr: str = "",
        raises: BaseException | None = None,
    ) -> None:
        self.result = ProcessResult(returncode, stdout, stderr)
        self.raises = raises
        self.calls: list[dict] = []

    async def __call__(self, args, stdin, env, cwd, timeout):
        import os

        self.calls.append(
            dict(
                args=args,
                stdin=stdin,
                env=dict(env),
                cwd=cwd,
                cwd_empty=os.listdir(cwd) == [],
                timeout=timeout,
            )
        )
        if self.raises:
            raise self.raises
        return self.result


def make_agent(runner, write_env=None, model="claude-opus-5-5", **env_vars):
    env = load_env(write_env(**env_vars)) if write_env else load_env(None)
    return ClaudeCodeAgent(
        ModelSettings(name="claude", provider="claude_code", model=model),
        SETTINGS.llm,
        env,
        render_system_prompt(SETTINGS),
        "BTC/CAD",
        runner,
    )


def test_command_is_locked_down() -> None:
    args = build_command("claude", "claude-opus-5-5", "SYSTEM")
    pairs = {args[i]: args[i + 1] for i in range(1, len(args) - 1) if args[i].startswith("--")}
    assert args[:2] == ["claude", "-p"]
    assert pairs["--system-prompt"] == "SYSTEM"
    assert pairs["--tools"] == ""
    assert pairs["--disallowedTools"] == "mcp__*"
    assert pairs["--setting-sources"] == ""
    assert pairs["--max-turns"] == "1"
    assert pairs["--output-format"] == "json"
    assert pairs["--model"] == "claude-opus-5-5"
    assert "--strict-mcp-config" in args and "--no-session-persistence" in args
    assert not any("dangerously" in a or "bypass" in a for a in args)


async def test_successful_decision(write_env) -> None:
    runner = FakeRunner(cli_output(json.dumps(BUY)))
    result = await make_agent(runner, write_env, CLAUDE_CODE_OAUTH_TOKEN=TOKEN).decide(SNAPSHOT)

    assert result.error is None
    assert result.proposal.action == "buy" and result.proposal.pair == "ETH/CAD"
    assert result.model == "claude_code/claude-haiku-5-5"  # the model that actually answered
    assert result.prompt_tokens == 1341 and result.completion_tokens == 10
    assert result.cost_usd == Decimal("0.000273")
    assert result.raw_response == json.dumps(BUY)

    [call] = runner.calls
    assert call["cwd_empty"]
    assert call["timeout"] == SETTINGS.llm.timeout_seconds
    assert '"total_equity": "10000"' in call["stdin"]
    assert call["args"][call["args"].index("--system-prompt") + 1] == render_system_prompt(SETTINGS)


async def test_child_env_uses_subscription_never_api_key(write_env, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.example")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:telegram-secret")
    monkeypatch.setenv("KRAKEN_API_SECRET", "kraken-secret")
    monkeypatch.setenv("HEALTHCHECK_URL", "https://hc-ping.com/secret-uuid")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    runner = FakeRunner(cli_output(json.dumps(BUY)))
    await make_agent(runner, write_env, CLAUDE_CODE_OAUTH_TOKEN=TOKEN).decide(SNAPSHOT)
    env = runner.calls[0]["env"]
    for secret in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "TELEGRAM_BOT_TOKEN",
        "KRAKEN_API_SECRET",
        "HEALTHCHECK_URL",
    ):
        assert secret not in env, secret
    assert env["HTTPS_PROXY"] == "http://proxy:3128"  # networking still works
    assert env["LC_ALL"] == "C.UTF-8"
    assert "PATH" in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


async def test_garbage_output_holds(write_env) -> None:
    runner = FakeRunner(cli_output("Buy ETH, it looks strong."))
    result = await make_agent(runner, write_env).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert "not a single JSON object" in result.error
    assert result.raw_response == "Buy ETH, it looks strong."
    assert len(runner.calls) == 1


async def test_cli_error_result_holds(write_env) -> None:
    stdout = cli_output(
        "Claude usage limit reached. Resets at 5pm.",
        is_error=True,
        subtype="error_during_execution",
    )
    result = await make_agent(FakeRunner(stdout, returncode=1), write_env).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert "usage limit" in result.error


async def test_non_json_output_holds_and_redacts_token(write_env) -> None:
    runner = FakeRunner("", returncode=1, stderr=f"auth failed for token {TOKEN}")
    result = await make_agent(runner, write_env, CLAUDE_CODE_OAUTH_TOKEN=TOKEN).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert "unreadable Claude Code output" in result.error
    assert TOKEN not in result.error and "***" in result.error


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (TimeoutError(), "timed out"),
        (FileNotFoundError(), "CLI not found"),
        (OSError("boom"), "Claude Code failed"),
    ],
)
async def test_process_failures_hold(write_env, exc, message) -> None:
    result = await make_agent(FakeRunner(raises=exc), write_env).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert message in result.error


async def test_placeholder_never_runs(write_env) -> None:
    runner = FakeRunner(cli_output(json.dumps(BUY)))
    result = await make_agent(runner, write_env, model="<model-id>").decide(SNAPSHOT)
    assert "placeholder" in result.error
    assert runner.calls == []


def test_rejects_litellm_model_settings() -> None:
    with pytest.raises(ValueError):
        ClaudeCodeAgent(
            ModelSettings(name="x", provider="litellm", model="anthropic/x"),
            SETTINGS.llm,
            load_env(None),
            "s",
            "BTC/CAD",
        )


# ----------------------------------------------------- real subprocess, fake CLI


def fake_cli(tmp_path, body: str):
    script = tmp_path / "fake-claude"
    script.write_text(f"#!{sys.executable}\nimport json, os, sys, time\n{body}\n")
    script.chmod(0o755)
    return script


async def test_real_subprocess_round_trip(tmp_path, write_env, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-not-leak")
    script = fake_cli(
        tmp_path,
        f"""
snapshot = sys.stdin.read()
assert "total_equity" in snapshot
assert "ANTHROPIC_API_KEY" not in os.environ
assert os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == {TOKEN!r}
assert os.listdir(".") == []
assert sys.argv[sys.argv.index("--tools") + 1] == ""
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                  "result": json.dumps({BUY!r}), "usage": {{}}, "modelUsage": {{}}}}))
""",
    )
    env_file = write_env(CLAUDE_CODE_OAUTH_TOKEN=TOKEN, CLAUDE_BIN=str(script))
    agent = ClaudeCodeAgent(
        ModelSettings(name="claude", provider="claude_code", model="opus"),
        SETTINGS.llm,
        load_env(env_file),
        render_system_prompt(SETTINGS),
        "BTC/CAD",
    )
    result = await agent.decide(SNAPSHOT)
    assert result.error is None, result.error
    assert result.proposal.action == "buy"
    assert result.model == "claude_code/opus"


async def test_real_subprocess_timeout_kills(tmp_path, write_env) -> None:
    script = fake_cli(tmp_path, "time.sleep(30)")
    llm = SETTINGS.llm.model_copy(update={"timeout_seconds": 1})
    agent = ClaudeCodeAgent(
        ModelSettings(name="claude", provider="claude_code", model="opus"),
        llm,
        load_env(write_env(CLAUDE_BIN=str(script))),
        "s",
        "BTC/CAD",
    )
    result = await agent.decide(SNAPSHOT)
    assert "timed out" in result.error


async def test_missing_cli(write_env) -> None:
    agent = ClaudeCodeAgent(
        ModelSettings(name="claude", provider="claude_code", model="opus"),
        SETTINGS.llm,
        load_env(write_env(CLAUDE_BIN="/nonexistent/claude")),
        "s",
        "BTC/CAD",
    )
    assert "CLI not found" in (await agent.decide(SNAPSHOT)).error
