from __future__ import annotations

import json
import logging
from decimal import Decimal

import litellm
import pytest

from ai_trader.ai.agent import LLMAgent, parse_proposal
from ai_trader.ai.prompt import render_system_prompt
from ai_trader.config import ModelSettings, load_env, load_trading_settings
from ai_trader.data.snapshot import MarketSnapshot

from .conftest import SETTINGS_PATH
from .fakes import T0, mock_llm

SETTINGS = load_trading_settings(SETTINGS_PATH)
SECRET = "sk-ant-test-SECRET-1234567890"
BUY = {
    "action": "buy",
    "pair": "BTC/CAD",
    "size_pct": 5,
    "confidence": 0.8,
    "reason": "uptrend",
    "stop_loss_pct": 4,
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


def agent(completion, model="anthropic/claude-opus-5-5", write_env=None, **model_kw):
    env = load_env(write_env(ANTHROPIC_API_KEY=SECRET)) if write_env else load_env(None)
    return LLMAgent(
        ModelSettings(name="m", provider="litellm", model=model, **model_kw),
        SETTINGS.llm,
        env,
        render_system_prompt(SETTINGS),
        "BTC/CAD",
        completion,
    )


# ----------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(BUY),
        "```json\n" + json.dumps(BUY) + "\n```",
        "```\n" + json.dumps(BUY) + "\n```",
        "  \n" + json.dumps(BUY, indent=2) + "\n ",
    ],
)
def test_parse_valid(text) -> None:
    proposal, error = parse_proposal(text, "BTC/CAD")
    assert error is None
    assert proposal.action == "buy"
    assert proposal.size_pct == Decimal(5)
    assert proposal.stop_loss_pct == Decimal(4)


@pytest.mark.parametrize(
    ("text", "error_part"),
    [
        (None, "empty"),
        ("", "empty"),
        ("I think you should buy BTC.", "not a single JSON object"),
        ("Sure! " + json.dumps(BUY), "not a single JSON object"),  # prose + JSON: no guessing
        (json.dumps(BUY) + "\nHope this helps", "not a single JSON object"),
        (json.dumps([BUY]), "expected a JSON object"),
        ('{"action": "buy", "pair": "BTC/CAD"', "not a single JSON object"),
        (json.dumps({**BUY, "action": "short"}), "action"),
        (json.dumps({**BUY, "size_pct": 150}), "size_pct"),
        (json.dumps({**BUY, "confidence": 2}), "confidence"),
        (json.dumps({**BUY, "leverage": 3}), "leverage"),
        (json.dumps({k: v for k, v in BUY.items() if k != "confidence"}), "confidence"),
    ],
)
def test_parse_invalid_becomes_hold(text, error_part) -> None:
    proposal, error = parse_proposal(text, "BTC/CAD")
    assert proposal.action == "hold"
    assert proposal.size_pct == 0
    assert error_part in error


def test_non_whitelisted_pair_parses_and_is_left_to_risk() -> None:
    proposal, error = parse_proposal(json.dumps({**BUY, "pair": "DOGE/CAD"}), "BTC/CAD")
    assert error is None and proposal.pair == "DOGE/CAD"  # RiskManager rejects it


# ------------------------------------------------------------------------- calls


async def test_successful_call_records_telemetry(write_env) -> None:
    completion = mock_llm(json.dumps(BUY))
    result = await agent(completion, write_env=write_env).decide(SNAPSHOT)

    assert result.error is None
    assert result.proposal.action == "buy"
    assert result.raw_response == json.dumps(BUY)
    assert result.model == "anthropic/claude-opus-5-5"
    assert result.prompt_tokens and result.completion_tokens
    assert result.cost_usd is not None and result.cost_usd > 0
    assert result.latency_ms is not None
    assert len(result.prompt_hash) == 16

    [call] = completion.calls
    assert call["num_retries"] == 0
    assert call["api_key"] == SECRET
    assert call["max_tokens"] == SETTINGS.llm.max_tokens
    assert call["timeout"] == SETTINGS.llm.timeout_seconds
    assert "temperature" not in call
    system, user = call["messages"]
    assert "If uncertain, choose hold." in system["content"]
    assert '"total_equity": "10000"' in user["content"]


async def test_model_overrides_are_passed(write_env) -> None:
    completion = mock_llm(json.dumps(BUY))
    await agent(
        completion,
        write_env=write_env,
        temperature=0.2,
        max_tokens=999,
        api_base="http://localhost:1",
    ).decide(SNAPSHOT)
    [call] = completion.calls
    assert (call["temperature"], call["max_tokens"], call["api_base"]) == (
        0.2,
        999,
        "http://localhost:1",
    )


async def test_garbage_output_holds_without_retry(write_env) -> None:
    completion = mock_llm("BUY BUY BUY!!!")
    result = await agent(completion, write_env=write_env).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert "not a single JSON object" in result.error
    assert result.raw_response == "BUY BUY BUY!!!"
    assert len(completion.calls) == 1


async def test_placeholder_model_is_never_called(write_env) -> None:
    completion = mock_llm(json.dumps(BUY))
    result = await agent(completion, model="openai/<model-id>", write_env=write_env).decide(
        SNAPSHOT
    )
    assert result.proposal.action == "hold"
    assert "placeholder" in result.error
    assert completion.calls == []


async def test_missing_api_key_is_never_called() -> None:
    completion = mock_llm(json.dumps(BUY))
    result = await agent(completion).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert result.error == "ANTHROPIC_API_KEY is not set"
    assert completion.calls == []


async def test_local_model_needs_no_key() -> None:
    completion = mock_llm(json.dumps(BUY))
    result = await agent(completion, model="ollama/llama3").decide(SNAPSHOT)
    assert result.error is None
    assert "api_key" not in completion.calls[0]


async def test_timeout_holds(write_env) -> None:
    timeout = litellm.Timeout(message="took too long", model="m", llm_provider="anthropic")
    result = await agent(mock_llm(timeout), write_env=write_env).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert "LLM call failed: Timeout" in result.error


async def test_api_key_redacted_from_errors_and_logs(write_env, caplog) -> None:
    async def leaky(**kwargs):
        raise RuntimeError(f"401 invalid x-api-key {kwargs['api_key']}")

    with caplog.at_level(logging.DEBUG):
        result = await agent(leaky, write_env=write_env).decide(SNAPSHOT)
    assert result.proposal.action == "hold"
    assert SECRET not in result.error
    assert "***" in result.error
    assert SECRET not in caplog.text
