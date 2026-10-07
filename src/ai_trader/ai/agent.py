"""LLM agent: snapshot in, ``TradeProposal`` out (Safety Invariants #2, #4).

* Any provider via litellm. One call per cycle, ``num_retries=0``, hard timeout.
* The raw text must be exactly one JSON object (markdown fences are tolerated).
  Anything else, including prose around the JSON, becomes ``hold``. There is no
  retry and no attempt to repair output: malformed output never turns into a trade.
* API keys come from ``EnvSettings`` and are passed per call; they are redacted from
  any error text before it is stored or logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import litellm
from pydantic import ValidationError

from ai_trader.ai.prompt import prompt_hash, render_user_prompt
from ai_trader.ai.schema import TradeProposal
from ai_trader.config import EnvSettings, LLMSettings, ModelSettings
from ai_trader.data.snapshot import MarketSnapshot

log = logging.getLogger(__name__)

litellm.suppress_debug_info = True

CompletionFn = Callable[..., Awaitable[Any]]

# Providers that need a key from .env. Others (e.g. ollama/) are called without one.
_PROVIDER_KEYS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "gemini_api_key",
}
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)
MAX_ERROR_CHARS = 500


@dataclass(frozen=True)
class AgentResult:
    proposal: TradeProposal
    model: str
    prompt_hash: str | None = None
    raw_response: str | None = None
    error: str | None = None  # set whenever the proposal is a fallback hold
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: Decimal | None = None


class DecisionMaker(Protocol):
    """Anything that turns a snapshot into a proposal: LLM agents and benchmarks."""

    name: str

    async def decide(self, snapshot: MarketSnapshot) -> AgentResult: ...


def parse_proposal(text: str | None, fallback_pair: str) -> tuple[TradeProposal, str | None]:
    """Parse raw model text. Returns ``(proposal, None)`` or ``(hold, error)``."""

    def hold(error: str) -> tuple[TradeProposal, str]:
        return TradeProposal.hold(fallback_pair, f"invalid model output: {error}"), error

    if text is None or not text.strip():
        return hold("empty response")
    body = text.strip()
    fenced = _FENCE_RE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return hold(f"not a single JSON object ({exc.msg})")
    if not isinstance(data, dict):
        return hold(f"expected a JSON object, got {type(data).__name__}")
    try:
        return TradeProposal.model_validate(data), None
    except ValidationError as exc:
        fields = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors(include_url=False)
        )
        return hold(f"schema validation failed: {fields}")


def _redact(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "***")
    return text[:MAX_ERROR_CHARS]


def _usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def _cost(response: Any) -> Decimal | None:
    hidden = getattr(response, "_hidden_params", None) or {}
    cost = hidden.get("response_cost")
    if cost is None:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            return None
    return None if cost is None else Decimal(str(cost))


class LLMAgent:
    def __init__(
        self,
        model: ModelSettings,
        llm: LLMSettings,
        env: EnvSettings,
        system_prompt: str,
        fallback_pair: str,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.name = model.litellm_model
        self._model = model
        self._llm = llm
        self._env = env
        self._system_prompt = system_prompt
        self._prompt_hash = prompt_hash(system_prompt)
        self._fallback_pair = fallback_pair
        self._completion = completion_fn or litellm.acompletion

    def _hold(self, error: str, **fields: Any) -> AgentResult:
        log.warning("%s: holding: %s", self.name, error)
        return AgentResult(
            proposal=TradeProposal.hold(self._fallback_pair, f"no decision: {error}"),
            model=self.name,
            prompt_hash=self._prompt_hash,
            error=error,
            **fields,
        )

    def _api_key(self) -> tuple[str | None, str | None]:
        """(key, error). Error is set when a required key is missing."""
        provider = self._model.litellm_model.split("/", 1)[0]
        attr = _PROVIDER_KEYS.get(provider)
        if attr is None:
            return None, None
        secret = getattr(self._env, attr)
        if secret is None:
            return None, f"{attr.upper()} is not set"
        return secret.get_secret_value(), None

    async def decide(self, snapshot: MarketSnapshot) -> AgentResult:
        if self._model.is_placeholder:
            return self._hold(f"model ID {self._model.litellm_model!r} is a placeholder")
        api_key, key_error = self._api_key()
        if key_error:
            return self._hold(key_error)

        kwargs: dict[str, Any] = {
            "model": self._model.litellm_model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": render_user_prompt(snapshot)},
            ],
            "timeout": self._llm.timeout_seconds,
            "max_tokens": self._model.max_tokens or self._llm.max_tokens,
            "num_retries": 0,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self._model.temperature is not None:
            kwargs["temperature"] = self._model.temperature
        if self._model.api_base:
            kwargs["api_base"] = self._model.api_base

        started = time.monotonic()
        try:
            # Belt and braces: litellm's own timeout plus an outer one.
            response = await asyncio.wait_for(
                self._completion(**kwargs), timeout=self._llm.timeout_seconds + 5
            )
        except Exception as exc:  # any provider/network failure means hold
            latency = int((time.monotonic() - started) * 1000)
            message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            return self._hold(f"LLM call failed: {_redact(message, api_key)}", latency_ms=latency)
        latency = int((time.monotonic() - started) * 1000)

        try:
            raw = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raw = None
        prompt_tokens, completion_tokens = _usage(response)
        proposal, error = parse_proposal(raw, self._fallback_pair)
        if error:
            log.warning("%s: unusable output, holding: %s", self.name, error)
        log.info(
            "%s: %s %s size=%s conf=%s (%d ms, %s/%s tokens)",
            self.name,
            proposal.action,
            proposal.pair,
            proposal.size_pct,
            proposal.confidence,
            latency,
            prompt_tokens,
            completion_tokens,
        )
        return AgentResult(
            proposal=proposal,
            model=self.name,
            prompt_hash=self._prompt_hash,
            raw_response=raw,
            error=error,
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_cost(response),
        )
