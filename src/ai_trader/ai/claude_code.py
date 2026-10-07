"""Claude via the Claude Code CLI (``claude -p``), billed to a Claude subscription seat.

The bot stays in control: one subprocess call per cycle, snapshot in on stdin, text out.
The call is locked down to a plain completion:

* ``--system-prompt`` replaces Claude Code's default prompt with ours (same prompt the
  litellm agent uses), ``--tools ""`` and ``--disallowedTools mcp__*`` with
  ``--strict-mcp-config`` remove every tool, ``--setting-sources ""`` skips user/project
  settings (and their hooks), and it runs in an empty temp directory so no CLAUDE.md
  is picked up. ``--max-turns 1``, no session persistence.
* ``ANTHROPIC_API_KEY`` and other API-routing variables are removed from the child
  environment so a call can never silently switch to per-token API billing. Auth is
  ``CLAUDE_CODE_OAUTH_TOKEN`` (from ``claude setup-token``).
* Output goes through the same strict parser as the litellm agent: anything but one
  JSON object is ``hold``. No retries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ai_trader.ai.agent import AgentResult, parse_proposal
from ai_trader.ai.prompt import prompt_hash, render_user_prompt
from ai_trader.ai.schema import TradeProposal
from ai_trader.config import EnvSettings, LLMSettings, ModelSettings
from ai_trader.data.snapshot import MarketSnapshot

log = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500
# Never let these reach the child: they would route the call to API billing or elsewhere.
_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_OAUTH_TOKEN",  # re-added from EnvSettings below
)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[list[str], str, Mapping[str, str], str, float], Awaitable[ProcessResult]]


async def run_subprocess(
    args: list[str], stdin: str, env: Mapping[str, str], cwd: str, timeout: float
) -> ProcessResult:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env),
        cwd=cwd,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin.encode()), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return ProcessResult(
        proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")
    )


def build_command(claude_bin: str, model: str, system_prompt: str) -> list[str]:
    return [
        claude_bin,
        "-p",
        "--system-prompt", system_prompt,
        "--tools", "",
        "--disallowedTools", "mcp__*",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--model", model,
        "--max-turns", "1",
        "--output-format", "json",
        "--no-session-persistence",
    ]  # fmt: skip


class ClaudeCodeAgent:
    def __init__(
        self,
        model: ModelSettings,
        llm: LLMSettings,
        env: EnvSettings,
        system_prompt: str,
        fallback_pair: str,
        runner: Runner = run_subprocess,
    ) -> None:
        if model.provider != "claude_code":
            raise ValueError("ClaudeCodeAgent needs provider: claude_code")
        self.name = f"claude_code/{model.model}"
        self._model = model
        self._llm = llm
        self._token = (
            env.claude_code_oauth_token.get_secret_value() if env.claude_code_oauth_token else None
        )
        self._claude_bin = env.claude_bin
        self._system_prompt = system_prompt
        self._prompt_hash = prompt_hash(system_prompt)
        self._fallback_pair = fallback_pair
        self._runner = runner

    def _child_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        if self._token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self._token
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env

    def _redact(self, text: str) -> str:
        if self._token:
            text = text.replace(self._token, "***")
        return text[:MAX_ERROR_CHARS]

    def _hold(self, error: str, **fields: Any) -> AgentResult:
        log.warning("%s: holding: %s", self.name, error)
        fields.setdefault("model", self.name)
        return AgentResult(
            proposal=TradeProposal.hold(self._fallback_pair, f"no decision: {error}"),
            prompt_hash=self._prompt_hash,
            error=error,
            **fields,
        )

    async def decide(self, snapshot: MarketSnapshot) -> AgentResult:
        if self._model.is_placeholder:
            return self._hold(f"model ID {self._model.model!r} is a placeholder")

        args = build_command(self._claude_bin, self._model.model, self._system_prompt)
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            with tempfile.TemporaryDirectory(prefix="ai-trader-claude-") as cwd:
                proc = await self._runner(
                    args,
                    render_user_prompt(snapshot),
                    self._child_env(),
                    cwd,
                    float(self._llm.timeout_seconds),
                )
        except FileNotFoundError:
            return self._hold(f"Claude Code CLI not found ({self._claude_bin!r})")
        except TimeoutError:
            latency = int((loop.time() - started) * 1000)
            return self._hold(f"timed out after {self._llm.timeout_seconds}s", latency_ms=latency)
        except Exception as exc:
            return self._hold(self._redact(f"Claude Code failed: {type(exc).__name__}: {exc}"))
        latency = int((loop.time() - started) * 1000)

        try:
            data = json.loads(proc.stdout)
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except ValueError:
            detail = self._redact((proc.stderr or proc.stdout).strip() or "no output")
            return self._hold(
                f"unreadable Claude Code output (exit {proc.returncode}): {detail}",
                latency_ms=latency,
            )

        raw = data.get("result")
        raw_text = raw if isinstance(raw, str) else None
        usage = data.get("usage") or {}
        prompt_tokens = (
            sum(
                int(usage.get(k) or 0)
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
            or None
        )
        completion_tokens = usage.get("output_tokens")
        cost = data.get("total_cost_usd")
        used = sorted((data.get("modelUsage") or {}).keys())
        telemetry: dict[str, Any] = {
            "model": f"claude_code/{','.join(used)}" if used else self.name,
            "raw_response": raw_text,
            "latency_ms": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": None if cost is None else Decimal(str(cost)),
        }

        if proc.returncode != 0 or data.get("is_error") or data.get("subtype") != "success":
            detail = self._redact(str(raw_text or data.get("subtype") or "unknown error"))
            return self._hold(f"Claude Code error: {detail}", **telemetry)

        proposal, error = parse_proposal(raw_text, self._fallback_pair)
        if error:
            log.warning("%s: unusable output, holding: %s", self.name, error)
        log.info(
            "%s: %s %s size=%s conf=%s (%d ms)",
            telemetry["model"],
            proposal.action,
            proposal.pair,
            proposal.size_pct,
            proposal.confidence,
            latency,
        )
        return AgentResult(
            proposal=proposal, prompt_hash=self._prompt_hash, error=error, **telemetry
        )
