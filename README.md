# TheTrader — AI crypto trading bot (Kraken spot, paper-first)

An LLM (Claude, GPT, Gemini, local models via `litellm`) analyzes market data and **proposes**
trades on Kraken spot CAD pairs. A deterministic risk layer approves, resizes, or rejects every
proposal. Paper and live trading share one code path; only the broker changes.

The goal is to find out whether an LLM can trade profitably **in paper trading** before any real
money is used. See [`PLAN.md`](PLAN.md) for the full design and phase plan.

> **Status:** Phase 0 (scaffolding, config, mode guard). Nothing trades yet.

## Safety

The full list is in `PLAN.md` §2. The ones enforced in code so far:

- **Paper by default.** Live mode needs both `MODE=live` and `LIVE_TRADING_CONFIRMED=yes`,
  otherwise the bot exits with code 1 before touching anything (`enforce_mode_guard` in
  `src/ai_trader/main.py`).
- **Strict config.** Unknown keys, out-of-range limits, non-spot symbols (e.g. `BTC/USD:USD`)
  and pairs not quoted in CAD are rejected at startup.
- **Secrets** load only from `.env` (gitignored) as `SecretStr`, so they never show up in logs
  or reprs.
- **No withdrawals, transfers, margin, or leverage.** A test scans `src/` and fails if any of
  those calls appear.

## Quick start

Requires Python 3.12+ (current numpy needs it).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env          # leave MODE=paper
pytest && ruff check . && ruff format --check .
ai-trader                     # validates config, runs the mode guard, exits
```

Configuration:

- `.env` holds the mode flags and secrets (see `.env.example`). Process environment variables
  override the file.
- `config/settings.yaml` holds pairs, schedule, paper fees, risk limits, models, and benchmarks.
  Set `SETTINGS_PATH` or pass `--settings` to use another file.

Before trading, verify the Kraken Pro fees for your volume tier and fill in real model IDs in
`settings.yaml`. The bot warns at startup while a model ID is still a `<model-id>` placeholder.

## Backtests and LLMs: important caveat

LLM backtests on historical data are **not trustworthy**. The models have probably seen past
prices during training (look-ahead bias). The backtester (Phase 3) exists to validate plumbing
and benchmark strategies only. The real test of the AI is forward paper trading (Phase 5).

## Disclaimer

Experimental software. Paper results do not guarantee live results. Every live disposition is a
taxable event (CRA and Revenu Québec).
