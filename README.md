# TheTrader — AI crypto trading bot (Kraken spot, paper-first)

An LLM (Claude, via your Claude Team seat; or any API model via `litellm`) analyzes market
data and **proposes** trades on Kraken spot CAD pairs. A deterministic risk layer approves, resizes, or rejects every
proposal. Paper and live trading share one code path; only the broker changes.

The goal is to find out whether an LLM can trade profitably **in paper trading** before any real
money is used. See [`PLAN.md`](PLAN.md) for the full design and phase plan.

> **Status:** Phase 2 done — LLM agents and the decision cycle run on demand
> (`ai-trader once`). No scheduler yet (Phase 4).

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
python scripts/print_snapshot.py BTC/CAD   # live MarketSnapshot from Kraken public data
ai-trader once                # one decision cycle per configured model (paper accounts)
ai-trader once --model claude # just one model
```

State goes to `DATABASE_URL` (default `data/trader.db`).

## Using Claude through your Claude Team seat

The default model (`provider: claude_code` in `settings.yaml`) runs the Claude Code CLI
(`claude -p`) on your Team seat, so there is no per-token API bill. Setup:

1. Install Claude Code where the bot runs (`npm install -g @anthropic-ai/claude-code`).
2. On any machine with a browser, run `claude setup-token` and put the token in `.env` as
   `CLAUDE_CODE_OAUTH_TOKEN` (valid one year).
3. `ai-trader once --model claude` to check it works.

Each call is locked down to a plain completion: our system prompt replaces Claude Code's,
all tools and MCP servers are disabled, user/project settings and hooks are not loaded, and it
runs in an empty temporary directory. `ANTHROPIC_API_KEY` is removed from the CLI's environment
so a call can't silently switch to API billing.

Caveats: calls count against your seat's usage limits (shared with normal Claude use); if a
limit is hit, that cycle holds. Anthropic's docs don't say whether personal automation on a
subscription is permitted; confirm with Anthropic before relying on it. To use the API
instead, set `provider: litellm`, `model: "anthropic/claude-opus-5-5"` and `ANTHROPIC_API_KEY`.

`scripts/record_fixtures.py` re-records the Kraken responses in `tests/fixtures/` that the
unit tests replay. Tests never touch the network.

Configuration:

- `.env` holds the mode flags and secrets (see `.env.example`). Process environment variables
  override the file.
- `config/settings.yaml` holds pairs, schedule, paper fees, risk limits, models, and benchmarks.
  Set `SETTINGS_PATH` or pass `--settings` to use another file.

Before trading, verify the Kraken Pro fees for your volume tier and fill in real model IDs in
`settings.yaml`. The bot warns at startup while a model ID is still a `<model-id>` placeholder.

## How paper trading is simulated

- **Fills** walk the live Kraken order book (asks for buys, bids for sells), so the average
  price includes slippage. If the fetched book (100 levels) can't fill the order, it is
  rejected, not partially filled.
- **Fees**: the taker fee from `settings.yaml` is charged on every fill, in CAD.
- **Exchange rules**: amounts round down to Kraken's step; min amount and min order value
  (e.g. 0.00005 BTC and 1 CAD) are enforced, as is available balance including the fee.
- **Stop-losses** rest inside the broker (one per pair, covering the whole position) and
  trigger as market sells when the best bid reaches the trigger.
- **Equity** marks positions at the best bid.
- **State** is rebuilt from the fill log on restart, so balances can't drift from fills.

## How a decision is made

Each cycle: check stop-losses → fetch market data once → for each account (in parallel):
skip if halted or over its daily LLM budget, build the snapshot, ask the model, re-fetch the
order books, run the RiskManager, place the order if approved, alert.

- **One JSON object or nothing.** The reply must be exactly one JSON object (markdown fences
  are tolerated). Prose around it, extra fields, or out-of-range values mean `hold`. There is
  no retry and no attempt to repair the output.
- **One call, no retries**, with a hard timeout and token cap (`llm:` in `settings.yaml`).
- **Daily limits** per account: `max_daily_calls`, and `max_daily_cost_usd` (for Claude Code
  this is its API-equivalent estimate). Once reached, the account holds without calling.
- **Buys are sized in CAD.** The broker spends at most the approved amount, so slippage can
  never push a trade past a risk cap.
- **Everything is logged** per decision: snapshot hash, model, prompt hash, raw response,
  parsed proposal, risk outcome and reason, tokens, latency, cost, and the order ID (fill
  details are in `fills`). The prompt hash changes whenever the prompt template or any
  risk/fee setting changes — useful for keeping the Phase 5 evaluation honest.
- **Kill switch** (`risk/killswitch.py`, wired to Telegram `/stop` in Phase 4) halts all
  trading and cancels open orders but keeps stop-losses, so open positions stay protected.

## Risk rules

Every proposal goes through `RiskManager.evaluate`, which returns approve / resize / reject:

- Halts (manual, drawdown, daily loss, errors) block all trading; `hold` is always allowed.
- Whitelisted pairs only, minimum confidence, max trades per UTC day, minimum minutes
  between trades. Stop-loss fills don't count as trades.
- Buys: `size_pct` is a % of total equity, then cut down to the tightest of: max per trade,
  max per pair, max total exposure, and cash after fees. Each resize says which limit bound.
- Sells: `size_pct` is a % of the position; never size-capped (selling reduces risk).
- Stop-loss: the model may tighten the default stop, never loosen it.
- Daily loss (vs. equity at UTC midnight) halts until the next UTC midnight; max drawdown
  (vs. peak since the last resume) halts until `/resume`.

## Backtests and LLMs: important caveat

LLM backtests on historical data are **not trustworthy**. The models have probably seen past
prices during training (look-ahead bias). The backtester (Phase 3) exists to validate plumbing
and benchmark strategies only. The real test of the AI is forward paper trading (Phase 5).

## Disclaimer

Experimental software. Paper results do not guarantee live results. Every live disposition is a
taxable event (CRA and Revenu Québec).
