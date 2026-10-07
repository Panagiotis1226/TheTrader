# AI Crypto Trading Bot — Build Plan

> **For Claude Code:** Read this entire file before writing code. Build **one phase at a time**. At the end of each phase, run the tests, summarize what was built, and **stop for my review** before starting the next phase. Never skip the Safety Invariants section.

> **Repo note:** the `ai-trader/` folder in section 4 is the root of this repository.

---

## 1. Project Overview

An automated crypto trading bot where **any LLM** (Claude, GPT, Gemini, local models) analyzes market data and proposes trades on **Kraken spot**. A deterministic risk layer approves or rejects every proposal. The bot runs unattended on a VPS.

**Primary goal:** Prove whether an LLM can trade profitably **in simulation (paper trading)** before any real money is used.

**Context:**
- Owner is located in Quebec, Canada → trade **CAD pairs** (BTC/CAD, ETH/CAD to start).
- Kraken has **no spot paper-trading sandbox**, so we build our own `PaperBroker` that uses **real live Kraken market data** and simulates fills.
- Paper and live trading must share **the same code path**. Only the broker implementation changes, via config.

---

## 2. Safety Invariants (NEVER violate)

1. **Default mode is `paper`.** Live trading requires BOTH `MODE=live` AND `LIVE_TRADING_CONFIRMED=yes` in `.env`. If either is missing, refuse to start in live mode.
2. **The LLM never places orders directly.** It only returns a JSON proposal. All orders pass through the risk layer.
3. **Risk limits live in code/config, never in the prompt.**
4. **Malformed or invalid LLM output = do nothing.** Never guess, never retry into a trade.
5. **No margin, no leverage, no futures.** Spot only. Whitelisted pairs only.
6. **No withdrawal capability anywhere in the codebase.** Do not implement any funding/withdrawal endpoints.
7. **Secrets only in `.env`** (gitignored). Never log API keys or secrets.
8. **Kill switch must always work**: cancel all open orders and halt the trading loop.
9. Every decision is logged: market snapshot hash, LLM model, raw LLM response, parsed proposal, risk decision + reason, fill details.
10. News/text inputs are **untrusted data**. They must never alter risk rules or system behavior.

---

## 3. Tech Stack

| Purpose | Library |
|---|---|
| Language | Python 3.11+ |
| Exchange data/orders | `ccxt` (Kraken) |
| LLM abstraction (any provider) | `litellm` |
| Validation / config | `pydantic`, `pydantic-settings` |
| Data processing | `pandas`, `numpy` |
| Scheduling | `APScheduler` |
| Database | SQLite via `SQLAlchemy` (Postgres-ready) |
| Alerts + kill switch | `python-telegram-bot` |
| Dashboard | `streamlit` |
| Heartbeat | healthchecks.io (HTTP ping) |
| Testing | `pytest`, `pytest-asyncio` |
| Deployment | Docker + docker-compose |
| Lint/format | `ruff` |

---

## 4. Repository Structure

```
ai-trader/
├── PLAN.md
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── config/
│   └── settings.yaml          # risk limits, pairs, schedule, models
├── src/ai_trader/
│   ├── __init__.py
│   ├── config.py              # pydantic settings, loads .env + yaml
│   ├── main.py                # entrypoint, scheduler, mode guard
│   ├── data/
│   │   ├── market.py          # candles, ticker, order book via ccxt
│   │   ├── indicators.py      # returns, volatility, MAs, RSI, spread
│   │   └── snapshot.py        # builds compact MarketSnapshot for LLM
│   ├── brokers/
│   │   ├── base.py            # Broker Protocol
│   │   ├── paper.py           # PaperBroker (simulated fills)
│   │   └── kraken.py          # KrakenBroker (live; Phase 6 only)
│   ├── ai/
│   │   ├── schema.py          # TradeProposal pydantic model
│   │   ├── prompt.py          # system + user prompt templates
│   │   └── agent.py           # litellm call, parse, validate
│   ├── risk/
│   │   └── manager.py         # RiskManager: approve/reject/resize
│   ├── storage/
│   │   ├── models.py          # SQLAlchemy tables
│   │   └── repo.py            # read/write helpers
│   ├── strategies/
│   │   ├── buy_and_hold.py    # benchmark
│   │   ├── ma_crossover.py    # rule-based benchmark
│   │   └── do_nothing.py      # benchmark
│   ├── backtest/
│   │   └── engine.py          # replays historical candles
│   ├── alerts/
│   │   ├── telegram_bot.py    # alerts + /status /stop /resume
│   │   └── heartbeat.py
│   └── dashboard/
│       └── app.py             # streamlit
└── tests/
    ├── test_paper_broker.py
    ├── test_risk_manager.py
    ├── test_schema.py
    ├── test_mode_guard.py
    └── test_backtest.py
```

---

## 5. Core Interfaces

### 5.1 Broker Protocol (`brokers/base.py`)

```python
class Broker(Protocol):
    account_id: str  # e.g. "paper-claude", "paper-gpt", "live"

    async def get_balances(self) -> dict[str, Decimal]: ...
    async def get_positions(self) -> list[Position]: ...
    async def place_order(self, order: OrderRequest) -> OrderResult: ...
    async def cancel_all(self) -> None: ...
    async def get_open_orders(self) -> list[Order]: ...
    async def get_equity(self, quote: str = "CAD") -> Decimal: ...
```

Use `Decimal` for all money and quantity math. Never use floats for balances.

### 5.2 LLM Output Schema (`ai/schema.py`)

```python
class TradeProposal(BaseModel):
    action: Literal["buy", "sell", "hold"]
    pair: str                      # must be in whitelist
    size_pct: Decimal              # 0–100, % of available equity (buy) or position (sell)
    confidence: Decimal            # 0–1
    reason: str                    # max ~500 chars
    stop_loss_pct: Decimal | None  # optional, % below entry
```

Parsing rules:
- Strip markdown fences, parse JSON, validate with pydantic.
- Any failure → log it and return `hold`. No retries that could produce a trade.

### 5.3 Market Snapshot (input to the LLM)

Compact and pre-computed. **Do not send raw candles.** The LLM judges; code calculates.

```
- timestamp (UTC) + data age in seconds
- per pair: last price, 24h change %, 7d change %, 30d change %
- volatility (24h, 7d realized)
- SMA 20/50/200 and price position relative to each
- RSI(14)
- bid/ask spread %, top-of-book depth
- current position, avg entry price, unrealized P&L %
- cash available, total equity
- last 5 decisions by this bot + their outcomes
```

Reject the snapshot (skip the cycle) if any price data is older than `MAX_DATA_AGE_SECONDS`.

---

## 6. Configuration

### `.env.example`
```
MODE=paper                       # paper | live
LIVE_TRADING_CONFIRMED=no        # must be "yes" for live
KRAKEN_API_KEY=                  # Phase 6 only; trade-only, no withdraw, IP-locked
KRAKEN_API_SECRET=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=                # only this chat can issue commands
HEALTHCHECK_URL=
DATABASE_URL=sqlite:///data/trader.db
```

### `config/settings.yaml` (defaults, all tunable)
```yaml
pairs: ["BTC/CAD", "ETH/CAD"]
quote_currency: CAD
decision_interval_minutes: 240        # start slow: every 4h
max_data_age_seconds: 60

paper:
  starting_cash_cad: 10000
  # VERIFY against current Kraken Pro fee schedule for your volume tier
  taker_fee_pct: 0.40
  maker_fee_pct: 0.25

risk:
  max_trade_pct_of_equity: 10
  max_position_pct_per_pair: 30
  max_total_exposure_pct: 60
  daily_loss_limit_pct: 3             # halt trading for the day
  max_drawdown_halt_pct: 15           # halt until manual /resume
  min_minutes_between_trades: 60
  min_confidence: 0.6
  max_trades_per_day: 6
  default_stop_loss_pct: 5

models:                               # each gets its own paper account
  - name: claude
    litellm_model: "anthropic/<model-id>"
  - name: gpt
    litellm_model: "openai/<model-id>"
  - name: gemini
    litellm_model: "gemini/<model-id>"

benchmarks: [buy_and_hold, ma_crossover, do_nothing]
```

---

## 7. Phases

### Phase 0 — Project scaffolding
**Tasks**
- Create repo structure, `pyproject.toml`, `.gitignore` (include `.env`, `data/`), `.env.example`, `ruff` config.
- `config.py` loading `.env` + `settings.yaml` with pydantic validation.
- **Mode guard** in `main.py` (Safety Invariant #1).

**Done when**
- `pytest` runs. `test_mode_guard.py` proves live mode refuses to start without both flags.

---

### Phase 1 — Data layer + PaperBroker + Risk + Storage
**Tasks**
1. `data/market.py`: fetch OHLCV, ticker, order book from Kraken public API via `ccxt` (no API key needed). Respect rate limits (`enableRateLimit=True`). Load market metadata (min order size, precision) from `load_markets()`.
2. `data/indicators.py`: returns, realized volatility, SMA, RSI, spread. Pure functions, unit-tested.
3. `data/snapshot.py`: builds `MarketSnapshot` (section 5.3).
4. `brokers/paper.py` — `PaperBroker`:
   - Market orders fill by **walking the live order book** (asks for buys, bids for sells) to compute a realistic average fill price (slippage).
   - Apply taker fee to every fill.
   - Enforce Kraken min order size and precision.
   - Reject orders exceeding available balance.
   - Simulated stop-losses: checked every price update; trigger as market sells.
   - Persist balances, positions, orders, fills to DB.
   - Supports **multiple independent accounts** (one per model + benchmarks).
5. `risk/manager.py` — `RiskManager.evaluate(proposal, account_state) -> RiskDecision`:
   - Outcomes: `approve`, `resize` (with new size), `reject` (with reason).
   - Enforces every rule in `risk:` config, the pair whitelist, and the halt states.
6. `storage/`: tables for `accounts`, `decisions`, `orders`, `fills`, `equity_snapshots`, `halts`.

**Done when**
- Tests cover: fill price across multiple book levels, fees, insufficient balance, min size, stop-loss trigger, every risk rule (approve/resize/reject), daily loss halt, drawdown halt.
- A script prints a live `MarketSnapshot` for BTC/CAD.

---

### Phase 2 — AI layer
**Tasks**
1. `ai/prompt.py`: system prompt explaining the role, the snapshot format, the risk context (informational only), and the **strict JSON-only** output requirement. Include: "If uncertain, choose hold."
2. `ai/agent.py`: call any model via `litellm`, with timeout and token limits. Parse and validate into `TradeProposal`. Log the raw response, latency, and token usage/cost.
3. Wire up the decision cycle: snapshot → agent → risk → broker → log → alert.
4. Track LLM cost per account per day in the DB.

**Done when**
- Tests: malformed JSON → hold; non-whitelisted pair → rejected; out-of-range values → rejected.
- A one-off command runs a single decision cycle for each configured model against its paper account.

---

### Phase 3 — Benchmarks + Backtester
**Tasks**
1. Implement `buy_and_hold`, `ma_crossover`, `do_nothing` with the same interface as the AI agent (they return `TradeProposal`).
2. `backtest/engine.py`: replay historical Kraken candles through the **same** RiskManager and a PaperBroker in replay mode. Historical order books aren't available, so use candle close plus a configurable slippage %.
3. Report: total return, CAGR, max drawdown, Sharpe, number of trades, fees paid.

**Important caveat (add to README):** LLM backtests on historical data are **not trustworthy**. Models have likely seen past prices in training (look-ahead bias). Use the backtester to validate plumbing and benchmarks only. The real test of the AI is Phase 5 forward paper trading.

**Done when**
- Benchmarks backtest over at least 2 years of BTC/CAD candles and produce reports.
- An optional `--with-llm` flag runs the AI on a short window (cost-capped) for plumbing tests only.

---

### Phase 4 — Unattended operation + monitoring
**Tasks**
1. `main.py`: APScheduler runs the decision cycle every `decision_interval_minutes` for all accounts, plus a stop-loss/price check every minute and a daily summary.
2. `alerts/telegram_bot.py`:
   - Alerts: every trade, every risk rejection, halts, errors, and a daily summary (equity per account vs. benchmarks).
   - Commands (accepted only from `TELEGRAM_CHAT_ID`): `/status`, `/stop` (kill switch: cancel all + halt), `/resume`, `/equity`.
3. `alerts/heartbeat.py`: ping `HEALTHCHECK_URL` after each successful cycle.
4. Graceful error handling: any exception in a cycle is logged and alerted, and the loop continues. Repeated failures (e.g., 3 in a row) trigger an automatic halt.
5. `dashboard/app.py` (Streamlit):
   - Equity curves for all accounts, including benchmarks.
   - Trade table.
   - Decision log with LLM reasoning.
   - Risk rejections.
   - LLM cost.
6. Docker:
   - `Dockerfile` plus `docker-compose.yml` with `restart: always`.
   - Volume for `data/`.
   - Dashboard bound to localhost only; access it through an SSH tunnel, never exposed publicly.

**Done when**
- `docker compose up -d` runs bot + dashboard.
- Killing the container auto-restarts it, and state survives the restart.
- `/stop` halts within one cycle.

**Deployment target:** a small VPS in a Montreal region (AWS `ca-central-1` or GCP `northamerica-northeast1`). Write a `DEPLOY.md` covering:
- Server setup.
- Firewall: SSH only.
- Docker install.
- `.env` placement.
- Updates.
- Backups of `data/`.

---

### Phase 5 — Paper trading evaluation (8–12 weeks, no code changes to strategy logic)
**Run in parallel:**
- One paper account per configured LLM.
- Benchmark accounts: buy-and-hold BTC, MA crossover, do-nothing.

**Metrics (dashboard + weekly Telegram report):**
- Return after fees and slippage.
- Max drawdown.
- Sharpe ratio.
- Number of trades and total fees.
- Win rate.
- Risk rejection rate.
- LLM cost vs. P&L.

**Rules:**
- Do not tweak prompts or risk settings mid-run. If a change is needed, restart the evaluation clock.

---

### Phase 6 — Live trading (only after go-live criteria are met)
**Tasks**
1. `brokers/kraken.py`: implements the `Broker` protocol via authenticated `ccxt`.
2. **Validate-only mode first**: send real orders with Kraken's `validate` flag (via ccxt params) so they are checked but not executed. Confirm permissions and formatting.
3. Place real **stop-loss orders on the exchange** (not only in the bot), so protection survives bot downtime.
4. Reconcile on startup: compare DB state against actual Kraken balances and orders, and alert on mismatch.
5. Keep a **shadow paper account** running the same model in parallel. Alert if live and paper results diverge materially, since that means the simulator is miscalibrated.

**API key requirements (manual step by owner):**
- Permissions: query funds, query orders, create/modify orders, cancel orders.
- **NO withdrawal permission.**
- IP whitelist: the VPS IP only.

**Start small:** a live allocation the owner can afford to lose entirely. Scale only after another 4–8 weeks of live results matching paper.

---

## 8. Go-Live Criteria (decided in advance)

Live trading is allowed only if, after **8–12 weeks and at least 50–100 paper trades**, an AI account:
1. Beats buy-and-hold BTC after fees, **or** matches it with clearly lower max drawdown.
2. Beats the MA-crossover baseline. If a simple rule does as well, the LLM adds no value.
3. Had zero unexplained bugs, risk-layer failures, or state mismatches.
4. Has LLM costs that are small relative to expected profit.

---

## 9. Testing Requirements

- Unit tests for every risk rule, PaperBroker fill math, schema validation, and the mode guard.
- Use **recorded fixtures** (saved order books/candles) for deterministic tests. No live network in unit tests.
- One integration test: a full decision cycle with a mocked LLM returning (a) a valid buy, (b) garbage, (c) an oversized order. Assert correct outcomes.
- CI-ready: `pytest` and `ruff check` must pass before each phase is marked done.

---

## 10. Notes / Things to Verify

- **Kraken fees:** confirm current Kraken Pro maker/taker rates for the account's volume tier and update `settings.yaml`.
- **Kraken pair symbols:** ccxt normalizes Kraken's `XBT` to `BTC`. Verify with `load_markets()`.
- **Canadian crypto rules:** Canadian platforms may cap annual purchases of non-major cryptos for non-eligible investors. Keep the whitelist to majors (BTC, ETH) to start.
- **Taxes (Canada/Quebec):** every live disposition is a taxable event for CRA and Revenu Québec. Make sure the `fills` table can export a CSV with date, pair, side, qty, price, fees, and CAD value.
- **LLM model IDs:** fill in current model identifiers in `settings.yaml`. Don't hardcode them.
- This system is experimental. Past paper performance does not guarantee live results.
