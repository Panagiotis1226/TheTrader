# Using ai-trader

Everything here is done on the machine the bot runs on, in a terminal. Telegram and
healthchecks.io are optional extras (see the end).

## 1. Where to run it

The Phase 5 evaluation runs for 8–12 weeks and needs a decision every 4 hours, so the bot
needs a machine that stays on.

- **A small VPS** (recommended): always on, about $10–20/month. Follow [`DEPLOY.md`](DEPLOY.md).
- **Your own computer**: works the same with Docker Desktop, but the computer must not sleep
  (on a Mac: keep it plugged in and turn off automatic sleep in *System Settings → Battery*
  or *Energy*). Cycles missed while it sleeps are simply skipped.

## 2. One-time setup

You need exactly one secret: a token that lets the bot use your Claude Team seat.

```bash
# on your computer (needs a browser); Claude Desktop alone doesn't include the CLI
curl -fsSL https://claude.ai/install.sh | bash
claude setup-token            # sign in, copy the token (valid one year)
```

Then, in the project folder on the machine that will run the bot:

```bash
cp .env.example .env
# edit .env: paste the token after CLAUDE_CODE_OAUTH_TOKEN=   (leave MODE=paper)
mkdir -p data
```

On Linux, also run `sudo chown 1000:1000 data` (the container's user).

## 3. Start and stop the bot

```bash
docker compose up -d --build     # start (bot + dashboard); restarts by itself after crashes/reboots
docker compose stop              # stop everything
docker compose start             # start again; it picks up where it left off
docker compose logs -f bot       # live log (Ctrl-C to leave)
```

## 4. Day-to-day commands

Run these in the project folder while the bot is running. They talk to the running bot
through its database.

| Command | What it does |
|---|---|
| `docker compose exec bot ai-trader status` | Accounts, positions, halts, last decision, recent alerts, evaluation progress |
| `docker compose exec bot ai-trader stop` | **Kill switch**: halts all trading now (also blocks a decision in progress). Stop-losses stay active. Add `--reason "..."` to record why. |
| `docker compose exec bot ai-trader resume` | Lifts all halts |
| `docker compose exec bot ai-trader report` | Performance per account: return, drawdown, Sharpe, trades, fees, win rate, rejections, LLM cost |
| `docker compose exec bot ai-trader evaluate` | Phase 5 go-live scorecard |
| `docker compose exec bot ai-trader once` | Runs one decision cycle right now (normally every 4 h) |

Tip: `alias at='docker compose exec bot ai-trader'` makes these `at status`, `at stop`, …

Without Docker (`pip install -e .` in a Python 3.12+ virtualenv): run `ai-trader run` in one
terminal (Ctrl-C stops it) and the same commands without the `docker compose exec bot` prefix
in another.

## 5. The dashboard

Open <http://localhost:8501> on the machine running the bot. On a VPS, open an SSH tunnel
first: `ssh -L 8501:127.0.0.1:8501 you@server`. It shows the status, return curves for every
account, the Phase 5 scorecard, trades, every decision with Claude's reasoning, risk
rejections, alerts and LLM cost. It is read-only; use the commands above to act.

## 6. What the bot does on its own

- Every 4 hours: Claude and the three benchmarks each get one decision; the risk manager
  approves, resizes or rejects; approved orders fill against Kraken's live order book (paper).
- Every minute: stop-losses are checked. Every hour: equity is recorded.
- Daily-loss limit (3%) blocks new buys until midnight UTC; a 15% drawdown blocks new buys
  until you `resume`. Sells and stop-losses keep working during these halts.
- 3 failed cycles in a row halt that account until you `resume`.
- Every alert is kept: `status` shows the latest, the dashboard shows the history.

## 7. Running the Phase 5 evaluation

1. **Freeze the configuration before starting.** The evaluation clock starts at Claude's first
   real answer and restarts by itself if the prompt, risk limits, fees, pairs or interval
   change. The bot warns at startup when that happens, and `evaluate` shows the
   configuration ID. Monitoring settings (report times etc.) don't affect it.
2. **Once a week**: `ai-trader status` (anything halted? alerts?) and `ai-trader evaluate`.
3. **If something breaks**: a halt or error alert means look at `status` and the logs, fix the
   cause, then `resume`. Write down anything you can't explain: criterion 3 asks for zero
   unexplained bugs.
4. **When it's done** (8–12 weeks and 50+ trades), `evaluate` shows every criterion:

   | Criterion (PLAN.md §8) | Judged |
   |---|---|
   | 8+ weeks, 50+ trades | automatically |
   | 1. Beats buy-and-hold, or matches it (±1 pt) with ≥25% lower max drawdown | automatically |
   | 2. Beats the MA crossover | automatically |
   | 3. No unexplained bugs or state mismatches | automatic checks + **you** |
   | 4. LLM cost small vs. profit | **you** (on a Claude seat the cost is the subscription) |
   | Model reached every cycle; same model throughout | flagged automatically |

   Thresholds are in `config/settings.yaml` under `evaluation:`. Decide them before you
   start, not after. Going live (Phase 6) is always your decision, never automatic.

## 8. Optional extras

- **Telegram** (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` in `.env`): alerts on your phone, and
  `/status`, `/equity`, `/stop`, `/resume` from anywhere. Setup steps in `DEPLOY.md` §5.
- **healthchecks.io** (`HEALTHCHECK_URL`): an email if the bot stops running.

Without them, nothing is lost: alerts are kept in the database and the log.

## 9. Troubleshooting

- **`disk I/O error` from SQLite at startup** (Docker Desktop on Mac/Windows, versions before
  this fix): stop the bot, delete the database files, update, and start again:
  ```bash
  docker compose down
  rm -f data/trader.db data/trader.db-wal data/trader.db-shm      # Windows: del data\trader.db*
  git pull && docker compose up -d --build
  ```
- **`Not logged in` in `status`**: the token in `.env` is missing or mistyped
  (`CLAUDE_CODE_OAUTH_TOKEN=...` on one line, no quotes). After fixing `.env`, run
  `docker compose up -d` again so the bot picks it up.
- **The bot keeps restarting**: `docker compose logs --tail 50 bot` shows why.
