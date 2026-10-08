# Deploying ai-trader on a VPS

Paper trading, unattended, on a small Linux server in Montreal. Allow about an hour.

## 1. Server

- **Where:** AWS `ca-central-1` (Montreal) or GCP `northamerica-northeast1` (Montreal).
- **Size:** 2 vCPU, 2 GB RAM, 20 GB disk (Claude Code + Streamlit don't fit well in 1 GB).
- **OS:** Ubuntu 24.04 LTS.

First login, as root or the default sudo user:

```bash
adduser trader && usermod -aG sudo trader          # your day-to-day user
# copy your SSH public key to /home/trader/.ssh/authorized_keys, then:
sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/; s/^#\?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
sudo apt update && sudo apt -y upgrade && sudo apt -y install unattended-upgrades git
```

## 2. Firewall: SSH only

In the cloud console, the security group / firewall rule allows **only TCP 22**, ideally only
from your own IP. On the server as well:

```bash
sudo ufw default deny incoming && sudo ufw default allow outgoing
sudo ufw allow OpenSSH && sudo ufw enable
```

Docker can bypass `ufw` for published ports. That's fine here: the only published port is the
dashboard, bound to `127.0.0.1`. Never change it to `0.0.0.0` or `8501:8501`.

## 3. Docker

Follow Docker's official Ubuntu instructions (<https://docs.docker.com/engine/install/ubuntu/>),
then:

```bash
sudo usermod -aG docker trader && newgrp docker
docker run --rm hello-world
sudo systemctl enable docker        # start Docker (and the bot) after a reboot
```

## 4. Code

```bash
cd ~ && git clone https://github.com/Panagiotis1226/TheTrader.git && cd TheTrader
```

Data (database, candle cache, backtests) goes to the Docker volume `trader-data`, created on
first start and owned by the container's user. `docker compose down` keeps it;
`docker compose down -v` deletes it, and with it the whole trading history.

## 5. `.env` (secrets)

```bash
cp .env.example .env && chmod 600 .env && nano .env
```

Keep `MODE=paper` and `LIVE_TRADING_CONFIRMED=no`. Only the Claude token is required;
Telegram and healthchecks.io are optional.

**`CLAUDE_CODE_OAUTH_TOKEN`**: lets the bot use your Claude Team seat. On your own computer
(it needs a browser), install the Claude Code CLI and create a token:

```bash
curl -fsSL https://claude.ai/install.sh | bash     # the CLI; Claude Desktop alone isn't enough
claude setup-token                                 # sign in, copy the token it prints
```

The token is valid for a year; put a reminder in your calendar to renew it.

**`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`** (optional): alerts on your phone and remote
commands. Without them, use `ai-trader status` / `stop` / `resume` on the server.
1. In Telegram, message **@BotFather**, send `/newbot`, and copy the token.
2. Open a **private** chat with your new bot and send it any message. (Don't use a group:
   every member could send `/stop`.)
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser; `"chat":{"id": ...}`
   is your chat ID.

**`HEALTHCHECK_URL`** (optional): emails you if the bot stops running. At <https://healthchecks.io>,
create a check with **period 4 hours** (the decision interval) and **grace 1 hour**, add your
email or Telegram as an alert channel, and copy its ping URL.

`KRAKEN_API_KEY` / `KRAKEN_API_SECRET` stay empty: paper trading only uses public data.

## 6. Start

```bash
docker compose up -d --build
docker compose logs -f bot        # Ctrl-C to stop following
```

Within a minute the first decision cycle runs. Check it with
`docker compose exec bot ai-trader status` (or `/status` in Telegram if configured).
Day-to-day use is in [`USAGE.md`](USAGE.md).

## 7. Dashboard (through an SSH tunnel)

From your computer:

```bash
ssh -L 8501:127.0.0.1:8501 trader@<server-ip>
```

Then open <http://localhost:8501>. Close the SSH session to close the tunnel.

## 8. Day-to-day

| | |
|---|---|
| Kill switch | `docker compose exec bot ai-trader stop` (or Telegram `/stop`): halts all trading at once and cancels open orders. Stop-losses stay active. `ai-trader resume` lifts all halts. |
| Status | `docker compose exec bot ai-trader status` / `report` / `evaluate` (or `/status`, `/equity`); daily summary at 08:00 Quebec time, weekly report on Mondays |
| Logs | `docker compose logs --tail 200 bot` |
| Stop / start the bot | `docker compose stop bot` / `docker compose start bot` |
| One-off cycle | `docker compose exec bot ai-trader once` |
| Backtest | `docker compose exec bot ai-trader backtest --refresh-data` |

Restarts: if the bot crashes, Docker restarts it within seconds and it carries on from the
database. A container you stop yourself (`docker compose stop`, `docker kill`) stays stopped
until you start it. After a server reboot everything comes back on its own.

If an account fails 3 cycles in a row, the bot halts it and raises an alert (`status`,
dashboard, Telegram if configured); fix the cause (logs), then `ai-trader resume`.

## 9. Updates

```bash
cd ~/TheTrader && git pull
docker compose build --pull && docker compose up -d
```

During the Phase 5 evaluation, don't change the prompt or the risk settings: that restarts
the evaluation clock. Every decision records a `prompt_hash`, so any change is visible in the
dashboard.

The Claude Code CLI inside the image doesn't auto-update. It updates when the image is rebuilt
from scratch (`docker compose build --no-cache`); pin a version with
`--build-arg CLAUDE_CODE_VERSION=x.y.z`.

## 10. Backups

Everything that matters is in `/app/data/trader.db` inside the `trader-data` volume (plus
your `.env`). SQLite must be backed up with its online backup command, not by copying the
file while the bot writes to it. This takes a consistent copy inside the container and
streams it out to `backups/` in the project folder:

```bash
mkdir -p backups
docker compose exec -T bot sh -c 'sqlite3 /app/data/trader.db ".backup /tmp/backup.db" && cat /tmp/backup.db && rm /tmp/backup.db' > backups/trader-$(date +%F).db
```

Daily at 03:00, keeping 30 days (`crontab -e` on the server; `%` must be escaped in cron):

```cron
0 3 * * * cd /home/trader/TheTrader && mkdir -p backups && docker compose exec -T bot sh -c 'sqlite3 /app/data/trader.db ".backup /tmp/backup.db" && cat /tmp/backup.db && rm /tmp/backup.db' > backups/trader-$(date +\%F).db && find backups -name 'trader-*.db' -mtime +30 -delete
```

Copy `backups/` off the server regularly (e.g. `rsync` or `scp` to your computer, or
`rclone` to cloud storage). Keep a copy of `.env` somewhere safe and private (a password
manager), never in git.

**Restore:** stream the backup into the volume while the bot is stopped, then start it:

```bash
docker compose stop bot
docker compose run --rm -T --no-deps bot sh -c 'cat > /app/data/trader.db && rm -f /app/data/trader.db-journal' < backups/trader-YYYY-MM-DD.db
docker compose start bot
```
