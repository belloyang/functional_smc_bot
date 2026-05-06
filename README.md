# Alpaca Trading Bot

This is a trading bot that uses Alpaca API to trade stocks and options.

## Features
- Backtesting with separate stock/option allocation
- Option trading with dynamic budgeting
- Stock trading with risk-based sizing
- Multi-instance safety (Run multiple tickers concurrently)
- Risk management (Daily caps, Per-ticker limits)
- Order management (Bracket orders with TP/SL)

## Prerequisites

- Python 3.12
- Alpaca API key & secret (Paper or Live)

## Installation
```bash
python3.12 -m pip install -r requirements.txt
```

## Docker
Build the image:
```bash
docker build -t smc-bot:latest .
```

Run the live bot with env file and persisted runtime state:
```bash
docker run -d \
  --name smc-bot \
  --restart unless-stopped \
  --env-file .env \
  -e APP_RUNTIME_DIR=/app/runtime \
  -v "$(pwd)/runtime:/app/runtime" \
  smc-bot:latest python -m app.bot QQQ --options
```

Run a backtest in the same image:
```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/backtest-output:/app/backtest-output" \
  smc-bot:latest \
  python scripts/backtest.py QQQ --days 90 --balance 2500 --options
```

Notes:
- `APP_RUNTIME_DIR` keeps `global_safety.json` and `trade_state_*.json` under a mounted folder so container restarts do not lose bot state.
- For Google Compute Engine, Docker on the VM is a better fit than Cloud Run because this bot is a long-running process.
- Override the container command to run `scripts/analyze_today.py` or other scripts as needed.
- Published images support both `linux/amd64` and `linux/arm64` when built through GitHub Actions.

## Docker Releases
This repo includes GitHub Actions workflows that publish Docker images to GitHub Container Registry (GHCR) whenever a version tag like `v1.5.0` is pushed or a manual release creates a new tag.

Image format:
```bash
ghcr.io/<owner>/<repo>:v1.5.0
ghcr.io/<owner>/<repo>:latest
```

Release flow:
1. Run the existing manual release workflow in GitHub Actions (`Manual Release`).
2. That workflow updates `app/__init__.py`, creates a git tag such as `v1.5.0`, pushes it, and publishes the Docker image to GHCR in the same workflow run.
3. The standalone `Publish Docker Image` workflow can still be used for manual tag-based publishes if needed.

Deploy from GHCR on your VM:
```bash
docker pull ghcr.io/<owner>/<repo>:v1.5.0

docker run -d \
  --name smc-bot \
  --restart unless-stopped \
  --env-file .env \
  -e APP_RUNTIME_DIR=/app/runtime \
  -v "$(pwd)/runtime:/app/runtime" \
  ghcr.io/<owner>/<repo>:v1.5.0 \
  python -m app.bot QQQ --options
```

Notes:
- The workflow uses `GITHUB_TOKEN`, so no separate Docker Hub credentials are required.
- If the repository or package is private, the VM will need a GHCR login before `docker pull`.
- `latest` is updated on every pushed version tag; use the versioned tag for stable deployments.

## Production With systemd
For a single GCE VM, a good production setup is Docker plus `systemd` supervision.

A ready-to-edit service template is included at:
```bash
deploy/systemd/smc-bot.service
```

Suggested VM layout:
```bash
/opt/smc-bot/.env
/opt/smc-bot/runtime/
```

Install flow on the VM:
```bash
sudo mkdir -p /opt/smc-bot/runtime
sudo cp deploy/systemd/smc-bot.service /etc/systemd/system/smc-bot.service
sudo systemctl daemon-reload
sudo systemctl enable smc-bot
sudo systemctl start smc-bot
```

Useful commands:
```bash
sudo systemctl status smc-bot
sudo systemctl restart smc-bot
sudo systemctl stop smc-bot
journalctl -u smc-bot -f
```

Before starting, edit the service file and set:
- `BOT_IMAGE` to the version you want, for example `ghcr.io/belloyang/functional_smc_bot:v1.5.0`
- `BOT_ENV_FILE` to your env file path
- `BOT_RUNTIME_DIR` to your persistent runtime directory
- `BOT_COMMAND` to the symbol/mode you want to run

## Configuration
Update `app/config.py` for global defaults:
- `STOCK_ALLOCATION_PCT`: Max % of equity for stock positions per ticker (default: 0.80).
- `OPTIONS_ALLOCATION_PCT`: Max % of equity for all option premiums (Global) (default: 0.20).
- `DEFAULT_DAILY_CAP`: Max number of trades per ticker per day.

Set your Alpaca API credentials:
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```

# Run backtest
## Options
```bash
python3.12 scripts/backtest.py QQQ --options --days 90 --balance 2500
```

## Stocks
```bash
python3.12 scripts/backtest.py QQQ --days 90 --balance 2500
```

### Parameters
- `--days`: Number of days to backtest (default: 30)
- `--balance`: Initial account balance (default: 10000.0)
- `--options`: Run with options simulation instead of stock
- `--cap`: Daily trade cap (-1 for unlimited)
- `--stock-budget`: % of balance for stock sizing (default: 0.80)
- `--option-budget`: % of balance for option sizing (default: 0.20)

# Run live trading

## Multi-Instance Support
You can run multiple instances of the bot for different tickers concurrently. By default, each uses 80% of capital, so for multiple instances you **must** manual allocate to share capital:
```bash
python3.12 -m app.bot SPY --stock-budget 0.40 &
python3.12 -m app.bot QQQ --stock-budget 0.40 &
```
Each instance respects the `MAX_TICKER_ALLOCATION` (default 80%) unless overridden.

## Session Management
### Run for a specific duration
```bash
# Run for 2 hours
python3.12 -m app.bot QQQ --session-duration 2
```

### Limit number of trades
```bash
# Run until 5 trades are executed
python3.12 -m app.bot QQQ --max-trades 5
```

### Graceful shutdown
Press `Ctrl+C` to gracefully stop the bot and see a session summary.

### All Parameters
- `symbol`: Symbol to trade (default: SPY)
- `--options`: Enable options trading
- `--cap`: Daily trade cap (-1 for unlimited)
- `--session-duration`: Session duration in hours
- `--max-trades`: Maximum number of trades per session
- `--stock-budget`: Manual % allocation for stock mode (overrides config)
- `--option-budget`: Manual % allocation for option mode (overrides config)

