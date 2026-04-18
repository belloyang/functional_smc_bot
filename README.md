# Alpaca Trading Bot

Trading and backtesting bot for US stocks/options with SMC-style signal logic.

## Features
- Live bot for stock or option mode
- Backtest engine using the same strategy functions as the live bot
- Runtime bot config via JSON file (`--config`)
- Batch backtest runner with config-driven `jobs`
- Risk controls: daily trade cap, option allocation, confidence filter, session limits

## Prerequisites
- Python 3.12
- Alpaca API key and secret

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

## Docker Releases
This repo includes a GitHub Actions workflow that publishes Docker images to GitHub Container Registry (GHCR) whenever a version tag like `v1.5.0` is pushed.

Image format:
```bash
ghcr.io/<owner>/<repo>:v1.5.0
ghcr.io/<owner>/<repo>:latest
```

Release flow:
1. Run the existing manual release workflow in GitHub Actions (`Manual Release`).
2. That workflow updates `app/__init__.py`, creates a git tag such as `v1.5.0`, and pushes it.
3. The `Publish Docker Image` workflow builds the Docker image and pushes it to GHCR automatically.

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

## Environment variables
The project loads `.env` from the current working directory through `app/config.py`.

Example `.env`:
```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
DISCORD_WEBHOOK_URL=
DISCORD_WEBHOOK_URL_LIVE_TRADING=
```

Important:
- Run commands from repo root so `.env` is found.
- Use `KEY=value` format in `.env` (not `export KEY=value`).

## Global defaults
You can adjust defaults in `app/config.py`:
- `OPTIONS_ALLOCATION_PCT`
- `DEFAULT_DAILY_CAP`
- `RISK_PER_TRADE` (default `0.01`, i.e. 1%)

## Run backtest
Stock mode:
```bash
python3.12 scripts/backtest.py QQQ --days 90 --balance 2500
```

Option mode:
```bash
python3.12 scripts/backtest.py QQQ --options --days 90 --balance 2500
```

Common params:
- `--days`
- `--balance`
- `--options`
- `--cap`
- `--option-allocation` (stock allocation is `1 - option-allocation`)
- `--max-option-contracts` (`-1` for unlimited, default)
- `--min-conf` (`all|low|medium|high`)

## Run batch backtests
Use `scripts/run_backtests.py` with either config or CLI lists.

Config file example:
```bash
python3.12 scripts/run_backtests.py --config scripts/backtest_batch_config.json
```

Direct CLI example:
```bash
python3.12 scripts/run_backtests.py --symbols SPY,QQQ --days 30,90 --balances 2500,10000 --modes stock,options
```

Notes:
- If `jobs` exists in config, it takes precedence over cartesian `symbols/days/balances/modes`.
- Logs are written under `<output>/logs`.
- Backtest charts are written under `<output>/backtest-output`.

## Run live bot
Basic:
```bash
python3.12 -m app.bot QQQ
```

Option mode:
```bash
python3.12 -m app.bot QQQ --options
```

With runtime JSON config:
```bash
python3.12 -m app.bot --config app/bot_runtime_config.json
```

You can still override config-file values from CLI:
```bash
python3.12 -m app.bot --config app/bot_runtime_config.json --min-conf high --cap 2
```

Supported runtime JSON keys:
- `symbol`
- `options` or `enable_options`
- `cap` or `daily_cap`
- `session_duration` or `session-duration`
- `option_allocation` or `option-allocation` (stock allocation is `1 - option-allocation`)
- `max_option_contracts` or `max-option-contracts` (`-1` for unlimited)
- `state_file` or `state-file`
- `min_conf` (`all|low|medium|high`)

## Multi-instance example
```bash
python3.12 -m app.bot SPY --option-allocation 0.60 &
python3.12 -m app.bot QQQ --option-allocation 0.60 &
```

## Graceful shutdown
Press `Ctrl+C` to stop and print a session summary.
