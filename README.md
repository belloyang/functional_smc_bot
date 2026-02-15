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
- `state_file` or `state-file`
- `min_conf` (`all|low|medium|high`)

## Multi-instance example
```bash
python3.12 -m app.bot SPY --option-allocation 0.60 &
python3.12 -m app.bot QQQ --option-allocation 0.60 &
```

## Graceful shutdown
Press `Ctrl+C` to stop and print a session summary.
