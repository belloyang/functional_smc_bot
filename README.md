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


