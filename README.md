# Alpaca Trading Bot

This is a trading bot that uses Alpaca API to trade stocks and options.

## Features
- Backtesting
- Option trading
- Stock trading
- Risk management
- Order management

## Prerequisites

- Python 3.12
- Alpaca API key
- Alpaca API secret
- Alpaca API paper trading

## Installation
```
python3.12 -m pip install -r requirements.txt
```


## Configuration
Set your Alpaca API credentials as environment variables for security:
```bash
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```
Alternatively, create a `.env` file in the root directory (it is ignored by git):
```text
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
```

# Run backtest
## Options
```bash
python3.12 scripts/backtest.py QQQ --options --days 30 --balance 10000
```

## Stocks
```bash
python3.12 scripts/backtest.py QQQ --days 30 --balance 10000
```

### Parameters
- `--days`: Number of days to backtest (default: 30)
- `--balance`: Initial account balance (default: 10000.0)
- `--options`: Run with options simulation instead of stock


# Run live trading
## Options
You can enable options trading by passing the `--options` flag:
```bash
python3.12 -m app.bot QQQ --options
```
Alternatively, update `ENABLE_OPTIONS` to `True` in `app/config.py`.

## Stocks
Default behavior (or ensure `ENABLE_OPTIONS` is `False` in `app/config.py`):
```bash
python3.12 -m app.bot QQQ 
```



