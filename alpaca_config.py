from alpaca.trading import TradingClient

API_KEY = "PKGYMDJH2IQXKAR26DBWFXYPW7"
API_SECRET = "7UKv2d3WfpyE2pxLRiZhNb8LkdZvLyp3bWn5g5z27REd"
BASE_URL = "https://paper-api.alpaca.markets"  # paper trading
SYMBOL = "SPY" # default symbol
TIMEFRAME_HTF = "15Min"
TIMEFRAME_LTF = "1Min"

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

RISK_PER_TRADE = 0.01  # 1%
ACCOUNT_BALANCE = trading_client.get_account().equity
