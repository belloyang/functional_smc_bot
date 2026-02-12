import os
from alpaca.trading import TradingClient

def load_env(file_path=".env"):
    """Simple helper to load .env variables without external dependencies."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Clean up key and value (remove spaces and quotes)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

# Load environment variables from .env if it exists
load_env()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_WEBHOOK_URL_LIVE_TRADING = os.getenv("DISCORD_WEBHOOK_URL_LIVE_TRADING")

if not API_KEY or not API_SECRET:
    print("WARNING: ALPACA_API_KEY or ALPACA_API_SECRET not found in environment variables.")

BASE_URL = "https://paper-api.alpaca.markets"  # paper trading
SYMBOL = "SPY" # default symbol
TIMEFRAME_HTF = "15Min"
TIMEFRAME_LTF = "1Min"

# Note: TradingClient will fail if keys are None, but we catch it here
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

RISK_PER_TRADE = 0.01  # 1%
try:
    ACCOUNT_BALANCE = trading_client.get_account().equity
except Exception:
    ACCOUNT_BALANCE = 10000 # Fallback for local testing without keys

ENABLE_OPTIONS = False # Set to True to trade options instead of shares

# --- RISK MANAGEMENT ---
STOCK_ALLOCATION_PCT = 0.80   # Max 80% of equity for stock positions per ticker
OPTIONS_ALLOCATION_PCT = 0.15 # Max 15% of equity for all option premiums (Global)
DEFAULT_DAILY_CAP = 5         # Default daily trade cap if none provided
