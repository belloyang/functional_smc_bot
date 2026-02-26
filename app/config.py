import os
from alpaca.trading import TradingClient
import random
from . import __version__

def load_env(file_path=".env"):
    """Simple helper to load .env variables without external dependencies."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    # Remove inline comments
                    if " #" in line: # Only split if there's a space before # to avoid issues with values containing #
                        line = line.split(" #", 1)[0]
                    elif line.startswith("#"):
                        continue
                        
                    key, value = line.split("=", 1)
                    # Clean up key and value (remove spaces and quotes)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

# Load environment variables from .env if it exists
load_env()

# --- IBKR SETTINGS ---
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497")) # 7497 for paper, 7496 for live
# set IBKR_CLIENT_ID to a random number to avoid conflicts
IBKR_CLIENT_ID = random.randint(1, 1000)
# 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen
IBKR_MARKET_DATA_TYPE = int(os.getenv("IBKR_MARKET_DATA_TYPE", "1"))
IBKR_HIST_TIMEOUT_SEC = float(os.getenv("IBKR_HIST_TIMEOUT_SEC", "20"))
IBKR_HIST_MAX_ATTEMPTS = int(os.getenv("IBKR_HIST_MAX_ATTEMPTS", "2"))
IBKR_HIST_BACKOFF_BASE_SEC = float(os.getenv("IBKR_HIST_BACKOFF_BASE_SEC", "1.0"))

# --- ALPACA (BACKWARD COMPATIBILITY) ---
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_WEBHOOK_URL_LIVE_TRADING = os.getenv("DISCORD_WEBHOOK_URL_LIVE_TRADING")

if not API_KEY or not API_SECRET:
    print("WARNING: ALPACA_API_KEY or ALPACA_API_SECRET not found in environment variables.")

BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL = "SPY" # default symbol
TIMEFRAME_HTF = "15Min"
TIMEFRAME_LTF = "1Min"

# Note: TradingClient will fail if keys are None, but we catch it here
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

RISK_PER_TRADE = 0.02  # 2% 
try:
    ACCOUNT_BALANCE = trading_client.get_account().equity
except Exception:
    ACCOUNT_BALANCE = 10000 # Fallback for local testing without keys

ENABLE_OPTIONS = False # Set to True to trade options instead of shares

# --- RISK MANAGEMENT ---
STOCK_ALLOCATION_PCT = 0.80   # Max 80% of equity for stock positions per ticker
OPTIONS_ALLOCATION_PCT = 0.15 # Max 15% of equity for all option premiums (Global)
MAX_OPTION_CONTRACTS = -1     # -1 means no explicit contract-count cap
DEFAULT_DAILY_CAP = 5         # Default daily trade cap if none provided

# --- DRAWDOWN & BEHAVIORAL SAFETY ---
MAX_GLOBAL_DRAWDOWN = 0.25    # Circuit breaker: Halt trading at 25% drop from peak
COOL_DOWN_MINUTES = 15        # Wait period after any losing trade
MAX_ALLOWED_IV = 0.40  # 40%
ALLOW_TRADE_WITHOUT_IV = True  # If True, bot will proceed when IV snapshot is missing
STALE_PENDING_ENTRY_MINUTES = 5  # Cancel unfilled entry orders after N minutes to avoid blocking
