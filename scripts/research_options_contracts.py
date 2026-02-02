from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, AssetExchange
from datetime import datetime, timedelta
import sys
import os
# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config

def research_options():
    print("Connecting to Alpaca...")
    client = TradingClient(config.API_KEY, config.API_SECRET, paper=True)
    
    # We want to find contracts for the underlying symbol
    underlying = config.SYMBOL
    print(f"Searching for options on {underlying}...")
    
    # Expiration range: next 7 days
    now = datetime.now()
    max_expiry = now + timedelta(days=7)
    
    # Note: providing too many filters might be safer or necessary
    try:
        # Check if GetOptionContractsRequest exists/works
        req = GetOptionContractsRequest(
            underlying_symbol=underlying,
            status=AssetStatus.ACTIVE,
            expiration_date_gte=now.date(),
            expiration_date_lte=max_expiry.date(),
            limit=10  # Just get a few to dry run
        )
        
        contracts = client.get_option_contracts(req)
        print(f"Found {len(contracts.option_contracts)} contracts.")
        
        for c in contracts.option_contracts[:5]:
            print(f"Symbol: {c.symbol}, Type: {c.type}, Strike: {c.strike_price}, Exp: {c.expiration_date}")
            
    except Exception as e:
        print(f"Error listing options: {e}")
        # Fallback or alternative method check
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    research_options()
