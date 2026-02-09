import asyncio
import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Set a unique client ID for testing before importing ibkr_mgr
os.environ["IBKR_CLIENT_ID"] = "99"

from unittest.mock import patch
from app import config
config.IBKR_CLIENT_ID = 99

from ib_insync import *
from app.ibkr_manager import ibkr_mgr
from app.bot import get_bars, get_latest_price_fallback

async def test_option_pricing():
    print("Connecting to IBKR...")
    connected = await ibkr_mgr.connect()
    if not connected:
        print("Failed to connect.")
        return

    ib = ibkr_mgr.ib
    
    # Test with a SPY Option (similar to the one in the error report)
    # Note: Strike and expiry might need to be adjusted for current date
    symbol = 'SPY'
    right = 'C'
    # Try to find a real contract first to be sure
    underlying = Stock(symbol, 'SMART', 'USD')
    [underlying] = await ib.qualifyContractsAsync(underlying)
    
    # Get chain details (minimal)
    chains = await ib.reqSecDefOptParamsAsync(underlying.symbol, '', underlying.secType, underlying.conId)
    if not chains:
        print("No option chains found. Ensure you have some market data connection.")
        ibkr_mgr.disconnect()
        return
        
    chain = next(c for c in chains if c.exchange == 'SMART')
    expiry = chain.expirations[0]
    strike = chain.strikes[len(chain.strikes)//2] # Middle strike
    
    contract = Option(symbol, expiry, strike, right, 'SMART')
    [contract] = await ib.qualifyContractsAsync(contract)
    
    print(f"Testing Pricing for {contract.localSymbol} (conId: {contract.conId})")
    
    # 1. Test Real-time Ticker (likely fails if no subscription)
    ib.reqMktData(contract)
    print("Waiting for ticker...")
    await asyncio.sleep(2)
    ticker = ib.ticker(contract)
    
    print(f"Ticker Ask: {ticker.ask}")
    print(f"Ticker Bid: {ticker.bid}")
    print(f"Ticker Last: {ticker.last}")
    print(f"Ticker MarketPrice: {ticker.marketPrice()}")
    
    # 1.5 Test Stock Pricing (to verify get_bars refactor)
    print("\nTesting Stock MIDPOINT Price for SPY...")
    stock_price = await get_latest_price_fallback('SPY')
    if stock_price:
        print(f"✅ SUCCESS: Stock Midpoint Price: {stock_price}")
    else:
        print("❌ FAILURE: Could not retrieve stock midpoint price.")

    # 2. Test Fallback (Historical Midpoint)
    print("\nTesting Historical MIDPOINT Fallback for Option...")
    # Try with useRTH=False for better coverage
    with patch('app.bot.ibkr_mgr.ib.reqHistoricalDataAsync', wraps=ib.reqHistoricalDataAsync) as mock_req:
        fallback_price = await get_latest_price_fallback(contract)
    
    if fallback_price:
        print(f"✅ SUCCESS: Historical Midpoint Price: {fallback_price}")
    else:
        print("❌ FAILURE: Could not retrieve historical midpoint price.")
        
    ibkr_mgr.disconnect()

if __name__ == "__main__":
    asyncio.run(test_option_pricing())
