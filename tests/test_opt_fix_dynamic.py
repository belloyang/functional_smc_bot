import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app import config
config.IBKR_CLIENT_ID = 97 # Unique ID for this test

from ib_insync import *
from app.ibkr_manager import ibkr_mgr
from app.bot import get_best_option_contract

async def test_dynamic_expiry_and_exchange():
    print("Connecting to IBKR...")
    connected = await ibkr_mgr.connect()
    if not connected:
        print("Failed to connect.")
        return

    ib = ibkr_mgr.ib
    
    symbols = ['QQQ', 'SPY']
    for symbol in symbols:
        print(f"\n--- Testing Symbol: {symbol} ---")
        try:
            # Test getting a CALL
            contract = await get_best_option_contract(symbol, 'buy')
            
            if contract:
                print(f"✅ SUCCESS: Found Contract: {contract.localSymbol}")
                print(f"  - ConId: {contract.conId}")
                print(f"  - Expiry: {contract.lastTradeDateOrContractMonth}")
                print(f"  - Strike: {contract.strike}")
                print(f"  - Exchange: '{contract.exchange}'")
                
                # Check if it passes the "exchange" requirement
                try:
                    print(f"  - Verifying Market Data (should not trigger Error 321)...")
                    ib.reqMktData(contract)
                    await asyncio.sleep(2)
                    ticker = ib.ticker(contract)
                    print(f"  - Market Price: {ticker.marketPrice()}")
                except Exception as ex:
                    print(f"  ❌ Error requesting market data: {ex}")
            else:
                print(f"❌ FAILURE: Could not find dynamic contract for {symbol}")
                
        except Exception as e:
            print(f"❌ Exception during test for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    ibkr_mgr.disconnect()

if __name__ == "__main__":
    asyncio.run(test_dynamic_expiry_and_exchange())
