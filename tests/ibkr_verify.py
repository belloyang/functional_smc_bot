import asyncio
import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ibkr_manager import ibkr_mgr
from ib_insync import Stock

async def verify():
    print("Starting IBKR Verification...")
    
    # Use a different client ID for verification to avoid conflicts with the bot
    from app import config
    config.IBKR_CLIENT_ID = 99
    
    # 1. Test Connection
    connected = await ibkr_mgr.connect()
    if not connected:
        print("❌ Failed to connect to IBKR. Ensure TWS/Gateway is running.")
        return
    
    print("✅ Connected successfully.")
    ib = ibkr_mgr.ib
    
    # 2. Test Account Data
    print("\nFetching Account Data...")
    acc_values = ib.accountValues()
    found_equity = False
    for v in acc_values:
        if v.tag == 'NetLiquidation':
            print(f"✅ Found Equity: ${v.value}")
            found_equity = True
            break
    if not found_equity:
        print("⚠️ Could not find NetLiquidation. This might happen if the account is new or empty.")

    # 3. Test Market Data (Historical)
    print("\nFetching Historical Data for SPY...")
    # NOTE: Historical data works without a subscription in Paper Trading mostly, 
    # but we will try with trades first.
    contract = Stock('SPY', 'SMART', 'USD')
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='MIDPOINT' if "Delayed" in str(ib.reqMarketDataType) else 'TRADES',
        useRTH=True
    )
    if bars:
        print(f"✅ Received {len(bars)} bars.")
        print(f"   Last Close: {bars[-1].close}")
    else:
        print("❌ Failed to fetch historical data. (If trading is closed, try MIDPOINT)")

    # 4. Test Ticker (Real-time/Delayed)
    print("\nFetching Ticker for SPY (Market Data Type 3: Delayed)...")
    ib.reqMktData(contract)
    print("Waiting 5 seconds for data to stream...")
    await asyncio.sleep(5)
    ticker = ib.ticker(contract)
    
    # Try multiple price fields as delayed data might populate differently
    price = ticker.marketPrice()
    if np.isnan(price) or price <= 0:
        price = ticker.last if ticker.last > 0 else ticker.close
        
    print(f"   Ticker Price Field: {price}")
    if price > 0:
        print("✅ Data received.")
    else:
        print("⚠️ Price is 0 or NaN. Delayed data may take a few minutes to start streaming if this is your first time.")

    print("\nVerification Complete.")
    ibkr_mgr.disconnect()

if __name__ == "__main__":
    asyncio.run(verify())
