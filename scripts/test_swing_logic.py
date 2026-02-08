import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config
from app.swing_strategy import get_swing_signal, SMCStructure, SMCLiquidity
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def test_swing_logic(symbol="SPY"):
    print(f"Testing Swing Logic for {symbol}...")
    
    # 1. Setup Client
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    
    # 2. Fetch Sample Data (Daily)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)
    
    print("Fetching Daily Data...")
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Day), start=start_dt, end=end_dt, feed="iex")
    d1_bars = client.get_stock_bars(req).df
    d1_bars = d1_bars.reset_index()
    
    # 3. Test Structure Analysis
    print("\n--- Testing Structure (SMC) ---")
    smc = SMCStructure()
    bias, high, low = smc.determine_bias(d1_bars)
    
    print(f"Computed Bias: {bias.upper()}")
    if high and low:
        # Debug print
        # print(f"DEBUG: Low Time Type: {type(low['time'])} Val: {low['time']}")
        
        low_date = low['time'].date() if hasattr(low['time'], 'date') else low['time']
        high_date = high['time'].date() if hasattr(high['time'], 'date') else high['time']
        
        print(f"Current Range: Low {low['price']} ({low_date}) -> High {high['price']} ({high_date})")
        
        current_price = d1_bars['close'].iloc[-1]
        zone = smc.check_premium_discount(current_price, high, low)
        print(f"Current Price: {current_price:.2f}")
        print(f"Zone:          {zone.upper()}")
    else:
        print("Not enough swings found to determine range.")
        
    # 4. Test Liquidity Analysis (Hourly)
    print("\n--- Testing Liquidity (H1) ---")
    start_dt_h1 = end_dt - timedelta(days=60)
    req_h = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Hour), start=start_dt_h1, end=end_dt, feed="iex")
    h1_bars = client.get_stock_bars(req_h).df
    h1_bars = h1_bars.reset_index()
    
    liq = SMCLiquidity()
    is_sweep = liq.detect_liquidity_sweep(h1_bars, bias, window=30)
    print(f"Liquidity Sweep Detected? {is_sweep}")
    
    ob = liq.find_order_blocks(h1_bars, bias)
    if ob:
        print(f"Nearest OB: {ob['type'].upper()} at {ob['high']:.2f}-{ob['low']:.2f} ({ob['time']})")
    else:
        print("No Order Block found nearby.")
        
    # 5. Full Signal Test
    print("\n--- Full Signal Orchestration ---")
    sig = get_swing_signal(symbol, client)
    print(f"Final Signal: {sig}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    test_swing_logic(target)
