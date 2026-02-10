import pandas as pd
import numpy as np
import pandas_ta as ta
import sys
import os

# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config
from app.bot import get_strategy_signal, detect_fvg, detect_order_block
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

def analyze_today_signals(symbol="SPY"):
    """
    Scans every minute of today's market session to find all strategy signals.
    """
    ET = ZoneInfo("US/Eastern")
    now_et = datetime.now(ET)
    today_str = now_et.strftime("%Y-%m-%d")
    
    print(f"🔍 Scanning {symbol} for ALL signals today ({today_str})...")
    print("="*80)
    
    # 1. Fetch Data
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    
    end_time = datetime.now(timezone.utc)
    # Get enough history for EMA50 (need at least 50 HTF bars = 750 mins + buffer)
    start_time = end_time - timedelta(days=5) 
    
    # Try SIP feed first, fallback to IEX
    feeds = ["sip", "iex"]
    htf_data = None
    ltf_data = None
    
    print("Fetching data (attempting SIP then IEX)...")
    for feed in feeds:
        try:
            # HTF Data (15 Min)
            htf_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time,
                feed=feed
            )
            htf_data = client.get_stock_bars(htf_req).df
            
            # LTF Data (1 Min)
            ltf_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time,
                feed=feed
            )
            ltf_data = client.get_stock_bars(ltf_req).df
            
            if not htf_data.empty and not ltf_data.empty:
                print(f"✅ Successfully loaded data using '{feed}' feed.")
                break
        except Exception as e:
            print(f"⚠️ Feed '{feed}' failed or not available.")
            continue

    if htf_data is None or htf_data.empty or ltf_data is None or ltf_data.empty:
        print("❌ CRITICAL: Could not fetch data from any feed. Check API keys.")
        return

    # 2. Process Data
    htf_data = htf_data.reset_index()
    ltf_data = ltf_data.reset_index()
    htf_data.set_index('timestamp', inplace=True)
    ltf_data.set_index('timestamp', inplace=True)
    htf_data.sort_index(inplace=True)
    ltf_data.sort_index(inplace=True)
    
    # Pre-calculate HTF indicators
    htf_data['ema50'] = ta.ema(htf_data['close'], length=50)
    
    # Pre-calculate LTF indicators
    ltf_data = detect_fvg(ltf_data)
    ltf_data = detect_order_block(ltf_data)
    
    # 3. Identify Scanning Window (Today 9:30 AM to Now)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et < market_open:
        print(f"⚠️ Market hasn't opened yet today. Checking yesterday's tail end for demo.")
        market_open = (now_et - timedelta(days=1)).replace(hour=9, minute=30)
    
    # Filter LTF to only include today's bars during market hours
    market_open_utc = market_open.astimezone(timezone.utc)
    now_utc = end_time
    
    # Iterate through every 1-minute bar that matches the timeframe
    scan_bars = ltf_data[ltf_data.index >= market_open_utc]
    
    print(f"Scanning {len(scan_bars)} 1-minute bars for signal triggers...")
    print("-" * 80)
    
    signals_found = 0
    last_signal_time = None
    
    for timestamp, bar in scan_bars.iterrows():
        # Get HTF bias at this specific time
        # Find the HTF bar that JUST CLOSED before or at this timestamp
        htf_closed_bars = htf_data[htf_data.index <= timestamp]
        if len(htf_closed_bars) < 50:
            continue
            
        last_htf = htf_closed_bars.iloc[-1]
        bias = "BULLISH" if last_htf['close'] > last_htf['ema50'] else "BEARISH"
        
        # Check Signal Conditions
        is_bull_trigger = bar['is_bull_ob_candle']
        is_bear_trigger = bar['is_bear_ob_candle']
        
        signal = None
        if bias == "BULLISH" and is_bull_trigger:
            signal = "BUY"
        elif bias == "BEARISH" and is_bear_trigger:
            signal = "SELL"
            
        if signal:
            # Deduplicate: Only report if at least 5 mins apart from same signal
            if last_signal_time and (timestamp - last_signal_time).total_seconds() < 300:
                continue
                
            last_signal_time = timestamp
            signals_found += 1
            time_et = timestamp.astimezone(ET).strftime("%I:%M %p")
            
            print(f"🚨 {signal} SIGNAL at {time_et}")
            print(f"   Price: ${bar['close']:.2f}")
            print(f"   HTF Bias: {bias} (EMA50: ${last_htf['ema50']:.2f})")
            print(f"   LTF Logic: {'Bullish OB Impulse' if signal == 'BUY' else 'Bearish OB Impulse'}")
            print(f"   OB Candle Impulse Size: {bar.get('impulse', False)}")
            print("-" * 40)
            
    if signals_found == 0:
        print("\n❌ No signals detected today using the core strategy.")
        print("   Checking current status for explanation...")
        
        # Get latest
        final_ltf = scan_bars.iloc[-1]
        final_htf = htf_data[htf_data.index <= scan_bars.index[-1]].iloc[-1]
        final_bias = "BULLISH" if final_htf['close'] > final_htf['ema50'] else "BEARISH"
        
        print(f"\n   Current Context ({scan_bars.index[-1].astimezone(ET).strftime('%I:%M %p')}):")
        print(f"   HTF: {final_bias} (Price: ${final_htf['close']:.2f}, EMA50: ${final_htf['ema50']:.2f})")
        print(f"   LTF Impulse: {final_ltf.get('impulse', False)}")
    else:
        print(f"\n✅ Total signals found: {signals_found}")

    print("\n" + "="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Exhaustive daily scan for strategy signals")
    parser.add_argument("symbol", nargs="?", default="SPY", help="Symbol to analyze (default: SPY)")
    
    args = parser.parse_args()
    
    analyze_today_signals(args.symbol)
