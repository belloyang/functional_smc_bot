import pandas as pd
import sys
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config
from app.bot import (
    get_strategy_signal,
    get_confidence_label,
    send_discord_notification
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def fetch_data(client, symbol, start_time, end_time):
    """Utility to fetch HTF and LTF data with feed fallback."""
    feeds = ["sip", "iex"]
    for feed in feeds:
        try:
            htf_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time,
                feed=feed
            )
            htf_data = client.get_stock_bars(htf_req).df
            
            ltf_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time,
                feed=feed
            )
            ltf_data = client.get_stock_bars(ltf_req).df
            
            if not htf_data.empty and not ltf_data.empty:
                return htf_data, ltf_data, feed
        except Exception:
            continue
    return None, None, None

def process_bars(htf_df, ltf_df):
    """
    Reset index for processing (matching live bot behavior).
    """
    htf_df = htf_df.reset_index()
    ltf_df = ltf_df.reset_index()
    return htf_df, ltf_df

def check_for_signal(timestamp, ltf_row, htf_data, ltf_data, ET, reported_signals, symbol, is_live=False, min_conf_val=0):
    """Checks if a signal exists at a specific timestamp and prints if new."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    # Slice data up to current timestamp (matching live bot behavior)
    htf_slice = htf_data[htf_data['timestamp'] <= ts]
    ltf_slice = ltf_data[ltf_data['timestamp'] <= ts]
    
    if len(htf_slice) < 50 or len(ltf_slice) < 5:
        return False
    
    # Use the same signal generation as live bot
    res = get_strategy_signal(htf_slice, ltf_slice)
    if not res:
        return False

    signal_raw, confidence = res if isinstance(res, tuple) else (res, 0)
    if signal_raw is None:
        return False

    # For logging: get HTF bias from the sliced data
    htf_causal = htf_slice
    if len(htf_causal) < 50:
        return False
    
    # Check Confidence Threshold
    if confidence < min_conf_val:
        return False
        
    signal = signal_raw.upper()
    
    # Avoid duplicate printing within 5 minutes of same signal
    sig_id = f"{signal}_{ts.date()}"
    last_sig_time = reported_signals.get(sig_id)
    
    if last_sig_time and (ts - last_sig_time).total_seconds() < 300:
        return False
        
    reported_signals[sig_id] = ts
    time_et = ts.tz_convert(ET).strftime("%I:%M %p")
    
    label = get_confidence_label(confidence)
    
    # Calculate bias from HTF data (get_strategy_signal already calculated EMA internally)
    # We'll determine bias from the signal itself
    bias = "BULLISH" if signal == "BUY" else "BEARISH"
    
    print(f"🚨 {signal} SIGNAL at {time_et} | Confidence: {confidence}% [{label}]")
    print(f"   Price: ${ltf_row['close']:.2f}")
    print(f"   HTF Bias: {bias}")
    print(f"   LTF Logic: {'Bullish OB Impulse' if signal == 'BUY' else 'Bearish OB Impulse'}")
    print("-" * 40)
    
    if is_live:
        send_discord_notification(signal, ltf_row['close'], time_et, symbol, bias, confidence)
        
    return True

def analyze_today_signals(symbol="SPY", min_conf_str="all"):
    ET = ZoneInfo("US/Eastern")
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    reported_signals = {} # To track and dedup signals
    
    # Map confidence choices to numeric thresholds
    CONF_THRESHOLDS = {
        'all': 0,
        'low': 20,
        'medium': 50,
        'high': 80
    }
    min_conf_val = CONF_THRESHOLDS.get(min_conf_str, 0)
    
    now_et = datetime.now(ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_live = (market_open <= now_et <= market_close)
    
    if is_live:
        print(f"🚀 [LIVE MODE] Monitoring {symbol} until market close ({market_close.strftime('%I:%M %p ET')})")
    else:
        print(f"📊 [HISTORICAL MODE] Analyzing {symbol} for today ({now_et.strftime('%Y-%m-%d')})")
    print("="*80)

    # 1. Initial Scan (From Open to Now)
    scan_start = market_open if now_et >= market_open else (now_et - timedelta(days=1)).replace(hour=9, minute=30)
    scan_start_utc = scan_start.astimezone(timezone.utc)
    
    print(f"Fetching data and scanning established signals since market open...")
    
    end_time_utc = datetime.now(timezone.utc)
    lookback_start_utc = end_time_utc - timedelta(days=14) # 14 days for stable EMA50
    
    htf_raw, ltf_raw, feed = fetch_data(client, symbol, lookback_start_utc, end_time_utc)
    if htf_raw is None:
        print("❌ CRITICAL: Could not fetch market data. Check API keys/connection.")
        return
    
    htf_processed, ltf_processed = process_bars(htf_raw, ltf_raw)
    
    # Perform Scan
    scan_bars = ltf_processed[ltf_processed['timestamp'] >= scan_start_utc]
    
    signals_count = 0
    for _, row in scan_bars.iterrows():
        if check_for_signal(row['timestamp'], row, htf_processed, ltf_processed, ET, reported_signals, symbol, is_live=False, min_conf_val=min_conf_val):
            signals_count += 1
            
    if signals_count == 0:
        print("No historical signals detected today so far.")
    else:
        print(f"✅ Found {signals_count} historical signals today.")

    # 2. Live Monitoring Loop
    if is_live:
        print("\n" + "-"*80)
        print(f"📡 Now monitoring {symbol} LIVE. Alerts will print as candles close.")
        print("-" * 80)
        
        try:
            while datetime.now(ET) <= market_close:
                # Wait for next minute candle to close (plus small buffer)
                now = datetime.now()
                sleep_seconds = 61 - now.second
                # print(f"DEBUG: Sleeping {sleep_seconds}s for next candle...")
                time.sleep(sleep_seconds)
                
                # Fetch latest
                end_live = datetime.now(timezone.utc)
                start_live = end_live - timedelta(days=2)
                
                h_live, l_live, _ = fetch_data(client, symbol, start_live, end_live)
                if h_live is None: continue
                
                h_proc, l_proc = process_bars(h_live, l_live)
                
                # Check the last 3 bars just in case of slight polling delay
                latest_bars = l_proc.tail(3)
                for _, row in latest_bars.iterrows():
                    check_for_signal(row['timestamp'], row, h_proc, l_proc, ET, reported_signals, symbol, is_live=True, min_conf_val=min_conf_val)
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")
            return

    print("\n" + "="*80)
    print("Analysis complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live or Historical signal analysis for SPY/SLV")
    parser.add_argument("symbol", nargs="?", default="SPY", help="Symbol to analyze")
    parser.add_argument("--min-conf", type=str, choices=['all', 'low', 'medium', 'high'], default='all', help="Minimum confidence level to display/alert (default: all)")
    args = parser.parse_args()
    analyze_today_signals(args.symbol, args.min_conf)
