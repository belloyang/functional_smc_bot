import pandas as pd
import numpy as np
import pandas_ta as ta
import sys
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config
from app.bot import get_strategy_signal, detect_fvg, detect_order_block
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
    """Pre-calculate indicators on dataframes."""
    htf = htf_df.reset_index()
    ltf = ltf_df.reset_index()
    htf.set_index('timestamp', inplace=True)
    ltf.set_index('timestamp', inplace=True)
    htf.sort_index(inplace=True)
    ltf.sort_index(inplace=True)
    
    htf['ema50'] = ta.ema(htf['close'], length=50)
    ltf = detect_fvg(ltf)
    ltf = detect_order_block(ltf)
    return htf, ltf

def send_discord_notification(signal, price, time_str, symbol, bias):
    """Sends a signal alert to Discord via Webhook."""
    if not config.DISCORD_WEBHOOK_URL:
        return
        
    color = 0x2ca02c if signal == "BUY" else 0xd62728 # Green for Buy, Red for Sell
    
    payload = {
        "embeds": [{
            "title": f"🚨 {signal} SIGNAL Detected!",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": f"**{symbol}**", "inline": True},
                {"name": "Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Time (ET)", "value": time_str, "inline": True},
                {"name": "Bias", "value": bias, "inline": True},
                {"name": "Strategy", "value": "SMC Order Block + Impulse", "inline": False}
            ],
            "footer": {"text": "Alpaca Live Monitor"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    
    try:
        response = requests.post(config.DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️ Failed to send Discord notification: {e}")

def check_for_signal(timestamp, ltf_row, htf_data, ET, reported_signals, symbol, is_live=False):
    """Checks if a signal exists at a specific timestamp and prints if new."""
    # Find the HTF bar that JUST CLOSED before or at this timestamp
    htf_closed_bars = htf_data[htf_data.index <= timestamp]
    if len(htf_closed_bars) < 50:
        return False
        
    last_htf = htf_closed_bars.iloc[-1]
    bias = "BULLISH" if last_htf['close'] > last_htf['ema50'] else "BEARISH"
    
    signal = None
    if bias == "BULLISH" and ltf_row['is_bull_ob_candle']:
        signal = "BUY"
    elif bias == "BEARISH" and ltf_row['is_bear_ob_candle']:
        signal = "SELL"
        
    if signal:
        # Avoid duplicate printing within 5 minutes of same signal
        sig_id = f"{signal}_{timestamp.date()}"
        last_sig_time = reported_signals.get(sig_id)
        
        if last_sig_time and (timestamp - last_sig_time).total_seconds() < 300:
            return False
            
        reported_signals[sig_id] = timestamp
        time_et = timestamp.astimezone(ET).strftime("%I:%M %p")
        
        print(f"🚨 {signal} SIGNAL at {time_et}")
        print(f"   Price: ${ltf_row['close']:.2f}")
        print(f"   HTF Bias: {bias} (EMA50: ${last_htf['ema50']:.2f})")
        print(f"   LTF Logic: {'Bullish OB Impulse' if signal == 'BUY' else 'Bearish OB Impulse'}")
        print("-" * 40)
        
        if is_live:
            send_discord_notification(signal, ltf_row['close'], time_et, symbol, bias)
            
        return True
    return False

def analyze_today_signals(symbol="SPY"):
    ET = ZoneInfo("US/Eastern")
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    reported_signals = {} # To track and dedup signals
    
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
    lookback_start_utc = end_time_utc - timedelta(days=7) # Enough history for EMA50 (needs 50 bars)
    
    htf_raw, ltf_raw, feed = fetch_data(client, symbol, lookback_start_utc, end_time_utc)
    if htf_raw is None:
        print("❌ CRITICAL: Could not fetch market data. Check API keys/connection.")
        return
    
    htf_processed, ltf_processed = process_bars(htf_raw, ltf_raw)
    
    # Perform Scan
    scan_bars = ltf_processed[ltf_processed.index >= scan_start_utc]
    
    signals_count = 0
    for ts, row in scan_bars.iterrows():
        if check_for_signal(ts, row, htf_processed, ET, reported_signals, symbol, is_live=False):
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
                latest_bars = l_proc.iloc[-3:]
                for ts, row in latest_bars.iterrows():
                    check_for_signal(ts, row, h_proc, ET, reported_signals, symbol, is_live=True)
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")
            return

    print("\n" + "="*80)
    print("Analysis complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live or Historical signal analysis for SPY/SLV")
    parser.add_argument("symbol", nargs="?", default="SPY", help="Symbol to analyze")
    args = parser.parse_args()
    analyze_today_signals(args.symbol)
