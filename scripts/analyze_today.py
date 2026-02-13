import pandas as pd
import sys
import os
import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config
from app.bot import (
    get_bars, get_strategy_signal, detect_fvg, detect_order_block,
    get_confidence_label, send_discord_notification
)
from app.ibkr_manager import ibkr_mgr
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

async def fetch_data_ibkr(symbol):
    """Fetch HTF/LTF via the same IBKR path used by app.bot."""
    htf_data = await get_bars(symbol, config.TIMEFRAME_HTF, 2000)
    ltf_data = await get_bars(symbol, config.TIMEFRAME_LTF, 5000)
    if htf_data.empty or ltf_data.empty:
        return None, None
    return htf_data, ltf_data


def fetch_data_alpaca(symbol, start_time, end_time):
    """Fallback historical fetch using Alpaca."""
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    try:
        htf_req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time,
            feed="iex"
        )
        ltf_req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time,
            feed="iex"
        )
        htf_data = client.get_stock_bars(htf_req).df
        ltf_data = client.get_stock_bars(ltf_req).df
        if htf_data.empty or ltf_data.empty:
            return None, None
        return htf_data.reset_index(), ltf_data.reset_index()
    except Exception:
        return None, None

def process_bars(htf_df, ltf_df):
    """
    Pre-calculate all indicators on the full history to ensure stability.
    """
    htf = htf_df.reset_index() if 'timestamp' not in htf_df.columns else htf_df.copy()
    ltf = ltf_df.reset_index() if 'timestamp' not in ltf_df.columns else ltf_df.copy()
    htf.set_index('timestamp', inplace=True)
    ltf.set_index('timestamp', inplace=True)
    htf.sort_index(inplace=True)
    ltf.sort_index(inplace=True)
    
    # Calculate indicators ONCE on the full history
    if 'ema50' not in htf.columns:
        htf['ema50'] = htf['close'].ewm(span=50, adjust=False).mean()
    
    # LTF indicators
    ltf = detect_fvg(ltf)
    ltf = detect_order_block(ltf)
    
    return htf, ltf

def check_for_signal(timestamp, ltf_row, htf_data, ltf_data, ET, reported_signals, symbol, is_live=False, min_conf_val=0):
    """Checks if a signal exists at a specific timestamp and prints if new."""
    # To avoid look-ahead bias, HTF bias must be based on the last bar that CLOSED before 'timestamp'.
    # In 15Min TF, at 10:00:00 AM, the 9:45:00 AM bar has just closed.
    htf_causal = htf_data[htf_data.index <= (timestamp - timedelta(minutes=15))]
    if len(htf_causal) < 50:
        return False
        
    # Slicing the pre-processed dataframes is now safe because markers are already calculated.
    ltf_slice = ltf_data[ltf_data.index <= timestamp].iloc[-200:]
    
    res = get_strategy_signal(htf_causal, ltf_slice)
    if not res: return False
    
    signal_raw, confidence = res if isinstance(res, tuple) else (res, 0)
    if signal_raw is None: return False
    
    # Check Confidence Threshold
    if confidence < min_conf_val:
        return False
        
    signal = signal_raw.upper()
    
    # Avoid duplicate printing within 5 minutes of same signal
    sig_id = f"{signal}_{timestamp.date()}"
    last_sig_time = reported_signals.get(sig_id)
    
    if last_sig_time and (timestamp - last_sig_time).total_seconds() < 300:
        return False
        
    reported_signals[sig_id] = timestamp
    time_et = timestamp.astimezone(ET).strftime("%I:%M %p")
    
    label = get_confidence_label(confidence)
    last_htf = htf_causal.iloc[-1]
    bias = "BULLISH" if last_htf['close'] > last_htf['ema50'] else "BEARISH"
    
    print(f"🚨 {signal} SIGNAL at {time_et} | Confidence: {confidence}% [{label}]")
    print(f"   Price: ${ltf_row['close']:.2f}")
    print(f"   HTF Bias: {bias} (EMA50: ${last_htf['ema50']:.2f})")
    print(f"   LTF Logic: {'Bullish OB Impulse' if signal == 'BUY' else 'Bearish OB Impulse'}")
    print("-" * 40)
    
    if is_live:
        send_discord_notification(signal, ltf_row['close'], time_et, symbol, bias, confidence)
        
    return True

async def analyze_today_signals(symbol="SPY", min_conf_str="all", data_source="ibkr"):
    ET = ZoneInfo("US/Eastern")
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
    
    htf_raw, ltf_raw = (None, None)
    if data_source == "ibkr":
        htf_raw, ltf_raw = await fetch_data_ibkr(symbol)
        if htf_raw is None:
            print("⚠️ IBKR data fetch failed. Falling back to Alpaca.")
            data_source = "alpaca"
    if data_source == "alpaca":
        htf_raw, ltf_raw = fetch_data_alpaca(symbol, lookback_start_utc, end_time_utc)
    if htf_raw is None:
        print("❌ CRITICAL: Could not fetch market data from IBKR or Alpaca.")
        return
    
    htf_processed, ltf_processed = process_bars(htf_raw, ltf_raw)
    
    # Perform Scan
    scan_bars = ltf_processed[ltf_processed.index >= scan_start_utc]
    
    signals_count = 0
    for ts, row in scan_bars.iterrows():
        if check_for_signal(ts, row, htf_processed, ltf_processed, ET, reported_signals, symbol, is_live=False, min_conf_val=min_conf_val):
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
                await asyncio.sleep(sleep_seconds)
                
                if data_source == "ibkr":
                    h_live, l_live = await fetch_data_ibkr(symbol)
                    if h_live is None:
                        print("⚠️ IBKR live fetch failed. Switching to Alpaca fallback.")
                        data_source = "alpaca"
                else:
                    h_live, l_live = (None, None)

                if data_source == "alpaca" and h_live is None:
                    end_live = datetime.now(timezone.utc)
                    start_live = end_live - timedelta(days=14)
                    h_live, l_live = fetch_data_alpaca(symbol, start_live, end_live)
                if h_live is None: continue
                
                h_proc, l_proc = process_bars(h_live, l_live)
                
                # Check the last 3 bars just in case of slight polling delay
                latest_bars = l_proc.iloc[-3:]
                for ts, row in latest_bars.iterrows():
                    check_for_signal(ts, row, h_proc, l_proc, ET, reported_signals, symbol, is_live=True, min_conf_val=min_conf_val)
                    
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
    async def _run():
        connected = await ibkr_mgr.connect()
        source = "ibkr" if connected else "alpaca"
        if not connected:
            print("⚠️ Failed to connect to IBKR. Using Alpaca fallback.")
        try:
            await analyze_today_signals(args.symbol, args.min_conf, data_source=source)
        finally:
            if connected:
                ibkr_mgr.disconnect()
    asyncio.run(_run())
