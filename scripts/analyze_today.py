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
    Detailed analysis of signal conditions for today's trading session.
    """
    ET = ZoneInfo("US/Eastern")
    today = datetime.now(ET)
    today_str = today.strftime("%Y-%m-%d")
    
    print(f"Analyzing {symbol} signal conditions for today ({today_str})...")
    print("="*80)
    
    # 1. Fetch Data
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=5)  # Get a few days for context
    
    # HTF Data (15 Min)
    print("Fetching HTF (15min) data...")
    htf_req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start_time,
        end=end_time,
        feed="iex"
    )
    htf_data = client.get_stock_bars(htf_req).df
    
    # LTF Data (1 Min)
    print("Fetching LTF (1min) data...")
    ltf_req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start_time,
        end=end_time,
        feed="iex"
    )
    ltf_data = client.get_stock_bars(ltf_req).df
    
    # Process data
    htf_data = htf_data.reset_index()
    ltf_data = ltf_data.reset_index()
    htf_data.set_index('timestamp', inplace=True)
    ltf_data.set_index('timestamp', inplace=True)
    htf_data.sort_index(inplace=True)
    ltf_data.sort_index(inplace=True)
    
    print(f"Data loaded. HTF: {len(htf_data)} bars, LTF: {len(ltf_data)} bars.")
    print("="*80)
    
    # Get today's date
    today_date = today.date()
    current_time = today
    
    # Analyze every 15 minutes during market hours
    # Start from market open (9:30 AM ET)
    market_open = today.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If current time is before market open, analyze from market open to now
    # If current time is after market open, start from market open
    if current_time < market_open:
        print(f"\n⚠️  Market hasn't opened yet. Current time: {current_time.strftime('%I:%M %p ET')}")
        print(f"Market opens at: {market_open.strftime('%I:%M %p ET')}")
        return
    
    analysis_start = market_open
    
    print(f"\nAnalyzing signal conditions from {analysis_start.strftime('%I:%M %p ET')} to {current_time.strftime('%I:%M %p ET')}")
    print("="*80)
    
    # Sample key times
    check_times = []
    check_time = analysis_start
    while check_time <= current_time:
        check_times.append(check_time)
        check_time += timedelta(minutes=15)
    
    for check_time in check_times:
        check_time_utc = check_time.astimezone(timezone.utc)
        
        # Get data up to this point
        htf_slice = htf_data[htf_data.index <= check_time_utc]
        ltf_slice = ltf_data[ltf_data.index <= check_time_utc]
        
        if len(htf_slice) < 50 or len(ltf_slice) < 5:
            continue
        
        # Limit to last 200 bars for performance
        htf_slice = htf_slice.iloc[-200:]
        ltf_slice = ltf_slice.iloc[-200:]
        
        # Calculate HTF bias
        htf_analysis = htf_slice.copy()
        htf_analysis['ema50'] = ta.ema(htf_analysis['close'], length=50)
        
        last_htf_close = htf_analysis['close'].iloc[-1]
        last_htf_ema50 = htf_analysis['ema50'].iloc[-1]
        htf_bias = "BULLISH" if last_htf_close > last_htf_ema50 else "BEARISH"
        
        # Calculate LTF indicators
        ltf_analysis = ltf_slice.copy()
        ltf_analysis = detect_fvg(ltf_analysis)
        ltf_analysis = detect_order_block(ltf_analysis)
        
        # Check last candle
        last_ltf = ltf_analysis.iloc[-1]
        
        # Get signal
        signal = get_strategy_signal(htf_slice, ltf_slice)
        
        # Detailed output
        print(f"\n⏰ Time: {check_time.strftime('%I:%M %p ET')}")
        print(f"   SPY Price: ${last_ltf['close']:.2f}")
        print(f"   HTF Bias: {htf_bias} (Price: ${last_htf_close:.2f}, EMA50: ${last_htf_ema50:.2f})")
        
        # Check for order blocks in recent candles
        recent_ltf = ltf_analysis.iloc[-10:]
        bull_ob_count = recent_ltf['is_bull_ob_candle'].sum()
        bear_ob_count = recent_ltf['is_bear_ob_candle'].sum()
        
        print(f"   LTF Last Candle:")
        print(f"      - Is Bullish OB Candle: {last_ltf['is_bull_ob_candle']}")
        print(f"      - Is Bearish OB Candle: {last_ltf['is_bear_ob_candle']}")
        print(f"      - Impulse: {last_ltf.get('impulse', False)}")
        print(f"   Recent 10 candles: {bull_ob_count} bull OB, {bear_ob_count} bear OB")
        
        if signal:
            print(f"   🚨 SIGNAL: {signal.upper()}")
        else:
            print(f"   ❌ No Signal")
            
            # Explain why no signal
            if htf_bias == "BULLISH" and not last_ltf['is_bull_ob_candle']:
                print(f"      Reason: HTF is bullish but no bullish order block detected on LTF")
            elif htf_bias == "BEARISH" and not last_ltf['is_bear_ob_candle']:
                print(f"      Reason: HTF is bearish but no bearish order block detected on LTF")
            elif htf_bias == "BULLISH":
                print(f"      Reason: HTF is bullish but waiting for bullish order block")
            elif htf_bias == "BEARISH":
                print(f"      Reason: HTF is bearish but waiting for bearish order block")
    
    print("\n" + "="*80)
    print("\n📋 STRATEGY REQUIREMENTS FOR BUY SIGNAL:")
    print("   1. HTF (15min) must be BULLISH: Price > EMA50")
    print("   2. LTF (1min) must show Bullish Order Block:")
    print("      - Previous candle was bearish (close < open)")
    print("      - Current candle is bullish (close > open)")
    print("      - Current candle is an 'impulse' (body > 1.5x avg body)")
    print("      - Current close > previous high (structure break)")
    print("\n📋 STRATEGY REQUIREMENTS FOR SELL SIGNAL:")
    print("   1. HTF (15min) must be BEARISH: Price < EMA50")
    print("   2. LTF (1min) must show Bearish Order Block:")
    print("      - Previous candle was bullish (close > open)")
    print("      - Current candle is bearish (close < open)")
    print("      - Current candle is an 'impulse' (body > 1.5x avg body)")
    print("      - Current close < previous low (structure break)")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze today's signal conditions for a given symbol")
    parser.add_argument("symbol", nargs="?", default="SPY", help="Symbol to analyze (default: SPY)")
    
    args = parser.parse_args()
    
    analyze_today_signals(args.symbol)
