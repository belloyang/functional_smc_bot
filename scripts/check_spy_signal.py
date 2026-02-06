import asyncio
import pandas as pd
import pandas_ta as ta
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ibkr_manager import ibkr_mgr
from app import config
from ib_insync import Stock, util

# Mock functions from bot.py
def detect_fvg(df):
    df['bull_fvg'] = (df['low'] > df['high'].shift(2))
    df['bear_fvg'] = (df['high'] < df['low'].shift(2))
    return df

def detect_order_block(df):
    body_size = abs(df['close'] - df['open'])
    avg_body = body_size.rolling(20).mean()
    df['impulse'] = (body_size > (avg_body * 1.5))
    
    df['is_bull_ob_candle'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                              (df['close'] > df['open']) & \
                              df['impulse'] & \
                              (df['close'] > df['high'].shift(1)) 
                              
    df['is_bear_ob_candle'] = (df['close'].shift(1) > df['open'].shift(1)) & \
                              (df['close'] < df['open']) & \
                              df['impulse'] & \
                              (df['close'] < df['low'].shift(1))
    return df

async def get_bars_debug(symbol, timeframe, limit=200):
    bar_size = '1 min'
    if '15Min' in timeframe:
        bar_size = '15 mins'
    elif '1Day' in timeframe:
        bar_size = '1 day'

    contract = Stock(symbol, 'SMART', 'USD')
    bars = await ibkr_mgr.ib.reqHistoricalDataAsync(
        contract,
        endDateTime='',
        durationStr='1 W',
        barSizeSetting=bar_size,
        whatToShow='MIDPOINT',
        useRTH=True,
        formatDate=1,
        keepUpToDate=False
    )
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    df = df.rename(columns={'date': 'timestamp'})
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df

async def analyze():
    print(f"Connecting to IBKR (ClientId 99) to analyze {config.SYMBOL}...")
    config.IBKR_CLIENT_ID = 99
    connected = await ibkr_mgr.connect()
    if not connected:
        print("Failed to connect.")
        return

    symbol = config.SYMBOL
    print(f"Fetching {config.TIMEFRAME_HTF} and {config.TIMEFRAME_LTF} bars...")
    htf_df = await get_bars_debug(symbol, config.TIMEFRAME_HTF)
    ltf_df = await get_bars_debug(symbol, config.TIMEFRAME_LTF)

    if htf_df.empty:
        print("HTF Data empty.")
        ibkr_mgr.disconnect()
        return
    if ltf_df.empty:
        print("LTF Data empty.")
        ibkr_mgr.disconnect()
        return

    print(f"Received {len(htf_df)} HTF bars and {len(ltf_df)} LTF bars.")

    print(f"\nHTF Analysis ({config.TIMEFRAME_HTF}):")
    ema_series = ta.ema(htf_df['close'], length=50)
    if ema_series is None or ema_series.empty:
        print("   EMA50 calculation failed (returned None or empty).")
        bias = "neutral"
    else:
        htf_df['ema50'] = ema_series
        last_htf = htf_df.iloc[-1]
        print(f"   Timestamp:  {last_htf['timestamp']}")
        print(f"   Last Close: {last_htf['close']:.2f}")
        
        ema_val = last_htf['ema50']
        if pd.isna(ema_val):
             print("   EMA50:      NaN (not enough bars?)")
             bias = "neutral"
        else:
             print(f"   EMA50:      {ema_val:.2f}")
             bias = "bullish" if last_htf['close'] > ema_val else "bearish"
    
    print(f"   Bias:       {bias.upper()}")

    print(f"\nLTF Analysis ({config.TIMEFRAME_LTF}):")
    ltf_df = detect_fvg(ltf_df)
    ltf_df = detect_order_block(ltf_df)
    
    # Analyze the last 15 bars for OBs
    last_count = 15
    last_bars = ltf_df.tail(last_count)
    print(f"Last {last_count} candles:")
    for idx, row in last_bars.iterrows():
        ob_status = "BULL" if row['is_bull_ob_candle'] else ("BEAR" if row['is_bear_ob_candle'] else "NONE")
        print(f"   [{row['timestamp']}] C: {row['close']:.2f} | Imp: {row['impulse']} | OB: {ob_status}")

    ob_detected = ltf_df[ltf_df['is_bull_ob_candle'] | ltf_df['is_bear_ob_candle']]
    if not ob_detected.empty:
        print("\nAll Order Blocks detected in the loaded data (200 bars):")
        for idx, row in ob_detected.iterrows():
            ob_type = "BULL" if row['is_bull_ob_candle'] else "BEAR"
            print(f"   {row['timestamp']}: {ob_type} OB")
    else:
        print("\nNo Order Blocks detected in the current data.")

    ibkr_mgr.disconnect()

if __name__ == "__main__":
    asyncio.run(analyze())
