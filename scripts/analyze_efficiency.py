import pandas as pd
import numpy as np
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas_ta as ta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
client = StockHistoricalDataClient(API_KEY, API_SECRET)

def detect_fvg(df):
    df = df.copy()
    df['bull_fvg'] = False
    df['bear_fvg'] = False
    df['fvg_top'] = np.nan
    df['fvg_bot'] = np.nan
    
    for i in range(2, len(df)):
        # Bullish FVG: Low of candle i > High of candle i-2
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            gap = df['low'].iloc[i] - df['high'].iloc[i-2]
            if gap > 0:
                df.at[df.index[i-1], 'bull_fvg'] = True
                df.at[df.index[i-1], 'fvg_top'] = df['low'].iloc[i]
                df.at[df.index[i-1], 'fvg_bot'] = df['high'].iloc[i-2]
                
        # Bearish FVG: High of candle i < Low of candle i-2
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            gap = df['low'].iloc[i-2] - df['high'].iloc[i]
            if gap > 0:
                df.at[df.index[i-1], 'bear_fvg'] = True
                df.at[df.index[i-1], 'fvg_top'] = df['low'].iloc[i-2]
                df.at[df.index[i-1], 'fvg_bot'] = df['high'].iloc[i]
    return df

def analyze_trade_efficiency(symbol, days=365):
    print(f"Analyzing efficiency for {symbol}...")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days + 10)
    
    req_1m = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute), start=start_dt)
    df = client.get_stock_bars(req_1m).df.reset_index()
    df = detect_fvg(df)
    
    # Simulate a simplified "Entry -> Target/Stop" result
    # Target = +1% of ATR, Stop = -1% of ATR (simplified proxy)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df = df.dropna(subset=['atr'])
    
    outcomes = []
    
    fvg_indices = df.index[df['bull_fvg'] == True].tolist()
    for idx in fvg_indices:
        entry_price = df.at[idx, 'fvg_top']
        atr = df.at[idx, 'atr']
        
        # Look ahead up to 240 minutes (4 hours)
        target = entry_price + (1.5 * atr)
        stop = entry_price - (1.5 * atr)
        
        reached_target = False
        reached_stop = False
        duration = 0
        
        for k in range(idx + 1, min(idx + 241, len(df))):
            duration += 1
            if df['high'].iloc[k] >= target:
                reached_target = True
                break
            if df['low'].iloc[k] <= stop:
                reached_stop = True
                break
        
        if reached_target:
            outcomes.append({"result": "win", "duration": duration})
        elif reached_stop:
            outcomes.append({"result": "loss", "duration": duration})
        else:
            outcomes.append({"result": "stale", "duration": duration})

    res_df = pd.DataFrame(outcomes)
    win_rate = (res_df['result'] == 'win').mean() * 100
    avg_win_dur = res_df[res_df['result'] == 'win']['duration'].mean()
    avg_loss_dur = res_df[res_df['result'] == 'loss']['duration'].mean()
    
    return {
        "Ticker": symbol,
        "FVG Success Rate (%)": round(win_rate, 2),
        "Avg Win Dur (min)": round(avg_win_dur, 1),
        "Avg Loss Dur (min)": round(avg_loss_dur, 1),
        "Efficiency (Win/Dur)": round(win_rate / avg_win_dur, 4)
    }

spy_eff = analyze_trade_efficiency("SPY")
qqq_eff = analyze_trade_efficiency("QQQ")

print("\n--- TRADE EFFICIENCY COMPARISON ---")
print(pd.DataFrame([spy_eff, qqq_eff]))
