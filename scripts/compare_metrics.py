import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np

def analyze_ticker(symbol, days=180):
    print(f"Analyzing {symbol}...")
    df = yf.download(symbol, period=f"{days}d", interval="15m")
    if df.empty:
        return None
    
    # Calculate Metrics
    df['ema50'] = ta.ema(df['Close'], length=50)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['adx'] = adx['ADX_14']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Metrics Analysis
    avg_adx = df['adx'].mean()
    pct_above_25 = (df['adx'] > 25).mean() * 100
    avg_atr_pct = (df['atr'] / df['Close']).mean() * 100
    
    # Trend "Cleanliness": How often is price above/below EMA for sustained periods?
    df['side'] = np.where(df['Close'] > df['ema50'], 1, -1)
    df['flips'] = (df['side'] != df['side'].shift(1)).astype(int)
    flips_per_100 = df['flips'].sum() / len(df) * 100
    
    return {
        "symbol": symbol,
        "avg_adx": avg_adx,
        "trend_consistency_pct": 100 - flips_per_100, # Higher is better (staying on one side of EMA)
        "pct_strong_trend": pct_above_25,
        "volatility_index (ATR%)": avg_atr_pct
    }

spy_stats = analyze_ticker("SPY")
qqq_stats = analyze_ticker("QQQ")

print("\n--- COMPARATIVE ANALYSIS (180 DAYS) ---")
print(pd.DataFrame([spy_stats, qqq_stats]))
