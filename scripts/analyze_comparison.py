import pandas as pd
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas_ta as ta
import numpy as np

# Load keys
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

def get_stats(symbol, days=180):
    print(f"Fetching data for {symbol}...")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days + 15) # Buffer for indicators
    
    # 15m Bars (HTF)
    req_15m = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(15, TimeFrameUnit.Minute), start=start_dt)
    bars_15m = client.get_stock_bars(req_15m).df.reset_index()
    
    # Calculate indicators (same as bot.py)
    bars_15m['ema50'] = ta.ema(bars_15m['close'], length=50)
    adx_df = ta.adx(bars_15m['high'], bars_15m['low'], bars_15m['close'], length=14)
    bars_15m['adx'] = adx_df['ADX_14']
    bars_15m['atr'] = ta.atr(bars_15m['high'], bars_15m['low'], bars_15m['close'], length=14)
    
    # Filter to requested timeframe
    df = bars_15m[bars_15m['timestamp'] >= pd.Timestamp(end_dt - timedelta(days=days), tz='UTC')].copy()
    
    # Metrics
    avg_adx = df['adx'].mean()
    pct_adx_25 = (df['adx'] > 25).mean() * 100
    avg_atr_pct = (df['atr'] / df['close']).mean() * 100
    
    # Trend "Cleanliness" (Side of EMA)
    df['side'] = np.where(df['close'] > df['ema50'], 1, -1)
    df['flips'] = (df['side'] != df['side'].shift(1)).astype(int)
    flips_per_day = df['flips'].sum() / days
    
    return {
        "Ticker": symbol,
        "Avg ADX": round(avg_adx, 2),
        "% ADX > 25 (Strong Trend)": round(pct_adx_25, 1),
        "Avg ATR %": round(avg_atr_pct, 4),
        "EMA Flips per Day": round(flips_per_day, 2),
        "Samples": len(df)
    }

tickers = ["QQQ", "SPY", "NVDA", "TSLA", "SMH", "XLK", "AMD", "MSFT"]
stats = [get_stats(ticker, days=365) for ticker in tickers]

print("\n--- MULTI-ASSET PERFORMANCE DYNAMICS (365 DAYS) ---")
results = pd.DataFrame(stats)
print(results.sort_values(by="% ADX > 25 (Strong Trend)", ascending=False))

# Look for best assets
df_results = pd.DataFrame(stats)
print(df_results.sort_values(by="% ADX > 25 (Strong Trend)", ascending=False).to_string())
diff_flips = spy['EMA Flips per Day'] - qqq['EMA Flips per Day']

print("\n--- KEY FINDINGS ---")
print(f"1. Trend Strength: QQQ is in a strong trend (ADX > 25) {diff_adx:.1f}% more often than SPY.")
print(f"2. Trend Cleanliness: SPY flips sides of the EMA50 {spy['EMA Flips per Day']} times per day vs QQQ {qqq['EMA Flips per Day']} times.")
print(f"3. Relative Volatility: QQQ is {qqq['Avg ATR %']/spy['Avg ATR %']:.1f}x more volatile (ATR%) than SPY.")
