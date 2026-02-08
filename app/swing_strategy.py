import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import logging

class SMCStructure:
    """
    Handles Market Structure Analysis (BOS, Swing Points, Premium/Discount).
    Timeframe: Daily (D1) & Weekly (W1)
    """
    def __init__(self):
        pass

    def get_swing_points(self, df: pd.DataFrame, window=3):
        """
        Identifies Swing Highs and Lows.
        Returns two lists: swing_highs, swing_lows.
        Each item is a dict: {index, price, time}
        """
        swing_highs = []
        swing_lows = []
        
        if len(df) < window * 2 + 1:
            return swing_highs, swing_lows

        # Iterate through the DataFrame to find local maxima and minima
        for i in range(window, len(df) - window):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Check for Swing High
            is_high = True
            for j in range(1, window + 1):
                if df['high'].iloc[i-j] > current_high or df['high'].iloc[i+j] > current_high:
                    is_high = False
                    break
            if is_high:
                swing_highs.append({'index': i, 'price': current_high, 'time': df['timestamp'].iloc[i]})

            # Check for Swing Low
            is_low = True
            for j in range(1, window + 1):
                if df['low'].iloc[i-j] < current_low or df['low'].iloc[i+j] < current_low:
                    is_low = False
                    break
            if is_low:
                swing_lows.append({'index': i, 'price': current_low, 'time': df['timestamp'].iloc[i]})

        return swing_highs, swing_lows

    def determine_bias(self, df: pd.DataFrame):
        """
        Determines market bias based on purely structural rules (HH/HL/LL/LH).
        Bullish: Making Higher Highs and Higher Lows.
        Bearish: Making Lower Lows and Lower Highs.
        """
        swing_highs, swing_lows = self.get_swing_points(df, window=5)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "neutral", None, None
            
        last_high = swing_highs[-1]
        prev_high = swing_highs[-2]
        
        last_low = swing_lows[-1]
        prev_low = swing_lows[-2]
        
        # Structure Analysis
        higher_high = last_high['price'] > prev_high['price']
        higher_low = last_low['price'] > prev_low['price']
        
        lower_low = last_low['price'] < prev_low['price']
        lower_high = last_high['price'] < prev_high['price']
        
        current_close = df['close'].iloc[-1]
        
        major_high = last_high
        major_low = last_low
        
        bias = "neutral"
        
        # Bullish Structure: HH + HL + Close > Last HL
        if higher_high and higher_low:
            if current_close > last_low['price']:
                bias = "bullish"
                # For Bullish Bias, the 'Range' is usually the last Swing Low to the new High
                major_low = last_low
                major_high = last_high
        
        # Bearish Structure: LL + LH + Close < Last LH
        elif lower_low and lower_high:
            if current_close < last_high['price']:
                bias = "bearish"
                major_high = last_high
                major_low = last_low
                
        # Trend Continuation Check (EMA Filter as confirmation)
        ema50 = ta.ema(df['close'], length=50)
        if ema50 is not None:
            if bias == "bullish" and current_close < ema50.iloc[-1]:
                # Weak Bullish / Pullback
                 pass
            elif bias == "bearish" and current_close > ema50.iloc[-1]:
                # Weak Bearish / Pullback
                 pass
                 
        return bias, major_high, major_low

    def check_premium_discount(self, price, swing_high, swing_low, bias="bullish"):
        """
        Returns 'premium', 'discount', or 'equilibrium'.
        """
        high_price = swing_high['price']
        low_price = swing_low['price']
        
        if high_price <= low_price:
            return "unknown"
            
        eq = (high_price + low_price) / 2
        
        if price < eq:
            return "discount"
        elif price > eq:
            return "premium"
        return "equilibrium"


class SMCLiquidity:
    """
    Handles Liquidity Detection (Sweeps of Swing Highs/Lows).
    """
    def detect_liquidity_sweep(self, df: pd.DataFrame, bias, window=20):
        """
        Checks if recent price action swept a key level (High/Low) and closed back inside.
        """
        if len(df) < window: return False
        
        # Look at the very recent price action (last 3-5 bars)
        recent_window = 5
        recent_df = df.iloc[-recent_window:]
        
        # Look for existing swing points in the larger window
        # We need to find "old" highs/lows that might have been swept
        # For efficiency, just look for min/max in [window] bars excluding recent_window
        
        historical_df = df.iloc[-(window + recent_window):-recent_window]
        if historical_df.empty: return False
        
        local_min = historical_df['low'].min()
        local_max = historical_df['high'].max()
        
        sweep_detected = False
        
        # Bullish Setup: Sweep of Sell-Side Liquidity (SSL)
        # Price goes below local_min but closes above it (or shows rejection)
        if bias == "bullish":
            # Check if any recent low went below historical low
            swept_low = recent_df['low'].min() < local_min
            # Check if we are now trading back up (current close > local_min is ideal, or just > sweep low)
            # Strict Sweep: Low < Old Low, Close > Old Low
            idx_sweep = recent_df['low'].idxmin()
            candle_sweep = recent_df.loc[idx_sweep]
            
            if swept_low and candle_sweep['close'] > candle_sweep['low']: # Pinbar-ish
                sweep_detected = True
                
        # Bearish Setup: Sweep of Buy-Side Liquidity (BSL)
        elif bias == "bearish":
            swept_high = recent_df['high'].max() > local_max
            idx_sweep = recent_df['high'].idxmax()
            candle_sweep = recent_df.loc[idx_sweep]
            
            if swept_high and candle_sweep['close'] < candle_sweep['high']:
                sweep_detected = True
                
        return sweep_detected

    def find_order_blocks(self, df: pd.DataFrame, bias):
        """
        Identifies valid Order Blocks (OB) near current price.
        """
        # We need the last OB created
        # Bullish OB: Last RED candle before a Green Impulse
        
        if len(df) < 5: return None
        
        # Iterate backwards
        for i in range(len(df)-2, 0, -1):
            curr = df.iloc[i]
            next_c = df.iloc[i+1] # Impulse candidate
            
            if bias == "bullish":
                # Find red candle
                if curr['close'] < curr['open']:
                    # Check next candle is Green and Engulfing/Impulsive
                    if next_c['close'] > next_c['open'] and next_c['close'] > curr['high']:
                        # Found Bullish OB
                        return {'high': curr['high'], 'low': curr['low'], 'type': 'bullish', 'time': curr['timestamp']}
                        
            elif bias == "bearish":
                # Find green candle
                if curr['close'] > curr['open']:
                    # Check next candle is Red and Impulsive
                    if next_c['close'] < next_c['open'] and next_c['close'] < curr['low']:
                        # Found Bearish OB
                        return {'high': curr['high'], 'low': curr['low'], 'type': 'bearish', 'time': curr['timestamp']}
                        
        return None


from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

def get_swing_signal(symbol, data_client: StockHistoricalDataClient):
    """
    Orchestrator for Swing Trading Signal.
    """
    try:
        # 1. Fetch Data (W1, D1, 4H - modeled here as D1 & 1H for data availability in standard plan)
        # Note: W1 can be Resampled from D1. 4H can be Resampled from 15Min or 1H.
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365) # 1 Year of data
        
        # Fetch Daily (D1)
        req_d1 = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Day), start=start_dt, end=end_dt, limit=1000)
        d1_bars = data_client.get_stock_bars(req_d1).df
        if d1_bars.empty: return None
        d1_bars = d1_bars.reset_index()
        
        # Fetch Hourly (H1) for Entry/Liquidity
        start_dt_h1 = end_dt - timedelta(days=60)
        req_h1 = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Hour), start=start_dt_h1, end=end_dt, limit=1000)
        h1_bars = data_client.get_stock_bars(req_h1).df
        if h1_bars.empty: return None
        h1_bars = h1_bars.reset_index()
        
        # 2. Analyze Structure (D1)
        smc = SMCStructure()
        bias_d1, swing_high, swing_low = smc.determine_bias(d1_bars)
        
        print(f"[{symbol}] D1 Bias: {bias_d1.upper()}")
        
        if bias_d1 == "neutral":
             return None
             
        # 3. Premium/Discount Check
        if not swing_high or not swing_low:
             return None
             
        current_price = d1_bars['close'].iloc[-1]
        pd_zone = smc.check_premium_discount(current_price, swing_high, swing_low)
        print(f"[{symbol}] Zone: {pd_zone.upper()} (Range: {swing_low['price']} - {swing_high['price']})")
        
        # Bias-Zone Alignment Rule
        if bias_d1 == "bullish" and pd_zone != "discount":
            print(f"[{symbol}] Skip: Bullish but not in Discount.")
            return None
        if bias_d1 == "bearish" and pd_zone != "premium":
            print(f"[{symbol}] Skip: Bearish but not in Premium.")
            return None
            
        # 4. Liquidity & Entry Check (H1)
        liq = SMCLiquidity()
        
        # Check for Sweep
        has_sweep = liq.detect_liquidity_sweep(h1_bars, bias_d1, window=30)
        print(f"[{symbol}] Liquidity Sweep: {has_sweep}")
        
        if not has_sweep:
            # Strict Rule: No Sweep = No Trade
            return None
            
        # Check for OB presence nearby
        ob = liq.find_order_blocks(h1_bars, bias_d1)
        if ob:
            print(f"[{symbol}] Valid OB Found: {ob['type']} at {ob['high']}-{ob['low']}")
            # Signal Generation
            if bias_d1 == "bullish":
                return "buy"
            elif bias_d1 == "bearish":
                return "sell"
                
        return None

    except Exception as e:
        print(f"Error in swing analysis: {e}")
        return None
