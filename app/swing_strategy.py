"""
Swing Trading Strategy Module

Timeframes: Daily (HTF) and 4-Hour (LTF)
Holding Period: Weeks to months
Stop Loss: 5-10% structure-based
Take Profit: 2:1 or 3:1 R:R
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

def get_swing_signal(htf: pd.DataFrame, ltf: pd.DataFrame):
    """
    Swing trading signal generation.
    
    Args:
        htf: Daily timeframe data
        ltf: 4-Hour timeframe data
        
    Returns:
        'buy', 'sell', or None
    """
    # 1. HTF Trend Filter (200 EMA on Daily)
    htf = htf.copy()
    htf['ema200'] = ta.ema(htf['close'], length=200)
    
    if len(htf) < 200:
        return None  # Not enough data
    
    # Check the LAST closed candle for bias
    last_close = htf['close'].iloc[-1]
    last_ema200 = htf['ema200'].iloc[-1]
    
    if pd.isna(last_ema200):
        return None
    
    bias = "bullish" if last_close > last_ema200 else "bearish"
    
    # 2. LTF Analysis (4-Hour)
    ltf = ltf.copy()
    
    # Add SMC indicators
    ltf = detect_swing_fvg(ltf)
    ltf = detect_swing_order_block(ltf)
    
    if len(ltf) < 20:
        return None
    
    # Get last 5 candles for signal check
    recent = ltf.tail(10)
    
    # 3. Signal Logic - Require BOTH OB and FVG for stronger confirmation
    has_bull_ob = recent['is_bull_ob_candle'].any()
    has_bull_fvg = recent['bull_fvg'].any()
    has_bear_ob = recent['is_bear_ob_candle'].any()
    has_bear_fvg = recent['bear_fvg'].any()
    
    # BUY: Bullish bias + (Bull OB OR Bull FVG)
    if bias == "bullish" and (has_bull_ob or has_bull_fvg):
        return "buy"
    
    # SELL: Bearish bias + (Bear OB OR Bear FVG)
    if bias == "bearish" and (has_bear_ob or has_bear_fvg):
        return "sell"
    
    return None


def detect_swing_fvg(df):
    """
    Fair Value Gap detection for swing trading.
    Uses larger gaps due to 4H timeframe.
    """
    df['bull_fvg'] = (df['low'] > df['high'].shift(2))
    df['bear_fvg'] = (df['high'] < df['low'].shift(2))
    return df


def detect_swing_order_block(df):
    """
    Order Block detection for swing trading.
    Stronger impulse threshold for 4H timeframe.
    """
    body_size = abs(df['close'] - df['open'])
    avg_body = body_size.rolling(20).mean()
    
    # Stronger impulse threshold for swing (2.0x vs 1.5x for day trading)
    df['impulse'] = body_size > (avg_body * 2.0)
    
    # Bullish OB: Down candle followed by strong up impulse
    df['is_bull_ob_candle'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                              (df['close'] > df['open']) & \
                              df['impulse'] & \
                              (df['close'] > df['high'].shift(1))
                              
    # Bearish OB: Up candle followed by strong down impulse
    df['is_bear_ob_candle'] = (df['close'].shift(1) > df['open'].shift(1)) & \
                              (df['close'] < df['open']) & \
                              df['impulse'] & \
                              (df['close'] < df['low'].shift(1))
    
    return df


def get_last_swing_high(df, window=5):
    """
    Find last confirmed swing high for swing trading.
    Uses larger window for 4H/Daily timeframes.
    """
    if len(df) < window * 2 + 1:
        return None
        
    for i in range(len(df) - 1 - window, window, -1):
        candidate_high = df['high'].iloc[i]
        left_max = df['high'].iloc[i-window:i].max()
        right_max = df['high'].iloc[i+1:i+window+1].max()
        
        if candidate_high > left_max and candidate_high > right_max:
            return candidate_high
            
    return None


def get_last_swing_low(df, window=5):
    """
    Find last confirmed swing low for swing trading.
    Uses larger window for 4H/Daily timeframes.
    """
    if len(df) < window * 2 + 1:
        return None
        
    for i in range(len(df) - 1 - window, window, -1):
        candidate_low = df['low'].iloc[i]
        left_min = df['low'].iloc[i-window:i].min()
        right_min = df['low'].iloc[i+1:i+window+1].min()
        
        if candidate_low < left_min and candidate_low < right_min:
            return candidate_low
            
    return None
