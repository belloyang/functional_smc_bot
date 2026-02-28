import pandas as pd
import numpy as np
import pandas_ta as ta
import re
import json
import os
import signal
import sys
import argparse
import time
import requests
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    OptionSnapshotRequest,
    OptionLatestQuoteRequest,
    OptionChainRequest,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide, OrderStatus, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, StopLossRequest, TakeProfitRequest, ReplaceOrderRequest
try:
    from . import config
except ImportError:
    import config


# ================= CONFIG =================

# Use config values
API_KEY = config.API_KEY
API_SECRET = config.API_SECRET
SYMBOL = config.SYMBOL
RISK_PER_TRADE = config.RISK_PER_TRADE

# ================= CLIENTS =================

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trade_client = config.trading_client

# ================= DATA =================

from alpaca.data.historical import OptionHistoricalDataClient
option_data_client = OptionHistoricalDataClient(API_KEY, API_SECRET)

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Throttle repetitive same-bias skip logs.
_LAST_SAME_BIAS_SKIP_LOG = {}

def get_bars(symbol, timeframe, limit):
    # Calculate a lookback to ensure we have enough data (e.g. 14 days)
    # This prevents the API from defaulting to "today only" which breaks indicators like EMA50
    # 14 days is safer for EMA50 stability.
    start_dt = datetime.now() - timedelta(days=14)
    
    # Request a large chunk of data starting from 14 days ago.
    # We increase the limit to ensure we don't truncate history needed for warm-up.
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, limit=10000, start=start_dt)
    bars = data_client.get_stock_bars(req).df
    bars = bars.reset_index()
    
    if bars.empty:
        print(f"Bars df for {symbol} is empty!")
        return bars

    # To be extremely consistent, we do NOT slice here anymore.
    # The strategy functions (get_strategy_signal) will handle looking at the 'last' bars.
    # Slicing here can break indicators if the limit is small.
        
    return bars

# ================= SMC LOGIC (CAUSAL) =================

def detect_swing_highs_lows(df, window=3):
    """
    Identifies if a specific candle *was* a swing point.
    A swing high at index `i` is confirmed at index `i + window`.
    """
    # This function is kept for structural compatibility but the logic is moved to specific getters
    # or we can add columns that mark the 'confirmation' of a swing.
    return df

def get_last_swing_high(df, window=3):
    # Returns the price of the last confirmed swing high looking backwards from the end
    # A swing high at 'i' is confirmed when we are at 'i + window'
    # So we iterate backwards.
    
    # We need at least window*2 + 1 bars
    if len(df) < window * 2 + 1:
        return None
        
    # Iterate backwards from current candle. 
    # for a candidate candle at index `c`, we need data up to `c + window`.
    # so the latest candidate we can check is at index `len(df) - 1 - window`.
    
    for i in range(len(df) - 1 - window, window, -1):
        # check if df[i] is local max of [i-window, i+window]
        candidate_high = df['high'].iloc[i]
        
        # Check left
        left_max = df['high'].iloc[i-window:i].max()
        # Check right
        right_max = df['high'].iloc[i+1:i+window+1].max()
        
        if candidate_high > left_max and candidate_high > right_max:
            return candidate_high
            
    return None

def get_last_swing_low(df, window=3):
    if len(df) < window * 2 + 1:
        return None
        
    for i in range(len(df) - 1 - window, window, -1):
        candidate_low = df['low'].iloc[i]
        left_min = df['low'].iloc[i-window:i].min()
        right_min = df['low'].iloc[i+1:i+window+1].min()
        
        if candidate_low < left_min and candidate_low < right_min:
            return candidate_low
            
    return None

def detect_fvg(df):
    """
    Fair Value Gap:
    Bullish: High of candle[i-2] < Low of candle[i]. Gap is between them.
    Bearish: Low of candle[i-2] > High of candle[i].
    Confirmed at close of candle[i].
    """
    # Create FVG columns. 
    # 'bull_fvg' at row `i` means an FVG formed between i-2 and i.
    df['bull_fvg'] = (df['low'] > df['high'].shift(2))
    df['bull_fvg_top'] = df['low'] # The top of the gap is the low of candle i
    df['bull_fvg_bottom'] = df['high'].shift(2) # The bottom of the gap is high of candle i-2
    
    df['bear_fvg'] = (df['high'] < df['low'].shift(2))
    df['bear_fvg_top'] = df['low'].shift(2) # The top of the gap is the low of candle i-2
    df['bear_fvg_bottom'] = df['high'] # The bottom of the gap is high of candle i
    return df

def detect_order_block(df):
    """
    Order Block: 
    Bullish: The last down candle before a strong up move (impulse).
    Bearish: The last up candle before a strong down move.
    """
    # Identify impulse (large body candles)
    body_size = abs(df['close'] - df['open'])
    avg_body = body_size.rolling(20).mean()
    avg_vol = df['volume'].rolling(20).mean()
    
    df['body_size'] = body_size
    df['avg_body'] = avg_body
    df['avg_vol'] = avg_vol
    df['impulse'] = body_size > (avg_body * 1.5) # Arbitrary threshold for "strong" move
    
    # We are looking for where an OB *was* formed. 
    # A Bullish OB is confirmed when we have a down candle followed by an impulse up.
    # At index `i` (impulse candle), the OB is at `i-1`.
    
    # Logic: 
    # 1. Previous candle (i-1) was bearish (Close < Open)
    # 2. Current candle (i) is bullish (Close > Open)
    # 3. Current candle is 'impulse'
    # 4. Current Close > Previous High (optional structure break)
    
    df['is_bull_ob_candle'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                              (df['close'] > df['open']) & \
                              df['impulse'] & \
                              (df['close'] > df['high'].shift(1)) 
                              
    df['is_bear_ob_candle'] = (df['close'].shift(1) > df['open'].shift(1)) & \
                              (df['close'] < df['open']) & \
                              df['impulse'] & \
                              (df['close'] < df['low'].shift(1))

    return df

def detect_structure_shift(df):
    # Not fully implemented in causal way for this snippet, relying on simpler OB/Impulse logic
    return df

def calculate_confidence(ltf_row, last_htf, fvg_touch=False):
    """Calculates a confidence score between 0-100%."""
    score = 0
    
    # 1. Impulse Intensity & Volume (20%)
    if ltf_row.get('avg_vol', 0) > 0:
        vol_ratio = ltf_row['volume'] / ltf_row['avg_vol']
        vol_score = min(20, max(0, (vol_ratio - 1.0) / 1.0 * 20))
        score += vol_score
        
    # 2. FVG Pullback Confluence (30%)
    if fvg_touch:
        score += 30
        
    # 3. Trend Proximity (20%)
    price = ltf_row['close']
    ema = last_htf.get('ema50')
    if ema:
        dist_pct = abs(price - ema) / ema
        # Pullbacks often touch or get close to the EMA.
        if dist_pct < 0.003:
            score += 20
        elif dist_pct < 0.005:
            score += 10
            
    # 4. Trend Strength (ADX) (30%)
    adx = last_htf.get('adx', 0)
    if not pd.isna(adx):
        if adx >= 25:
            score += 30
        elif adx > 20:
            score += 15
        elif adx < 15:
            score -= 20 # Penalize very choppy markets
            
    return int(min(100, max(0, score)))

def get_confidence_label(score):
    if score >= 80: return "🔥🔥 High"
    if score >= 60: return "📈 Medium"
    return "⚖️ Low"

def send_discord_notification(signal, price, time_str, symbol, bias, confidence):
    """Sends a signal alert to Discord via Webhook."""
    webhook_url = getattr(config, 'DISCORD_WEBHOOK_URL', None)
    if not webhook_url:
        return
        
    color = 0x2ca02c if signal.upper() == "BUY" else 0xd62728
    label = get_confidence_label(confidence)
    
    payload = {
        "embeds": [{
            "title": f"🚀 [{symbol}] {signal.upper()} Signal Detected!",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": f"**{symbol}**", "inline": True},
                {"name": "Confidence", "value": f"**{confidence}%** ({label})", "inline": True},
                {"name": "Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Time (ET)", "value": time_str, "inline": True},
                {"name": "Bias", "value": bias.upper(), "inline": True},
                {"name": "Strategy", "value": "SMC Order Block + Impulse", "inline": False}
            ],
            "footer": {"text": f"Alpaca Bot Live v{getattr(config, '__version__', '?.?.?')}"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Discord Notification Failed: {e}")

def send_discord_live_trading_notification(signal, symbol, order_details, confidence, strategy_bias):
    """Sends a detailed live trading notification to Discord."""
    webhook_url = getattr(config, 'DISCORD_WEBHOOK_URL_LIVE_TRADING', None)
    if not webhook_url:
        return
        
    now_et = datetime.now(ZoneInfo("America/New_York"))
    time_et_str = now_et.strftime("%I:%M %p")
    
    label = get_confidence_label(confidence)
    # Color based on signal
    color = 0x00ff00 if signal.lower() == "buy" else 0xff0000
    if "close" in signal.lower() or "cleanup" in signal.lower() or "flip" in signal.lower():
        color = 0x0000ff # Blue for neutral/closure
    
    fields = [
        {"name": "Symbol", "value": f"**{symbol}**", "inline": True},
        {"name": "Time (ET)", "value": time_et_str, "inline": True},
        {"name": "HTF Bias", "value": strategy_bias.upper(), "inline": True},
        {"name": "Confidence", "value": f"{confidence}% [{label}]", "inline": True},
    ]
    
    # Add order details
    if isinstance(order_details, dict):
        for k, v in order_details.items():
            fields.append({"name": k, "value": str(v), "inline": True})
    else:
        fields.append({"name": "Order Details", "value": str(order_details), "inline": False})
    
    payload = {
        "embeds": [{
            "title": f"📈 LIVE ORDER: {signal.upper()}",
            "color": color,
            "fields": fields,
            "footer": {"text": f"SMC Bot Live Execution (Alpaca) v{getattr(config, '__version__', '?.?.?')}"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        print(f"⚠️ Live Notification Failed: {e}")

def liquidity_sweep(df):
    # Not fully implemented in causal way for this snippet
    return df

# ================= STRATEGY =================

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ...


def get_strategy_signal(htf: pd.DataFrame, ltf: pd.DataFrame):
    """
    Pure strategy function.
    Takes historical dataframes and returns a signal ('buy', 'sell', or None).
    """
    # 1. HTF Direction
    htf = htf.copy()
    
    # Only calculate EMA if it doesn't exist (prevents drift if already pre-calculated on full history)
    if 'ema50' not in htf.columns:
        if len(htf) < 50: return None
        htf['ema50'] = ta.ema(htf['close'], length=50)
    
    if htf['ema50'].isnull().all():
        return None
        
    # Check the LAST closed candle for bias. 
    last_htf = htf.iloc[-1]
    
    # ADX Filter: If ADX is below 25, the market is considered choppy.
    adx_val = last_htf.get('adx', 0)
    is_choppy = False
    if not pd.isna(adx_val) and adx_val < 25:
        is_choppy = True
        
    if is_choppy:
        bias = "choppy"
    else:
        # Bias Hysteresis: Use ATR to scale the buffer (10% of ATR)
        price = last_htf['close']
        ema = last_htf['ema50']
        atr = last_htf.get('atr', price * 0.001) 
        
        buffer = 0.1 * (atr / price)
        buffer = max(0.0002, min(0.001, buffer)) # Sanity bounds (0.02% to 0.1%)
        
        if price > (ema + buffer):
            bias = "bullish"
        elif price < (ema - buffer):
            bias = "bearish"
        else:
            bias = "choppy" # Inside the noise band

        # --- BALANCED EXTENSION FILTER ---
        # Don't chase trends if price is already significantly extended, UNLESS the move is parabolic.
        dist_from_ema = abs(price - ema)
        
        # Parabolic Override: If ADX is extremely high (> 40), extension is a sign of strength, not exhaustion.
        is_parabolic = adx_val > 40
        
        # Buffer kept at 3.0x ATR for balanced momentum capture
        # NOTE: 2.5x was too tight and blocked entries during strong summer 2025 QQQ trend
        extension_limit = 3.0 * atr
        
        if dist_from_ema > extension_limit and not is_parabolic:
            return None, 0

    # --- RSI OVERBOUGHT / OVERSOLD FILTER ---
    # Avoid entering longs into overbought HTF conditions and shorts into oversold ones.
    rsi = last_htf.get('rsi14', None)
    if rsi is not None and not pd.isna(rsi):
        if bias == "bullish" and rsi > 77.5:
            return None, 0  # Overbought — risk of mean reversion against the trade
        elif bias == "bearish" and rsi < 22.5:
            return None, 0  # Oversold — risk of bounce against the trade

    # 2. LTF Analysis
    ltf = ltf.copy()
    # Add indicators if not present
    if 'bull_fvg' not in ltf.columns:
        ltf = detect_fvg(ltf)
    if 'is_bull_ob_candle' not in ltf.columns:
        ltf = detect_order_block(ltf)
    
    if len(ltf) < 5: return None
    
    # Get last formed candle for signal check
    last_closed = ltf.iloc[-1]
    
    # --- FALLING KNIFE / MOMENTUM PROTECTION ---
    # Don't enter if the signal candle is a massive impulse AGAINST our bias.
    # e.g. buying when a giant red candle just slammed into the FVG.
    is_opposite_impulse = False
    if last_closed.get('impulse', False):
        if bias == "bullish" and last_closed['close'] < last_closed['open']:
            is_opposite_impulse = True
        elif bias == "bearish" and last_closed['close'] > last_closed['open']:
            is_opposite_impulse = True
            
    # Volume Confirmation: Pullbacks on extreme volume are risky (reversal threat)
    vol_ratio = 1.0
    if last_closed.get('avg_vol', 0) > 0:
        vol_ratio = last_closed['volume'] / last_closed['avg_vol']
    
    signal = None
    confidence = 0
    fvg_touch = False

    # Block if high volume is moving AGAINST our bias (reversal threat).
    # High volume WITH our bias is fine — it signals strong directional interest on the pullback.
    vol_against_bias = False
    if vol_ratio > 2.0:
        if bias == "bullish" and last_closed['close'] < last_closed['open']:
            vol_against_bias = True  # Big red candle on high vol in a bull trend
        elif bias == "bearish" and last_closed['close'] > last_closed['open']:
            vol_against_bias = True  # Big green candle on high vol in a bear trend

    if is_opposite_impulse or vol_against_bias:
        return None, 0

    # We look back over the last N candles to find a recent valid FVG/OB structure
    lookback = 15
    recent_ltf = ltf.iloc[-lookback-1:-1]
    
    # Threshold for "close enough" (Scaled with volatility)
    atr = last_htf.get('atr', last_closed['close'] * 0.005)
    buffer = 0.1 * (atr / last_closed['close'])
    buffer = max(0.0002, min(0.001, buffer)) # Sanity bounds
    
    if bias == "bullish":
        # Search for a recent Bullish FVG associated with an Impulse/OB
        for i in range(len(recent_ltf) - 1, -1, -1):
            row = recent_ltf.iloc[i]
            if row.get('bull_fvg', False):
                ob_nearby = any(recent_ltf.iloc[max(0, i-2):min(len(recent_ltf), i+1)]['is_bull_ob_candle'])
                
                if ob_nearby:
                    fvg_top = row['bull_fvg_top']
                    fvg_bot = row['bull_fvg_bottom']
                    
                    # Entry condition: Price pulls back to touch or get very close to FVG top
                    # But must stay above FVG bottom (structure preservation)
                    px_limit_top = fvg_top * (1 + buffer)
                    px_limit_bot = fvg_bot * (1 - buffer)
                    
                    if last_closed['low'] <= px_limit_top and last_closed['close'] > px_limit_bot:
                        signal = "buy"
                        fvg_touch = True
                        break
             
    elif bias == "bearish":
        for i in range(len(recent_ltf) - 1, -1, -1):
            row = recent_ltf.iloc[i]
            if row.get('bear_fvg', False):
                ob_nearby = any(recent_ltf.iloc[max(0, i-2):min(len(recent_ltf), i+1)]['is_bear_ob_candle'])
                
                if ob_nearby:
                    fvg_top = row['bear_fvg_top']
                    fvg_bot = row['bear_fvg_bottom']
                    
                    px_limit_top = fvg_top * (1 + buffer)
                    px_limit_bot = fvg_bot * (1 - buffer)
                    
                    if last_closed['high'] >= px_limit_bot and last_closed['close'] < px_limit_top:
                        signal = "sell"
                        fvg_touch = True
                        break

    elif bias == "choppy":
         signal = None

    if signal:
        confidence = calculate_confidence(last_closed, htf.iloc[-1], fvg_touch=fvg_touch)
        # --- CONFIDENCE FLOOR ---
        # Discard low-confidence signals. A valid signal (FVG+OB with ADX>=25) scores at least 60,
        # so 50 only filters edge cases where indicator data is unreliable (e.g. ADX=0).
        # NOTE: 60% was too aggressive and blocked 70% of valid trend entries in summer 2025.
        if confidence < 50:
            return None, 0

    return signal, confidence


def _normalize_timestamp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a UTC `timestamp` column exists for stable causal slicing."""
    out = df.copy()
    if 'timestamp' not in out.columns:
        out = out.reset_index()

    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True, errors='coerce')
    out = out.dropna(subset=['timestamp'])
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out


def precompute_strategy_features(htf_df: pd.DataFrame, ltf_df: pd.DataFrame):
    """Compute strategy features once on full history to avoid signal drift."""
    htf = _normalize_timestamp_frame(htf_df)
    ltf = _normalize_timestamp_frame(ltf_df)

    if 'ema50' not in htf.columns:
        htf['ema50'] = ta.ema(htf['close'], length=50)

    # Compute ADX for trend strength (using standard 14 period)
    if 'adx' not in htf.columns:
        # pandas_ta.adx returns a DataFrame with ADX_14, DMP_14, DMN_14
        adx_df = ta.adx(htf['high'], htf['low'], htf['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            # We just need the main ADX column. It's usually the first one 'ADX_14'
            adx_col = [c for c in adx_df.columns if c.startswith('ADX')][0]
            htf['adx'] = adx_df[adx_col]
        else:
            htf['adx'] = 0

    if 'atr' not in htf.columns:
        htf['atr'] = htf.ta.atr(length=14)

    if 'rsi14' not in htf.columns:
        htf['rsi14'] = ta.rsi(htf['close'], length=14)

    if 'bull_fvg' not in ltf.columns or 'bear_fvg' not in ltf.columns:
        ltf = detect_fvg(ltf)
    if 'is_bull_ob_candle' not in ltf.columns or 'is_bear_ob_candle' not in ltf.columns:
        ltf = detect_order_block(ltf)

    return htf, ltf


def get_causal_signal_from_precomputed(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, evaluation_ts, ltf_window: int = 200):
    """
    Evaluate signal at a specific closed 1m candle timestamp.
    Uses only 15m candles that were already closed at that timestamp.
    """
    if evaluation_ts is None:
        return None

    ts = pd.Timestamp(evaluation_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    ltf_causal = ltf_df[ltf_df['timestamp'] <= ts]
    if ltf_causal.empty:
        return None

    htf_causal = htf_df[htf_df['timestamp'] <= (ts - timedelta(minutes=15))]
    if htf_causal.empty or len(htf_causal) < 50:
        return None

    return get_strategy_signal(htf_causal, ltf_causal.iloc[-ltf_window:])

def generate_signal(symbol=None):
    if symbol is None:
        symbol = SYMBOL
        
    try:
        # Convert config strings to TimeFrame objects
        # Assuming simple mapping for now based on known config
        tf_htf = TimeFrame(15, TimeFrameUnit.Minute) if config.TIMEFRAME_HTF == "15Min" else TimeFrame(1, TimeFrameUnit.Hour) # Fallback/Example
        tf_ltf = TimeFrame(1, TimeFrameUnit.Minute) if config.TIMEFRAME_LTF == "1Min" else TimeFrame(5, TimeFrameUnit.Minute)
        
        # Override with exact parsing if needed, but for SPY/15Min/1Min typical usage:
        
        htf = get_bars(symbol, tf_htf, config.HTF_BARS if hasattr(config, 'HTF_BARS') else 200)
        ltf = get_bars(symbol, tf_ltf, config.LTF_BARS if hasattr(config, 'LTF_BARS') else 200)
        
        if htf.empty or ltf.empty:
            print("Not enough data fetched.")
            return None

        htf, ltf = precompute_strategy_features(htf, ltf)

        # Last fully closed LTF bar: drop newest potentially-forming row.
        if len(ltf) < 2:
            return None

        evaluation_ts = ltf['timestamp'].iloc[-2]
        return get_causal_signal_from_precomputed(htf, ltf, evaluation_ts)

    except Exception as e:
        print(f"Error generating signal for {symbol}: {e}")
        return None

# ================= RISK =================

from alpaca.trading.requests import StopLossRequest, TakeProfitRequest

def calculate_smart_quantity(symbol, price, stop_loss_price, budget_override=None, risk_allocation_pct=None):
    """
    Calculates position size based on:
    1. Risk Amount (1% of Equity)
    2. Distance to Stop Loss
    3. Caps by Buying Power
    """
    try:
        account = trade_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
    except Exception as e:
        print(f"Error fetching account info: {e}")
        equity = 10000.0
        buying_power = 10000.0
        
    alloc_risk_pct = 1.0 if risk_allocation_pct is None else float(risk_allocation_pct)
    alloc_risk_pct = min(1.0, max(0.0, alloc_risk_pct))
    risk_amount = equity * alloc_risk_pct * RISK_PER_TRADE
    stop_distance = abs(price - stop_loss_price)
    
    if stop_distance <= 0:
        print("Stop distance is 0 or negative! Defaulting to min qty.")
        return 0
        
    # 1. Risk-Based Qty
    qty_risk = risk_amount / stop_distance
    
    # 2. Ticker-Based Cap (Multi-Instance Safety)
    if budget_override is not None:
        max_ticker_pct = budget_override
    else:
        max_ticker_pct = getattr(config, 'STOCK_ALLOCATION_PCT', 0.80)
        
    max_ticker_amt = equity * max_ticker_pct
    
    # Calculate current exposure for this ticker (Stock Only)
    current_exposure = 0.0
    try:
        positions = trade_client.get_all_positions()
        for p in positions:
            if p.symbol == symbol and p.asset_class.value == 'us_equity':
                current_exposure += abs(float(p.market_value))
    except Exception as e:
        print(f"Error calculating exposure for {symbol}: {e}")
        
    remaining_ticker_budget = max(0, max_ticker_amt - current_exposure)
    qty_ticker = remaining_ticker_budget / price
    
    # 3. Buying Power Cap
    max_cost = buying_power * 0.95
    qty_bp = max_cost / price
    
    # Final
    qty = int(min(qty_risk, qty_ticker, qty_bp))
    
    # Log
    print(f"DEBUG: Equity: ${equity:.2f} | RiskAlloc: {alloc_risk_pct:.2f} | Risk($): ${risk_amount:.2f} | Entry: {price} | SL: {stop_loss_price} | Dist: {stop_distance:.2f}")
    print(f"DEBUG: RiskQty: {int(qty_risk)} | BPQty: {int(qty_bp)} -> Final: {qty}")
    
    return max(0, qty)

# ================= HELPERS =================

from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType

def get_best_option_contract(symbol, signal_type, known_price=None):
    """
    Finds the best option contract for the given signal.
    """
    try:
        if known_price:
             current_price = known_price
        else:
            current_price_df = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 1)
            if current_price_df.empty:
                print("Could not fetch current price for options.")
                return None
            current_price = current_price_df['close'].iloc[-1]
            
        now = datetime.now()
        start_date = now + timedelta(days=3)
        end_date = now + timedelta(days=14)
        
        contract_type = ContractType.CALL if signal_type == "buy" else ContractType.PUT
        
        req = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=start_date.date(),
            expiration_date_lte=end_date.date(),
            type=contract_type,
            limit=100 
        )
        
        print(f"DEBUG: Searching options for {symbol} | Type: {contract_type} | Range: {start_date.date()} to {end_date.date()}")
        contracts = trade_client.get_option_contracts(req).option_contracts
        
        if not contracts:
            print(f"No option contracts found for {symbol}")
            return None
            
        contracts.sort(key=lambda x: x.expiration_date)
        nearest_expiry = contracts[0].expiration_date
        expiry_contracts = [c for c in contracts if c.expiration_date == nearest_expiry]
        expiry_contracts.sort(key=lambda c: abs(float(c.strike_price) - current_price))
        
        return expiry_contracts[0]

    except Exception as e:
        print(f"Error getting option contract: {e}")
        return None

def get_current_position(symbol):
    """
    Fetches the current position for the given symbol.
    Returns the Position object if found, else None.
    """
    try:
        return trade_client.get_open_position(symbol)
    except Exception:
        return None

def cancel_all_orders_for_symbol(symbol):
    """
    Safely cancels all open orders for a specific symbol.
    """
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbol=symbol)
        orders = trade_client.get_orders(req)
        for o in orders:
            trade_client.cancel_order_by_id(o.id)
            print(f"Cancelled order {o.id} for {symbol}")
    except Exception as e:
        print(f"⚠️ Error cancelling orders for {symbol}: {e}")

def _should_log_same_bias_skip(symbol, signal, interval_minutes=15):
    """Rate-limit repeated 'same bias, skipping' logs for cleaner live output."""
    key = f"{symbol}:{signal}"
    now = datetime.now(timezone.utc)
    last = _LAST_SAME_BIAS_SKIP_LOG.get(key)
    if last is None or (now - last).total_seconds() >= (interval_minutes * 60):
        _LAST_SAME_BIAS_SKIP_LOG[key] = now
        return True
    return False

def _log_close_submission(symbol, close_result, context):
    """Log close-order telemetry in a normalized way."""
    try:
        if isinstance(close_result, dict):
            oid = close_result.get("id") or close_result.get("order_id") or "?"
            status = close_result.get("status", "?")
            print(f"✅ {context}: close submitted for {symbol} | id={oid} status={status}")
            return

        oid = getattr(close_result, "id", "?")
        status = getattr(close_result, "status", "?")
        filled_qty = getattr(close_result, "filled_qty", None)
        msg = f"✅ {context}: close submitted for {symbol} | id={oid} status={status}"
        if filled_qty is not None:
            msg += f" filled_qty={filled_qty}"
        print(msg)
    except Exception as e:
        print(f"✅ {context}: close submitted for {symbol} (details unavailable: {e})")

def place_trade(signal, symbol, confidence=0, use_daily_cap=True, daily_cap_value=None, option_allocation_override=None, max_option_contracts_override=None):
    # Determine bias for notifications
    bias = "bullish" if signal == "buy" else "bearish"

    # --- HARD FILTER CHECK ---
    if confidence <= 0:
        print(f"🛑 REJECTED: Confidence is 0 (Blocked by Strategy Filters) for {symbol}")
        return False

    # --- GLOBAL SAFETY CHECKS ---
    safety = load_global_safety_state()
    if safety.get("halted", False):
        print("🚨 TRADING HALTED: Global Drawdown Circuit Breaker is ACTIVE. No new trades allowed.")
        return False
        
    if safety.get("last_loss_time"):
        last_loss = datetime.fromisoformat(safety["last_loss_time"])
        wait_mins = getattr(config, 'COOL_DOWN_MINUTES', 60)
        time_diff = (datetime.now(timezone.utc) - last_loss).total_seconds() / 60
        if time_diff < wait_mins:
            print(f"🕒 COOL-DOWN: Last loss was {time_diff:.1f} mins ago. Waiting {wait_mins} total. Skipping.")
            return False

    # --- TIME FILTER (10:00 AM - 3:30 PM ET) ---
    now_et = datetime.now(ZoneInfo("America/New_York"))
    current_time = now_et.time()
    start_time = datetime.strptime("09:40:00", "%H:%M:%S").time()
    end_time = datetime.strptime("15:55:00", "%H:%M:%S").time()
    
    if current_time < start_time or current_time > end_time:
        print(f"🕒 TIME FILTER: Current time {current_time} is outside the 09:40-15:55 window. Skipping.")
        return False

    # --- DAILY TRADE CAP ---
    state = load_trade_state()
    today_str = now_et.strftime("%Y-%m-%d")
    
    # Initialize or reset daily count
    ticker_state = state.get(symbol, {})
    last_date = ticker_state.get("last_trade_date")
    daily_count = ticker_state.get("daily_trade_count", 0)
    
    if last_date != today_str:
        daily_count = 0
        ticker_state["last_trade_date"] = today_str
    
    # Determine the cap to use
    if daily_cap_value is None:
        cap_limit = getattr(config, 'DEFAULT_DAILY_CAP', 5)  # Use config or fallback to 5
    else:
        cap_limit = daily_cap_value
    
    if use_daily_cap and cap_limit >= 0 and daily_count >= cap_limit:
        print(f"🛑 DAILY CAP: Already placed {daily_count} trades for {symbol} today (limit: {cap_limit}). Skipping.")
        return False

    # Get latest price and ample history for Swing Point detection
    # We need enough history to find a swing point (e.g. 50-100 bars)
    price_df = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 100)
    if price_df.empty:
        print("Could not fetch price for placement.")
        return False
    
    price = price_df['close'].iloc[-1]
    last_close = price # Current price estimate
    option_allocation_pct = option_allocation_override if option_allocation_override is not None else getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20)
    option_allocation_pct = min(1.0, max(0.0, float(option_allocation_pct)))
    stock_allocation_pct = 1.0 - option_allocation_pct
    max_option_contracts = max_option_contracts_override if max_option_contracts_override is not None else getattr(config, 'MAX_OPTION_CONTRACTS', -1)
    
    # 1. Fetch CURRENT state for the base symbol and all related contracts
    all_positions = []
    try:
        all_positions = trade_client.get_all_positions()
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        return False

    qty_held = 0
    side_held = None
    any_options_held = False

    for p in all_positions:
        if not p.symbol.startswith(symbol): continue
        if p.asset_class.value == 'us_equity' and p.symbol == symbol:
            qty_held = float(p.qty)
            side_held = p.side
        if p.asset_class.value == 'us_option':
            any_options_held = True

    print(f"DEBUG: Current Position for {symbol}: {qty_held} shares (Side: {side_held}) | Options: {any_options_held}")

    # 2. Global Position Cleanup (Cross-Asset Signal Flip)
    same_bias_held = False
    for pos in all_positions:
        if not pos.symbol.startswith(symbol): continue
        
        # Identify if it's the underlying or an option
        is_stock = (pos.asset_class.value == 'us_equity' and pos.symbol == symbol)
        is_option = (pos.asset_class.value == 'us_option')
        
        # Identify Bias
        is_bullish = False
        if is_stock:
            is_bullish = (pos.side == PositionSide.LONG)
        elif is_option:
            m = re.search(r'\d{6}([CP])\d{8}', pos.symbol)
            is_bullish = (m and m.group(1) == 'C')

        # Conflict Detection: Close if signal is opposite of current holding bias
        if (signal == "buy" and not is_bullish) or (signal == "sell" and is_bullish):
            print(f"🔄 CROSS-ASSET CLEANUP: Closing {pos.symbol} bias conflict with {signal.upper()} signal.")
            try:
                cancel_all_orders_for_symbol(pos.symbol)
                close_result = trade_client.close_position(pos.symbol)
                _log_close_submission(pos.symbol, close_result, "CROSS-ASSET CLEANUP")
                
                # Notify
                send_discord_live_trading_notification(
                    signal=f"cleanup_close_{pos.side}",
                    symbol=pos.symbol,
                    order_details={
                        "Action": "CLOSE (Market)",
                        "Side": pos.side,
                        "Price": price,
                        "Type": "Cross-Asset Cleanup"
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                # UPDATE LOCAL STATE IMMEDIATELY (Treat as closed for logic follow-through)
                if is_stock:
                    qty_held = 0
                    side_held = None
                if is_option:
                    # Caution: This assumes we closed ALL options. 
                    # If we had multiple and only closed one, this would need more precision.
                    # But for this bot's MVP, we only ever open one contract.
                    any_options_held = False
            except Exception as close_err:
                print(f"❌ Cleanup failed for {pos.symbol}: {close_err}")
        
        elif (signal == "buy" and is_bullish) or (signal == "sell" and not is_bullish):
            # Matches signal bias!
            if is_option:
                same_bias_held = True



    # 3. Mix-up Guards (Prevent holding both Stock and Options for same symbol)
    if getattr(config, 'ENABLE_OPTIONS', False):
        if qty_held != 0:
             print(f"Warning: Holding underlying shares ({qty_held}) while in Options Mode. Skipping new option entries to avoid mix-up.")
             return False
    else:
        if any_options_held:
             print(f"Warning: Holding option contracts while in Stock Mode. Skipping new stock entries to avoid mix-up.")
             return False

    # ================= OPTIONS MODE =================
    if getattr(config, 'ENABLE_OPTIONS', False):

        if same_bias_held:
            if _should_log_same_bias_skip(symbol, signal):
                print(f"Existing {signal.upper()} option bias detected for {symbol}. Skipping redundant entry.")
            return False

        print(f"Options Trading Enabled. Searching for contract for {signal.upper()}...")
        contract = get_best_option_contract(symbol, signal)
        
        if contract:
            trade_symbol = contract.symbol
            # Check if we already hold THIS specific contract
            contract_pos = get_current_position(trade_symbol)
            if contract_pos and float(contract_pos.qty) > 0:
                 print(f"Already hold {trade_symbol}. Holding.")
                 return False

            # Fetch Current Option Price & Snapshot to calculate Brackets and check IV
            from alpaca.data.requests import OptionLatestQuoteRequest, OptionSnapshotRequest
            try:
                # 1. Check IV from Snapshot first
                try:
                    snap_req = OptionSnapshotRequest(symbol_or_symbols=trade_symbol)
                    snap = option_data_client.get_option_snapshot(snap_req)
                    iv = snap[trade_symbol].implied_volatility if snap and trade_symbol in snap else None
                    max_iv = getattr(config, 'MAX_ALLOWED_IV', 0.40)
                    allow_no_iv = getattr(config, 'ALLOW_TRADE_WITHOUT_IV', False)

                    if iv is None:
                        if allow_no_iv:
                            print("⚠️ IV unavailable; proceeding because ALLOW_TRADE_WITHOUT_IV=True.")
                        else:
                            print("🛑 REJECTED: Option IV unavailable; cannot enforce MAX_ALLOWED_IV. Set ALLOW_TRADE_WITHOUT_IV=True to override.")
                            return False
                    else:
                        # Defensive: some feeds may return IV in percent (e.g., 55 for 55%)
                        iv_value = iv / 100.0 if iv > 2 else iv
                        if iv_value > max_iv:
                            print(f"🛑 REJECTED: Option IV too high ({iv_value:.2f} > {max_iv:.2f}). Market too volatile.")
                            return False
                except Exception as e:
                    print(f"⚠️ Could not fetch option snapshot for IV check: {e}")
                    if not getattr(config, 'ALLOW_TRADE_WITHOUT_IV', False):
                        print("🛑 REJECTED: Missing IV data and ALLOW_TRADE_WITHOUT_IV is False.")
                        return False

                # 2. We need the quote to determine Entry Price approximation
                quote_req = OptionLatestQuoteRequest(symbol_or_symbols=trade_symbol)
                quote = option_data_client.get_option_latest_quote(quote_req)
                
                # Use Ask price as likely entry (buying)
                entry_est = quote[trade_symbol].ask_price
                if entry_est <= 0:
                    print(f"Invalid option ask price {entry_est}. Skipping.")
                    return False
                    
                # SL/TP levels used for Discord notification and virtual stop tracking only.
                # Alpaca options orders don't support bracket params — enforcement is via manage_trade_updates().
                sl_price = round(entry_est * 0.80, 2)
                tp_price = round(entry_est * 1.50, 2)

                print(f"Options Levels: Est.Entry: {entry_est} | Virtual SL: {sl_price} (-20%) | Virtual TP: {tp_price} (+50%)")

                # --- GLOBAL OPTION EXPOSURE (Sum of all held premiums) ---
                account = trade_client.get_account()
                equity = float(account.equity)
                budget_pct = option_allocation_pct
                total_budget = equity * budget_pct
                
                # Fetch all current positions to sum exposure
                all_positions = trade_client.get_all_positions()
                
                # 1. Global Option Exposure Check
                existing_option_exposure = 0.0
                for p in all_positions:
                    if p.asset_class == 'us_option':
                        existing_option_exposure += abs(float(p.market_value))
                
                global_option_remaining = total_budget - existing_option_exposure
                
                # 2. Ticker-Specific Option Exposure Check
                max_ticker_option_pct = option_allocation_pct
                
                max_ticker_option_amt = equity * max_ticker_option_pct
                ticker_option_exposure = 0.0
                for p in all_positions:
                    if p.symbol.startswith(symbol) and p.asset_class.value == 'us_option':
                        ticker_option_exposure += abs(float(p.market_value))
                
                ticker_remaining = max(0, max_ticker_option_amt - ticker_option_exposure)
                
                # 3. Final Budget for this trade
                available_budget = min(global_option_remaining, ticker_remaining)
                cost_per_contract = entry_est * 100
                
                if available_budget < cost_per_contract:
                    print(f"⚠️ WARNING: Insufficient Budget for {symbol} option. Global Remain: ${global_option_remaining:.2f}, Ticker Remain: ${ticker_remaining:.2f}. Need ${cost_per_contract:.2f}. Skipping.")
                    return False
                
                # 3. Risk-Based Position Sizing (mode allocation aware)
                current_risk_target = equity * option_allocation_pct * RISK_PER_TRADE
                risk_per_contract = entry_est * 0.20 * 100  # 20% SL on premium
                qty_risk = int(current_risk_target // risk_per_contract) if risk_per_contract > 0 else 0
                
                # Still capped by available budget
                qty_cap = int(available_budget // cost_per_contract)
                
                qty = min(qty_risk, qty_cap)
                if max_option_contracts != -1 and qty > max_option_contracts:
                    print(f"DEBUG: Capping contracts from {qty} to {max_option_contracts}.")
                    qty = max_option_contracts
                    
                if qty < 1:
                    print("⚠️ WARNING: Calculated contracts < 1. Skipping.")
                    return False

                print(f"Sizing: Equity ${equity:.2f} | Risk Target ${current_risk_target:.2f} | Risk/Ctr ${risk_per_contract:.2f} | Budget ${available_budget:.2f} | Qty: {qty}")

                # Note: Alpaca options orders do not support bracket parameters.
                # SL/TP is enforced virtually via manage_trade_updates() polling loop.
                order = MarketOrderRequest(
                    symbol=trade_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                trade_client.submit_order(order)
                
                # Notify
                send_discord_live_trading_notification(
                    signal=f"buy_option_{contract.symbol[-9] if len(contract.symbol) > 9 else '?'}",
                    symbol=trade_symbol,
                    order_details={
                        "Action": "BUY",
                        "Qty": qty,
                        "Entry": entry_est,
                        "SL": sl_price,
                        "TP": tp_price,
                        "Type": "Market Bracket"
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                print(f"✅ OPTION BRACKET SUBMITTED: {trade_symbol}")
                # Update daily count
                ticker_state["daily_trade_count"] = daily_count + 1
                state[symbol] = ticker_state
                save_trade_state(state, symbol=symbol)
                return True

            except Exception as e:
                print(f"❌ Option Order failed: {e}")
                print("Falling back to share trading...")
        else:
             print("Could not find suitable option contract. Falling back to shares.")
        
    # --- STOCK TRADING LOGIC ---
    
    if signal == "buy":
        if qty_held == 0:
            # 1.1 Determine Stop Loss (Recent Swing Low)
            # Look back 50 bars for a swing low
            swing_low = get_last_swing_low(price_df, window=5)
            
            if swing_low and swing_low < price:
                sl_price = swing_low
                print(f"Structure Stop: Swing Low at {sl_price}")
            else:
                # Fallback: 0.5% below price
                sl_price = price * 0.995
                print(f"Fallback Stop: 0.5% at {sl_price}")
                
            # 1.2 Determine Take Profit (1:2.5 Risk Reward)
            risk_dist = price - sl_price
            tp_price = price + (risk_dist * 2.5)
            
            # 1.3 Calculate Quantity
            qty = calculate_smart_quantity(
                symbol,
                price,
                sl_price,
                budget_override=stock_allocation_pct,
                risk_allocation_pct=stock_allocation_pct
            )
            if qty <= 0:
                print("Calculated Quantity is 0, skipping trade.")
                return False

            print(f"Placing BUY Bracket: Entry ~{price} | SL {sl_price:.2f} | TP {tp_price:.2f} | Qty {qty}")
            
            try:
                # Construct Bracket Objects
                # Note: stop_loss and take_profit params in MarketOrderRequest expect simple objects or dicts
                # using the dedicated Request classes is safest.
                
                sl_req = StopLossRequest(stop_price=round(sl_price, 2))
                tp_req = TakeProfitRequest(limit_price=round(tp_price, 2))
                
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    stop_loss=sl_req,
                    take_profit=tp_req
                )
                trade_client.submit_order(order)
                print("✅ BUY BRACKET ORDER SUBMITTED")
                
                # Notify
                send_discord_live_trading_notification(
                    signal="buy_stock",
                    symbol=symbol,
                    order_details={
                        "Action": "BUY",
                        "Qty": qty,
                        "Entry": price,
                        "SL": sl_price,
                        "TP": tp_price,
                        "Type": "Market Bracket"
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                # Update daily count
                ticker_state["daily_trade_count"] = daily_count + 1
                state[symbol] = ticker_state
                save_trade_state(state, symbol=symbol)
                return True
            except Exception as e:
                print(f"❌ Order failed: {e}")
                return False
        
        elif side_held == PositionSide.SHORT: 
             print(f"Closing Short Position ({qty_held} shares) due to BUY signal.")
             try:
                # Safety: Cancel any existing open orders (SL/TP) for this symbol first
                cancel_all_orders_for_symbol(symbol)
                
                order = MarketOrderRequest(symbol=symbol, qty=abs(qty_held), side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                close_order = trade_client.submit_order(order)
                print("✅ SHORT CLOSE SUBMITTED")
                _log_close_submission(symbol, close_order, "TREND FLIP")
                
                # Notify
                send_discord_live_trading_notification(
                    signal="flip_close_short_stock",
                    symbol=symbol,
                    order_details={
                        "Action": "BUY (Market)",
                        "Qty": abs(qty_held),
                        "Price": price,
                        "Type": "Trend Flip Exit"
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
             except Exception as e:
                print(f"❌ Close Short failed: {e}")
                return False
        else:
            print(f"Already Long {qty_held} shares. Ignoring BUY signal.")
            return False

    elif signal == "sell":
        if qty_held == 0:
            print("SELL Signal detected but no position held. Skipping Short Entry (Long-Only Mode).")
            return False
            
        elif side_held == PositionSide.LONG: # We are Long
            print(f"Closing Long Position ({qty_held} shares) due to SELL signal.")
            try:
                # Safety: Cancel any existing open orders (SL/TP) for this symbol first
                cancel_all_orders_for_symbol(symbol)
                
                order = MarketOrderRequest(symbol=symbol, qty=abs(qty_held), side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                close_order = trade_client.submit_order(order)
                print("✅ LONG CLOSE SUBMITTED")
                _log_close_submission(symbol, close_order, "TREND FLIP")
                
                # Notify
                send_discord_live_trading_notification(
                    signal="flip_close_long_stock",
                    symbol=symbol,
                    order_details={
                        "Action": "SELL (Market)",
                        "Qty": abs(qty_held),
                        "Price": price,
                        "Type": "Trend Flip Exit"
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
            except Exception as e:
                 print(f"❌ Close Long failed: {e}")
                 return False
                 
        else:
            print(f"Already Short {qty_held} shares. Ignoring SELL signal.")
            return False

    # 4. Final Notification
    if signal in ["buy", "sell"]:
        time_str = datetime.now(ZoneInfo("America/New_York")).strftime("%I:%M %p")
        # We only send notification if a trade was actually SUBMITTED, 
        # but place_trade is called whenever there's a signal.
        # So we check if we submitted an order in the logic above? 
        # Actually it's better to send it here to confirm the SIGNAL was processed.
        bias = "BULLISH" if signal == "buy" else "BEARISH"
        # Do not send discord notification for now
        # send_discord_notification(signal, price, time_str, symbol, bias, confidence)

    return False

def load_initial_ticker_state(symbol):
    """Helper to load state for a specific symbol at startup or loop start."""
    state = load_trade_state(symbol=symbol)
    return state.get(symbol, {}), state

def parse_option_expiry(symbol):
    # Regex to capture YYMMDD from SPY251219C00500000
    # Format: Root(up to 6 chars) + YYMMDD + Type(C/P) + Strike
    match = re.search(r'[A-Z]+(\d{6})[CP]', symbol)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, "%y%m%d")
    return None

def manage_option_expiry(target_symbol=None):
    """
    Checks all open option positions for the target symbol.
    1. Critical (<= 1 Day): Force Close.
    2. Stale (<= 3 Days): Decay Exit (2/3 time rule).
    """
    if not getattr(config, 'ENABLE_OPTIONS', False):
        return
        
    try:
        positions = trade_client.get_all_positions()
        now = datetime.now()
        
        for pos in positions:
            if pos.asset_class != 'us_option':
                continue
            
            if target_symbol and not pos.symbol.startswith(target_symbol):
                continue
                
            expiry = parse_option_expiry(pos.symbol)
            if not expiry:
                print(f"Could not parse expiry for {pos.symbol}, skipping.")
                continue
                
            # Precision Check: Only close if today IS the expiration day or later
            is_expiry_day = (now.date() >= expiry.date())
            
            if is_expiry_day:
                print(f"⚠️ CRITICAL: {pos.symbol} expires TODAY ({expiry.date()})! Force Closing.")
                trade_client.close_position(pos.symbol)
                
                # Notify
                send_discord_live_trading_notification(
                    signal="expiry_close",
                    symbol=pos.symbol,
                    order_details={
                        "Action": "CLOSE (Market)",
                        "Reason": "Option Expiry",
                        "Expiry": str(expiry.date())
                    },
                    confidence=100,
                    strategy_bias="NEUTRAL"
                )
            else:
                days_left = (expiry.date() - now.date()).days
                print(f"DEBUG: {pos.symbol} DTE: {days_left} days (Expires: {expiry.date()}) - Safe")

    except Exception as e:
        print(f"Error in manage_option_expiry: {e}")

from alpaca.trading.requests import ReplaceOrderRequest

# ================= STATE MANAGEMENT =================

STATE_FILE = "trade_state.json"  # Default fallback
GLOBAL_SAFETY_FILE = "global_safety.json"

def load_global_safety_state():
    if os.path.exists(GLOBAL_SAFETY_FILE):
        try:
            with open(GLOBAL_SAFETY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"peak_equity": 0.0, "halted": False, "last_loss_time": None}

def save_global_safety_state(state):
    try:
        with open(GLOBAL_SAFETY_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error saving global safety state: {e}")

def update_peak_equity(current_equity):
    state = load_global_safety_state()
    if current_equity > state.get("peak_equity", 0.0):
        state["peak_equity"] = current_equity
        save_global_safety_state(state)
        print(f"📈 New Peak Equity: ${current_equity:.2f}")
    
    # Check for drawdown halt
    max_dd = getattr(config, 'MAX_GLOBAL_DRAWDOWN', 0.15)
    peak = state.get("peak_equity", 0.0)
    if peak > 0:
        dd = (peak - current_equity) / peak
        if dd >= max_dd and not state.get("halted", False):
            state["halted"] = True
            save_global_safety_state(state)
            print(f"🚨 GLOBAL DRAWDOWN CIRCUIT BREAKER HIT ({dd*100:.1f}%)! Trading Halted.")
    return state

def mark_loss():
    state = load_global_safety_state()
    state["last_loss_time"] = datetime.now(timezone.utc).isoformat()
    save_global_safety_state(state)
    cool_down_minutes = getattr(config, 'COOL_DOWN_MINUTES', 60)
    print(f"🕒 Post-Loss Cool-down Triggered ({cool_down_minutes} min).")

def get_state_file_path(symbol=None):
    """Returns the state file path, prioritized by --state-file then symbol-specific."""
    if hasattr(args, 'state_file') and args.state_file:
        return args.state_file
    if symbol:
        return f"trade_state_{symbol}.json"
    return STATE_FILE

def load_trade_state(symbol=None):
    file_path = get_state_file_path(symbol)
    if os.path.exists(file_path):
        for attempt in range(3):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                if attempt < 2:
                    time.sleep(0.5)
                    continue
                print(f"⚠️ Error loading state from {file_path}: {e}")
    return {}

def save_trade_state(state, symbol=None):
    file_path = get_state_file_path(symbol)
    for attempt in range(3):
        try:
            with open(file_path, "w") as f:
                json.dump(state, f, indent=4)
            return
        except IOError as e:
            if attempt < 2:
                time.sleep(0.5)
                continue
            print(f"⚠️ Error saving state to {file_path}: {e}")

def manage_trade_updates(target_symbol=None):
    """
    Monitors active trades and adjusts Stop Losses.
    1. Stocks: Stepped Trailing Stop (10% increments).
    2. Options: Hybrid Trailing (BE at 10%, Lock 10% at 20%, Lock 20% at 30%).
    """
    try:
        positions = trade_client.get_all_positions()
        for pos in positions:
            symbol = pos.symbol
            if target_symbol and not symbol.startswith(target_symbol):
                continue
            # Skip if we can't determine direction clearly (assuming Long for simplicity or checking side)
            # Both Stock Long and Option Long (Call/Put) have side='long' in Position object typically
            is_long = pos.side == 'long'
            if not is_long: continue # Skip short management for MVP simplicity

            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            pl_pct = float(pos.unrealized_plpc)
            
            # --- OPTIONS RISK MANAGEMENT (Manual) ---
            if pos.asset_class == 'us_option':
                state = load_trade_state(symbol=symbol)
                symbol_state = state.get(symbol, {"virtual_stop": -0.20}) # Default -20% SL
                virtual_stop = symbol_state.get("virtual_stop", -0.20)
                
                # Hard TP Threshold
                OPTION_TP = 0.50
                
                # 1. Check Exit Triggers
                if pl_pct <= virtual_stop:
                    reason = "VIRTUAL STOP" if virtual_stop > -0.20 else "STOP LOSS"
                    print(f"🛑 OPTION {reason} HIT: {symbol} at {pl_pct*100:.1f}% (Stop: {virtual_stop*100:.1f}%). Closing.")
                    if pl_pct < 0:
                        mark_loss()
                    close_result = trade_client.close_position(symbol)
                    _log_close_submission(symbol, close_result, f"OPTION {reason}")
                    
                    # Notify
                    send_discord_live_trading_notification(
                        signal=f"option_{reason.lower().replace(' ', '_')}_hit",
                        symbol=symbol,
                        order_details={
                            "Action": "CLOSE (Market)",
                            "PnL": f"{pl_pct*100:.1f}%",
                            "Stop": f"{virtual_stop*100:.1f}%",
                            "Type": reason
                        },
                        confidence=0,
                        strategy_bias="NEUTRAL"
                    )
                    # Cleanup state
                    if symbol in state:
                        del state[symbol]
                        save_trade_state(state, symbol=symbol)
                    continue
                elif pl_pct >= OPTION_TP:
                    print(f"🎯 OPTION TAKE PROFIT HIT: {symbol} at {pl_pct*100:.1f}%. Closing.")
                    close_result = trade_client.close_position(symbol)
                    _log_close_submission(symbol, close_result, "OPTION TAKE PROFIT")
                    
                    # Notify
                    send_discord_live_trading_notification(
                        signal="option_tp_hit",
                        symbol=symbol,
                        order_details={
                            "Action": "CLOSE (Market)",
                            "PnL": f"{pl_pct*100:.1f}%",
                            "Type": "Take Profit"
                        },
                        confidence=0,
                        strategy_bias="NEUTRAL"
                    )
                    if symbol in state:
                        del state[symbol]
                        save_trade_state(state, symbol=symbol)
                    continue
                
                # 2. Update Virtual Stop Thresholds (Hybrid Trailing)
                updated = False
                
                # Hybrid Strategy: +10% BE, +20% -> +10%, +30% -> +20%, +40% -> +30%
                if pl_pct >= 0.40 and virtual_stop < 0.30:
                    virtual_stop = 0.30
                    updated = True
                    print(f"💰 OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to +30%.")
                elif pl_pct >= 0.30 and virtual_stop < 0.20:
                    virtual_stop = 0.20
                    updated = True
                    print(f"💰 OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to +20%.")
                elif pl_pct >= 0.20 and virtual_stop < 0.10:
                    virtual_stop = 0.10
                    updated = True
                    print(f"💰 OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to +10%.")
                elif pl_pct >= 0.10 and virtual_stop < 0.0:
                    virtual_stop = 0.0
                    updated = True
                    print(f"🛡️ OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to BE (0%).")
                
                if updated:
                    state[symbol] = {"virtual_stop": virtual_stop}
                    save_trade_state(state, symbol=symbol)
                
                continue
            
            # --- STEPPED TRAILING STOP LOGIC (STOCKS) ---
            # 1. Fetch current stop-loss order if it exists
            stop_order = None
            current_stop = 0.0
            try:
                orders = trade_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbol=symbol))
                for o in orders:
                    if o.type == 'stop' or o.type == 'stop_limit':
                        stop_order = o
                        current_stop = float(o.stop_price)
                        break
            except Exception as e:
                print(f"DEBUG: Could not fetch stop order for {symbol}: {e}")

            if not stop_order:
                continue

            # 2. Stepped Trailing Update
            # Break Even if Profit > 10%
            # Lock 10% if Profit > 20%
            # Lock 20% if Profit > 30%
            # ... and so on (Step 10%)
            
            STEP_SIZE = 0.10
            new_stop = None
            
            if pl_pct >= STEP_SIZE:
                # Calculate the milestone we have passed
                milestone = int(pl_pct * 10) / 10.0
                lock_pct = milestone - 0.10
                
                target_stop = entry_price * (1 + lock_pct)
                
                # Only update if the new target is meaningfully higher than current stop
                if target_stop > current_stop + 0.01:
                    new_stop = target_stop
                    if lock_pct <= 0.001: 
                         print(f"🛡️ BREAK EVEN: {symbol} is up {pl_pct*100:.1f}%. Moving SL to Entry {new_stop:.2f}")
                    else:
                         print(f"💰 PROFIT LOCK: {symbol} is up {pl_pct*100:.1f}%. Moving SL to {new_stop:.2f} (+{lock_pct*100:.0f}%)")

            # Apply Update
            if new_stop:
                try:
                    # We use replace_order
                    # Note: For Bracket orders, replacing the Stop Leg is supported.
                    replace_req = ReplaceOrderRequest(
                         stop_price=round(new_stop, 2)
                    )
                    trade_client.replace_order(stop_order.id, replace_req)
                    print(f"✅ Stop Loss Updated for {symbol}")
                except Exception as e:
                    # Check if the stop was actually hit (e.g. order filled/cancelled)
                    # This is a bit complex in live, usually handled by events, but here
                    # if the replace fails because the order is gone, it might have been hit.
                    pass
                    print(f"❌ Failed to update Stop Loss for {symbol}: {e}")

    except Exception as e:
        print(f"Error in manage_trade_updates: {e}")
    
    # Final Cleanup: Remove state for any symbols we no longer hold
    try:
        current_positions = trade_client.get_all_positions()
        held_symbols = {p.symbol for p in current_positions}
        state = load_trade_state(symbol=target_symbol)
        state_symbols = list(state.keys())
        cleaned = False
        for s in state_symbols:
            entry = state.get(s, {})
            # Only cleanup entries that represent active position-tracking state.
            # Daily cap bookkeeping (e.g. {"last_trade_date", "daily_trade_count"})
            # should not be treated as an externally closed position.
            is_position_tracking_entry = isinstance(entry, dict) and ("virtual_stop" in entry)
            if not is_position_tracking_entry:
                continue

            if s not in held_symbols:
                # This was likely closed externally (e.g. Stop Loss hit on Alpaca)
                print(f"📡 EXTERNAL CLOSE DETECTED: {s} is no longer in positions. Cleaning state.")
                
                # Notify
                send_discord_live_trading_notification(
                    signal="external_stop_detected",
                    symbol=s,
                    order_details={
                        "Action": "CLEANUP",
                        "Reason": "Position closed externally (Automated Stop-Loss?)",
                        "System": "Alpaca Core"
                    },
                    confidence=0,
                    strategy_bias="NEUTRAL"
                )
                
                del state[s]
                cleaned = True
        if cleaned:
            save_trade_state(state, symbol=target_symbol)
    except Exception:
        pass

# ================= SESSION MANAGEMENT =================

class TradingSession:
    """Manages trading session state and statistics."""
    
    def __init__(self, duration_hours=None):
        self.start_time = datetime.now()
        self.duration_hours = duration_hours
        self.trades_executed = 0
        self.should_stop = False
        
        # Enhanced tracking
        self.opening_equity = None
        self.opening_positions = []
        self.closed_trades = []  # List of dicts with {symbol, pnl, win}
        self.day_starting_equity = None
        self.daily_loss_limit_hit = False
        
        # Capture opening state
        try:
            account = trade_client.get_account()
            self.opening_equity = float(account.equity)
            self.day_starting_equity = float(account.equity) # Initialize daily starting equity
            
            # Capture opening positions
            positions = trade_client.get_all_positions()
            for pos in positions:
                self.opening_positions.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': pos.side.value,
                    'entry_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl)
                })
        except Exception as e:
            print(f"⚠️ Could not capture opening state: {e}")
            self.opening_equity = 0.0
        
    def record_trade(self):
        """Record that a trade was executed."""
        self.trades_executed += 1
    
    def record_closed_trade(self, symbol, pnl):
        """Record a closed trade with its P&L."""
        self.closed_trades.append({
            'symbol': symbol,
            'pnl': pnl,
            'win': pnl > 0
        })

    def _get_closed_trades_today_stats(self):
        """
        Reconstruct today's closed trade outcomes from filled closed orders.
        Uses FIFO matching of buy->sell lots per symbol/asset_class.
        """
        ET = ZoneInfo("US/Eastern")
        now_utc = datetime.now(timezone.utc)
        start_today_et = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
        start_today_utc = start_today_et.astimezone(timezone.utc)

        try:
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500,
                after=start_today_utc,
                until=now_utc,
                nested=True,
            )
            orders = trade_client.get_orders(req)
        except Exception:
            return {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "breakeven": 0,
                "realized_pnl": 0.0,
            }

        # Seed lots from opening positions so sells today can close previously-opened positions.
        lots = {}
        for pos in self.opening_positions:
            symbol = pos.get('symbol', '')
            is_option = bool(re.match(r'^[A-Z]+\d{6}[CP]\d{8}$', symbol))
            key = (symbol, 'us_option' if is_option else 'us_equity')
            qty = abs(float(pos.get('qty', 0.0)))
            if qty <= 0:
                continue
            lots.setdefault(key, []).append({
                "qty": qty,
                "price": float(pos.get('entry_price', 0.0)),
            })

        realized = []
        # Process in fill-time order for deterministic FIFO matching.
        filled_orders = sorted(
            [
                o for o in orders
                if getattr(o, 'filled_at', None) is not None
                and float(getattr(o, 'filled_qty', 0) or 0) > 0
                and float(getattr(o, 'filled_avg_price', 0) or 0) > 0
            ],
            key=lambda o: o.filled_at
        )

        for o in filled_orders:
            symbol = getattr(o, 'symbol', None)
            asset_class_raw = str(getattr(o, 'asset_class', '')).lower()
            asset_class = 'us_option' if 'option' in asset_class_raw else 'us_equity'
            side = str(getattr(o, 'side', '')).lower()
            qty = abs(float(getattr(o, 'filled_qty', 0) or 0))
            px = float(getattr(o, 'filled_avg_price', 0) or 0)
            if not symbol or qty <= 0 or px <= 0:
                continue

            key = (symbol, asset_class)
            multiplier = 100.0 if asset_class == 'us_option' else 1.0

            if side == 'buy':
                lots.setdefault(key, []).append({"qty": qty, "price": px})
                continue

            if side != 'sell':
                continue

            remaining = qty
            sell_pnl = 0.0
            matched = 0.0
            queue = lots.setdefault(key, [])
            while remaining > 1e-9 and queue:
                lot = queue[0]
                take = min(remaining, lot["qty"])
                sell_pnl += (px - lot["price"]) * take * multiplier
                lot["qty"] -= take
                remaining -= take
                matched += take
                if lot["qty"] <= 1e-9:
                    queue.pop(0)

            if matched > 0:
                realized.append(sell_pnl)

        wins = sum(1 for p in realized if p > 0.01)
        losses = sum(1 for p in realized if p < -0.01)
        breakeven = sum(1 for p in realized if -0.01 <= p <= 0.01)
        return {
            "count": len(realized),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "realized_pnl": sum(realized),
        }
        
    def should_continue(self):
        """Check if the session should continue running."""
        if self.should_stop:
            return False
            
        # Check duration limit
        if self.duration_hours is not None:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours >= self.duration_hours:
                return False
                
        return True
    
    def get_summary(self):
        """Get session summary statistics."""
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        
        # Get current account state
        closing_equity = 0.0
        account_equity_valid = False
        closing_positions = []
        try:
            account = trade_client.get_account()
            closing_equity = float(account.equity)
            account_equity_valid = closing_equity > 0
            
            positions = trade_client.get_all_positions()
            for pos in positions:
                closing_positions.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': pos.side.value,
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                })
        except Exception as e:
            print(f"⚠️ Could not fetch closing state: {e}")
        
        # Calculate statistics
        closed_today = self._get_closed_trades_today_stats()
        opening_unrealized = sum(float(p.get('unrealized_pl', 0.0)) for p in self.opening_positions)
        closing_unrealized = sum(float(p.get('unrealized_pl', 0.0)) for p in closing_positions)

        # Fallback when broker equity is missing/invalid (avoid bogus -100% report).
        if self.opening_equity and not account_equity_valid:
            closing_equity = self.opening_equity + closed_today["realized_pnl"] + (closing_unrealized - opening_unrealized)

        total_pnl = closing_equity - self.opening_equity if self.opening_equity else 0.0
        pnl_pct = (total_pnl / self.opening_equity * 100) if self.opening_equity else 0.0
        
        # Win rate from closed trades
        wins = sum(1 for t in self.closed_trades if t['win'])
        losses = len(self.closed_trades) - wins
        win_rate = (wins / len(self.closed_trades) * 100) if self.closed_trades else 0.0
        
        # P&L stats from closed trades
        closed_pnl = sum(t['pnl'] for t in self.closed_trades)
        winning_trades = [t['pnl'] for t in self.closed_trades if t['win']]
        losing_trades = [t['pnl'] for t in self.closed_trades if not t['win']]
        
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        # Build summary
        summary = [
            "=" * 60,
            "SESSION SUMMARY",
            "=" * 60,
            f"Start Time:      {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration:        {hours}h {minutes}m",
            "",
            "ACCOUNT PERFORMANCE",
            "-" * 60,
            f"Opening Equity:  ${self.opening_equity:,.2f}",
            f"Closing Equity:  ${closing_equity:,.2f}",
            f"Total P&L:       ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)",
            "",
            "TRADING ACTIVITY",
            "-" * 60,
            f"Trades Executed: {self.trades_executed}",
            f"Trades Closed:   {closed_today['count']}",
            f"Today W/L/BE:    {closed_today['wins']} / {closed_today['losses']} / {closed_today['breakeven']}",
        ]
        
        if self.closed_trades:
            summary.extend([
                f"Win Rate:        {win_rate:.1f}% ({wins}W / {losses}L)",
                "",
                "CLOSED TRADES P&L",
                "-" * 60,
                f"Total Realized:  ${closed_pnl:+,.2f}",
                f"Average Win:     ${avg_win:+,.2f}",
                f"Average Loss:    ${avg_loss:+,.2f}",
                f"Largest Win:     ${largest_win:+,.2f}",
                f"Largest Loss:    ${largest_loss:+,.2f}",
            ])
        
        # Opening positions
        if self.opening_positions:
            summary.extend([
                "",
                "OPENING POSITIONS",
                "-" * 60,
            ])
            for pos in self.opening_positions:
                summary.append(
                    f"  {pos['symbol']:20s} {pos['side']:5s} {pos['qty']:>8.0f} @ ${pos['entry_price']:>8.2f} "
                    f"| Value: ${pos['market_value']:>10,.2f} | P&L: ${pos['unrealized_pl']:>+10,.2f}"
                )
        
        # Closing positions
        if closing_positions:
            summary.extend([
                "",
                "CLOSING POSITIONS",
                "-" * 60,
            ])
            for pos in closing_positions:
                summary.append(
                    f"  {pos['symbol']:20s} {pos['side']:5s} {pos['qty']:>8.0f} @ ${pos['entry_price']:>8.2f} "
                    f"| Current: ${pos['current_price']:>8.2f} | P&L: ${pos['unrealized_pl']:>+10,.2f} ({pos['unrealized_plpc']*100:+.2f}%)"
                )
        
        # Limits
        if self.duration_hours:
            summary.extend([
                "",
                "SESSION LIMITS",
                "-" * 60,
            ])
            if self.duration_hours:
                summary.append(f"Duration Limit:  {self.duration_hours} hours")
        
        summary.append("=" * 60)
        return "\n".join(summary)
    
    def request_stop(self):
        """Request session to stop gracefully."""
        self.should_stop = True
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print("\n\n🛑 Shutdown signal received. Stopping session gracefully...")
            self.request_stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

def interruptible_sleep(seconds, session):
    """Sleeps for the specified time, but returns early if session should stop."""
    end_time = time.time() + seconds
    while time.time() < end_time and session.should_continue():
        # Sleep in small increments to stay responsive
        remaining = end_time - time.time()
        time.sleep(min(1.0, remaining))

# ================= MAIN LOOP =================

def _load_runtime_json_config(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("❌ Error: runtime config JSON root must be an object.")
            sys.exit(1)
        return data
    except FileNotFoundError:
        print(f"❌ Error: config file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: invalid JSON config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading config file {path}: {e}")
        sys.exit(1)


def _cfg_value(cfg, *keys, default=None):
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SMC trading bot.")
    parser.add_argument("symbol", nargs="?", default=None, help="Symbol to trade (default: SPY)")
    parser.add_argument("--config", type=str, help="Path to runtime JSON config file")
    parser.add_argument("--options", action="store_true", help="Enable options trading (overrides config)")
    parser.add_argument("--cap", type=int, metavar="N", help="Daily trade cap: -1 for unlimited, 0 for no trades, positive for max trades per day (default: 5)")
    parser.add_argument("--session-duration", type=float, help="Session duration in hours (runs indefinitely if not specified)")
    parser.add_argument("--option-allocation", type=float, help="Fraction of equity allocated to options (0.0 to 1.0). Stock allocation is 1 - option-allocation.")
    parser.add_argument("--max-option-contracts", type=int, help="Maximum option contracts per trade (-1 for no limit)")
    parser.add_argument("--state-file", type=str, help="Override state file path (default: trade_state_{symbol}.json)")
    parser.add_argument("--min-conf", type=str, choices=['all', 'low', 'medium', 'high'], default=None, help="Minimum confidence level to take a signal (default: all)")
    
    args = parser.parse_args()
    runtime_cfg = _load_runtime_json_config(args.config)

    target_symbol = (args.symbol or str(_cfg_value(runtime_cfg, "symbol", default=SYMBOL))).upper()
    enable_options = bool(_cfg_value(runtime_cfg, "options", "enable_options", default=config.ENABLE_OPTIONS))
    if args.options:
        print("Overriding ENABLE_OPTIONS to True from command line.")
        enable_options = True
    config.ENABLE_OPTIONS = enable_options

    # Resolve JSON/CLI settings with CLI taking precedence.
    option_allocation = args.option_allocation if args.option_allocation is not None else _cfg_value(runtime_cfg, "option_allocation", "option-allocation")
    if option_allocation is None:
        option_allocation = float(getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20))
    option_allocation = float(option_allocation)
    if option_allocation < 0 or option_allocation > 1:
        print("❌ Error: option allocation must be within [0.0, 1.0].")
        sys.exit(1)
    stock_allocation = 1.0 - option_allocation
    max_option_contracts = args.max_option_contracts if args.max_option_contracts is not None else _cfg_value(runtime_cfg, "max_option_contracts", "max-option-contracts", default=getattr(config, "MAX_OPTION_CONTRACTS", -1))
    max_option_contracts = int(max_option_contracts)
    if max_option_contracts != -1 and max_option_contracts < 1:
        print("❌ Error: max option contracts must be -1 (unlimited) or >= 1.")
        sys.exit(1)

    daily_cap = args.cap if args.cap is not None else _cfg_value(runtime_cfg, "cap", "daily_cap", default=5)
    session_duration = args.session_duration if args.session_duration is not None else _cfg_value(runtime_cfg, "session_duration", "session-duration")
    state_file = args.state_file if args.state_file is not None else _cfg_value(runtime_cfg, "state_file", "state-file")
    min_conf = args.min_conf if args.min_conf is not None else _cfg_value(runtime_cfg, "min_conf", "min-conf", default="all")
    if min_conf not in {"all", "low", "medium", "high"}:
        print(f"❌ Error: invalid min_conf value '{min_conf}'. Use one of: all, low, medium, high.")
        sys.exit(1)
    args.min_conf = min_conf
    if state_file:
        args.state_file = str(state_file)
    
    # Handle daily trade cap
    if daily_cap == 0:
        print("⚠️  WARNING: Daily trade cap is set to 0. No trades will be executed.")
        response = input("Do you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Exiting...")
            sys.exit(0)
    
    # Initialize session
    session = TradingSession(duration_hours=session_duration)
    
    # Setup signal handlers for graceful shutdown
    session.setup_signal_handlers()
    
    print(f"Starting SMC Bot for {target_symbol} (Options: {config.ENABLE_OPTIONS})...")
    if daily_cap == -1:
        print(f"Daily Trade Cap: Unlimited")
    elif daily_cap == 0:
        print(f"Daily Trade Cap: 0 (No trades allowed)")
    else:
        print(f"Daily Trade Cap: {daily_cap} trades per day")
    print(f"Option Allocation: {option_allocation:.2f} | Stock Allocation: {stock_allocation:.2f}")
    print(f"Max Option Contracts/Trade: {'Unlimited' if max_option_contracts == -1 else max_option_contracts}")
    if session_duration:
        print(f"Session Duration: {session_duration} hours")
    print(f"Min Confidence Filter: {min_conf.upper()}")
    
    # Map confidence choices to numeric thresholds
    CONF_THRESHOLDS = {
        'all': 0,
        'low': 20,
        'medium': 60,
        'high': 80
    }
    min_conf_threshold = CONF_THRESHOLDS.get(min_conf, 0)
    
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")

    try:
        while session.should_continue():
            try:
                # 1. Check if market is open FIRST
                clock = trade_client.get_clock()
                timestamp_et = clock.timestamp.astimezone(ET)
                
                if not clock.is_open:
                    next_open_et = clock.next_open.astimezone(ET)
                    print(f"🕒 Market is CLOSED. Next open: {next_open_et.strftime('%Y-%m-%d %H:%M:%S')}. Waiting 15 minutes...")
                    interruptible_sleep(900, session) # Wait 15 minutes
                    continue

                # 2. Daily Loss Limit Check (3%)
                account = trade_client.get_account()
                current_equity = float(account.equity)
                
                # Update Peak Equity and Check Drawdown Halt
                update_peak_equity(current_equity)
                
                current_date = timestamp_et.date()
                
                # Reset daily starting equity at start of new trading day
                if session.day_starting_equity is None or (hasattr(session, 'last_trading_date') and session.last_trading_date != current_date):
                    session.day_starting_equity = current_equity
                    session.daily_loss_limit_hit = False
                    session.last_trading_date = current_date
                    print(f"🔄 New trading day. Starting equity: ${session.day_starting_equity:,.2f}")
                
                # Check daily loss limit
                daily_pnl_pct = (current_equity - session.day_starting_equity) / session.day_starting_equity if session.day_starting_equity > 0 else 0
                if daily_pnl_pct <= -0.03 and not session.daily_loss_limit_hit:
                    print(f"🛑 DAILY LOSS LIMIT HIT (-3%). Current: ${current_equity:,.2f} | Start: ${session.day_starting_equity:,.2f} | Loss: {daily_pnl_pct*100:.2f}%")
                    session.daily_loss_limit_hit = True
                    mark_loss()
                    # Close all positions
                    try:
                        positions = trade_client.get_all_positions()
                        for pos in positions:
                            trade_client.close_position(pos.symbol)
                            print(f"Closed {pos.symbol}")
                            # Notify each liquidation
                            send_discord_live_trading_notification(
                                signal="daily_halt_liquidation",
                                symbol=pos.symbol,
                                order_details={
                                    "Action": "CLOSE (Market)",
                                    "Reason": "Daily Loss Limit Hit (-3%)",
                                    "Equity": current_equity
                                },
                                confidence=0,
                                strategy_bias="NEUTRAL"
                            )
                    except Exception as e:
                        print(f"⚠️ Error closing positions: {e}")
                    session.daily_loss_limit_hit = True
                
                if session.daily_loss_limit_hit:
                    print("⏸️ Daily loss limit active. Waiting until next trading day...")
                    interruptible_sleep(900, session)  # Wait 15 minutes
                    continue

                # 3. Maintenance Tasks (Only run during market hours)
                manage_option_expiry(target_symbol=target_symbol)
                manage_trade_updates(target_symbol=target_symbol)

                # 2. Market is open, look for signals
                timestamp_et = clock.timestamp.astimezone(ET)
                print(f"Analyzing {target_symbol} at {timestamp_et}...")
                res = generate_signal(target_symbol)
                if res and res[0] is not None:
                    sig, conf = res if isinstance(res, tuple) else (res, 0)
                    
                    if conf < min_conf_threshold:
                        print(f"⚠️  Signal {sig.upper()} detected but SKIPPED (Confidence {conf}% < {min_conf.upper()} threshold)")
                        # Wait 1 minute and check again
                        interruptible_sleep(60, session)
                        continue

                    print(f"🚀 Signal detected: {sig.upper()}! Confidence: {conf}%")
                    # Pass daily_cap: -1 = unlimited, 0 = no trades, positive = cap
                    use_cap = (daily_cap != -1)
                    entry_submitted = place_trade(
                        sig, 
                        target_symbol, 
                        confidence=conf,
                        use_daily_cap=use_cap, 
                        daily_cap_value=daily_cap if use_cap else None,
                        option_allocation_override=option_allocation,
                        max_option_contracts_override=max_option_contracts
                    )
                    if entry_submitted:
                        session.record_trade()
                        # After a successful entry, sleep to avoid rapid double-entry.
                        print("Trade placed. Cooling down for 5 minutes...")
                        interruptible_sleep(300, session)
                    else:
                        # No entry was submitted for this signal (skipped/managed-only); re-check normally.
                        interruptible_sleep(60, session)
                else:
                    # No signal, wait 1 minute before checking again
                    interruptible_sleep(60, session)

            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                print("Restarting loop in 60 seconds...")
                interruptible_sleep(60, session)
    except KeyboardInterrupt:
        # If signal handler didn't catch it for some reason
        session.request_stop()
    
    # Session ended
    print("\n")
    print(session.get_summary())
