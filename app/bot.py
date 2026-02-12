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
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
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
    df['bear_fvg'] = (df['high'] < df['low'].shift(2))
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

def calculate_confidence(ltf_row, last_htf):
    """Calculates a confidence score between 0-100%."""
    score = 0
    
    # 1. Impulse Intensity (40%)
    if ltf_row.get('avg_body', 0) > 0:
        ratio = ltf_row['body_size'] / ltf_row['avg_body']
        impulse_score = min(40, max(0, (ratio - 1.5) / 1.5 * 40))
        score += impulse_score

    # 2. Volume Confirmation (20%)
    if ltf_row.get('avg_vol', 0) > 0:
        vol_ratio = ltf_row['volume'] / ltf_row['avg_vol']
        vol_score = min(20, max(0, (vol_ratio - 1.0) / 1.0 * 20))
        score += vol_score
        
    # 3. FVG Confluence (20%)
    if ltf_row.get('bull_fvg') or ltf_row.get('bear_fvg'):
        score += 20
        
    # 4. Trend Proximity (20%)
    price = ltf_row['close']
    ema = last_htf.get('ema50')
    if ema:
        dist_pct = abs(price - ema) / ema
        if dist_pct < 0.002:
            score += 20
        elif dist_pct < 0.005:
            score += 10
    
    return int(min(100, score))

def get_confidence_label(score):
    if score >= 80: return "🔥🔥 High"
    if score >= 50: return "📈 Medium"
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
    bias = "bullish" if last_htf['close'] > last_htf['ema50'] else "bearish"
    
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
    
    signal = None
    confidence = 0
    
    if bias == "bullish":
        # Trigger: We just had a bullish impulse (OB creation)
        if last_closed['is_bull_ob_candle']:
             signal = "buy"
             
    elif bias == "bearish":
        if last_closed['is_bear_ob_candle']:
             signal = "sell"

    if signal:
        confidence = calculate_confidence(last_closed, htf.iloc[-1])

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

def calculate_smart_quantity(symbol, price, stop_loss_price, budget_override=None):
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
        
    risk_amount = equity * RISK_PER_TRADE
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
    print(f"DEBUG: Equity: ${equity:.2f} | Risk($): ${risk_amount:.2f} | Entry: {price} | SL: {stop_loss_price} | Dist: {stop_distance:.2f}")
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

def place_trade(signal, symbol, confidence=0, use_daily_cap=True, daily_cap_value=None, stock_budget_override=None, option_budget_override=None):
    # --- TIME FILTER (10:00 AM - 3:30 PM ET) ---
    now_et = datetime.now(ZoneInfo("America/New_York"))
    current_time = now_et.time()
    start_time = datetime.strptime("09:40:00", "%H:%M:%S").time()
    end_time = datetime.strptime("15:55:00", "%H:%M:%S").time()
    
    if current_time < start_time or current_time > end_time:
        print(f"🕒 TIME FILTER: Current time {current_time} is outside the 09:40-15:55 window. Skipping.")
        return

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
        return

    # Get latest price and ample history for Swing Point detection
    # We need enough history to find a swing point (e.g. 50-100 bars)
    price_df = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 100)
    if price_df.empty:
        print("Could not fetch price for placement.")
        return
    
    price = price_df['close'].iloc[-1]
    last_close = price # Current price estimate
    
    # 1. Fetch CURRENT state for the base symbol and all related contracts
    all_positions = []
    try:
        all_positions = trade_client.get_all_positions()
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        return

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
                trade_client.close_position(pos.symbol)
                
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
             return
    else:
        if any_options_held:
             print(f"Warning: Holding option contracts while in Stock Mode. Skipping new stock entries to avoid mix-up.")
             return

    # ================= OPTIONS MODE =================
    if getattr(config, 'ENABLE_OPTIONS', False):

        if same_bias_held:
            print(f"Existing {signal.upper()} option bias detected for {symbol}. Skipping redundant entry.")
            return

        print(f"Options Trading Enabled. Searching for contract for {signal.upper()}...")
        contract = get_best_option_contract(symbol, signal)
        
        if contract:
            trade_symbol = contract.symbol
            # Check if we already hold THIS specific contract
            contract_pos = get_current_position(trade_symbol)
            if contract_pos and float(contract_pos.qty) > 0:
                 print(f"Already hold {trade_symbol}. Holding.")
                 return

            # Fetch Current Option Price to calculate Brackets
            from alpaca.data.requests import OptionLatestQuoteRequest
            try:
                # We need the quote to determine Entry Price approximation
                quote_req = OptionLatestQuoteRequest(symbol_or_symbols=trade_symbol)
                quote = option_data_client.get_option_latest_quote(quote_req)
                
                # Use Ask price as likely entry (buying)
                entry_est = quote[trade_symbol].ask_price
                if entry_est <= 0:
                    print(f"Invalid option ask price {entry_est}. Skipping.")
                    return
                    
                # Calculate Levels (-20% SL, +50% TP)
                # Recommendation: Tighter stops (-20%) often preserve capital better in automated systems.
                sl_price = entry_est * 0.80
                tp_price = entry_est * 1.50
                
                # Round to 2 decimals
                sl_price = round(sl_price, 2)
                tp_price = round(tp_price, 2)
                
                print(f"Options Bracket: Est.Entry: {entry_est} | SL: {sl_price} (-20%) | TP: {tp_price} (+50%)")
                
                # --- GLOBAL OPTION EXPOSURE (Sum of all held premiums) ---
                account = trade_client.get_account()
                equity = float(account.equity)
                if option_budget_override is not None:
                    budget_pct = option_budget_override
                else:
                    budget_pct = getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20)
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
                if option_budget_override is not None:
                     max_ticker_option_pct = option_budget_override
                else:
                     max_ticker_option_pct = getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20)
                
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
                    return
                
                # 3. Risk-Based Position Sizing (Max 2% equity risk per trade)
                current_risk_target = equity * 0.02  # Risk 2% of total equity
                risk_per_contract = entry_est * 0.20 * 100  # 20% SL on premium
                qty_risk = int(current_risk_target // risk_per_contract) if risk_per_contract > 0 else 0
                
                # Still capped by available budget
                qty_cap = int(available_budget // cost_per_contract)
                
                qty = min(qty_risk, qty_cap)
                if qty > 5:
                    print(f"DEBUG: Capping contracts from {qty} to 5.")
                    qty = 5
                    
                if qty < 1:
                    print("⚠️ WARNING: Calculated contracts < 1. Skipping.")
                    return

                print(f"Sizing: Equity ${equity:.2f} | Risk Target ${current_risk_target:.2f} | Risk/Ctr ${risk_per_contract:.2f} | Budget ${available_budget:.2f} | Qty: {qty}")
                
                sl_req = StopLossRequest(stop_price=sl_price)
                tp_req = TakeProfitRequest(limit_price=tp_price)
                
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
                return

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
            qty = calculate_smart_quantity(symbol, price, sl_price, budget_override=stock_budget_override)
            if qty <= 0:
                print("Calculated Quantity is 0, skipping trade.")
                return

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
            except Exception as e:
                print(f"❌ Order failed: {e}")
        
        elif side_held == PositionSide.SHORT: 
             print(f"Closing Short Position ({qty_held} shares) due to BUY signal.")
             try:
                # Safety: Cancel any existing open orders (SL/TP) for this symbol first
                cancel_all_orders_for_symbol(symbol)
                
                order = MarketOrderRequest(symbol=symbol, qty=abs(qty_held), side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                trade_client.submit_order(order)
                print("✅ SHORT CLOSE SUBMITTED")
                
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
        else:
            print(f"Already Long {qty_held} shares. Ignoring BUY signal.")

    elif signal == "sell":
        if qty_held == 0:
            print("SELL Signal detected but no position held. Skipping Short Entry (Long-Only Mode).")
            return
            
        elif side_held == PositionSide.LONG: # We are Long
            print(f"Closing Long Position ({qty_held} shares) due to SELL signal.")
            try:
                # Safety: Cancel any existing open orders (SL/TP) for this symbol first
                cancel_all_orders_for_symbol(symbol)
                
                order = MarketOrderRequest(symbol=symbol, qty=abs(qty_held), side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                trade_client.submit_order(order)
                print("✅ LONG CLOSE SUBMITTED")
                
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
                 
        else:
            print(f"Already Short {qty_held} shares. Ignoring SELL signal.")

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
            else:
                days_left = (expiry.date() - now.date()).days
                print(f"DEBUG: {pos.symbol} DTE: {days_left} days (Expires: {expiry.date()}) - Safe")

    except Exception as e:
        print(f"Error in manage_option_expiry: {e}")

from alpaca.trading.requests import ReplaceOrderRequest

# ================= STATE MANAGEMENT =================

STATE_FILE = "trade_state.json"  # Default fallback

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
                    trade_client.close_position(symbol)
                    # Cleanup state
                    if symbol in state:
                        del state[symbol]
                        save_trade_state(state, symbol=symbol)
                    continue
                elif pl_pct >= OPTION_TP:
                    print(f"🎯 OPTION TAKE PROFIT HIT: {symbol} at {pl_pct*100:.1f}%. Closing.")
                    trade_client.close_position(symbol)
                    if symbol in state:
                        del state[symbol]
                        save_trade_state(state, symbol=symbol)
                    continue
                
                # 2. Update Virtual Stop Thresholds (Hybrid Trailing)
                updated = False
                
                # Hybrid Strategy: +10% BE, +20% -> +10%, +30% -> +20%
                if pl_pct >= 0.30 and virtual_stop < 0.20:
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
                orders = trade_client.get_orders(GetOrdersRequest(status=OrderStatus.OPEN, symbol=symbol))
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
            if s not in held_symbols:
                del state[s]
                cleaned = True
        if cleaned:
            save_trade_state(state, symbol=target_symbol)
    except Exception:
        pass

# ================= SESSION MANAGEMENT =================

class TradingSession:
    """Manages trading session state and statistics."""
    
    def __init__(self, duration_hours=None, max_trades=None):
        self.start_time = datetime.now()
        self.duration_hours = duration_hours
        self.max_trades = max_trades
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
        
    def should_continue(self):
        """Check if the session should continue running."""
        if self.should_stop:
            return False
            
        # Check duration limit
        if self.duration_hours is not None:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours >= self.duration_hours:
                return False
                
        # Check trades limit
        if self.max_trades is not None:
            if self.trades_executed >= self.max_trades:
                return False
                
        return True
    
    def get_summary(self):
        """Get session summary statistics."""
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        
        # Get current account state
        closing_equity = 0.0
        closing_positions = []
        try:
            account = trade_client.get_account()
            closing_equity = float(account.equity)
            
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
            f"Trades Closed:   {len(self.closed_trades)}",
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
        if self.max_trades or self.duration_hours:
            summary.extend([
                "",
                "SESSION LIMITS",
                "-" * 60,
            ])
            if self.max_trades:
                summary.append(f"Trade Limit:     {self.max_trades}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SMC trading bot.")
    parser.add_argument("symbol", nargs="?", default=SYMBOL, help="Symbol to trade (default: SPY)")
    parser.add_argument("--options", action="store_true", help="Enable options trading (overrides config)")
    parser.add_argument("--cap", type=int, metavar="N", help="Daily trade cap: -1 for unlimited, 0 for no trades, positive for max trades per day (default: 5)")
    parser.add_argument("--session-duration", type=float, help="Session duration in hours (runs indefinitely if not specified)")
    parser.add_argument("--max-trades", type=int, help="Maximum number of trades per session (unlimited if not specified)")
    parser.add_argument("--stock-budget", type=float, help="Stock allocation budget override (0.0 to 1.0)")
    parser.add_argument("--option-budget", type=float, help="Option allocation budget override (0.0 to 1.0)")
    parser.add_argument("--state-file", type=str, help="Override state file path (default: trade_state_{symbol}.json)")
    parser.add_argument("--min-conf", type=str, choices=['all', 'low', 'medium', 'high'], default='all', help="Minimum confidence level to take a signal (default: all)")
    
    args = parser.parse_args()
    target_symbol = args.symbol
    
    if args.options:
        print("Overriding ENABLE_OPTIONS to True from command line.")
        config.ENABLE_OPTIONS = True
    
    # Validation for budgets
    if config.ENABLE_OPTIONS and args.stock_budget is not None:
        print("❌ Error: --stock-budget is only valid in Stock Mode. Use --option-budget for Options Mode.")
        sys.exit(1)
    if not config.ENABLE_OPTIONS and args.option_budget is not None:
        print("❌ Error: --option-budget is only valid in Options Mode. Use --stock-budget for Stock Mode.")
        sys.exit(1)
    
    # Handle daily trade cap
    daily_cap = args.cap if args.cap is not None else 5  # Default to 5
    
    if daily_cap == 0:
        print("⚠️  WARNING: Daily trade cap is set to 0. No trades will be executed.")
        response = input("Do you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Exiting...")
            sys.exit(0)
    
    # Initialize session
    session = TradingSession(duration_hours=args.session_duration, max_trades=args.max_trades)
    
    # Setup signal handlers for graceful shutdown
    session.setup_signal_handlers()
    
    print(f"Starting SMC Bot for {target_symbol} (Options: {config.ENABLE_OPTIONS})...")
    if daily_cap == -1:
        print(f"Daily Trade Cap: Unlimited")
    elif daily_cap == 0:
        print(f"Daily Trade Cap: 0 (No trades allowed)")
    else:
        print(f"Daily Trade Cap: {daily_cap} trades per day")
    if args.session_duration:
        print(f"Session Duration: {args.session_duration} hours")
    if args.max_trades:
        print(f"Max Trades: {args.max_trades}")
    print(f"Min Confidence Filter: {args.min_conf.upper()}")
    
    # Map confidence choices to numeric thresholds
    CONF_THRESHOLDS = {
        'all': 0,
        'low': 20,
        'medium': 50,
        'high': 80
    }
    min_conf_threshold = CONF_THRESHOLDS.get(args.min_conf, 0)
    
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
                    print("🛑 Closing all positions and stopping trading for today.")
                    # Close all positions
                    try:
                        positions = trade_client.get_all_positions()
                        for pos in positions:
                            trade_client.close_position(pos.symbol)
                            print(f"Closed {pos.symbol}")
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
                        print(f"⚠️  Signal {sig.upper()} detected but SKIPPED (Confidence {conf}% < {args.min_conf.upper()} threshold)")
                        # Wait 1 minute and check again
                        interruptible_sleep(60, session)
                        continue

                    print(f"🚀 Signal detected: {sig.upper()}! Confidence: {conf}%")
                    # Pass daily_cap: -1 = unlimited, 0 = no trades, positive = cap
                    use_cap = (daily_cap != -1)
                    place_trade(
                        sig, 
                        target_symbol, 
                        confidence=conf,
                        use_daily_cap=use_cap, 
                        daily_cap_value=daily_cap if use_cap else None,
                        stock_budget_override=args.stock_budget,
                        option_budget_override=args.option_budget
                    )
                    session.record_trade()
                    # After a trade, sleep for 5 minutes to avoid rapid double-entry
                    print("Trade placed. Cooling down for 5 minutes...")
                    interruptible_sleep(300, session)
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
