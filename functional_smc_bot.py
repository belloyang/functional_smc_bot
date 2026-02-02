import pandas as pd
import numpy as np
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import alpaca_config as config

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

def get_bars(symbol, timeframe, limit):
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, limit=limit)
    bars = data_client.get_stock_bars(req).df
    bars = bars.reset_index()
    print(f"Bars columns: {bars.columns}")
    if bars.empty:
        print("Bars df is empty!")
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
    # Ensure we work on copies to avoid SettingWithCopy warnings on the original df if needed
    htf = htf.copy()
    htf['ema50'] = ta.ema(htf['close'], length=50)
    
    if len(htf) < 50: return None # Not enough data
    
    # Check the LAST closed candle for bias
    bias = "bullish" if htf['close'].iloc[-1] > htf['ema50'].iloc[-1] else "bearish"
    
    # 2. LTF Analysis
    ltf = ltf.copy()
    # Add indicators
    ltf = detect_fvg(ltf)
    ltf = detect_order_block(ltf)
    
    if len(ltf) < 5: return None
    
    # Get last formed candle for signal check
    last_closed = ltf.iloc[-1]
    
    signal = None
    
    if bias == "bullish":
        # Trigger: We just had a bullish impulse (OB creation)
        if last_closed['is_bull_ob_candle']:
             signal = "buy"
             
    elif bias == "bearish":
        if last_closed['is_bear_ob_candle']:
             signal = "sell"

    return signal

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

    except Exception as e:
        print("Error fetching data:", e)
        return None

    return get_strategy_signal(htf, ltf)

# ================= RISK =================

def calculate_position_size(price):
    if hasattr(config, 'ACCOUNT_BALANCE'):
        balance = config.ACCOUNT_BALANCE
    else:
        try:
            balance = float(trade_client.get_account().equity)
        except:
            balance = 10000 # Fallback
        
    risk_amount = float(balance) * RISK_PER_TRADE
    stop_loss_distance = price * 0.005  # 0.5% stop
    if stop_loss_distance == 0: return 0
    qty = risk_amount / stop_loss_distance
    return int(qty)

# ================= EXECUTION =================

def place_trade(signal, symbol):
    # Get latest price
    price = get_bars(symbol, "1Min", 1)['close'].iloc[-1]
    qty = calculate_position_size(price)
    
    if qty <= 0:
        print("Quantity is 0, skipping trade.")
        return

    side = OrderSide.BUY if signal == "buy" else OrderSide.SELL
    
    print(f"Placing {signal.upper()} order for {qty} shares of {symbol} at approx {price}")
    
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        trade_client.submit_order(order)
        print(f"✅ {signal.upper()} ORDER SUBMITTED")
    except Exception as e:
        print(f"❌ Order failed: {e}")

# ================= MAIN LOOP =================

import sys

if __name__ == "__main__":
    target_symbol = sys.argv[1] if len(sys.argv) > 1 else SYMBOL
    print(f"Analyzing market for {target_symbol}...")
    
    try:
        sig = generate_signal(target_symbol)
        if sig:
            print(f"Signal detected: {sig}")
            place_trade(sig, target_symbol)
        else:
            print("No signal detected.")
    except Exception as e:
        print(f"An error occurred: {e}")