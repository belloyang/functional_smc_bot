import pandas as pd
import numpy as np
import pandas_ta as ta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
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

from datetime import datetime, timedelta

def get_bars(symbol, timeframe, limit):
    # Calculate a lookback to ensure we have enough data (e.g. 5 days)
    # This prevents the API from defaulting to "today only" which breaks indicators like EMA50
    start_dt = datetime.now() - timedelta(days=5)
    
    # Request a large chunk of data starting from 5 days ago to ensure we capture the most recent data
    # limit=10000 is typically sufficient for 1Min bars over 5 days (~2000 bars)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, limit=10000, start=start_dt)
    bars = data_client.get_stock_bars(req).df
    bars = bars.reset_index()
    
    if bars.empty:
        print("Bars df is empty!")
        return bars

    # Slice the dataframe to return only the requested 'limit' of MOST RECENT bars
    if len(bars) > limit:
        bars = bars.iloc[-limit:]
        
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

# ================= EXECUTION =================

from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType
from datetime import datetime, timedelta

def get_best_option_contract(symbol, signal_type, known_price=None):
    """
    Finds the best option contract for the given signal.
    Criteria:
    - Type: CALL for 'buy', PUT for 'sell'
    - Expiration: 2-7 days out (Weekly)
    - Strike: At-The-Money (closest to current price)
    """
    # Get current price
    try:
        if known_price:
             current_price = known_price
        else:
            current_price_df = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 1)
            if current_price_df.empty:
                print("Could not fetch current price for options.")
                return None
            current_price = current_price_df['close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for options: {e}")
        return None
        
        now = datetime.now()
        # Look for expirations between 2 and 14 days out to ensure we find something
        start_date = now + timedelta(days=2)
        end_date = now + timedelta(days=14)
        
        contract_type = ContractType.CALL if signal_type == "buy" else ContractType.PUT
        
        req = GetOptionContractsRequest(
            underlying_symbol=symbol,
            status=AssetStatus.ACTIVE,
            expiration_date_gte=start_date.date(),
            expiration_date_lte=end_date.date(),
            type=contract_type,
            limit=100 
        )
        
        print(f"DEBUG: Searching options for {symbol} | Type: {contract_type} | Range: {start_date.date()} to {end_date.date()}")
        contracts = trade_client.get_option_contracts(req).option_contracts
        print(f"DEBUG: Found {len(contracts)} raw contracts")
        
        if not contracts:
            print(f"No option contracts found for {symbol}")
            return None
            
        # Filter and Sort
        # 1. Sort by expiration (nearest first)
        # 2. Sort by strike difference to current price
        
        # We want the nearest expiration that isn't today/tomorrow (handled by query date range)
        # Group by expiration? Or just sort all by distance to now, then distance to price.
        
        # Let's pick the closest expiration first
        contracts.sort(key=lambda x: x.expiration_date)
        nearest_expiry = contracts[0].expiration_date
        
        # Filter for this expiry
        expiry_contracts = [c for c in contracts if c.expiration_date == nearest_expiry]
        
        # Find ATM strike
        # Sort by abs(strike - price)
        expiry_contracts.sort(key=lambda c: abs(float(c.strike_price) - current_price))
        
        best_contract = expiry_contracts[0]
        print(f"Selected Contract: {best_contract.symbol} | Strike: {best_contract.strike_price} | Exp: {best_contract.expiration_date}")
        return best_contract

    except Exception as e:
        print(f"Error getting option contract: {e}")
        return None

def get_current_position(symbol):
    """
    Fetches the current position for the given symbol.
    Returns the Position object if found, else None.
    """
    try:
        position = trade_client.get_open_position(symbol)
        return position
    except Exception as e:
        # 404 error typically means no position found, which is fine
        return None

def place_trade(signal, symbol):
    # Get latest price
    # Fix: Use TimeFrame object
    price_df = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 1)
    if price_df.empty:
        print("Could not fetch price for placement.")
        return
    price = price_df['close'].iloc[-1]
    
    # 1. Check for Existing Position
    current_position = get_current_position(symbol)
    qty_held = float(current_position.qty) if current_position else 0
    side_held = current_position.side if current_position else None # OrderSide.BUY (Long) or OrderSide.SELL (Short)
    
    print(f"DEBUG: Current Position for {symbol}: {qty_held} shares (Side: {side_held})")

    # ================= OPTIONS MODE =================
    if getattr(config, 'ENABLE_OPTIONS', False):
        # NOTE: Position handling for Options is complex because we trade contracts (diff symbols), 
        # not the underlying symbol directly.
        # For this MVP, we will only trade if we don't have ANY position in the underlying to keep it simple.
        # A full Option Bot would need to track contract symbols.
        
        # Improvement: Check if we hold any options for this symbol?
        # For now, let's stick to the previous simple logic but slightly safer:
        # If we have a position in the underlying (e.g. assigned shares), don't trade options on top blindly.
        if qty_held != 0:
             print(f"Warning: Holding underlying shares ({qty_held}) while in Options Mode. Skipping new option entries to avoid mix-up.")
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

            # MVP: Trade 1 contract
            qty = 1 
            # Both Call and Put are "BUY" to Open
            side = OrderSide.BUY 
            
            try:
                order = MarketOrderRequest(
                    symbol=trade_symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                trade_client.submit_order(order)
                print(f"✅ OPTION ORDER SUBMITTED: {trade_symbol}")
                return
            except Exception as e:
                print(f"❌ Option Order failed: {e}")
                print("Falling back to share trading...")
        else:
             print("Could not find suitable option contract. Falling back to shares.")

    # ================= STOCK MODE =================
    
    # Logic matrix:
    # BUY Signal:
    #   - No Position: BUY
    #   - Short Position: CLOSE Short (Buy to Cover), then BUY Long? -> MVP: Just Close Short.
    #   - Long Position: HOLD (Don't pyramid)
    
    # SELL Signal:
    #   - No Position: IGNORE (No Shorting allowed per user rule)
    #   - Long Position: SELL ALL (Exit)
    #   - Short Position: HOLD
    
    if signal == "buy":
        if qty_held == 0:
            # Enter Long
            qty = calculate_position_size(price)
            if qty <= 0:
                print("Calculated Quantity is 0, skipping trade.")
                return

            print(f"Placing BUY order for {qty} shares of {symbol} at approx {price}")
            try:
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trade_client.submit_order(order)
                print("✅ BUY ORDER SUBMITTED")
            except Exception as e:
                print(f"❌ Order failed: {e}")
        
        elif side_held == OrderSide.SELL: # We are Short
             print(f"Closing Short Position ({qty_held} shares) due to BUY signal.")
             try:
                # To close a Short, we BUY same qty
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs(qty_held),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trade_client.submit_order(order)
                print("✅ SHORT CLOSE SUBMITTED")
             except Exception as e:
                print(f"❌ Close Short failed: {e}")
        
        else: # We are Long
            print(f"Already Long {qty_held} shares. Ignoring BUY signal (No Pyramiding).")

    elif signal == "sell":
        if qty_held == 0:
            # No Shorting allowed
            print("SELL Signal detected but no position held. Skipping Short Entry (Long-Only Mode).")
            return
            
        elif side_held == OrderSide.BUY: # We are Long
            print(f"Closing Long Position ({qty_held} shares) due to SELL signal.")
            try:
                # To close a Long, we SELL same qty
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs(qty_held),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                trade_client.submit_order(order)
                print("✅ LONG CLOSE SUBMITTED")
            except Exception as e:
                 print(f"❌ Close Long failed: {e}")
                 
        else: # We are Short
            print(f"Already Short {qty_held} shares. Ignoring SELL signal.")

# ================= MAIN LOOP =================

import sys
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SMC trading bot.")
    parser.add_argument("symbol", nargs="?", default=SYMBOL, help="Symbol to trade (default: SPY)")
    parser.add_argument("--options", action="store_true", help="Enable options trading (overrides config)")
    
    args = parser.parse_args()
    target_symbol = args.symbol
    
    if args.options:
        print("Overriding ENABLE_OPTIONS to True from command line.")
        config.ENABLE_OPTIONS = True
        
    print(f"Starting SMC Bot for {target_symbol} (Options: {config.ENABLE_OPTIONS})...")
    
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")

    while True:
        try:
            # 1. Check if market is open
            clock = trade_client.get_clock()
            if not clock.is_open:
                next_open_et = clock.next_open.astimezone(ET)
                print(f"Market is CLOSED. Next open: {next_open_et}. Waiting 15 minutes...")
                time.sleep(900) # Wait 15 minutes
                continue

            # 2. Market is open, look for signals
            timestamp_et = clock.timestamp.astimezone(ET)
            print(f"Analyzing {target_symbol} at {timestamp_et}...")
            sig = generate_signal(target_symbol)
            
            if sig:
                print(f"🚀 Signal detected: {sig.upper()}!")
                place_trade(sig, target_symbol)
                # After a trade, sleep for 5 minutes to avoid rapid double-entry
                print("Trade placed. Cooling down for 5 minutes...")
                time.sleep(300)
            else:
                # No signal, wait 1 minute before checking again
                time.sleep(60)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            print("Restarting loop in 60 seconds...")
            time.sleep(60)