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

def place_trade(signal, symbol):
    # Get latest price
    # Fix: Use TimeFrame object
    price_df = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 1)
    if price_df.empty:
        print("Could not fetch price for placement.")
        return
    price = price_df['close'].iloc[-1]
    
    # Check if Options Trading is Enabled
    if getattr(config, 'ENABLE_OPTIONS', False):
        print(f"Options Trading Enabled. Searching for contract for {signal.upper()}...")
        contract = get_best_option_contract(symbol, signal)
        
        if contract:
            # For Options, the symbol to trade IS the contract symbol
            trade_symbol = contract.symbol
            # MVP: Trade 1 contract
            qty = 1 
            print(f"Placing OPTION order: {signal.upper()} {qty}x {trade_symbol}")
            
            # Note: For options, we typically BUY to Open.
            # If signal is 'buy' (bullish) -> Buy Call
            # If signal is 'sell' (bearish) -> Buy Put (Bot logic generates 'sell' signal for bearish bias)
            # So in both cases side is BUY? 
            # WAIT. The bot generates 'sell' signal. 
            # get_best_option_contract handles the type (PUT).
            # So we just need to BUY the contract (Long Call or Long Put).
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

    # FALLBACK / STANDARD SHARE TRADING
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
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SMC trading bot.")
    parser.add_argument("symbol", nargs="?", default=SYMBOL, help="Symbol to trade (default: SPY)")
    parser.add_argument("--options", action="store_true", help="Enable options trading (overrides config)")
    
    args = parser.parse_args()
    target_symbol = args.symbol
    
    if args.options:
        print("Overriding ENABLE_OPTIONS to True from command line.")
        config.ENABLE_OPTIONS = True
        
    print(f"Analyzing market for {target_symbol} (Options: {config.ENABLE_OPTIONS})...")
    
    try:
        sig = generate_signal(target_symbol)
        if sig:
            print(f"Signal detected: {sig}")
            place_trade(sig, target_symbol)
        else:
            print("No signal detected.")
    except Exception as e:
        print(f"An error occurred: {e}")