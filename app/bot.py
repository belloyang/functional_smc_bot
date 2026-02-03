import pandas as pd
import numpy as np
import pandas_ta as ta
import re
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide, OrderStatus, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, StopLossRequest, TakeProfitRequest
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

from alpaca.trading.requests import StopLossRequest, TakeProfitRequest

def calculate_smart_quantity(price, stop_loss_price):
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
    
    # 2. Buying Power Cap
    max_cost = buying_power * 0.95
    qty_bp = max_cost / price
    
    # Final
    qty = int(min(qty_risk, qty_bp))
    
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

def place_trade(signal, symbol):
    # Get latest price and ample history for Swing Point detection
    # We need enough history to find a swing point (e.g. 50-100 bars)
    price_df = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 100)
    if price_df.empty:
        print("Could not fetch price for placement.")
        return
    
    price = price_df['close'].iloc[-1]
    last_close = price # Current price estimate
    
    # 1. Check for Existing Position
    current_position = get_current_position(symbol)
    qty_held = float(current_position.qty) if current_position else 0
    side_held = current_position.side if current_position else None 
    
    print(f"DEBUG: Current Position for {symbol}: {qty_held} shares (Side: {side_held})")


    # 1. Global Position Cleanup (Cross-Asset Signal Flip)
    any_options_held = False
    try:
        all_positions = trade_client.get_all_positions()
        for pos in all_positions:
            if not pos.symbol.startswith(symbol): continue
            
            # Identify if it's the underlying or an option
            is_stock = (pos.asset_class.value == 'us_equity' and pos.symbol == symbol)
            is_option = (pos.asset_class.value == 'us_option')
            
            if is_option: any_options_held = True

            # Identify Bias
            is_bullish = False
            if is_stock:
                is_bullish = (pos.side == PositionSide.LONG)
            elif is_option:
                # Parse for C/P
                m = re.search(r'\d{6}([CP])\d{8}', pos.symbol)
                is_bullish = (m and m.group(1) == 'C')

            # Conflict Detection: Close if signal is opposite of current holding bias
            if (signal == "buy" and not is_bullish) or (signal == "sell" and is_bullish):
                print(f"🔄 CROSS-ASSET CLEANUP: Closing {pos.symbol} bias conflict with {signal.upper()} signal.")
                try:
                    cancel_all_orders_for_symbol(pos.symbol)
                    trade_client.close_position(pos.symbol)
                    if is_stock: 
                        qty_held = 0
                        side_held = None
                    if is_option:
                        any_options_held = False
                except Exception as close_err:
                    print(f"❌ Cleanup failed for {pos.symbol}: {close_err}")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

    # 2. Mix-up Guards (Prevent holding both Stock and Options for same symbol)
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
                    
                # Calculate Levels (-20% SL, +40% TP)
                # Recommendation: Tighter stops (-20%) often preserve capital better in automated systems.
                sl_price = entry_est * 0.80
                tp_price = entry_est * 1.40
                
                # Round to 2 decimals
                sl_price = round(sl_price, 2)
                tp_price = round(tp_price, 2)
                
                print(f"Options Bracket: Est.Entry: {entry_est} | SL: {sl_price} (-20%) | TP: {tp_price} (+40%)")
                
                qty = 1 # MVP size
                
                sl_req = StopLossRequest(stop_price=sl_price)
                tp_req = TakeProfitRequest(limit_price=tp_price)
                
                order = MarketOrderRequest(
                    symbol=trade_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    stop_loss=sl_req,
                    take_profit=tp_req
                )
                
                trade_client.submit_order(order)
                print(f"✅ OPTION BRACKET SUBMITTED: {trade_symbol}")
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
                
            # 1.2 Determine Take Profit (1:2 Risk Reward)
            risk_dist = price - sl_price
            tp_price = price + (risk_dist * 2.0)
            
            # 1.3 Calculate Quantity
            qty = calculate_smart_quantity(price, sl_price)
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
            except Exception as e:
                 print(f"❌ Close Long failed: {e}")
                 
        else:
            print(f"Already Short {qty_held} shares. Ignoring SELL signal.")


import re

def parse_option_expiry(symbol):
    # Regex to capture YYMMDD from SPY251219C00500000
    # Format: Root(up to 6 chars) + YYMMDD + Type(C/P) + Strike
    match = re.search(r'[A-Z]+(\d{6})[CP]', symbol)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, "%y%m%d")
    return None

def manage_option_expiry():
    """
    Checks all open option positions.
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
                
            expiry = parse_option_expiry(pos.symbol)
            if not expiry:
                print(f"Could not parse expiry for {pos.symbol}, skipping.")
                continue
                
            dte = (expiry - now).days
            
            # Logic:
            # expiry is e.g. 2025-12-19 00:00:00.
            # now is 2025-12-17 15:00:00.
            # delta is ~1.4 days. .days returns 1.
            
            if dte <= 1:
                print(f"⚠️ CRITICAL: {pos.symbol} expires in {dte} days! Force Closing.")
                trade_client.close_position(pos.symbol)
            elif dte <= 3:
                print(f"⚠️ THETA DECAY: {pos.symbol} has {dte} days left (Stale). Closing to preserve value.")
                trade_client.close_position(pos.symbol)
            else:
                # DEBUG log only occasionally?
                # print(f"DEBUG: {pos.symbol} DTE: {dte} (Safe)")
                pass

    except Exception as e:
        print(f"Error in manage_option_expiry: {e}")

from alpaca.trading.requests import ReplaceOrderRequest

def manage_trade_updates():
    """
    Monitors active trades and adjusts Stop Losses.
    1. Break-Even: If Profit > 15%, move SL to Entry.
    2. Profit Lock: If Profit > 30%, move SL to +15% Profit.
    """
    try:
        positions = trade_client.get_all_positions()
        for pos in positions:
            symbol = pos.symbol
            # Skip if we can't determine direction clearly (assuming Long for simplicity or checking side)
            # Both Stock Long and Option Long (Call/Put) have side='long' in Position object typically
            is_long = pos.side == 'long'
            if not is_long: continue # Skip short management for MVP simplicity

            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            pl_pct = float(pos.unrealized_plpc)
            
            # Thresholds
            BE_THRESHOLD = 0.15
            LOCK_THRESHOLD = 0.30
            
            if pl_pct < BE_THRESHOLD:
                continue
                
            # Find the existing Stop Loss Order
            # Warning: This finds ANY open Stop/StopLimit sell order for this symbol.
            # In a complex bot with multiple positions per symbol, this is risky. 
            # But for this bot (one pos per symbol), it is safe.
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import OrderStatus, QueryOrderStatus
            
            orders_req = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbol=symbol,
                side=OrderSide.SELL # Assuming we are closing a Long
            )
            open_orders = trade_client.get_orders(orders_req)
            
            stop_order = None
            for o in open_orders:
                # We look for STOP or STOP_LIMIT orders
                if o.order_type in ['stop', 'stop_limit']:
                    stop_order = o
                    break
            
            if not stop_order:
                continue
                
            current_stop = float(stop_order.stop_price) if stop_order.stop_price else 0.0
            
            # Logic: Move Stop UP only
            new_stop = None
            
            # 2. Profit Lock (>30% gain -> Lock 15%)
            if pl_pct > LOCK_THRESHOLD:
                target_stop = entry_price * 1.15
                if current_stop < target_stop:
                    new_stop = target_stop
                    print(f"💰 PROFIT LOCK: {symbol} is up {pl_pct*100:.1f}%. Moving SL to {new_stop:.2f} (+15%)")
            
            # 1. Break Even (>15% gain -> Move to Entry)
            elif pl_pct > BE_THRESHOLD:
                target_stop = entry_price
                # Give it a tiny buffer so fees don't eat us? Entry * 1.005?
                # Let's stick to raw entry for now.
                if current_stop < target_stop:
                    new_stop = target_stop
                    print(f"🛡️ BREAK EVEN: {symbol} is up {pl_pct*100:.1f}%. Moving SL to Entry {new_stop:.2f}")

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
            # 0. Maintenance Tasks
            manage_option_expiry()
            manage_trade_updates() # <--- NEW Call
            
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