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
import asyncio
from ib_insync import *
try:
    from . import config
    from .ibkr_manager import ibkr_mgr
except ImportError:
    import config
    from ibkr_manager import ibkr_mgr


# ================= CONFIG =================

SYMBOL = config.SYMBOL
RISK_PER_TRADE = config.RISK_PER_TRADE

# ================= CLIENTS =================

# Global ib is replaced by dynamic ibkr_mgr.ib in functions

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

async def get_bars(symbol, timeframe, limit):
    """
    Fetch historical bars from IBKR.
    """
    bar_size = '1 min'
    if '15Min' in timeframe:
        bar_size = '15 mins'
    elif '1Day' in timeframe:
        bar_size = '1 day'

    # Create Contract
    contract = Stock(symbol, 'SMART', 'USD')
    
    # Using reqHistoricalDataAsync for non-blocking fetch
    # Using 'MIDPOINT' which is often available without a full subscription
    bars = await ibkr_mgr.ib.reqHistoricalDataAsync(
        contract,
        endDateTime='',
        durationStr='1 W',
        barSizeSetting=bar_size,
        whatToShow='MIDPOINT',
        useRTH=True,
        formatDate=1,
        keepUpToDate=False
    )
    
    if not bars:
        print(f"No bars returned for {symbol}")
        return pd.DataFrame()

    df = util.df(bars)
    # Map column names to strategy expectations
    df = df.rename(columns={'date': 'timestamp'})
    
    if len(df) > limit:
        df = df.iloc[-limit:]
        
    return df

async def get_latest_price_fallback(symbol):
    """
    Fetches the latest close price from historical data if real-time ticker fails.
    """
    try:
        df = await get_bars(symbol, "1Min", 1)
        if not df.empty:
            return float(df['close'].iloc[-1])
    except Exception as e:
        print(f"Error in price fallback for {symbol}: {e}")
    return None

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

async def generate_signal(symbol=None):
    if symbol is None:
        symbol = SYMBOL
        
    try:
        # Fetch bars for both timeframes
        htf_df = await get_bars(symbol, config.TIMEFRAME_HTF, 200)
        ltf_df = await get_bars(symbol, config.TIMEFRAME_LTF, 200)
        
        if htf_df.empty or ltf_df.empty:
            print(f"Not enough data fetched for {symbol}.")
            return None

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

    return get_strategy_signal(htf_df, ltf_df)

# ================= RISK =================

from alpaca.trading.requests import StopLossRequest, TakeProfitRequest

async def calculate_smart_quantity(symbol, price, stop_loss_price, budget_override=None):
    """
    Calculates position size using IBKR account data.
    """
    try:
        # In ib_insync, account values are stored in 'accountValues'
        # NetLiquidation is usually the equity.
        acc_values = ibkr_mgr.ib.accountValues()
        equity = 10000.0
        buying_power = 10000.0
        
        for v in acc_values:
            if v.tag == 'NetLiquidation':
                equity = float(v.value)
            if v.tag == 'BuyingPower':
                buying_power = float(v.value)
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
        # ib.positions() returns a list of Position objects
        positions = ibkr_mgr.ib.positions()
        for p in positions:
            # We identify the ticker by checking contract.symbol
            if p.contract.symbol == symbol and isinstance(p.contract, Stock):
                current_exposure += abs(p.position * price)
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

async def get_best_option_contract(symbol, signal_type, known_price=None):
    ib = ibkr_mgr.ib
    """
    Finds the best option contract using IBKR reqContractDetails.
    """
    try:
        if known_price:
            current_price = known_price
        else:
            current_price_df = await get_bars(symbol, "1Min", 1)
            if current_price_df.empty:
                return None
            current_price = current_price_df['close'].iloc[-1]

        # Define Option Right and Target Strike
        right = 'C' if signal_type == "buy" else 'P'
        
        # In a real scenario, we'd query chains. For IBKR MVP, we'll try to find a near-ATM
        # expiring in 1-2 weeks.
        target_strike = round(current_price)
        
        # Search for contracts
        contract = Option(symbol, '20260213', target_strike, right, 'SMART') # Hardcoded expiry for example, should be dynamic
        # Real logic: use ib.reqSecDefOptParams and then find best
        
        details = await ib.reqContractDetailsAsync(contract)
        if not details:
            print(f"No option contracts found for {symbol} {right}")
            return None
            
        return details[0].contract

    except Exception as e:
        print(f"Error getting option contract: {e}")
        return None

def get_current_position(symbol):
    """
    Returns the current position for the given symbol from the live list.
    """
    positions = ibkr_mgr.ib.positions()
    for p in positions:
        if p.contract.symbol == symbol:
            return p
    return None

async def cancel_all_orders_for_symbol(symbol):
    open_trades = ibkr_mgr.ib.openTrades()
    for t in open_trades:
        if t.contract.symbol == symbol:
            ibkr_mgr.ib.cancelOrder(t.order)
            print(f"Cancelled order for {symbol}")

async def place_trade(signal, symbol, use_daily_cap=True, daily_cap_value=5, stock_budget_override=None, option_budget_override=None):
    ib = ibkr_mgr.ib
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
    
    ticker_state = state.get(symbol, {})
    last_date = ticker_state.get("last_trade_date")
    daily_count = ticker_state.get("daily_trade_count", 0)
    
    if last_date != today_str:
        daily_count = 0
        ticker_state["last_trade_date"] = today_str
    
    if daily_cap_value is None:
        cap_limit = getattr(config, 'DEFAULT_DAILY_CAP', 5)
    else:
        cap_limit = daily_cap_value
    
    if use_daily_cap and cap_limit >= 0 and daily_count >= cap_limit:
        print(f"🛑 DAILY CAP: Already placed {daily_count} trades for {symbol} today (limit: {cap_limit}). Skipping.")
        return

    # Get latest price
    price_df = await get_bars(symbol, "1Min", 100)
    if price_df.empty:
        print("Could not fetch price for placement.")
        return
    
    price = price_df['close'].iloc[-1]
    
    # 1. Fetch CURRENT positions
    positions = ib.positions()
    qty_held = 0
    side_held = None
    any_options_held = False
    
    for p in positions:
        if p.contract.symbol != symbol: continue
        if isinstance(p.contract, Stock):
            qty_held = p.position # Positive for long, negative for short in IBKR
            side_held = 'long' if qty_held > 0 else 'short' if qty_held < 0 else None
        elif isinstance(p.contract, Option):
            any_options_held = True

    print(f"DEBUG: Current Position for {symbol}: {qty_held} units (Side: {side_held}) | Options: {any_options_held}")

    # 2. Global Position Cleanup (Cross-Asset Signal Flip)
    same_bias_held = False
    for p in positions:
        if p.contract.symbol != symbol: continue
        
        is_stock = isinstance(p.contract, Stock)
        is_option = isinstance(p.contract, Option)
        
        # Identify Bias
        is_bullish = False
        if is_stock:
            is_bullish = (p.position > 0)
        elif is_option:
            is_bullish = (p.contract.right == 'C')

        # Conflict Detection: Close if signal is opposite of current holding bias
        if (signal == "buy" and not is_bullish) or (signal == "sell" and is_bullish):
            print(f"🔄 CROSS-ASSET CLEANUP: Closing {p.contract.localSymbol} bias conflict with {signal.upper()} signal.")
            try:
                # Cancel open orders for this contract
                ib.reqGlobalCancel() # Quickest way in IBKR for all
                # Close position with a Market Order
                action = 'SELL' if p.position > 0 else 'BUY'
                close_order = MarketOrder(action, abs(p.position))
                ib.placeOrder(p.contract, close_order)
                
                # Update local state
                if is_stock:
                    qty_held = 0
                    side_held = None
                if is_option:
                    any_options_held = False
            except Exception as close_err:
                print(f"❌ Cleanup failed for {p.contract.localSymbol}: {close_err}")
        
        elif (signal == "buy" and is_bullish) or (signal == "sell" and not is_bullish):
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
        opt_contract = await get_best_option_contract(symbol, signal)
        
        if opt_contract:
            # Check if we already hold THIS specific contract
            positions = ib.positions()
            for p in positions:
                if p.contract.conId == opt_contract.conId and p.position != 0:
                     print(f"Already hold {opt_contract.localSymbol}. Holding.")
                     return

            try:
                # Fetch Current Option Price (Tick)
                ib.reqMktData(opt_contract)
                await asyncio.sleep(0.5) # Wait for tick
                ticker = ib.ticker(opt_contract)
                
                if not ticker:
                    print(f"❌ Ticker not found for {opt_contract.localSymbol}. Skipping.")
                    return
                    
                entry_est = ticker.ask if ticker.ask > 0 else ticker.last
                
                if not entry_est or entry_est <= 0 or np.isnan(entry_est):
                    # For options, MIDPOINT historical data via get_bars might not work without subscription
                    # but we try ticker.marketPrice() as a last resort
                    entry_est = ticker.marketPrice()
                    
                if not entry_est or entry_est <= 0 or np.isnan(entry_est):
                    print(f"Invalid option ask price for {opt_contract.localSymbol}. Skipping.")
                    return
                    
                entry_est = float(entry_est)
                sl_price = float(round(entry_est * 0.80, 2))
                tp_price = float(round(entry_est * 1.50, 2))
                
                print(f"Options Bracket: Est.Entry: {entry_est} | SL: {sl_price} | TP: {tp_price}")
                
                # --- GLOBAL OPTION EXPOSURE ---
                acc_values = ib.accountValues()
                equity = 10000.0
                for v in acc_values:
                    if v.tag == 'NetLiquidation': equity = float(v.value)
                
                budget_pct = option_budget_override if option_budget_override is not None else getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20)
                total_budget = equity * budget_pct
                
                existing_option_exposure = 0.0
                ticker_option_exposure = 0.0
                for p in positions:
                    if isinstance(p.contract, Option):
                        existing_option_exposure += abs(p.position * entry_est * 100) # Rough estimate
                        if p.contract.symbol == symbol:
                            ticker_option_exposure += abs(p.position * entry_est * 100)
                
                available_budget = min(total_budget - existing_option_exposure, (equity * budget_pct) - ticker_option_exposure)
                cost_per_contract = entry_est * 100
                
                if available_budget < cost_per_contract:
                    print(f"⚠️ Insufficient Budget for {symbol} option.")
                    return
                
                qty = int(available_budget // cost_per_contract)
                qty = min(qty, 5) # Cap at 5
                
                if qty < 1: return

                print(f"Sizing: Qty {qty}")
                
                # Place Bracket
                action = 'BUY'
                bracket = ib.bracketOrder(action, qty, entry_est, tp_price, sl_price)
                for o in bracket:
                    o.tif = 'DAY' # Ensure TIF is explicit
                    ib.placeOrder(opt_contract, o)
                
                print(f"✅ OPTION BRACKET SUBMITTED: {opt_contract.localSymbol}")
                # Update state
                ticker_state["daily_trade_count"] = daily_count + 1
                state[symbol] = ticker_state
                save_trade_state(state)
                return

            except Exception as e:
                print(f"❌ Option Order failed: {e}")
        else:
             print("Could not find suitable option contract. Falling back to shares.")
        
    # --- STOCK TRADING LOGIC ---
    if signal == "buy":
        if qty_held == 0:
            price = float(price)
            sl_price = float(round(price * 0.95, 2))
            tp_price = float(round(price * 1.125, 2))
            
            qty = await calculate_smart_quantity(symbol, price, sl_price, budget_override=stock_budget_override)
            if qty <= 0: return

            print(f"Placing BUY Bracket: Entry ~{price} | SL {sl_price:.2f} | TP {tp_price:.2f} | Qty {qty}")
            
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                bracket = ib.bracketOrder('BUY', qty, price, tp_price, sl_price)
                for o in bracket:
                    o.tif = 'DAY' # Ensure TIF is explicit
                    ib.placeOrder(contract, o)
                
                print("✅ BUY BRACKET ORDER SUBMITTED")
                ticker_state["daily_trade_count"] = daily_count + 1
                state[symbol] = ticker_state
                save_trade_state(state)
            except Exception as e:
                print(f"❌ Order failed: {e}")
        
        elif qty_held < 0: # We are Short in IBKR
             print(f"Closing Short Position ({qty_held}) due to BUY signal.")
             try:
                ib.reqGlobalCancel()
                close_order = MarketOrder('BUY', abs(qty_held))
                ib.placeOrder(Stock(symbol, 'SMART', 'USD'), close_order)
                print("✅ SHORT CLOSE SUBMITTED")
             except Exception as e:
                print(f"❌ Close Short failed: {e}")
        else:
            print(f"Already Long {qty_held}. Ignoring BUY signal.")

    elif signal == "sell":
        if qty_held == 0:
            print("SELL Signal detected but no position held. Skipping.")
            return
            
        elif qty_held > 0: # We are Long
            print(f"Closing Long Position ({qty_held}) due to SELL signal.")
            try:
                ib.reqGlobalCancel()
                close_order = MarketOrder('SELL', abs(qty_held))
                ib.placeOrder(Stock(symbol, 'SMART', 'USD'), close_order)
                print("✅ LONG CLOSE SUBMITTED")
            except Exception as e:
                 print(f"❌ Close Long failed: {e}")
                 
        else:
            print(f"Already Short {qty_held} shares. Ignoring SELL signal.")



def parse_option_expiry(symbol):
    # Regex to capture YYMMDD from SPY251219C00500000
    # Format: Root(up to 6 chars) + YYMMDD + Type(C/P) + Strike
    match = re.search(r'[A-Z]+(\d{6})[CP]', symbol)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, "%y%m%d")
    return None

async def manage_option_expiry():
    ib = ibkr_mgr.ib
    """
    Checks all open option positions for expiry in IBKR.
    """
    if not getattr(config, 'ENABLE_OPTIONS', False):
        return
        
    try:
        positions = ib.positions()
        now = datetime.now()
        
        for p in positions:
            if not isinstance(p.contract, Option):
                continue
                
            # IBKR Option contracts have 'lastTradeDateOrContractMonth' (YYYYMMDD)
            expiry_str = p.contract.lastTradeDateOrContractMonth
            expiry = datetime.strptime(expiry_str, "%Y%m%d")
            
            is_expiry_day = (now.date() >= expiry.date())
            
            if is_expiry_day:
                print(f"⚠️ CRITICAL: {p.contract.localSymbol} expires TODAY! Force Closing.")
                # Close with Market Order
                action = 'SELL' if p.position > 0 else 'BUY'
                close_order = MarketOrder(action, abs(p.position))
                ib.placeOrder(p.contract, close_order)
            else:
                days_left = (expiry.date() - now.date()).days
                print(f"DEBUG: {p.contract.localSymbol} DTE: {days_left} days - Safe")

    except Exception as e:
        print(f"Error in manage_option_expiry: {e}")

from alpaca.trading.requests import ReplaceOrderRequest

# ================= STATE MANAGEMENT =================

STATE_FILE = "trade_state.json"

def load_trade_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading state: {e}")
    return {}

def save_trade_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error saving state: {e}")

async def manage_trade_updates():
    ib = ibkr_mgr.ib
    """
    Asynchronous trade updates for IBKR (Hybrid Trailing SL).
    """
    try:
        positions = ib.positions()
        for p in positions:
            symbol = p.contract.symbol
            if symbol != SYMBOL: continue
            
            ticker = ib.ticker(p.contract)
            # Only request if not already streaming (checking all tickers)
            if ticker not in ib.tickers():
                print(f"DEBUG: Starting data stream for {p.contract.localSymbol}...")
                ib.reqMktData(p.contract)
            
            if not ticker: continue
                
            curr_price = ticker.marketPrice()
            if not curr_price or curr_price <= 0 or np.isnan(curr_price):
                # Fallback for delayed data
                curr_price = ticker.last if ticker.last > 0 else ticker.close
            
            if not curr_price or curr_price <= 0 or np.isnan(curr_price):
                # Final fallback to historical MIDPOINT
                print(f"ℹ️ Market Data (Error 10089) - Falling back to Historical MIDPOINT for {symbol}...")
                curr_price = await get_latest_price_fallback(symbol)
                
            if not curr_price or curr_price <= 0 or np.isnan(curr_price): continue
            
            curr_price = float(curr_price)
            avg_cost = float(p.avgCost)
            if avg_cost <= 0: continue
            
            is_long = (p.position > 0)
            pl_pct = (curr_price - avg_cost) / avg_cost if is_long else (avg_cost - curr_price) / avg_cost

            # --- OPTIONS RISK MANAGEMENT ---
            if isinstance(p.contract, Option):
                state = load_trade_state()
                symbol_state = state.get(symbol, {"virtual_stop": -0.20})
                virtual_stop = symbol_state.get("virtual_stop", -0.20)
                
                # Exit Triggers
                if pl_pct <= virtual_stop:
                    print(f"🛑 OPTION STOP HIT: {p.contract.localSymbol} at {pl_pct*100:.1f}%. Closing.")
                    ib.reqGlobalCancel()
                    action = 'SELL' if p.position > 0 else 'BUY'
                    ib.placeOrder(p.contract, MarketOrder(action, abs(p.position)))
                    if symbol in state: del state[symbol]; save_trade_state(state)
                    continue
                elif pl_pct >= 0.50: # TP 50%
                    print(f"🎯 OPTION TP HIT: {p.contract.localSymbol} at {pl_pct*100:.1f}%. Closing.")
                    ib.reqGlobalCancel()
                    action = 'SELL' if p.position > 0 else 'BUY'
                    ib.placeOrder(p.contract, MarketOrder(action, abs(p.position)))
                    if symbol in state: del state[symbol]; save_trade_state(state)
                    continue

                # Trailing logic
                updated = False
                if pl_pct >= 0.40 and virtual_stop < 0.20:
                    virtual_stop = 0.20; updated = True
                elif pl_pct >= 0.30 and virtual_stop < 0.10:
                    virtual_stop = 0.10; updated = True
                elif pl_pct >= 0.15 and virtual_stop < 0.0:
                    virtual_stop = 0.0; updated = True
                
                if updated:
                    state[symbol] = {"virtual_stop": virtual_stop}
                    save_trade_state(state)
                continue

            # --- STOCK TRAILING ---
            if pl_pct < 0.15: continue
            
            open_trades = ib.openTrades()
            stop_order_trade = None
            for t in open_trades:
                if t.contract.conId == p.contract.conId and t.order.orderType in ['STP', 'STP LMT']:
                    stop_order_trade = t
                    break
            
            if not stop_order_trade: continue
            
            curr_stop = stop_order_trade.order.auxPrice
            new_stop = None
            
            if pl_pct > 0.09: # Lock 5%
                target_stop = avg_cost * 1.05 if is_long else avg_cost * 0.95
                if (is_long and curr_stop < target_stop) or (not is_long and curr_stop > target_stop):
                    new_stop = target_stop
            elif pl_pct > 0.05: # Break Even
                target_stop = avg_cost
                if (is_long and curr_stop < target_stop) or (not is_long and curr_stop > target_stop):
                    new_stop = target_stop
            
            if new_stop:
                stop_order_trade.order.auxPrice = round(new_stop, 2)
                ib.placeOrder(p.contract, stop_order_trade.order)
                print(f"✅ SL Updated for {symbol} to {new_stop:.2f}")

    except Exception as e:
        print(f"Error in manage_trade_updates: {e}")

    except Exception as e:
        print(f"Error in manage_trade_updates: {e}")
    
    # Final Cleanup: Remove state for any symbols we no longer hold
    try:
        current_positions = ib.positions()
        held_symbols = {p.contract.symbol for p in current_positions}
        state = load_trade_state()
        state_symbols = list(state.keys())
        cleaned = False
        for s in state_symbols:
            if s not in held_symbols:
                del state[s]
                cleaned = True
        if cleaned:
            save_trade_state(state)
    except Exception:
        pass

def is_market_open():
    """
    Checks if US markets are currently open (9:30 AM - 4:00 PM ET, Mon-Fri).
    Returns (bool, status_message)
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    
    # Weekend Check
    if now_et.weekday() >= 5:
        return False, "Market is CLOSED (Weekend)"
    
    # Time Check
    current_time = now_et.time()
    market_open = datetime.strptime("09:30:00", "%H:%M:%S").time()
    market_close = datetime.strptime("16:00:00", "%H:%M:%S").time()
    
    if current_time < market_open:
        return False, "Market is CLOSED (Pre-Market)"
    if current_time > market_close:
        return False, "Market is CLOSED (After-Hours)"
        
    return True, "Market is OPEN"

# ================= SESSION MANAGEMENT =================

class TradingSession:
    """Manages trading session state and statistics."""
    
    def __init__(self, duration_hours=None, max_trades=None):
        ib = ibkr_mgr.ib
        self.start_time = datetime.now()
        self.duration_hours = duration_hours
        self.max_trades = max_trades
        self.trades_executed = 0
        self.should_stop = False
        
        # Enhanced tracking
        self.opening_equity = None
        self.opening_positions = []
        self.closed_trades = []  # List of dicts with {symbol, pnl, win}
        
        # Capture opening state
        try:
            acc_values = ib.accountValues()
            for v in acc_values:
                if v.tag == 'NetLiquidation':
                    self.opening_equity = float(v.value)
                    break
            
            positions = ib.positions()
            for p in positions:
                self.opening_positions.append({
                    'symbol': p.contract.localSymbol,
                    'qty': float(p.position),
                    'side': 'long' if p.position > 0 else 'short',
                    'entry_price': float(p.avgCost),
                    'market_value': float(p.position * p.avgCost), # Rough
                    'unrealized_pl': 0.0 # Requires ticker
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
        ib = ibkr_mgr.ib
        """Get session summary statistics."""
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        
        # Get current account state
        closing_equity = 0.0
        closing_positions = []
        try:
            acc_values = ib.accountValues()
            for v in acc_values:
                if v.tag == 'NetLiquidation':
                    closing_equity = float(v.value)
                    break
            
            positions = ib.positions()
            for p in positions:
                closing_positions.append({
                    'symbol': p.contract.localSymbol,
                    'qty': float(p.position),
                    'side': 'long' if p.position > 0 else 'short',
                    'entry_price': float(p.avgCost),
                    'current_price': 0.0, # Requires ticker
                    'market_value': float(p.position * p.avgCost),
                    'unrealized_pl': 0.0,
                    'unrealized_plpc': 0.0
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

async def interruptible_sleep(seconds, session):
    """Sleeps for the specified time, but returns early if session should stop."""
    end_time = time.time() + seconds
    while time.time() < end_time and session.should_continue():
        # Sleep in small increments to stay responsive
        remaining = end_time - time.time()
        await asyncio.sleep(min(1.0, remaining))

async def main():
    parser = argparse.ArgumentParser(description="Run the SMC trading bot.")
    parser.add_argument("symbol", nargs="?", default=SYMBOL, help="Symbol to trade (default: SPY)")
    parser.add_argument("--options", action="store_true", help="Enable options trading (overrides config)")
    parser.add_argument("--cap", type=int, metavar="N", help="Daily trade cap: -1 for unlimited, 0 for no trades, positive for max trades per day (default: 5)")
    parser.add_argument("--session-duration", type=float, help="Session duration in hours (runs indefinitely if not specified)")
    parser.add_argument("--max-trades", type=int, help="Maximum number of trades per session (unlimited if not specified)")
    parser.add_argument("--stock-budget", type=float, help="Stock allocation budget override (0.0 to 1.0)")
    parser.add_argument("--option-budget", type=float, help="Option allocation budget override (0.0 to 1.0)")
    
    args = parser.parse_args()
    target_symbol = args.symbol
    
    if args.options:
        print("Overriding ENABLE_OPTIONS to True from command line.")
        config.ENABLE_OPTIONS = True
    
    # Validation
    if config.ENABLE_OPTIONS and args.stock_budget is not None:
        print("❌ Error: --stock-budget is only valid in Stock Mode.")
        sys.exit(1)
    
    daily_cap = args.cap if args.cap is not None else 5
    
    # Connect to IBKR
    connected = await ibkr_mgr.connect()
    if not connected:
        print("❌ Failed to connect to IBKR. Exiting.")
        return

    # Initialize session
    session = TradingSession(duration_hours=args.session_duration, max_trades=args.max_trades)
    session.setup_signal_handlers()
    
    print(f"Starting SMC Bot for {target_symbol} (Options: {config.ENABLE_OPTIONS})...")
    
    try:
        while session.should_continue():
            try:
                # 0. Maintenance Tasks
                await manage_option_expiry()
                await manage_trade_updates()
                
                # 1. Market Hours Filter
                is_open, status_msg = is_market_open()
                if not is_open:
                    print(f"🕒 {status_msg}. Waiting 15 minutes...")
                    await interruptible_sleep(900, session)
                    continue

                # 2. Market is open, look for signals
                print(f"Analyzing {target_symbol} at {now_et}...")
                sig = await generate_signal(target_symbol)
                
                if sig:
                    print(f"🚀 Signal detected: {sig.upper()}!({target_symbol})")
                    use_cap = (daily_cap != -1)
                    await place_trade(
                        sig, 
                        target_symbol, 
                        use_daily_cap=use_cap, 
                        daily_cap_value=daily_cap if use_cap else None,
                        stock_budget_override=args.stock_budget,
                        option_budget_override=args.option_budget
                    )
                    session.record_trade()
                    print("Trade placed. Cooling down for 5 minutes...")
                    await interruptible_sleep(300, session)
                else:
                    await interruptible_sleep(60, session)

            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                await interruptible_sleep(60, session)
    except KeyboardInterrupt:
        session.request_stop()
    finally:
        print("\n")
        print(session.get_summary())
        ibkr_mgr.disconnect()

if __name__ == "__main__":
    asyncio.run(main())