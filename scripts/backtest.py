import pandas as pd
import numpy as np
# from scipy.stats import norm 
import math
import matplotlib.pyplot as plt
import builtins
from alpaca.data.historical import StockHistoricalDataClient

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta, timezone

import sys
import os
# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config
from app.bot import (
    get_last_swing_low,
    get_last_swing_high,
    precompute_strategy_features,
    get_causal_signal_from_precomputed,
)


def print(*args, **kwargs):  # noqa: A001
    """Windows-safe print: fallback when console encoding can't render emojis."""
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(a.encode("ascii", "replace").decode("ascii"))
            else:
                safe_args.append(a)
        builtins.print(*safe_args, **kwargs)

def black_scholes_price(S, K, T, r, sigma, type='call'):
    """
    S: Current Stock Price
    K: Strike Price
    T: Time to maturity (in years)
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    """
    if T <= 0:
        return max(0, S - K) if type == 'call' else max(0, K - S)
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if type == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
    return price

def black_scholes_delta(S, K, T, r, sigma, type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = 1.0 if (type == 'call' and S > K) else -1.0 if (type == 'put' and S < K) else 0.0
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if type == 'call':
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0

def black_scholes_theta(S, K, T, r, sigma, type='call'):
    """
    Returns theta per day (negative for long options).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    first = -(S * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if type == 'call':
        second = -r * K * math.exp(-r * T) * norm_cdf(d2)
    else:
        second = r * K * math.exp(-r * T) * norm_cdf(-d2)
    return (first + second) / 365.0

def select_backtest_option_contract(S, sigma, signal, r=0.04):
    """
    Greek-aware synthetic contract picker for backtest parity with live intent:
    prefer moderate delta, lower theta decay, and slightly longer DTE (overnight-friendly).
    """
    min_dte = int(getattr(config, "OPTION_MIN_DTE", 5))
    max_dte = int(getattr(config, "OPTION_MAX_DTE", 21))
    target_dte = float(getattr(config, "OPTION_TARGET_DTE", 12))
    target_delta = float(getattr(config, "OPTION_TARGET_DELTA", 0.55))

    # Strike ladder around spot (1-point spacing) to mimic listed strikes.
    lo = int(round(S * 0.94))
    hi = int(round(S * 1.06))
    dtes = sorted({d for d in [min_dte, 7, 10, 14, max_dte] if min_dte <= d <= max_dte})
    if not dtes:
        dtes = [7]

    right = 'call' if signal == "buy" else 'put'
    best = None
    best_score = float("inf")
    for dte in dtes:
        T = dte / 365.0
        for K in range(lo, hi + 1):
            premium = black_scholes_price(S, K, T, r, sigma, type=right)
            if premium <= 0:
                continue
            delta = black_scholes_delta(S, K, T, r, sigma, type=right)
            if right == 'put':
                delta = abs(delta)
            theta_day = abs(black_scholes_theta(S, K, T, r, sigma, type=right))

            dte_penalty = abs(dte - target_dte) / max(1.0, target_dte)
            delta_penalty = abs(delta - target_delta)
            theta_penalty = theta_day / max(0.001, premium)  # time decay efficiency
            moneyness_penalty = abs(K - S) / max(1.0, S)

            score = (
                1.25 * delta_penalty
                + 1.10 * theta_penalty
                + 0.75 * dte_penalty
                + 0.60 * moneyness_penalty
            )
            if score < best_score:
                best_score = score
                best = (K, dte, premium, right)

    if best is None:
        # Fallback
        dte = max(7, min_dte)
        K = round(S)
        T = dte / 365.0
        premium = black_scholes_price(S, K, T, r, sigma, type=right)
        return K, dte, premium, right
    return best

def run_backtest(days_back=30, symbol=None, trade_type="stock", initial_balance=10000.0, use_daily_cap=True, daily_cap_value=5, option_allocation_pct=0.20, max_option_contracts=-1, min_conf_val=0, max_iv=0.40, allow_no_iv=False, ignore_pdt=False):
    if symbol is None:
        symbol = config.SYMBOL
        
    option_allocation_pct = float(option_allocation_pct)
    if option_allocation_pct < 0 or option_allocation_pct > 1:
        raise ValueError("option_allocation_pct must be within [0.0, 1.0]")
    max_option_contracts = int(max_option_contracts)
    if max_option_contracts != -1 and max_option_contracts < 1:
        raise ValueError("max_option_contracts must be -1 (unlimited) or >= 1")
    stock_budget_pct = 1.0 - option_allocation_pct

    print(f"Starting backtest for {symbol} over last {days_back} days (Type: {trade_type})...")
    
    # 1. Fetch Data
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    
    # End at current UTC time to better reflect live/realtime market state.
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back + 20) # Extra data for vol calc
    
    # HTF Data (15 Min)
    print("Fetching HTF data...")
    htf_req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start_time,
        end=end_time,
        feed="iex"
    )
    htf_data = client.get_stock_bars(htf_req).df
    if htf_data.empty:
        print("No HTF data found.")
        return

    # LTF Data (1 Min)
    print("Fetching LTF data...")
    ltf_req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start_time,
        end=end_time,
        feed="iex"
    )
    ltf_data = client.get_stock_bars(ltf_req).df
    if ltf_data.empty:
        print("No LTF data found.")
        return

    # Prepare bars and precompute strategy features once (same causal path as live bot)
    htf_data = htf_data.reset_index()
    ltf_data = ltf_data.reset_index()

    # Use index temporarily for volatility calc.
    htf_data = htf_data.set_index('timestamp').sort_index()
    
    # Calculate Historical Volatility (for Options)
    # Annualized Volatility using daily logs of HTF (resampled to daily approx or rolling window)
    # Using 15min close to approximate daily is rough, let's use rolling 20-day on 15min returns
    # scaled by sqrt(252 * 26) assuming 26 15-min bars/day? No, 6.5 hours = 26 bars.
    
    htf_data['log_ret'] = np.log(htf_data['close'] / htf_data['close'].shift(1))
    
    # Rolling standard deviation of returns * sqrt(bars_per_year)
    # 252 trading days * 26 (15-min bars) = 6552 bars/year
    # Let's use a simpler rolling Vol estimation:
    # 50-period rolling std dev of log returns * sqrt(6552)
    htf_data['volatility'] = htf_data['log_ret'].rolling(window=50).std() * math.sqrt(252 * 26)
    htf_data['volatility'] = htf_data['volatility'].fillna(0.20) # Default 20%

    # Precompute signal features on full history to mirror live-bot signal evaluation.
    htf_data = htf_data.reset_index()
    htf_precomputed, ltf_precomputed = precompute_strategy_features(htf_data, ltf_data)

    # Use timestamp index for simulation loop mechanics.
    htf_data = htf_precomputed.set_index('timestamp').sort_index()
    ltf_data = ltf_precomputed.set_index('timestamp').sort_index()

    print(f"Data loaded. HTF: {len(htf_data)} bars, LTF: {len(ltf_data)} bars.")

    # 2. Simulation Loop
    
    balance = initial_balance
    position = 0 # Quantity (Shares or Contracts)
    entry_price = 0 # Stock price or Option premium per share
    
    # Trade specific state (Stock or Option)
    active_trade = None # {'strike': K, 'expiry': date, 'type': 'call'/'put'/'stock', ...}
    
    equity_curve = []
    trades = []
    print(f"Initial balance: {balance}")
    
    daily_trade_count = 0
    last_trade_date = None
    last_exit_time = None # New: Track cool-down
    next_entry_allowed_time = None # Mirror live bot: wait 5 minutes after each entry
    day_starting_balance = initial_balance # Initialize for the first day
    current_equity = initial_balance       # Initialize for first bar
    daily_stop_hit = False # Initialize for the first day
    
    # --- GLOBAL SAFETY INITIALIZATION ---
    global_drawdown_hit = False
    last_loss_exit_time = None
    COOL_DOWN_MINUTES = getattr(config, 'COOL_DOWN_MINUTES', 15)
    MAX_DRAWDOWN_PCT = getattr(config, 'MAX_GLOBAL_DRAWDOWN', 0.20)
    peak_equity = initial_balance
    
    # --- PDT RULE ---
    # If equity < $25k, allow max 3 day-trades in a rolling 5-business-day window.
    pdt_enabled = (initial_balance < 25000) and (not ignore_pdt)
    pdt_daytrade_limit = 3
    pdt_day_trade_dates = []  # one entry per same-day round-trip close (date in ET)
    pdt_blocked_entries = 0
    last_pdt_block_log_date = None
    
    start_idx = 400 # Warmup for vol and indicators
    equity_curve.append({'time': ltf_data.index[start_idx-1], 'balance': balance})
    
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")

    def _register_day_trade(entry_ts_utc, exit_ts_utc):
        if not pdt_enabled or entry_ts_utc is None or exit_ts_utc is None:
            return
        entry_date = entry_ts_utc.astimezone(ET).date()
        exit_date = exit_ts_utc.astimezone(ET).date()
        if entry_date == exit_date:
            pdt_day_trade_dates.append(exit_date)

    def _rolling_day_trade_count(current_date_et):
        if not pdt_enabled:
            return 0
        window_start = (pd.Timestamp(current_date_et) - pd.tseries.offsets.BDay(4)).date()
        return sum(1 for d in pdt_day_trade_dates if window_start <= d <= current_date_et)
    
    
    for i in range(start_idx, len(ltf_data)):
        current_bar = ltf_data.iloc[i]
        current_time_utc = current_bar.name
        
        # Convert to US/Eastern for filtering and logging
        current_time_et = current_time_utc.astimezone(ET)
        
        # 1. Global Safety Checks
        if global_drawdown_hit:
            break
            
        # 2. Cool-down Period Check (Post-Loss)
        if last_loss_exit_time:
            time_since_loss = (current_time_utc - last_loss_exit_time).total_seconds() / 60
            if time_since_loss < COOL_DOWN_MINUTES:
                # Still in cool-down, but we continue so we can manage open positions
                pass
            else:
                last_loss_exit_time = None
                
        # --- REFINED MARKET HOURS FILTER (9:40 AM - 3:55 PM ET) ---
        trade_window_start = current_time_et.replace(hour=9, minute=40, second=0, microsecond=0)
        trade_window_end = current_time_et.replace(hour=15, minute=55, second=0, microsecond=0)
        
        in_window = (trade_window_start <= current_time_et <= trade_window_end)
        
        # Regular Market Hours (9:30 AM - 4:00 PM ET) for managing existing positions
        market_open = current_time_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time_et.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_open = (market_open <= current_time_et <= market_close)

        # We don't 'continue' here because we still want to manage open positions
        # outside the entry window (e.g. for stops/targets during the first 30 mins),
        # but we MUST stay within regular market hours for options.

        # --- DAYS BACK ENFORCEMENT ---
        # We fetched extra data for warmup, but we only want to trade starting from days_back
        simulation_start_time = end_time - timedelta(days=days_back)
        if current_time_utc < simulation_start_time:
            continue
            
        # Generate signal using the same causal helper as live bot.
        res = get_causal_signal_from_precomputed(htf_precomputed, ltf_precomputed, current_time_utc, ltf_window=200)
        signal, confidence = res if isinstance(res, tuple) else (res, 0)

        # Confidence gate follows CLI/runtime min_conf directly to mirror live bot behavior.
        current_min_conf = min_conf_val
        
        # For option revaluation and diagnostics.
        htf_slice = htf_data[htf_data.index <= (current_time_utc - timedelta(minutes=15))]
        if len(htf_slice) < 50:
            continue
        current_vol = htf_slice.iloc[-1]['volatility']
        ltf_slice = ltf_data.iloc[max(0, i-200):i+1]

        price = current_bar['close']
        
        # --- RISK CHECKS (Before Signal) ---
        current_option_price = 0
        if position != 0 and active_trade and is_market_open:
            if trade_type == "stock":
                # 1. Stop Loss Hit
                if price <= active_trade['stop_loss']:
                    exit_price = active_trade['stop_loss']
                    proceeds = position * exit_price
                    balance += proceeds
                    pnl = (exit_price - entry_price) * position
                    trades.append({'time': current_time_et, 'type': 'stop_loss_stock', 'price': exit_price, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time_et}] 🛑 STOCK STOP HIT   @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    if pnl < 0:
                        last_loss_exit_time = current_time_utc
                    _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                    position = 0
                    active_trade = None
                    last_exit_time = current_time_utc
                    continue

                # 2. Take Profit Hit
                if price >= active_trade['take_profit']:
                    exit_price = active_trade['take_profit']
                    proceeds = position * exit_price
                    balance += proceeds
                    pnl = (exit_price - entry_price) * position
                    trades.append({'time': current_time_et, 'type': 'take_profit_stock', 'price': exit_price, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time_et}] 🎯 STOCK TARGET HIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                    position = 0
                    active_trade = None
                    last_exit_time = current_time_utc
                    continue

                # 3. Active Management
                pl_pct = (price - entry_price) / entry_price
                # --- STEPPED TRAILING STOP LOGIC ---
                STEP_SIZE = 0.10
                if pl_pct >= STEP_SIZE:
                    milestone = int(pl_pct * 10) / 10.0
                    lock_pct = milestone - 0.10
                    target_stop = entry_price * (1 + lock_pct)
                    
                    if active_trade['stop_loss'] < target_stop:
                        active_trade['stop_loss'] = target_stop
                        if lock_pct <= 0.001:
                             print(f"[{current_time_et}] 🛡️ BREAK EVEN: up {pl_pct*100:.1f}%. SL -> Entry ({target_stop:.2f})")
                        else:
                             print(f"[{current_time_et}] 💰 PROFIT LOCK: up {pl_pct*100:.1f}%. SL -> {target_stop:.2f} (+{lock_pct*100:.0f}%)")

            elif trade_type == "options":
                time_held = current_time_utc - active_trade['entry_time']
                days_passed = time_held.total_seconds() / (24 * 3600)
                T_remain = max(0.0001, (active_trade['expiry_days'] - days_passed) / 365.0)
                sigma = current_vol if current_vol > 0 else 0.2
                
                # Recalculate Option Value
                current_option_price = black_scholes_price(price, active_trade['strike'], T_remain, 0.04, sigma, type=active_trade['type'])
                
                # 1. Stop Loss Hit
                if current_option_price <= active_trade['stop_loss']:
                    exit_price = active_trade['stop_loss']
                    balance += exit_price * 100 * abs(position)
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': f"stop_loss_{active_trade['type']}", 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] 🛑 OPTION STOP HIT   @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    if pnl < 0: last_loss_exit_time = current_time_utc
                    _register_day_trade(active_trade.get('entry_time'), current_time_utc)
                    position = 0; active_trade = None; last_exit_time = current_time_utc
                    continue

                # 2. Take Profit Hit
                elif current_option_price >= active_trade['take_profit']:
                    exit_price = active_trade['take_profit']
                    balance += exit_price * 100 * abs(position)
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': f"take_profit_{active_trade['type']}", 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] 🎯 OPTION TARGET HIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    _register_day_trade(active_trade.get('entry_time'), current_time_utc)
                    position = 0; active_trade = None; last_exit_time = current_time_utc
                    continue
                    
                # 3. Active Management (Hybrid Trailing)
                pl_pct = (current_option_price - entry_price) / entry_price
                
                # Hybrid Strategy: +10% BE, +20% -> +10%, +30% -> +20%, +40% -> +30%
                if pl_pct >= 0.40:
                    lock_price = entry_price * 1.30 # +30% SL
                    if active_trade['stop_loss'] < lock_price:
                        active_trade['stop_loss'] = lock_price
                        print(f"[{current_time_et}] 💰 OPTION TRAILING (HYBRID): LOCKED +30% ({lock_price:.2f})")
                elif pl_pct >= 0.30:
                    lock_price = entry_price * 1.20 # +20% SL
                    if active_trade['stop_loss'] < lock_price:
                        active_trade['stop_loss'] = lock_price
                        print(f"[{current_time_et}] 💰 OPTION TRAILING (HYBRID): LOCKED +20% ({lock_price:.2f})")
                elif pl_pct >= 0.20:
                    lock_price = entry_price * 1.10 # +10% SL
                    if active_trade['stop_loss'] < lock_price:
                        active_trade['stop_loss'] = lock_price
                        print(f"[{current_time_et}] 💰 OPTION TRAILING (HYBRID): LOCKED +10% ({lock_price:.2f})")
                elif pl_pct >= 0.10 and active_trade['stop_loss'] < entry_price:
                    active_trade['stop_loss'] = entry_price
                    print(f"[{current_time_et}] 🛡️ OPTION TRAILING (HYBRID): MOVED TO BE ({entry_price:.2f})")
                
                # 4. Expiry Guard (Sync with bot.py: Close ON expiration day)
                is_expiry_day = (current_time_utc.date() >= active_trade['expiry_date'].date())
                if is_expiry_day:
                     exit_price = current_option_price
                     proceeds = exit_price * 100 * abs(position)
                     balance += proceeds
                     pnl = (exit_price - entry_price) * 100 * abs(position)
                     trades.append({'time': current_time_et, 'type': 'expiry_close', 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                     print(f"[{current_time_et}] ⚠️ EXPIRY EXIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                     _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                     position = 0
                     active_trade = None
                     last_exit_time = current_time_utc
                     continue

        # --- DAILY RESET AND LOSS LIMIT CHECK ---
        current_date = current_time_et.date()
        if last_trade_date != current_date:
            daily_trade_count = 0
            last_trade_date = current_date
            day_starting_balance = current_equity # Reset for new day
            daily_stop_hit = False
            
        # --- EQUITY TRACKING (REAL-TIME) ---
        current_equity = balance
        if position != 0 and active_trade:
            if trade_type == "stock":
                current_equity += position * price
            elif trade_type == "options":
                # Recalculate already handled during management, but for consistency:
                time_held = current_time_utc - active_trade['entry_time']
                days_passed = time_held.total_seconds() / (24 * 3600)
                T_remain = max(0.0001, (active_trade['expiry_days'] - days_passed) / 365.0)
                tracking_premium = black_scholes_price(price, active_trade['strike'], T_remain, 0.04, current_vol, type=active_trade['type'])
                current_equity += abs(position) * tracking_premium * 100
        
        equity_curve.append({'time': current_time_et, 'balance': current_equity})

        # Update Peak Equity
        if current_equity > peak_equity:
            peak_equity = current_equity
            
        # Global Drawdown Breaker Check
        current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        if current_drawdown >= MAX_DRAWDOWN_PCT:
            print(f"[{current_time_et}] 🚨 GLOBAL DRAWDOWN BREAKER HIT ({current_drawdown*100:.1f}%). Peak Equity was: ${peak_equity:.2f}")
            print(f"[{current_time_et}] 🚨 HALTING ALL TRADING.")
            global_drawdown_hit = True
            break # Exit loop immediately

        # Portfolio Loss Circuit Breaker (3% Daily Max) — matches live bot threshold
        daily_pnl_pct = (current_equity - day_starting_balance) / day_starting_balance if day_starting_balance > 0 else 0
        if daily_pnl_pct <= -0.03 and is_market_open:
             if not daily_stop_hit:
                 print(f"[{current_time_et}] 🛑 DAILY LOSS LIMIT HIT (-3%). STOPPING FOR DAY.")
                 daily_stop_hit = True
                 if position != 0:
                     # Close position if hit
                     if trade_type == "stock":
                         balance += position * price
                         pnl = (price - entry_price) * position
                         trades.append({'time': current_time_et, 'type': 'daily_stop_exit', 'price': price, 'qty': position, 'pnl': pnl})
                         _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                         position = 0
                         active_trade = None
                     elif trade_type == "options":
                         # Re-calc option exit
                         time_held = current_time_utc - active_trade['entry_time']
                         days_passed = time_held.total_seconds() / (24 * 3600)
                         T_remain = max(0.0001, (active_trade['expiry_days'] - days_passed) / 365.0)
                         exit_pr = black_scholes_price(price, active_trade['strike'], T_remain, 0.04, current_vol, type=active_trade['type'])
                         balance += exit_pr * 100 * abs(position)
                         pnl = (exit_pr - entry_price) * 100 * abs(position)
                         trades.append({'time': current_time_et, 'type': 'daily_stop_exit', 'price': exit_pr, 'qty': abs(position), 'pnl': pnl})
                         _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                         position = 0
                         active_trade = None
                 daily_stop_hit = True
             continue
             
        if daily_stop_hit:
             continue

        # --- ENTRY LOGIC ---
        pdt_day_trades_used = _rolling_day_trade_count(current_date)
        pdt_entry_blocked = pdt_enabled and pdt_day_trades_used >= pdt_daytrade_limit
        if pdt_entry_blocked and position == 0 and last_pdt_block_log_date != current_date:
            print(f"[{current_time_et}] 🛑 PDT BLOCK: {pdt_day_trades_used} day-trades in rolling 5 business days (<$25k).")
            last_pdt_block_log_date = current_date
        if pdt_entry_blocked and position == 0 and signal in ("buy", "sell") and in_window:
            pdt_blocked_entries += 1

        if signal == "buy":
            # 1. HARD BLOCK CHECK (Confidence 0)
            if confidence <= 0:
                # print(f"[{current_time_et}] 🛑 SIGNAL BLOCKED: Hard Block (Confidence 0) for BUY")
                continue

            # --- BUY SIGNAL ACTION ---
            # Check confidence threshold and daily cap
            conf_ok = (confidence >= current_min_conf)
            trade_count_ok = (not use_daily_cap) or (daily_trade_count < daily_cap_value)
            
            # --- COOL-DOWN CHECK ---
            if last_loss_exit_time:
                continue
            
            can_enter_by_time = (next_entry_allowed_time is None or current_time_utc >= next_entry_allowed_time)
            if position == 0 and in_window and trade_count_ok and conf_ok and can_enter_by_time and not pdt_entry_blocked:
                # Enter Long (Stock or Call)
                if trade_type == "stock":
                    # Determine SL/TP (Sync with bot.py)
                    swing_low = get_last_swing_low(ltf_slice, window=5)
                    if swing_low and swing_low < price:
                        sl = swing_low
                    else:
                        sl = price * 0.995 # Fallback 0.5%
                    
                    risk_dist = price - sl
                    tp = price + (risk_dist * 2.5)
                    
                    # Calculate Qty based on allocation-aware risk
                    risk_amt = balance * stock_budget_pct * config.RISK_PER_TRADE
                    qty_risk = int(risk_amt / risk_dist) if risk_dist > 0 else 0
                    
                    # Capped by Stock Allocation (e.g. 40%)
                    max_stock_cost = balance * stock_budget_pct
                    qty_cap = int(max_stock_cost / price)
                    
                    qty = min(qty_risk, qty_cap)
                    
                    if qty > 0:
                        cost = qty * price
                        balance -= cost
                        position = qty
                        entry_price = price
                        daily_trade_count += 1

                        # Store trade metadata for management
                        active_trade = {
                            'type': 'stock',
                            'stop_loss': sl,
                            'take_profit': tp,
                            'entry_time': current_time_utc
                        }

                        trades.append({'time': current_time_et, 'type': 'buy_stock', 'price': price, 'qty': qty, 'sl': sl, 'tp': tp})
                        print(f"[{current_time_et}] BUY STOCK @ {price:.2f} | Qty: {qty} | SL: {sl:.2f} | TP: {tp:.2f}")
                        next_entry_allowed_time = current_time_utc + timedelta(minutes=5)

                elif trade_type == "options":

                    iv = None
                    if current_vol is not None and not np.isnan(current_vol):
                        iv = current_vol
                    if iv is None:
                        if not allow_no_iv:
                            # Missing IV and strict mode: skip entry
                            continue
                        sigma = 0.20  # fallback when allowed
                    else:
                        iv_val = iv / 100.0 if iv > 2 else iv
                        if iv_val > max_iv:
                            continue  # Skip trade when IV exceeds cap
                        sigma = iv_val
                    
                    res_opt = select_backtest_option_contract(price, sigma, "buy")
                    strike, days_to_expiry, premium, opt_type = res_opt
                    if premium <= 0:
                        continue
                    
                    T = days_to_expiry / 365.0
                    contract_cost = premium * 100

                    total_budget = balance * option_allocation_pct
                    current_risk_target = balance * option_allocation_pct * config.RISK_PER_TRADE
                    
                    # Risk per contract is 20% premium (our SL)
                    risk_per_contract = premium * 0.20 * 100
                    qty_risk = int(current_risk_target // risk_per_contract) if risk_per_contract > 0 else 0
                    
                    # Still capped by total option allocation budget
                    qty_cap = int(total_budget // contract_cost)
                    
                    qty = min(qty_risk, qty_cap)
                    if max_option_contracts != -1 and qty > max_option_contracts:
                        qty = max_option_contracts
                    
                    # Block entry if in cool-down
                    if last_loss_exit_time:
                        qty = 0
                    
                    if qty >= 1 and balance >= (contract_cost * qty):
                        total_cost = contract_cost * qty
                        balance -= total_cost
                        position = qty
                        entry_price = premium
                        daily_trade_count += 1

                        active_trade = {
                            'strike': strike,
                            'expiry_days': days_to_expiry,
                            'expiry_date': current_time_utc + timedelta(days=days_to_expiry),
                            'entry_time': current_time_utc,
                            'type': 'call',
                            'stop_loss': premium * 0.80, # -20%
                            'take_profit': premium * 1.50 # +50%
                        }

                        trades.append({'time': current_time_et, 'type': 'buy_call', 'price': premium, 'qty': qty, 'strike': strike})
                        print(f"[{current_time_et}] BUY CALL  @ {premium:.2f} (Qty: {qty}, Strk: {strike}) | SL: {active_trade['stop_loss']:.2f} | TP: {active_trade['take_profit']:.2f}")
                        next_entry_allowed_time = current_time_utc + timedelta(minutes=5)

            elif position < 0 or (trade_type == "options" and active_trade and active_trade['type'] == 'put'):
                # Exit Bearish Position due to Bullish Signal Flip
                if trade_type == "stock":
                    # Stock shorting not simulated
                    pass
                elif trade_type == "options":
                    exit_price = current_option_price
                    if exit_price == 0: # If we just entered or logic skipped BS calc
                        time_held = current_time_utc - active_trade['entry_time']
                        days_passed = time_held.total_seconds() / (24 * 3600)
                        T_remain = max(0.0001, (active_trade['expiry_days'] - days_passed) / 365.0)
                        exit_price = black_scholes_price(price, active_trade['strike'], T_remain, 0.04, current_vol, type='put')

                    balance += exit_price * 100 * abs(position)
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': 'flip_exit_put', 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] FLIP EXIT PUT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                    position = 0
                    active_trade = None

        elif signal == "sell":
            # 1. HARD BLOCK CHECK (Confidence 0)
            if confidence <= 0:
                # print(f"[{current_time_et}] 🛑 SIGNAL BLOCKED: Hard Block (Confidence 0) for SELL")
                continue

            # --- SELL SIGNAL ACTION ---
            # Check confidence threshold and daily cap
            conf_ok = (confidence >= current_min_conf)
            trade_count_ok = (not use_daily_cap) or (daily_trade_count < daily_cap_value)

            # --- COOL-DOWN CHECK ---
            if last_loss_exit_time:
                continue
            
            can_enter_by_time = (next_entry_allowed_time is None or current_time_utc >= next_entry_allowed_time)
            if position == 0 and in_window and trade_count_ok and conf_ok and can_enter_by_time and not pdt_entry_blocked:
                # Enter Short (Options only)
                if trade_type == "options":

                    iv = None
                    if current_vol is not None and not np.isnan(current_vol):
                        iv = current_vol
                    if iv is None:
                        if not allow_no_iv:
                            continue
                        sigma = 0.20
                    else:
                        iv_val = iv / 100.0 if iv > 2 else iv
                        if iv_val > max_iv:
                            continue
                        sigma = iv_val
                        
                    res_opt = select_backtest_option_contract(price, sigma, "sell")
                    strike, days_to_expiry, premium, opt_type = res_opt
                    if premium <= 0:
                        continue
                    
                    T = days_to_expiry / 365.0
                    contract_cost = premium * 100

                    total_budget = balance * option_allocation_pct
                    current_risk_target = balance * option_allocation_pct * config.RISK_PER_TRADE
                    
                    # Risk per contract is 20% premium (our SL)
                    risk_per_contract = premium * 0.20 * 100
                    qty_risk = int(current_risk_target // risk_per_contract) if risk_per_contract > 0 else 0
                    
                    qty_cap = int(total_budget // contract_cost)
                    
                    qty = min(qty_risk, qty_cap)
                    if max_option_contracts != -1 and qty > max_option_contracts:
                        qty = max_option_contracts
                    
                    # Block entry if in cool-down
                    if last_loss_exit_time:
                        qty = 0
                    
                    if qty >= 1 and balance >= (contract_cost * qty):
                        total_cost = contract_cost * qty
                        balance -= total_cost
                        position = -qty # Negative denotes Put for logic
                        entry_price = premium
                        daily_trade_count += 1

                        active_trade = {
                            'strike': strike,
                            'expiry_days': days_to_expiry,
                            'expiry_date': current_time_utc + timedelta(days=days_to_expiry),
                            'entry_time': current_time_utc,
                            'type': 'put',
                            'stop_loss': premium * 0.80,
                            'take_profit': premium * 1.50 # +50%
                        }
                        trades.append({'time': current_time_et, 'type': 'buy_put', 'price': premium, 'qty': qty, 'strike': strike})
                        print(f"[{current_time_et}] BUY PUT   @ {premium:.2f} (Qty: {qty}, Strk: {strike}) | SL: {active_trade['stop_loss']:.2f} | TP: {active_trade['take_profit']:.2f}")
                        next_entry_allowed_time = current_time_utc + timedelta(minutes=5)

            elif position > 0 or (trade_type == "options" and active_trade and active_trade['type'] == 'call'):
                # Exit Bullish Position due to Bearish Signal Flip
                if trade_type == "stock":
                    balance += position * price
                    pnl = (price - entry_price) * position
                    trades.append({'time': current_time_et, 'type': 'sell_stock', 'price': price, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time_et}] SELL STOCK @ {price:.2f} | PnL: {pnl:.2f}")
                    _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                    position = 0
                    active_trade = None
                elif trade_type == "options":
                    exit_price = current_option_price
                    if exit_price == 0:
                        time_held = current_time_utc - active_trade['entry_time']
                        days_passed = time_held.total_seconds() / (24 * 3600)
                        T_remain = max(0.0001, (active_trade['expiry_days'] - days_passed) / 365.0)
                        exit_price = black_scholes_price(price, active_trade['strike'], T_remain, 0.04, current_vol, type='call')

                    balance += exit_price * 100 * abs(position)
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': 'flip_exit_call', 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] FLIP EXIT CALL @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    _register_day_trade(active_trade.get('entry_time') if active_trade else None, current_time_utc)
                    position = 0
                    active_trade = None

        # --- (Removed) Redundant Equity Tracking Block ---

    # End of backtest - Mark to Market
    if position != 0 and active_trade:
        # Use values from the LAST processed bar to avoid look-ahead bias if we halted early
        last_price = price 
        last_time_utc = current_time_utc
        last_time_et = current_time_et

        if trade_type == "stock":
            pnl = (last_price - entry_price) * position
            balance += position * last_price
            trades.append({'time': last_time_et, 'type': 'mtm_stock', 'price': last_price, 'qty': position, 'pnl': pnl})
        elif trade_type == "options":
            time_held = last_time_utc - active_trade['entry_time']
            days_passed = time_held.total_seconds() / (24 * 3600)
            T_remain = max(0, (active_trade['expiry_days'] - days_passed) / 365.0)
            # Use last available volatility
            sigma = htf_data.iloc[-1]['volatility']
            exit_premium = black_scholes_price(last_price, active_trade['strike'], T_remain, 0.04, sigma, type=active_trade['type'])
            pnl = (exit_premium - entry_price) * 100 * abs(position)
            balance += exit_premium * 100 * abs(position)
            trades.append({'time': last_time_et, 'type': f'mtm_{active_trade["type"]}', 'price': exit_premium, 'qty': abs(position), 'pnl': pnl})
        position = 0
        active_trade = None

    # Keep equity curve consistent with reported final balance after forced liquidation.
    if equity_curve:
        final_curve_time = locals().get('last_time_et', locals().get('current_time_et', ltf_data.index[-1].astimezone(ET)))
        equity_curve.append({'time': final_curve_time, 'balance': balance})

    # --- ROUND TRIP STATISTICS ---
    # In this script, 'trades' contains entries and exits. 
    # Exits have 'pnl' field.
    exit_trades = [t for t in trades if 'pnl' in t]
    wins = [t for t in exit_trades if t['pnl'] > 0]
    losses = [t for t in exit_trades if t['pnl'] < 0]
    breakevens = [t for t in exit_trades if t['pnl'] == 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    be_count = len(breakevens)
    total_exit_trades = len(exit_trades)
    
    win_rate = (win_count / total_exit_trades * 100) if total_exit_trades > 0 else 0

    print("="*30)
    print(f"Backtest Complete ({trade_type.upper()}).")
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance:   ${balance:.2f}")
    if initial_balance > 0:
        print(f"Return:          {((balance - initial_balance)/initial_balance)*100:.2f}%")
    print(f"Total Trades:    {total_exit_trades} (Round Trips)")
    print(f"Wins:            {win_count}")
    print(f"Losses:          {loss_count}")
    print(f"Break-evens:     {be_count}")
    print(f"Win Rate:        {win_rate:.1f}%")
    if pdt_enabled:
        print(f"PDT Rule:        ON (<$25k, max {pdt_daytrade_limit} day-trades / 5 business days)")
        print(f"PDT Day-Trades:  {len(pdt_day_trade_dates)}")
        print(f"PDT Blocks:      {pdt_blocked_entries} entry attempts")
    elif initial_balance < 25000 and ignore_pdt:
        print("PDT Rule:        OFF (explicitly ignored by --ignore-pdt)")
    
    # --- VISUALIZATION ---
    if equity_curve:
        df_equity = pd.DataFrame(equity_curve)
        df_equity.set_index('time', inplace=True)
        
        # Calculate stats for the plot
        final_return_pct = ((balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
        peak_equity = df_equity['balance'].max()
        peak_return_pct = ((peak_equity - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
        
        # Drawdown calculation
        rolling_max = df_equity['balance'].cummax()
        drawdown = (df_equity['balance'] - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100
        
        plt.figure(figsize=(12, 7))
        plt.plot(df_equity.index, df_equity['balance'], label='Account Equity', color='#2ca02c', linewidth=2)
        plt.fill_between(df_equity.index, df_equity['balance'], initial_balance, where=(df_equity['balance'] >= initial_balance), color='#2ca02c', alpha=0.2)
        plt.fill_between(df_equity.index, df_equity['balance'], initial_balance, where=(df_equity['balance'] < initial_balance), color='#d62728', alpha=0.2)
        
        # Summary Box
        stats_text = (
            f"Software Version: v{getattr(config, '__version__', '?.?.?')}\n"
            f"Initial Balance: ${initial_balance:,.2f}\n"
            f"Final Balance: ${balance:,.2f}\n"
            f"Total Return: {final_return_pct:.2f}%\n"
            f"Peak Return: {peak_return_pct:.2f}%\n"
            f"Max Drawdown: {max_drawdown_pct:.2f}%\n"
            f"Total Trades: {total_exit_trades}\n"
            f"Wins: {win_count} | Losses: {loss_count} | BE: {be_count}\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(f"SMC Bot Backtest: {symbol} ({trade_type.upper()})", fontsize=14, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_dir = os.path.join(os.getcwd(), "backtest-output", trade_type, symbol)
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"backtest_{mode}_{symbol}_{initial_balance}_{days_back}.png")
        plt.savefig(plot_path)
        print(f"\n📊 Equity curve saved to: {plot_path}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default=config.SYMBOL, help="Symbol to backtest")
    parser.add_argument("--options", action="store_true", help="Run backtest with Options instead of Stock")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest (default: 30)")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial account balance (default: 10000)")
    parser.add_argument("--cap", type=int, metavar="N", help="Daily trade cap: -1 for unlimited, positive for max trades per day (default: 5)")
    parser.add_argument("--option-allocation", type=float, help="Fraction of equity allocated to options (0.0 to 1.0). Stock allocation is 1 - option-allocation.")
    parser.add_argument("--max-option-contracts", type=int, default=-1, help="Maximum option contracts per trade (-1 for no limit)")
    parser.add_argument("--min-conf", type=str, choices=['all', 'low', 'medium', 'high'], default='all', help="Minimum confidence level to take a signal (default: all)")
    parser.add_argument("--max-iv", type=float, default=0.40, help="Maximum Implied Volatility allowed to enter a trade. Defaults to 0.40 (40%).")
    parser.add_argument("--allow-no-iv", action="store_true", help="Allow trades when IV data is unavailable (falls back to 20%%).")
    parser.add_argument("--ignore-pdt", action="store_true", help="Ignore PDT rule in backtest even if balance is below $25,000 (default: PDT enforced).")
    
    args = parser.parse_args()
    
    # Handle daily cap
    if args.cap is not None:
        if args.cap == -1:
            use_daily_cap = False
            daily_cap_value = 5  # Doesn't matter when cap is disabled
        else:
            use_daily_cap = True
            daily_cap_value = args.cap
    else:
        use_daily_cap = True  # Default behavior
        daily_cap_value = 5
    
    option_allocation = args.option_allocation if args.option_allocation is not None else getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20)
    option_allocation = float(option_allocation)
    if option_allocation < 0 or option_allocation > 1:
        print("❌ Error: --option-allocation must be within [0.0, 1.0].")
        sys.exit(1)
    if args.max_option_contracts != -1 and args.max_option_contracts < 1:
        print("❌ Error: --max-option-contracts must be -1 (unlimited) or >= 1.")
        sys.exit(1)
    
    # Handle confidence thresholds
    CONF_THRESHOLDS = {
        'all': 0,
        'low': 20,
        'medium': 60,
        'high': 80
    }
    min_conf_val = CONF_THRESHOLDS.get(args.min_conf, 0)
    
    mode = "options" if args.options else "stock"
    print(f"Max IV: {args.max_iv*100:.1f}%")
    run_backtest(
        days_back=args.days,
        symbol=args.symbol,
        trade_type=mode,
        initial_balance=args.balance,
        use_daily_cap=use_daily_cap,
        daily_cap_value=daily_cap_value,
        option_allocation_pct=option_allocation,
        max_option_contracts=args.max_option_contracts,
        min_conf_val=min_conf_val,
        max_iv=args.max_iv,
        allow_no_iv=args.allow_no_iv or getattr(config, 'ALLOW_TRADE_WITHOUT_IV', False),
        ignore_pdt=args.ignore_pdt or getattr(config, 'IGNORE_PDT', False))
