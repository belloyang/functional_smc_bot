import pandas as pd
import numpy as np
# from scipy.stats import norm 
import math
import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta, timezone

import sys
import os
# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config
from app.bot import get_strategy_signal, get_last_swing_low, get_last_swing_high

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

def run_backtest(days_back=30, symbol=None, trade_type="stock", initial_balance=10000.0):
    if symbol is None:
        symbol = config.SYMBOL
        
    print(f"Starting backtest for {symbol} over last {days_back} days (Type: {trade_type})...")
    
    # 1. Fetch Data
    # ... (Data fetching code omitted for brevity as it is unchanged) ...
    # Note: replace_file_content replaces contiguous blocks. I need to be careful not to delete the data fetching block if I bridge the gap.
    # Actually, the user asked to update "initial balance and days_back". 
    # I will replace the signature line and the hardcoded balance line separately or in a large block if safe.
    # It is safer here to do two replaces or one large block if I know the content.
    # I'll just change the signature line first.
    
    # 1. Fetch Data
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    
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

    # Ensure indices are handled
    htf_data = htf_data.reset_index() 
    ltf_data = ltf_data.reset_index()
    
    # Set timestamp as index for easier slicing
    htf_data.set_index('timestamp', inplace=True)
    ltf_data.set_index('timestamp', inplace=True)
    
    # Sort just in case
    htf_data.sort_index(inplace=True)
    ltf_data.sort_index(inplace=True)
    
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
    
    start_idx = 400 # Warmup for vol and indicators
    equity_curve.append({'time': ltf_data.index[start_idx-1], 'balance': balance})
    
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")
    
    for i in range(start_idx, len(ltf_data)):
        current_bar = ltf_data.iloc[i]
        current_time_utc = current_bar.name
        
        # Convert to US/Eastern for filtering and logging
        current_time_et = current_time_utc.astimezone(ET)
        
        # --- MARKET HOURS FILTER (9:30 AM - 4:00 PM ET) ---
        market_open = current_time_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if not (market_open <= current_time_et <= market_close):
            continue

        # --- DAYS BACK ENFORCEMENT ---
        # We fetched extra data for warmup, but we only want to trade starting from days_back
        simulation_start_time = end_time - timedelta(days=days_back)
        if current_time_utc < simulation_start_time:
            continue
            
        # Get HTF data up to current_time
        htf_slice = htf_data[htf_data.index <= current_time_utc]
        if len(htf_slice) < 50:
            continue
            
        current_vol = htf_slice.iloc[-1]['volatility']
        htf_slice = htf_slice.iloc[-200:] # Limit to last 200 HTF for speed
        ltf_slice = ltf_data.iloc[max(0, i-200):i+1] # Pass last 200 LTF
        
        # Run Strategy
        signal = get_strategy_signal(htf_slice, ltf_slice)
        
        price = current_bar['close']
        
        # --- RISK CHECKS (Before Signal) ---
        current_option_price = 0
        if position != 0 and active_trade:
            if trade_type == "stock":
                # 1. Stop Loss Hit
                if price <= active_trade['stop_loss']:
                    exit_price = active_trade['stop_loss']
                    proceeds = position * exit_price
                    balance += proceeds
                    pnl = (exit_price - entry_price) * position
                    trades.append({'time': current_time_et, 'type': 'stop_loss_stock', 'price': exit_price, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time_et}] 🛑 STOCK STOP HIT   @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    active_trade = None
                    continue

                # 2. Take Profit Hit
                if price >= active_trade['take_profit']:
                    exit_price = active_trade['take_profit']
                    proceeds = position * exit_price
                    balance += proceeds
                    pnl = (exit_price - entry_price) * position
                    trades.append({'time': current_time_et, 'type': 'take_profit_stock', 'price': exit_price, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time_et}] 🎯 STOCK TARGET HIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    active_trade = None
                    continue

                # 3. Active Management
                pl_pct = (price - entry_price) / entry_price
                if pl_pct > 0.15 and active_trade['stop_loss'] < entry_price:
                    active_trade['stop_loss'] = entry_price
                    print(f"[{current_time_et}] 🛡️ STOCK MOVED SL TO BE ({entry_price:.2f})")
                if pl_pct > 0.30:
                    lock_price = entry_price * 1.15
                    if active_trade['stop_loss'] < lock_price:
                        active_trade['stop_loss'] = lock_price
                        print(f"[{current_time_et}] 💰 STOCK LOCKED PROFIT ({lock_price:.2f})")

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
                    proceeds = exit_price * 100 * abs(position)
                    balance += proceeds
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': f"stop_loss_{active_trade['type']}", 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] 🛑 OPTION STOP HIT   @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    active_trade = None
                    continue

                # 2. Take Profit Hit
                if current_option_price >= active_trade['take_profit']:
                    exit_price = active_trade['take_profit']
                    proceeds = exit_price * 100 * abs(position)
                    balance += proceeds
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': f"take_profit_{active_trade['type']}", 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] 🎯 OPTION TARGET HIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    active_trade = None
                    continue
                    
                # 3. Active Management
                pl_pct = (current_option_price - entry_price) / entry_price
                if pl_pct > 0.15 and active_trade['stop_loss'] < entry_price:
                    active_trade['stop_loss'] = entry_price
                    print(f"[{current_time_et}] 🛡️ OPTION MOVED SL TO BE ({entry_price:.2f})")
                if pl_pct > 0.30:
                    lock_price = entry_price * 1.15
                    if active_trade['stop_loss'] < lock_price:
                        active_trade['stop_loss'] = lock_price
                        print(f"[{current_time_et}] 💰 OPTION LOCKED PROFIT ({lock_price:.2f})")
                
                # 4. Expiry Guard (Sync with bot.py: Close ON expiration day)
                is_expiry_day = (current_time_utc.date() >= active_trade['expiry_date'].date())
                if is_expiry_day:
                     exit_price = current_option_price
                     proceeds = exit_price * 100 * abs(position)
                     balance += proceeds
                     pnl = (exit_price - entry_price) * 100 * abs(position)
                     trades.append({'time': current_time_et, 'type': 'expiry_close', 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                     print(f"[{current_time_et}] ⚠️ EXPIRY EXIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                     position = 0
                     active_trade = None
                     continue

        # --- ENTRY LOGIC ---
        if signal == "buy":
            # --- BUY SIGNAL ACTION ---
            if position == 0:
                # Enter Long (Stock or Call)
                if trade_type == "stock":
                    # Determine SL/TP (Sync with bot.py)
                    swing_low = get_last_swing_low(ltf_slice, window=5)
                    if swing_low and swing_low < price:
                        sl = swing_low
                    else:
                        sl = price * 0.995 # Fallback 0.5%
                    
                    risk_dist = price - sl
                    tp = price + (risk_dist * 2.0)
                    
                    # Calculate Qty based on Risk (1% of balance)
                    risk_amt = balance * 0.01
                    qty = int(risk_amt / risk_dist) if risk_dist > 0 else 0
                    
                    if qty > 0:
                        cost = qty * price
                        balance -= cost
                        position = qty
                        entry_price = price

                        # Store trade metadata for management
                        active_trade = {
                            'type': 'stock',
                            'stop_loss': sl,
                            'take_profit': tp,
                            'entry_time': current_time_utc
                        }

                        trades.append({'time': current_time_et, 'type': 'buy_stock', 'price': price, 'qty': qty, 'sl': sl, 'tp': tp})
                        print(f"[{current_time_et}] BUY STOCK @ {price:.2f} | Qty: {qty} | SL: {sl:.2f} | TP: {tp:.2f}")

                elif trade_type == "options":
                    strike = round(price)
                    days_to_expiry = 7
                    T = days_to_expiry / 365.0
                    sigma = current_vol if current_vol > 0 else 0.2
                    premium = black_scholes_price(price, strike, T, 0.04, sigma, type='call')
                    contract_cost = premium * 100

                    budget = balance * 0.10
                    qty = int(budget // contract_cost)
                    if qty > 5: qty = 5
                    
                    if qty >= 1 and balance >= (contract_cost * qty):
                        total_cost = contract_cost * qty
                        balance -= total_cost
                        position = qty
                        entry_price = premium

                        active_trade = {
                            'strike': strike,
                            'expiry_days': days_to_expiry,
                            'expiry_date': current_time_utc + timedelta(days=days_to_expiry),
                            'entry_time': current_time_utc,
                            'type': 'call',
                            'stop_loss': premium * 0.80, # -20%
                            'take_profit': premium * 1.40 # +40%
                        }

                        trades.append({'time': current_time_et, 'type': 'buy_call', 'price': premium, 'qty': qty, 'strike': strike})
                        print(f"[{current_time_et}] BUY CALL  @ {premium:.2f} (Qty: {qty}, Strk: {strike}) | SL: {active_trade['stop_loss']:.2f} | TP: {active_trade['take_profit']:.2f}")

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
                    position = 0
                    active_trade = None

        elif signal == "sell":
            # --- SELL SIGNAL ACTION ---
            if position == 0:
                # Enter Short (Options only)
                if trade_type == "options":
                    strike = round(price)
                    days_to_expiry = 7
                    sigma = current_vol if current_vol > 0 else 0.2
                    premium = black_scholes_price(price, strike, days_to_expiry/365.0, 0.04, sigma, type='put')
                    contract_cost = premium * 100

                    budget = balance * 0.10
                    qty = int(budget // contract_cost)
                    if qty > 5: qty = 5
                    
                    if qty >= 1 and balance >= (contract_cost * qty):
                        total_cost = contract_cost * qty
                        balance -= total_cost
                        position = -qty # Negative denotes Put for logic
                        entry_price = premium

                        active_trade = {
                            'strike': strike,
                            'expiry_days': days_to_expiry,
                            'expiry_date': current_time_utc + timedelta(days=days_to_expiry),
                            'entry_time': current_time_utc,
                            'type': 'put',
                            'stop_loss': premium * 0.80,
                            'take_profit': premium * 1.40
                        }
                        trades.append({'time': current_time_et, 'type': 'buy_put', 'price': premium, 'qty': qty, 'strike': strike})
                        print(f"[{current_time_et}] BUY PUT   @ {premium:.2f} (Qty: {qty}, Strk: {strike}) | SL: {active_trade['stop_loss']:.2f} | TP: {active_trade['take_profit']:.2f}")

            elif position > 0 or (trade_type == "options" and active_trade and active_trade['type'] == 'call'):
                # Exit Bullish Position due to Bearish Signal Flip
                if trade_type == "stock":
                    balance += position * price
                    pnl = (price - entry_price) * position
                    trades.append({'time': current_time_et, 'type': 'sell_stock', 'price': price, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time_et}] SELL STOCK @ {price:.2f} | PnL: {pnl:.2f}")
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
                    position = 0
                    active_trade = None

        # --- EQUITY TRACKING (END OF BAR) ---
        current_equity = balance
        if position != 0 and active_trade:
            if trade_type == "stock":
                current_equity += position * price
            elif trade_type == "options":
                time_held = current_time_utc - active_trade['entry_time']
                days_passed = time_held.total_seconds() / (24 * 3600)
                T_remain = max(0.0001, (active_trade['expiry_days'] - days_passed) / 365.0)
                tracking_premium = black_scholes_price(price, active_trade['strike'], T_remain, 0.04, current_vol, type=active_trade['type'])
                current_equity += abs(position) * tracking_premium * 100

        equity_curve.append({'time': current_time_et, 'balance': current_equity})

    # End of backtest - Mark to Market
    if position != 0 and active_trade:
        last_price = ltf_data.iloc[-1]['close']
        last_time_utc = ltf_data.index[-1]
        last_time_et = last_time_utc.astimezone(ET)

        if trade_type == "stock":
            balance += position * last_price
            trades.append({'time': last_time_et, 'type': 'mtm_stock', 'price': last_price, 'qty': position})
        elif trade_type == "options":
            time_held = last_time_utc - active_trade['entry_time']
            days_passed = time_held.total_seconds() / (24 * 3600)
            T_remain = max(0, (active_trade['expiry_days'] - days_passed) / 365.0)
            # Use last available volatility
            sigma = htf_data.iloc[-1]['volatility']
            exit_premium = black_scholes_price(last_price, active_trade['strike'], T_remain, 0.04, sigma, type=active_trade['type'])
            balance += exit_premium * 100 * abs(position)
            trades.append({'time': last_time_et, 'type': f'mtm_{active_trade["type"]}', 'price': exit_premium, 'qty': abs(position)})
        position = 0
        active_trade = None

    print("="*30)
    print(f"Backtest Complete ({trade_type.upper()}).")
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance:   ${balance:.2f}")
    if initial_balance > 0:
        print(f"Return:          {((balance - initial_balance)/initial_balance)*100:.2f}%")
    print(f"Total Trades:    {len(trades)}")
    
    # --- VISUALIZATION ---
    if equity_curve:
        df_equity = pd.DataFrame(equity_curve)
        df_equity.set_index('time', inplace=True)
        
        # Calculate stats for the plot
        final_return_pct = ((balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
        peak_equity = df_equity['balance'].max()
        max_profit = peak_equity - initial_balance
        
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
            f"Initial Balance: ${initial_balance:,.2f}\n"
            f"Final Balance: ${balance:,.2f}\n"
            f"Total Return: {final_return_pct:.2f}%\n"
            f"Max Profit: ${max_profit:,.2f}\n"
            f"Max Drawdown: {max_drawdown_pct:.2f}%\n"
            f"Total Trades: {len(trades)}"
        )
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(f"SMC Bot Backtest: {symbol} ({trade_type.upper()})", fontsize=14, pad=20)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_dir = os.path.join(os.getcwd(), "backtest-output")
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
    
    args = parser.parse_args()
    
    mode = "options" if args.options else "stock"
    run_backtest(days_back=args.days, symbol=args.symbol, trade_type=mode, initial_balance=args.balance)
