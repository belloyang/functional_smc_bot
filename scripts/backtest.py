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
from app.bot import get_strategy_signal

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
    
    # Option specific state
    option_contract = None # {'strike': K, 'expiry': date, 'type': 'call'/'put', 'bars_to_expiry_start': N}
    
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
        
        # --- OPTIONS PRICING SIMULATION ---
        current_option_price = 0
        if position != 0 and trade_type == "options" and option_contract:
            time_held = current_time_utc - option_contract['entry_time']
            days_passed = time_held.total_seconds() / (24 * 3600)
            T_remain = max(0.0001, (option_contract['expiry_days'] - days_passed) / 365.0)
            sigma = current_vol if current_vol > 0 else 0.2
            
            # Recalculate Option Value
            current_option_price = black_scholes_price(price, option_contract['strike'], T_remain, 0.04, sigma, type=option_contract['type'])
            
            # --- RISK CHECKS (Before Signal) ---
            
            # 1. Stop Loss Hit
            if current_option_price <= option_contract['stop_loss']:
                exit_price = option_contract['stop_loss'] # In simulation, we slip to SL price
                proceeds = exit_price * 100 * abs(position)
                balance += proceeds
                pnl = (exit_price - entry_price) * 100 * abs(position)
                exit_type = f"stop_loss_{option_contract['type']}"
                trades.append({'time': current_time_et, 'type': exit_type, 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                print(f"[{current_time_et}] 🛑 STOP HIT   @ {exit_price:.2f} | PnL: {pnl:.2f}")
                position = 0
                option_contract = None
                continue # Trade closed, next bar

            # 2. Take Profit Hit
            if current_option_price >= option_contract['take_profit']:
                exit_price = option_contract['take_profit']
                proceeds = exit_price * 100 * abs(position)
                balance += proceeds
                pnl = (exit_price - entry_price) * 100 * abs(position)
                exit_type = f"take_profit_{option_contract['type']}"
                trades.append({'time': current_time_et, 'type': exit_type, 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                print(f"[{current_time_et}] 🎯 TARGET HIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                position = 0
                option_contract = None
                continue
                
            # 3. Active Management (Update Stops)
            unrealized_pl_pct = (current_option_price - entry_price) / entry_price
            
            # Rule A: Break-Even (>15% Profit)
            if unrealized_pl_pct > 0.15 and option_contract['stop_loss'] < entry_price:
                option_contract['stop_loss'] = entry_price
                print(f"[{current_time_et}] 🛡️ MOVED SL TO BE ({entry_price:.2f})")
                
            # Rule B: Profit Lock (>30% Profit -> Lock 15%)
            if unrealized_pl_pct > 0.30:
                lock_price = entry_price * 1.15
                if option_contract['stop_loss'] < lock_price:
                    option_contract['stop_loss'] = lock_price
                    print(f"[{current_time_et}] 💰 LOCKED PROFIT ({lock_price:.2f})")
            
            # 4. Expiry Guard (<1 Day left)
            if T_remain * 365.0 < 1.0:
                 exit_price = current_option_price
                 proceeds = exit_price * 100 * abs(position)
                 balance += proceeds
                 pnl = (exit_price - entry_price) * 100 * abs(position)
                 trades.append({'time': current_time_et, 'type': 'expiry_close', 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                 print(f"[{current_time_et}] ⚠️ EXPIRY EXIT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                 position = 0
                 option_contract = None
                 continue

        # --- ENTRY LOGIC ---
        if signal == "buy":
            # --- BUY SIGNAL ACTION ---
            if position == 0:
                # Enter Long (Stock or Call)
                if trade_type == "stock":
                    qty = int(balance * 0.01 / (price * 0.005))
                    if qty > 0:
                        cost = qty * price
                        balance -= cost
                        position = qty
                        entry_price = price
                        trades.append({'time': current_time_et, 'type': 'buy_stock', 'price': price, 'qty': qty})
                        print(f"[{current_time_et}] BUY STOCK @ {price:.2f} | Qty: {qty}")
                
                elif trade_type == "options":
                    strike = round(price)
                    days_to_expiry = 7
                    T = days_to_expiry / 365.0
                    r = 0.04
                    sigma = current_vol if current_vol > 0 else 0.2
                    premium = black_scholes_price(price, strike, T, r, sigma, type='call')
                    contract_cost = premium * 100
                    
                    if balance >= contract_cost:
                        balance -= contract_cost
                        position = 1
                        entry_price = premium
                        
                        # Set SL/TP
                        sl = premium * 0.80
                        tp = premium * 1.40
                        
                        option_contract = {
                            'strike': strike, 
                            'expiry_days': days_to_expiry, 
                            'entry_time': current_time_utc, 
                            'type': 'call',
                            'stop_loss': sl,
                            'take_profit': tp
                        }
                        
                        trades.append({'time': current_time_et, 'type': 'buy_call', 'price': premium, 'qty': 1, 'strike': strike})
                        print(f"[{current_time_et}] BUY CALL  @ {premium:.2f} (Strk: {strike}) | SL: {sl:.2f} | TP: {tp:.2f}")

            elif position < 0 or (trade_type == "options" and option_contract and option_contract['type'] == 'put'):
                # Exit Bearish Position due to Signal Flip
                if trade_type == "stock":
                    # (Stock doesn't support short in this script yet, but for logic consistency)
                    pass
                elif trade_type == "options":
                    exit_price = current_option_price # Calculated above
                    proceeds = exit_price * 100 * abs(position)
                    balance += proceeds
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': 'flip_exit_put', 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] FLIP EXIT PUT @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    option_contract = None
                    
                    # Immediate Entry Logic for Flip?
                    # For simulation simplicity, we wait for next bar to enter to avoid complex state updates in one loop.
                    # or simply copy entry logic here. Let's wait.

        elif signal == "sell":
            # --- SELL SIGNAL ACTION ---
            if position == 0:
                # Enter Bearish (Put)
                if trade_type == "options":
                    strike = round(price)
                    days_to_expiry = 7
                    T = days_to_expiry / 365.0
                    r = 0.04
                    sigma = current_vol if current_vol > 0 else 0.2
                    premium = black_scholes_price(price, strike, T, r, sigma, type='put')
                    contract_cost = premium * 100
                    
                    if balance >= contract_cost:
                        balance -= contract_cost
                        position = -1 # Negative for tracking direction conceptually, though qty is 1 contract
                        entry_price = premium
                        
                        # Set SL/TP
                        sl = premium * 0.80
                        tp = premium * 1.40
                        
                        option_contract = {
                            'strike': strike, 
                            'expiry_days': days_to_expiry, 
                            'entry_time': current_time_utc, 
                            'type': 'put',
                            'stop_loss': sl,
                            'take_profit': tp
                        }
                        trades.append({'time': current_time_et, 'type': 'buy_put', 'price': premium, 'qty': 1, 'strike': strike})
                        print(f"[{current_time_et}] BUY PUT   @ {premium:.2f} (Strk: {strike}) | SL: {sl:.2f} | TP: {tp:.2f}")
                
                elif trade_type == "stock":
                    # Sell stock if held (handled below)
                    pass

            elif position > 0 or (trade_type == "options" and option_contract and option_contract['type'] == 'call'):
                # Exit Bullish Position due to Signal Flip
                if trade_type == "stock":
                    proceeds = position * price
                    balance += proceeds
                    pnl = (price - entry_price) * position
                    trades.append({'time': current_time_et, 'type': 'sell_stock', 'price': price, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time_et}] SELL STOCK @ {price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                elif trade_type == "options":
                    exit_price = current_option_price
                    proceeds = exit_price * 100 * abs(position)
                    balance += proceeds
                    pnl = (exit_price - entry_price) * 100 * abs(position)
                    trades.append({'time': current_time_et, 'type': 'flip_exit_call', 'price': exit_price, 'qty': abs(position), 'pnl': pnl})
                    print(f"[{current_time_et}] FLIP EXIT CALL @ {exit_price:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    option_contract = None

        # --- EQUITY TRACKING (END OF BAR) ---
        current_equity = balance
        if position != 0:
            if trade_type == "stock":
                current_equity += position * price
            elif trade_type == "options" and option_contract:
                # We need to ensure current_option_price is available or calculated
                time_held_curr = current_time_utc - option_contract['entry_time']
                days_passed_curr = time_held_curr.total_seconds() / (24 * 3600)
                T_remain_curr = max(0.0001, (option_contract['expiry_days'] - days_passed_curr) / 365.0)
                # Use current_vol if it was calculated in the risk checks, else fallback
                sigma_curr = current_vol if 'current_vol' in locals() and current_vol > 0 else 0.2
                tracking_premium = black_scholes_price(price, option_contract['strike'], T_remain_curr, 0.04, sigma_curr, type=option_contract['type'])
                current_equity += abs(position) * tracking_premium * 100
        
        equity_curve.append({'time': current_time_et, 'balance': current_equity})

    # End of backtest - Mark to Market
    if position != 0:
        last_price = ltf_data.iloc[-1]['close']
        last_time_utc = ltf_data.index[-1]
        last_time_et = last_time_utc.astimezone(ET)
        
        if trade_type == "stock":
            balance += position * last_price
            trades.append({'time': last_time_et, 'type': 'mtm_stock', 'price': last_price, 'qty': position})
        elif trade_type == "options":
             if option_contract:
                time_held = last_time_utc - option_contract['entry_time']
                days_passed = time_held.total_seconds() / (24 * 3600)
                T_remain = max(0, (option_contract['expiry_days'] - days_passed) / 365.0)
                sigma = htf_data.iloc[-1]['volatility']
                exit_premium = black_scholes_price(last_price, option_contract['strike'], T_remain, 0.04, sigma, type=option_contract['type'])
                balance += exit_premium * 100 * abs(position)
                trades.append({'time': last_time_et, 'type': f'mtm_{option_contract["type"]}', 'price': exit_premium, 'qty': abs(position)})
        
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
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_equity.index, df_equity['balance'], label='Account Equity', color='#2ca02c')
        plt.fill_between(df_equity.index, df_equity['balance'], initial_balance, where=(df_equity['balance'] >= initial_balance), color='#2ca02c', alpha=0.3)
        plt.fill_between(df_equity.index, df_equity['balance'], initial_balance, where=(df_equity['balance'] < initial_balance), color='#d62728', alpha=0.3)
        
        plt.title(f"SMC Bot Backtest: {symbol} ({trade_type.upper()})", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(os.getcwd(), "backtest_equity.png")
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
