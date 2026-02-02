import pandas as pd
import numpy as np
# from scipy.stats import norm 
import math
from alpaca.data.historical import StockHistoricalDataClient

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta, timezone
import alpaca_config as config
from functional_smc_bot import get_strategy_signal

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

def run_backtest(days_back=30, symbol=None, trade_type="stock"):
    if symbol is None:
        symbol = config.SYMBOL
        
    print(f"Starting backtest for {symbol} over last {days_back} days (Type: {trade_type})...")
    
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
    initial_balance = 10000.0 
    
    balance = initial_balance
    position = 0 # Quantity (Shares or Contracts)
    entry_price = 0 # Stock price or Option premium per share
    
    # Option specific state
    option_contract = None # {'strike': K, 'expiry': date, 'type': 'call'/'put', 'bars_to_expiry_start': N}
    
    trades = []
    print(f"Initial balance: {balance}")
    
    start_idx = 400 # Warmup for vol and indicators
    
    for i in range(start_idx, len(ltf_data)):
        current_bar = ltf_data.iloc[i]
        current_time = current_bar.name
        
        # Get HTF data up to current_time
        htf_slice = htf_data[htf_data.index <= current_time]
        if len(htf_slice) < 50:
            continue
            
        current_vol = htf_slice.iloc[-1]['volatility']
        htf_slice = htf_slice.iloc[-200:] # Limit to last 200 HTF for speed
        ltf_slice = ltf_data.iloc[max(0, i-200):i+1] # Pass last 200 LTF

        # Run Strategy
        signal = get_strategy_signal(htf_slice, ltf_slice)
        
        price = current_bar['close']
        
        # --- ENTRY LOGIC ---
        if signal == "buy" and position == 0:
            if trade_type == "stock":
                # Stock Entry
                qty = int(balance * 0.01 / (price * 0.005)) # Risk-based sizing
                if qty > 0:
                    cost = qty * price
                    balance -= cost
                    position = qty
                    entry_price = price
                    trades.append({'time': current_time, 'type': 'buy_stock', 'price': price, 'qty': qty})
                    print(f"[{current_time}] BUY STOCK @ {price:.2f} | Qty: {qty}")
            
            elif trade_type == "options":
                # Option Entry (Bullish -> Buy Call)
                # Select Contract: ATM, 7 Days out (approx 1 week)
                strike = round(price)
                days_to_expiry = 7
                T = days_to_expiry / 365.0
                r = 0.04 # Risk free rate 4%
                sigma = current_vol if current_vol > 0 else 0.2
                
                # Black-Scholes Price
                premium = black_scholes_price(price, strike, T, r, sigma, type='call')
                
                # Buy 1 Contract (100 shares)
                # Realistically we might buy more, but let's stick to 1 or risk-based
                # Cost = premium * 100
                contract_cost = premium * 100
                
                if balance >= contract_cost:
                    balance -= contract_cost
                    position = 1 # 1 Contract
                    entry_price = premium
                    option_contract = {
                        'strike': strike,
                        'expiry_days': days_to_expiry,
                        'entry_time': current_time,
                        'type': 'call'
                    }
                    trades.append({'time': current_time, 'type': 'buy_call', 'price': premium, 'qty': 1, 'strike': strike})
                    print(f"[{current_time}] BUY CALL  @ {premium:.2f} (Strk: {strike}) | Vol: {sigma:.2f}")

        # --- EXIT LOGIC ---
        elif signal == "sell" and position > 0:
            # Strategies for "sell" signal usually mean exit long or go short
            # Here assuming exit long.
            
            if trade_type == "stock":
                proceeds = position * price
                balance += proceeds
                pnl = (price - entry_price) * position
                trades.append({'time': current_time, 'type': 'sell_stock', 'price': price, 'qty': position, 'pnl': pnl})
                print(f"[{current_time}] SELL STOCK @ {price:.2f} | PnL: {pnl:.2f}")
                position = 0
                
            elif trade_type == "options":
                if option_contract and option_contract['type'] == 'call':
                    # Sell to Close Call
                    # Calculate remaining time T
                    # How many days passed?
                    time_held = current_time - option_contract['entry_time']
                    days_passed = time_held.total_seconds() / (24 * 3600)
                    T_remain = (option_contract['expiry_days'] - days_passed) / 365.0
                    
                    if T_remain <= 0:
                        # Expired
                         exit_premium = max(0, price - option_contract['strike'])
                    else:
                        sigma = current_vol if current_vol > 0 else 0.2
                        exit_premium = black_scholes_price(price, option_contract['strike'], T_remain, 0.04, sigma, type='call')
                        
                    proceeds = exit_premium * 100 * position
                    balance += proceeds
                    pnl = (exit_premium - entry_price) * 100 * position
                    
                    trades.append({'time': current_time, 'type': 'sell_call', 'price': exit_premium, 'qty': position, 'pnl': pnl})
                    print(f"[{current_time}] SELL CALL @ {exit_premium:.2f} | PnL: {pnl:.2f}")
                    position = 0
                    option_contract = None

    # End of backtest - Mark to Market
    if position > 0:
        last_price = ltf_data.iloc[-1]['close']
        if trade_type == "stock":
            balance += position * last_price
        elif trade_type == "options":
             # Value remaining options
             if option_contract:
                time_held = ltf_data.index[-1] - option_contract['entry_time']
                days_passed = time_held.total_seconds() / (24 * 3600)
                T_remain = max(0, (option_contract['expiry_days'] - days_passed) / 365.0)
                sigma = htf_data.iloc[-1]['volatility']
                exit_premium = black_scholes_price(last_price, option_contract['strike'], T_remain, 0.04, sigma, type=option_contract['type'])
                balance += exit_premium * 100 * position
        
    print("="*30)
    print(f"Backtest Complete ({trade_type.upper()}).")
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance:   ${balance:.2f}")
    if initial_balance > 0:
        print(f"Return:          {((balance - initial_balance)/initial_balance)*100:.2f}%")
    print(f"Total Trades:    {len(trades)}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default=config.SYMBOL, help="Symbol to backtest")
    parser.add_argument("--options", action="store_true", help="Run backtest with Options instead of Stock")
    args = parser.parse_args()
    
    mode = "options" if args.options else "stock"
    run_backtest(symbol=args.symbol, trade_type=mode)
