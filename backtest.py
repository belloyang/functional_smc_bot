import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta, timezone
import alpaca_config as config
from functional_smc_bot import get_strategy_signal

def run_backtest(days_back=300, symbol=None):
    if symbol is None:
        symbol = config.SYMBOL
        
    print(f"Starting backtest for {symbol} over last {days_back} days...")
    
    # 1. Fetch Data
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    
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

    # Reset index to access 'timestamp' as a column, but keep symbol index if present
    # Alpaca multi-symbol returns MultiIndex [symbol, timestamp]. Single symbol might be same.
    # We used single symbol request.
    
    # Ensure indices are handled
    htf_data = htf_data.reset_index() 
    ltf_data = ltf_data.reset_index()
    
    # Set timestamp as index for easier slicing
    htf_data.set_index('timestamp', inplace=True)
    ltf_data.set_index('timestamp', inplace=True)
    
    # Sort just in case
    htf_data.sort_index(inplace=True)
    ltf_data.sort_index(inplace=True)

    print(f"Data loaded. HTF: {len(htf_data)} bars, LTF: {len(ltf_data)} bars.")

    # 2. Simulation Loop
    # Best Practice: Use a fixed balance for backtesting to standardize results (e.g., $100k)
    # This makes it easier to compare strategy performance regardless of your actual account size.
    initial_balance = 10000.0 
    
    # If you prefer to use your actual account balance:
    # try:
    #     initial_balance = float(config.ACCOUNT_BALANCE)
    # except (ValueError, TypeError):
    #     initial_balance = 100000.0
        
    balance = initial_balance
    position = 0 # Quantity
    entry_price = 0
    trades = []
    print(f"Initial balance: {balance}")
    
    # We iterate through LTF candles.
    # To run strategy, we need "past" data relative to current time.
    # Optimization: Instead of sliding window every single minute, we can just pass slice.
    
    # We need at least enough data for indicators
    # HTF requires 50 bars. LTF requires ~20 (for swing/FVG checks).
    
    start_idx = 200 # Skip first N bars to allow indicators to warm up
    
    for i in range(start_idx, len(ltf_data)):
        current_bar = ltf_data.iloc[i]
        current_time = current_bar.name
        
        # Slice data up to current time (exclusive of future)
        # Note: In real live trading at time T, we have closed bars up to T-1 (or T if just closed).
        # We simulate "just closed bar i".
        
        ltf_slice = ltf_data.iloc[max(0, i-200):i+1] # Pass last 200 LTF
        
        # Get HTF data up to current_time
        htf_slice = htf_data[htf_data.index <= current_time]
        if len(htf_slice) < 50:
            continue
            
        htf_slice = htf_slice.iloc[-200:] # Limit to last 200 HTF for speed

        # Run Strategy
        signal = get_strategy_signal(htf_slice, ltf_slice)
        
        price = current_bar['close']
        
        if signal == "buy" and position == 0:
            # Enter Long
            qty = int(balance * 0.01 / (price * 0.005)) # Simple risk sizing from bot
            if qty > 0:
                cost = qty * price
                balance -= cost # Deduct cash (simplified, ignoring margin)
                position = qty
                entry_price = price
                trades.append({'time': current_time, 'type': 'buy', 'price': price, 'qty': qty})
                print(f"[{current_time}] BUY  @ {price:.2f} | Qty: {qty}")

        elif signal == "sell" and position > 0:
            # Exit Long (simplified: signal 'sell' closes long, doesn't short for now unless strategy implies flip)
            # The bot logic was: bias bearish -> sell signal. 
            # If we are long and get a sell signal (bearish bias + bear OB), we should close.
             
            proceeds = position * price
            balance += proceeds
            pnl = (price - entry_price) * position
            
            trades.append({'time': current_time, 'type': 'sell', 'price': price, 'qty': position, 'pnl': pnl})
            print(f"[{current_time}] SELL @ {price:.2f} | PnL: {pnl:.2f}")
            position = 0
            
    
    # End of backtest
    # Mark to market if still holding
    if position > 0:
        last_price = ltf_data.iloc[-1]['close']
        balance += position * last_price
        
    print("="*30)
    print(f"Backtest Complete.")
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance:   ${balance:.2f}")
    print(f"Return:          {((balance - initial_balance)/initial_balance)*100:.2f}%")
    print(f"Total Trades:    {len(trades)}")

if __name__ == "__main__":
    import sys
    target_symbol = sys.argv[1] if len(sys.argv) > 1 else config.SYMBOL
    run_backtest(symbol=target_symbol)
