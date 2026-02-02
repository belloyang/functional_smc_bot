from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import datetime
import sys
import os
# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import config

def test_historical_fetch():
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    
    # Try to fetch data from a month ago
    end_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    start_time = end_time - datetime.timedelta(days=2)
    
    print(f"Fetching from {start_time} to {end_time}")
    
    req = StockBarsRequest(
        symbol_or_symbols=config.SYMBOL,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),
        start=start_time,
        end=end_time
    )
    
    try:
        bars = client.get_stock_bars(req).df
        print("Successfully fetched bars:")
        print(bars.head())
        print(f"Total rows: {len(bars)}")
    except Exception as e:
        print(f"Error fetching historical data: {e}")

if __name__ == "__main__":
    test_historical_fetch()
