from alpaca.trading import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

api_key = "PKGYMDJH2IQXKAR26DBWFXYPW7"
api_secret = "7UKv2d3WfpyE2pxLRiZhNb8LkdZvLyp3bWn5g5z27REd"
trading_client = TradingClient(api_key, api_secret, paper=True)
selected_tickers = [
    "SLV", "USAR", "TSLA", "AAPL",
    "MSFT", "NVDA", "AMZN", "GOOGL",
     "META", "TSLA"]

# cancel all existing orders
orders = trading_client.get_orders()
print('There are currently', len(orders), 'orders:')
for order in orders:
    trading_client.cancel_order_by_id(order.id)
    print('Order cancelled:', order.symbol, order.qty, order.side, order.limit_price, order.id)

def submit_limit_order(symbol, qty, side, limit_price, take_profit_price=None, stop_loss_price=None):
    
    take_profit_request = TakeProfitRequest(limit_price=take_profit_price) if take_profit_price else None
    stop_loss_request = StopLossRequest(stop_price=stop_loss_price) if stop_loss_price else None

    limit_order = LimitOrderRequest(
    symbol=symbol,
    qty=qty,
    side=side,
    type=OrderType.LIMIT,
    time_in_force=TimeInForce.DAY,
    limit_price=limit_price,
    take_profit=take_profit_request,
    stop_loss=stop_loss_request
    )
    trading_client.submit_order(limit_order)
    print('Limit order submitted:', limit_order)

def submit_market_order(symbol, qty, side, take_profit_price=None, stop_loss_price=None):
    
    take_profit_request = TakeProfitRequest(limit_price=take_profit_price) if take_profit_price else None
    stop_loss_request = StopLossRequest(stop_price=stop_loss_price) if stop_loss_price else None

    market_order = MarketOrderRequest(
    symbol=symbol,
    qty=qty,
    side=side,
    type=OrderType.MARKET,
    time_in_force=TimeInForce.DAY,
    take_profit=take_profit_request,
    stop_loss=stop_loss_request
    )
    trading_client.submit_order(market_order)
    print('Market order submitted:', market_order)

# read my account from alpaca
account = trading_client.get_account()
print('Account equity:', account.equity)
print('Account buying power:', account.buying_power)
print('Account cash:', account.cash)

# submit_limit_order("SLV", 100, OrderSide.BUY, 95, take_profit_price=105, stop_loss_price=90);

