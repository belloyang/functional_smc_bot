import asyncio
import sys
import os

# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.bot import send_discord_live_trading_notification
from app import __version__

def test_live_notification():
    print(f"Sending test live trading notification to Discord (Main Branch) v{__version__}...")
    
    # Test case 1: Buy Option
    send_discord_live_trading_notification(
        signal="buy_option_C",
        symbol="SPY260320C00561000",
        order_details={
            "Action": "BUY",
            "Qty": 2,
            "Entry": 4.50,
            "SL": 3.60,
            "TP": 6.75,
            "Type": "Market Bracket"
        },
        confidence=85,
        strategy_bias="bullish"
    )
    
    # Test case 2: Flip Exit
    send_discord_live_trading_notification(
        signal="flip_close_long_stock",
        symbol="SPY",
        order_details={
            "Action": "SELL (Market)",
            "Qty": 100,
            "Price": 560.25,
            "Type": "Trend Flip Exit"
        },
        confidence=65,
        strategy_bias="bearish"
    )

    # Test case 3: Virtual Stop Hit
    send_discord_live_trading_notification(
        signal="option_virtual_stop_hit",
        symbol="SPY260320P00550000",
        order_details={
            "Action": "CLOSE (Market)",
            "PnL": "-20.5%",
            "Stop": "-20.0%",
            "Type": "VIRTUAL STOP"
        },
        confidence=0,
        strategy_bias="NEUTRAL"
    )

    # Test case 4: Daily Halt
    send_discord_live_trading_notification(
        signal="daily_halt_liquidation",
        symbol="SPY",
        order_details={
            "Action": "CLOSE (Market)",
            "Reason": "Daily Loss Limit Hit (-3%)",
            "Equity": 9700.50
        },
        confidence=0,
        strategy_bias="NEUTRAL"
    )
    
    print("✅ Test notifications sent. Please check your Discord channel.")

if __name__ == "__main__":
    test_live_notification()
