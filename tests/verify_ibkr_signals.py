import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.ibkr_manager import ibkr_mgr
from app.bot import generate_signal, get_confidence_label

async def verify_signal_parity(symbol="QQQ"):
    print(f"📡 Verifying signal parity for {symbol} on IBKR branch...")
    connected = await ibkr_mgr.connect()
    if not connected:
        print("❌ Failed to connect to IBKR.")
        return

    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
        print(f"Actual Time: {now_et.strftime('%I:%M:%S %p ET')}")
        
        # Call generate_signal (which now drops the last candle and is strictly causal)
        res = await generate_signal(symbol)
        
        if res:
            signal, confidence = res
            if signal:
                label = get_confidence_label(confidence)
                print(f"🚨 SIGNAL DETECTED: {signal.upper()} | Confidence: {confidence}% [{label}]")
            else:
                print("ℹ️ No signal currently active according to bot logic.")
        else:
            print("❌ generate_signal returned None (likely not enough history or bias filtering).")

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ibkr_mgr.disconnect()

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    asyncio.run(verify_signal_parity(symbol))
