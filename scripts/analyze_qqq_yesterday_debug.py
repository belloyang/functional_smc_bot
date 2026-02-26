import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.bot import get_bars, precompute_strategy_features
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

symbol = "QQQ"

htf_raw = get_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), 500)
ltf_raw = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 2000)

htf, ltf = precompute_strategy_features(htf_raw, ltf_raw)
et_tz = ZoneInfo("US/Eastern")
open_dt = datetime.strptime("2026-02-24 09:40:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=et_tz)
close_dt = datetime.strptime("2026-02-24 15:55:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=et_tz)

scan_htf = htf[(htf['timestamp'] >= open_dt) & (htf['timestamp'] <= close_dt)]

print("HTF Bars during yesterday's session:")
for i in range(len(scan_htf)):
    row = scan_htf.iloc[i]
    ts_et = row['timestamp'].astimezone(et_tz).strftime("%I:%M %p")
    adx = row.get('adx', 0)
    print(f"{ts_et} | Close: {row['close']:.2f} | EMA50: {row.get('ema50', 0):.2f} | ADX: {adx:.2f}")

