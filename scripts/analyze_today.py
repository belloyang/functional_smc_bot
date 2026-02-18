import sys
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from app import config
from app.bot import (
    generate_signal,
    get_confidence_label,
    precompute_strategy_features,
    get_causal_signal_from_precomputed,
)


CONF_THRESHOLDS = {
    "all": 0,
    "low": 20,
    "medium": 50,
    "high": 80,
}


def _format_signal_line(symbol: str, signal: str, confidence: int) -> str:
    et = ZoneInfo("US/Eastern")
    now_et = datetime.now(et).strftime("%Y-%m-%d %I:%M:%S %p ET")
    label = get_confidence_label(confidence)
    return f"{now_et} | {symbol} | {signal.upper()} | confidence={confidence}% [{label}]"


def run(symbol: str, min_conf: str = "all", watch: bool = False, interval_sec: int = 60):
    threshold = CONF_THRESHOLDS.get(min_conf, 0)
    symbol = symbol.upper()
    print(f"Signal source: app.bot.generate_signal")
    print(f"Symbol: {symbol} | min_conf: {min_conf} ({threshold}) | watch: {watch}")

    last_seen_key = None
    while True:
        res = generate_signal(symbol)
        signal, confidence = res if isinstance(res, tuple) else (res, 0)

        if signal and confidence >= threshold:
            key = (signal, confidence)
            line = _format_signal_line(symbol, signal, confidence)
            # In watch mode, avoid printing the exact same result repeatedly every poll.
            if not watch or key != last_seen_key:
                print(f"🚨 {line}")
            last_seen_key = key
        else:
            if not watch:
                print("No signal.")

        if not watch:
            return

        time.sleep(max(1, int(interval_sec)))


def _fetch_for_scan(symbol: str, start_utc: datetime, end_utc: datetime):
    client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
    for feed in ("sip", "iex"):
        try:
            htf_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start_utc,
                end=end_utc,
                feed=feed,
            )
            ltf_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=start_utc,
                end=end_utc,
                feed=feed,
            )
            htf_raw = client.get_stock_bars(htf_req).df
            ltf_raw = client.get_stock_bars(ltf_req).df
            if not htf_raw.empty and not ltf_raw.empty:
                return htf_raw, ltf_raw, feed
        except Exception:
            continue
    return None, None, None


def scan_today(symbol: str, min_conf: str = "all"):
    threshold = CONF_THRESHOLDS.get(min_conf, 0)
    symbol = symbol.upper()
    et = ZoneInfo("US/Eastern")
    now_et = datetime.now(et)
    market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_open_utc = market_open_et.astimezone(timezone.utc)
    end_utc = datetime.now(timezone.utc)
    lookback_start = end_utc - timedelta(days=14)

    htf_raw, ltf_raw, feed = _fetch_for_scan(symbol, lookback_start, end_utc)
    if htf_raw is None:
        print("❌ Could not fetch data for scan.")
        return

    htf, ltf = precompute_strategy_features(htf_raw, ltf_raw)
    scan_rows = ltf[ltf["timestamp"] >= market_open_utc]
    printed = {}
    count = 0

    print(f"Scan mode: historical replay since market open | feed={feed}")
    for _, row in scan_rows.iterrows():
        ts = row["timestamp"]
        res = get_causal_signal_from_precomputed(htf, ltf, ts, ltf_window=200)
        signal, confidence = res if isinstance(res, tuple) else (res, 0)
        if not signal or confidence < threshold:
            continue
        key = (signal, pd.Timestamp(ts).floor("5min"))
        if key in printed:
            continue
        printed[key] = True
        count += 1
        t_et = pd.Timestamp(ts).tz_convert(et).strftime("%I:%M %p")
        label = get_confidence_label(confidence)
        print(f"🚨 {symbol} | {signal.upper()} at {t_et} | confidence={confidence}% [{label}]")

    if count == 0:
        print("No historical signals detected since market open.")
    else:
        print(f"✅ Historical signals found: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Signal check using app.bot.generate_signal (kept in sync with live bot logic)."
    )
    parser.add_argument("symbol", nargs="?", default="SPY", help="Symbol to analyze")
    parser.add_argument(
        "--min-conf",
        type=str,
        choices=["all", "low", "medium", "high"],
        default="all",
        help="Minimum confidence level to display (default: all)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously check signal every interval (default: one-shot).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds for --watch mode (default: 60).",
    )
    parser.add_argument(
        "--scan-today",
        action="store_true",
        help="Replay today's closed 1m bars and report historical detections (non-live-parity mode).",
    )
    args = parser.parse_args()

    if args.scan_today:
        scan_today(args.symbol, args.min_conf)
    else:
        run(args.symbol, args.min_conf, args.watch, args.interval)
