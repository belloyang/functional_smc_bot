import sys
import os
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

# Add root directory to path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from app.bot import (
    generate_signal,
    get_confidence_label,
    precompute_strategy_features,
    get_causal_signal_from_precomputed,
    get_bars,
    send_discord_notification
)


CONF_THRESHOLDS = {
    "all": 0,
    "low": 20,
    "medium": 60,
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
        # --- MARKET HOURS CHECK ---
        et = ZoneInfo("US/Eastern")
        now_et = datetime.now(et)
        is_weekend = now_et.weekday() >= 5
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        if watch and (is_weekend or now_et < market_open or now_et > market_close):
            reason = "Weekend" if is_weekend else "Outside 9:30 AM - 4:00 PM ET"
            print(f"[{now_et.strftime('%I:%M %p')}] 🛑 {reason}. Exiting --watch mode.")
            return

        res = generate_signal(symbol)
        signal, confidence = res if isinstance(res, tuple) else (res, 0)

        if signal and confidence >= threshold:
            key = (signal, confidence)
            line = _format_signal_line(symbol, signal, confidence)
            emoji = "🟢" if signal == "buy" else "🔴"
            # In watch mode, avoid printing the exact same result repeatedly every poll.
            if not watch or key != last_seen_key:
                
                # Get current price for notification
                try:
                    bars = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 1)
                    price = bars['close'].iloc[-1] if bars is not None and not bars.empty else 0
                except Exception:
                    price = 0
                time_str = datetime.now(ZoneInfo("US/Eastern")).strftime("%I:%M %p")
                bias = "BULLISH" if signal == "buy" else "BEARISH"
                send_discord_notification(signal, price, time_str, symbol, bias, confidence)
                print(f"{emoji} |price={price} | {line}")
            last_seen_key = key
        else:
            if not watch:
                print("No signal.")

        if not watch:
            return

        time.sleep(max(1, int(interval_sec)))


def _fetch_for_scan(symbol: str):
    try:
        htf_raw = get_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), 10000)
        ltf_raw = get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), 20000)
    except Exception:
        return None, None, None
    if htf_raw is None or ltf_raw is None or htf_raw.empty or ltf_raw.empty:
        return None, None, None
    return htf_raw, ltf_raw, "app.bot.get_bars"


def scan_today(symbol: str, min_conf: str = "all", days_ago: int = 0):
    threshold = CONF_THRESHOLDS.get(min_conf, 0)
    symbol = symbol.upper()
    et = ZoneInfo("US/Eastern")
    now_et = datetime.now(et)
    if days_ago > 0:
        now_et -= timedelta(days=days_ago)
        
    market_open_et = now_et.replace(hour=9, minute=40, second=0, microsecond=0)
    market_open_utc = market_open_et.astimezone(timezone.utc)
    market_close_et = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
    market_close_utc = market_close_et.astimezone(timezone.utc)
    
    if days_ago > 0:
        end_utc = market_close_utc
    else:
        end_utc = min(datetime.now(timezone.utc), market_close_utc)

    htf_raw, ltf_raw, feed = _fetch_for_scan(symbol)
    if htf_raw is None:
        print("❌ Could not fetch data for scan.")
        return

    htf, ltf = precompute_strategy_features(htf_raw, ltf_raw)
    scan_rows = ltf[(ltf["timestamp"] >= market_open_utc) & (ltf["timestamp"] <= end_utc)]
    count = 0

    print(f"Scan mode: historical replay in 09:40-15:55 ET | source={feed}")
    for _, row in scan_rows.iterrows():
        ts = row["timestamp"]
        res = get_causal_signal_from_precomputed(htf, ltf, ts, ltf_window=200)
        signal, confidence = res if isinstance(res, tuple) else (res, 0)
        if not signal or confidence < threshold:
            continue
        count += 1
        t_et = pd.Timestamp(ts).tz_convert(et).strftime("%I:%M %p")
        label = get_confidence_label(confidence)
        # put green light for buy, red light for sell, and double light for buy since it's more actionable
        emoji = "🟢" if signal == "buy" else "🔴"
        print(f"{emoji} {symbol} | {signal.upper()} at {t_et} | price={row['close']} | confidence={confidence}% [{label}]")

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
    parser.add_argument(
        "--days-ago",
        type=int,
        default=0,
        help="Number of days ago to scan when using --scan-today (default: 0).",
    )
    args = parser.parse_args()

    if args.scan_today:
        scan_today(args.symbol, args.min_conf, args.days_ago)
    else:
        run(args.symbol, args.min_conf, args.watch, args.interval)
