#!/usr/bin/env python3
import argparse
import asyncio
from dataclasses import dataclass
from typing import Optional
import os
import sys

from ib_insync import Stock

# Add project root to path so `app` imports work when script is run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ibkr_manager import ibkr_mgr


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str

@dataclass
class IbError:
    req_id: int
    code: int
    message: str
    symbol: str


class IbErrorCollector:
    def __init__(self):
        self.errors: list[IbError] = []

    def on_error(self, req_id, code, message, contract):
        sym = getattr(contract, "symbol", "") if contract is not None else ""
        self.errors.append(IbError(req_id=int(req_id), code=int(code), message=str(message), symbol=str(sym)))

    def has(self, code: int, symbol: str = "") -> bool:
        s = (symbol or "").upper()
        for e in self.errors:
            if e.code != code:
                continue
            if not s or e.symbol.upper() == s:
                return True
        return False


async def fetch_hist_once(contract, duration, bar_size, what_to_show, timeout_sec):
    try:
        bars = await asyncio.wait_for(
            ibkr_mgr.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            ),
            timeout=timeout_sec,
        )
        if not bars:
            return False, "no bars returned"
        return True, f"{len(bars)} bars"
    except asyncio.TimeoutError:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


async def check_live_quote(symbol: str, timeout_sec: float, errors: IbErrorCollector) -> CheckResult:
    contract = Stock(symbol, "SMART", "USD")
    try:
        qualified = await ibkr_mgr.ib.qualifyContractsAsync(contract)
        if not qualified:
            return CheckResult("Contract qualification", False, "failed")
    except Exception as e:
        return CheckResult("Contract qualification", False, str(e))

    ticker = ibkr_mgr.ib.reqMktData(contract, "", False, False)
    should_cancel = True
    try:
        for _ in range(max(1, int(timeout_sec / 0.5))):
            await asyncio.sleep(0.5)
            mp = ticker.marketPrice()
            if mp is not None and mp > 0:
                return CheckResult("Live quote", True, f"marketPrice={mp}")
        bid = ticker.bid
        ask = ticker.ask
        if (bid is not None and bid > 0) or (ask is not None and ask > 0):
            return CheckResult("Live quote", True, f"bid={bid}, ask={ask}")
        if errors.has(10089, symbol):
            should_cancel = False
            return CheckResult(
                "Live quote",
                False,
                "missing live API subscription (10089). Delayed data may still be available.",
            )
        return CheckResult("Live quote", False, "no valid market price/bid/ask in time window")
    finally:
        if should_cancel:
            ibkr_mgr.ib.cancelMktData(contract)


async def check_hist(symbol: str, duration: str, timeout_sec: float) -> list[CheckResult]:
    contract = Stock(symbol, "SMART", "USD")
    checks = [
        ("Hist 15m TRADES", "15 mins", "TRADES"),
        ("Hist 15m MIDPOINT", "15 mins", "MIDPOINT"),
        ("Hist 1m TRADES", "1 min", "TRADES"),
        ("Hist 1m MIDPOINT", "1 min", "MIDPOINT"),
    ]
    out: list[CheckResult] = []
    for name, bar_size, wts in checks:
        ok, detail = await fetch_hist_once(contract, duration, bar_size, wts, timeout_sec)
        out.append(CheckResult(name, ok, detail))
    return out


def print_summary(results: list[CheckResult], symbol: str):
    print("\n=== IBKR Data Check Summary ===")
    print(f"Symbol: {symbol}")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"- {status:4} | {r.name}: {r.detail}")

    live_ok = any(r.ok for r in results if r.name == "Live quote")
    hist_results = [r for r in results if r.name.startswith("Hist ")]
    hist_pass = sum(1 for r in hist_results if r.ok)

    print("\nInterpretation:")
    if not live_ok and hist_pass == 0:
        print("- Live quote and historical both failed. Likely connection/session/pacing issue, or broad entitlement issue.")
    elif live_ok and hist_pass == 0:
        print("- Live quote works but historical failed. Likely HMDS/pacing/session issue, and possibly historical entitlement scope.")
    elif live_ok and hist_pass > 0:
        print("- Account has at least partial usable data access for this symbol. Timeouts are likely intermittent HMDS/session/pacing.")
    else:
        print("- Mixed result. Re-run this check and inspect TWS/Gateway status and subscriptions.")


async def _run(symbol: str, duration: str, timeout_sec: float):
    print(f"Connecting to IBKR and checking {symbol}...")
    connected = await ibkr_mgr.connect()
    if not connected:
        print("❌ Could not connect to IBKR. Check TWS/Gateway host/port/clientId.")
        return 2

    results: list[CheckResult] = []
    err = IbErrorCollector()
    ibkr_mgr.ib.errorEvent += err.on_error
    try:
        results.append(await check_live_quote(symbol, timeout_sec, err))
        results.extend(await check_hist(symbol, duration, timeout_sec))
        if err.has(10089, symbol):
            results.append(CheckResult("Subscription hint", False, "IBKR code 10089 detected: live API subscription required for this instrument/feed."))
        print_summary(results, symbol)
    finally:
        ibkr_mgr.ib.errorEvent -= err.on_error
        ibkr_mgr.disconnect()

    return 0 if any(r.ok for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description="Check IBKR live/historical market-data accessibility for a symbol.")
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker to test, e.g. SPY or QQQ")
    parser.add_argument("--duration", default="14 D", help="Historical lookback duration (default: '14 D')")
    parser.add_argument("--timeout", type=float, default=12.0, help="Per-request timeout seconds (default: 12)")
    args = parser.parse_args()
    exit_code = asyncio.run(_run(args.symbol.upper(), args.duration, args.timeout))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
