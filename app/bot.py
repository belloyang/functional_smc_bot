import pandas as pd
import numpy as np
import pandas_ta as ta
import re
import json
from unittest.mock import patch
import os
import signal
import sys
import argparse
import time
import asyncio
import requests
from ib_insync import *
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    OptionSnapshotRequest,
    OptionLatestQuoteRequest,
    OptionChainRequest,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide, OrderStatus, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, StopLossRequest, TakeProfitRequest, ReplaceOrderRequest
# Setup ib_insync utility loop for async environments
util.patchAsyncio()
try:
    from . import config
    from .ibkr_manager import ibkr_mgr
except ImportError:
    import config
    from ibkr_manager import ibkr_mgr


# ================= CONFIG =================

SYMBOL = config.SYMBOL
RISK_PER_TRADE = config.RISK_PER_TRADE

# ================= CLIENTS =================

# Global ib is replaced by dynamic ibkr_mgr.ib in functions

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

async def _req_historical_with_retry(
    req_contract,
    *,
    duration_str,
    bar_size,
    what_to_show,
    use_rth=False,
    format_date=1,
    keep_up_to_date=False,
    log_symbol=None,
):
    """
    Wrapper around IBKR historical requests with timeout/backoff.
    Reduces long hangs and noisy cascades of cancelled requests.
    """
    timeout_sec = float(getattr(config, "IBKR_HIST_TIMEOUT_SEC", 12))
    max_attempts = max(1, int(getattr(config, "IBKR_HIST_MAX_ATTEMPTS", 2)))
    backoff_base = float(getattr(config, "IBKR_HIST_BACKOFF_BASE_SEC", 0.8))

    symbol = log_symbol or getattr(req_contract, "symbol", "?")
    exchange = getattr(req_contract, "exchange", "?")

    for attempt in range(1, max_attempts + 1):
        if not ibkr_mgr.ib or not ibkr_mgr.ib.isConnected():
            print(f"⚠️ IBKR not connected. Skip historical request for {symbol}.")
            return []

        try:
            bars = await asyncio.wait_for(
                ibkr_mgr.ib.reqHistoricalDataAsync(
                    req_contract,
                    endDateTime="",
                    durationStr=duration_str,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=format_date,
                    keepUpToDate=keep_up_to_date,
                ),
                timeout=timeout_sec,
            )
            if bars:
                return bars
        except asyncio.TimeoutError:
            print(
                f"⚠️ Historical timeout {attempt}/{max_attempts} for "
                f"{symbol} {bar_size} {what_to_show} @{exchange}"
            )
        except Exception as e:
            msg = str(e).lower()
            # 162 cancellations are common after timeouts/retries; keep logs concise.
            if "historical data query cancelled" not in msg:
                print(
                    f"⚠️ Historical request failed {attempt}/{max_attempts} "
                    f"for {symbol} {bar_size} {what_to_show} @{exchange}: {e}"
                )

        if attempt < max_attempts:
            await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))

    return []

async def get_bars(contract_or_symbol, timeframe, limit):
    """
    Fetch historical bars from IBKR.
    """
    bar_size = '1 min'
    if '15Min' in timeframe:
        bar_size = '15 mins'
    elif '1Day' in timeframe:
        bar_size = '1 day'

    # Create or use Contract
    if isinstance(contract_or_symbol, Contract):
        contract = contract_or_symbol
        symbol = contract.symbol
    else:
        symbol = contract_or_symbol
        contract = Stock(symbol, 'SMART', 'USD')

    is_option_contract = isinstance(contract, Option)

    # Sequential fallback tuned by security type:
    # - Stocks: request only stock-appropriate exchange/data combos.
    # - Options: broader fallback across whatToShow and exchanges.
    if is_option_contract:
        what_to_show_options = ['MIDPOINT', 'TRADES', 'BID', 'ASK']
        exchanges = ['SMART', 'CBOE', 'PHLX', 'ARCA']
    else:
        # Avoid option-only exchanges for stocks; these produce repetitive 162 errors.
        what_to_show_options = ['TRADES', 'MIDPOINT']
        exchanges = ['SMART']

    bars = None
    
    for wts in what_to_show_options:
        for exch in exchanges:
            # Do not mutate the original contract object between retries.
            if isinstance(contract, Stock):
                req_contract = Stock(symbol, exch, 'USD')
            elif isinstance(contract, Option):
                req_contract = Option(
                    contract.symbol,
                    contract.lastTradeDateOrContractMonth,
                    contract.strike,
                    contract.right,
                    exch
                )
                req_contract.tradingClass = contract.tradingClass
                req_contract.multiplier = contract.multiplier
            else:
                req_contract = contract
                req_contract.exchange = exch

            bars = await _req_historical_with_retry(
                req_contract,
                duration_str="14 D",  # Match analyze_today.py for EMA50 stability
                bar_size=bar_size,
                what_to_show=wts,
                use_rth=False,
                format_date=1,
                keep_up_to_date=False,
                log_symbol=symbol,
            )
            if bars:
                break
        if bars: break
    
    if not bars:
        # Final attempt: try a larger bar size just to get ANY price
        bars = await _req_historical_with_retry(
            contract,
            duration_str="14 D",
            bar_size="1 day",
            what_to_show="MIDPOINT",
            use_rth=False,
            format_date=1,
            keep_up_to_date=False,
            log_symbol=symbol,
        )
    
    if not bars:
        print(f"No bars returned for {symbol}")
        return pd.DataFrame()

    df = util.df(bars)
    # Map column names to strategy expectations
    df = df.rename(columns={'date': 'timestamp'})
    
    if len(df) > limit:
        df = df.iloc[-limit:]
        
    return df

async def get_latest_price_fallback(contract_or_symbol):
    """
    Fetches the latest close price from historical data if real-time ticker fails.
    """
    try:
        # 1. Try a quick ticker update wait first (if it's a contract)
        if isinstance(contract_or_symbol, Contract):
            ibkr_mgr.ib.reqMktData(contract_or_symbol)
            # Wait up to 5s specifically for ticker update
            for _ in range(10):
                await asyncio.sleep(0.5)
                ticker = ibkr_mgr.ib.ticker(contract_or_symbol)
                if ticker and ticker.marketPrice() > 0 and not np.isnan(ticker.marketPrice()):
                    return float(ticker.marketPrice())

        # 2. Try Historical Bars (1Min)
        df = await get_bars(contract_or_symbol, "1Min", 1)
        if not df.empty:
            return float(df['close'].iloc[-1])
            
        # 3. Try Historical Bars (1Day) as ultimate IBKR fallback
        df_day = await get_bars(contract_or_symbol, "1Day", 1)
        if not df_day.empty:
            return float(df_day['close'].iloc[-1])
        
        # 3.5 Try 'TRADES' for Options (Sometimes MIDPOINT is restricted)
        if isinstance(contract_or_symbol, Option):
            bars = await _req_historical_with_retry(
                contract_or_symbol,
                duration_str="1 D",
                bar_size="1 min",
                what_to_show="TRADES",
                use_rth=False,
                format_date=1,
                keep_up_to_date=False,
                log_symbol=contract_or_symbol.symbol,
            )
            if bars:
                return float(bars[-1].close)
            
        # 4. Try Alpaca Fallback (Last Resort)
        try:
            # Note: We need to parse the IBKR contract/symbol into Alpaca's format
            if isinstance(contract_or_symbol, Option):
                # Format: SPY260320C00561000
                symbol = contract_or_symbol.symbol
                # lastTradeDateOrContractMonth is YYYYMMDD. Alpaca needs YYMMDD.
                raw_exp = contract_or_symbol.lastTradeDateOrContractMonth
                expiry = raw_exp[2:] # YYMMDD if YYYYMMDD
                right = contract_or_symbol.right
                # OCC format for Alpaca: [SYM][YYMMDD][C/P][STRIKE (8 digits, 3 decimal implied)]
                strike_val = int(round(contract_or_symbol.strike * 1000))
                strike_str = f"{strike_val:08d}"
                alp_occ = f"{symbol}{expiry}{right}{strike_str}"
                
                print(f"ℹ️ IBKR failed for {contract_or_symbol.localSymbol}. Using Alpaca fallback for {alp_occ}...")
                data_client = OptionHistoricalDataClient(config.API_KEY, config.API_SECRET)
                
                # Try COMPACT format first
                try:
                    latest_quote = data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=alp_occ))
                    if latest_quote and alp_occ in latest_quote:
                        q = latest_quote[alp_occ]
                        if q.ask_price > 0: return float(q.ask_price)
                except Exception as e1:
                    # print(f"Alpaca compact format failed: {e1}")
                    pass
                
                # Try padded format
                try:
                    alp_occ_padded = f"{symbol.ljust(6)}{expiry}{right}{strike_str}"
                    latest_quote = data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=alp_occ_padded))
                    if latest_quote and alp_occ_padded in latest_quote:
                        q = latest_quote[alp_occ_padded]
                        if q.ask_price > 0: return float(q.ask_price)
                except Exception as e2:
                    # print(f"Alpaca padded format failed: {e2}")
                    pass
            else:
                symbol = contract_or_symbol.symbol if isinstance(contract_or_symbol, Contract) else contract_or_symbol
                print(f"ℹ️ IBKR failed. Using Alpaca fallback for {symbol}...")
                data_client = StockHistoricalDataClient(config.API_KEY, config.API_SECRET)
                latest_quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
                if latest_quote and symbol in latest_quote:
                    q = latest_quote[symbol]
                    price = (q.ask_price + q.bid_price) / 2 if q.ask_price > 0 and q.bid_price > 0 else q.ask_price
                    if price > 0: return float(price)
        except Exception as alp_e:
            print(f"Alpaca fallback failed: {alp_e}")
            
    except Exception as e:
        sym = contract_or_symbol.symbol if isinstance(contract_or_symbol, Contract) else contract_or_symbol
        print(f"Error in price fallback for {sym}: {e}")


def _round_to_tick(price: float, tick: float) -> float:
    """Round a price to the nearest valid IBKR tick increment."""
    if tick <= 0:
        tick = 0.01
    return round(round(price / tick) * tick, 8)
    return None


def _build_fill_friendly_bracket(ib, action, qty, entry_price, tp_price, sl_price, min_tick=0.01):
    """
    Build an IBKR bracket order with a fill-friendly entry policy.

    Config:
    - BRACKET_ENTRY_MODE: 'market' (default) or 'loose_limit'
    - BRACKET_ENTRY_SLIPPAGE_BPS: extra limit looseness for loose_limit mode
    """
    action = str(action).upper()
    mode = str(getattr(config, 'BRACKET_ENTRY_MODE', 'market')).strip().lower()

    bracket = ib.bracketOrder(action, qty, float(entry_price), float(tp_price), float(sl_price))
    parent = bracket[0]

    if mode in {'market', 'mkt'}:
        parent.orderType = 'MKT'
        if hasattr(parent, 'lmtPrice'):
            parent.lmtPrice = 0
        print(f"ℹ️ Bracket entry mode=market ({action})")
        return bracket

    slippage_bps = float(getattr(config, 'BRACKET_ENTRY_SLIPPAGE_BPS', 15))
    slippage_bps = max(0.0, slippage_bps)
    if action == 'BUY':
        loose_entry = float(entry_price) * (1.0 + slippage_bps / 10000.0)
    else:
        loose_entry = float(entry_price) * (1.0 - slippage_bps / 10000.0)
    parent.lmtPrice = _round_to_tick(loose_entry, min_tick)
    print(f"ℹ️ Bracket entry mode=loose_limit ({action}) bps={slippage_bps:.1f} -> {parent.lmtPrice}")
    return bracket


def _is_terminal_order_status(status: str) -> bool:
    status = str(status or "").lower()
    return status in {"filled", "cancelled", "inactive", "api cancelled"}


async def _wait_for_parent_fill(ib, parent_trade, timeout_sec: float) -> bool:
    end_time = time.monotonic() + max(0.0, timeout_sec)
    while time.monotonic() < end_time:
        status = str(getattr(parent_trade.orderStatus, "status", "") or "")
        filled = float(getattr(parent_trade.orderStatus, "filled", 0) or 0)
        remaining = float(getattr(parent_trade.orderStatus, "remaining", 0) or 0)
        if filled > 0 and remaining <= 0:
            return True
        if _is_terminal_order_status(status) and status.lower() != "filled":
            return False
        await asyncio.sleep(0.25)
    return False


async def _chase_or_cancel_unfilled_parent(
    ib,
    contract,
    parent_trade,
    *,
    min_tick=0.01,
    label="",
):
    """
    Give the parent entry order a short window to fill. If it stays passive, chase once
    to the current marketable price and cancel if it still does not fill.
    """
    if parent_trade is None:
        return False

    if await _wait_for_parent_fill(
        ib,
        parent_trade,
        float(getattr(config, "BRACKET_ENTRY_INITIAL_WAIT_SEC", 8)),
    ):
        return True

    parent_order = getattr(parent_trade, "order", None)
    parent_status = str(getattr(parent_trade.orderStatus, "status", "") or "")
    if parent_order is None or _is_terminal_order_status(parent_status):
        return parent_status.lower() == "filled"

    chase_enabled = bool(getattr(config, "BRACKET_ENTRY_CHASE_ENABLED", True))
    order_type = str(getattr(parent_order, "orderType", "") or "").upper()
    action = str(getattr(parent_order, "action", "") or "").upper()

    if chase_enabled and order_type == "LMT":
        ib.reqMktData(contract)
        ticker = None
        for _ in range(6):
            await asyncio.sleep(0.5)
            ticker = ib.ticker(contract)
            bid = float(getattr(ticker, "bid", 0) or 0) if ticker else 0.0
            ask = float(getattr(ticker, "ask", 0) or 0) if ticker else 0.0
            last = float(getattr(ticker, "last", 0) or 0) if ticker else 0.0
            market = float(ticker.marketPrice() or 0) if ticker else 0.0
            if action == "BUY":
                chase_price = ask or last or market
                if chase_price > 0:
                    new_lmt = _round_to_tick(chase_price, min_tick)
                    if new_lmt > float(getattr(parent_order, "lmtPrice", 0) or 0):
                        parent_order.lmtPrice = new_lmt
                        ib.placeOrder(contract, parent_order)
                        print(
                            f"ℹ️ Repriced unfilled entry for {label or getattr(contract, 'localSymbol', '?')} "
                            f"to {new_lmt:.2f} to improve fill odds."
                        )
                    break
            else:
                chase_price = bid or last or market
                if chase_price > 0:
                    new_lmt = _round_to_tick(chase_price, min_tick)
                    curr_lmt = float(getattr(parent_order, "lmtPrice", 0) or 0)
                    if curr_lmt <= 0 or new_lmt < curr_lmt:
                        parent_order.lmtPrice = new_lmt
                        ib.placeOrder(contract, parent_order)
                        print(
                            f"ℹ️ Repriced unfilled entry for {label or getattr(contract, 'localSymbol', '?')} "
                            f"to {new_lmt:.2f} to improve fill odds."
                        )
                    break

        if await _wait_for_parent_fill(
            ib,
            parent_trade,
            float(getattr(config, "BRACKET_ENTRY_FINAL_WAIT_SEC", 5)),
        ):
            return True

    _cancel_ibkr_orders_by_conid(ib, getattr(contract, "conId", 0))
    print(
        f"⚠️ Entry did not fill for {label or getattr(contract, 'localSymbol', '?')}. "
        "Cancelled resting bracket so later signals are not blocked."
    )
    return False


async def _submit_bracket_with_guard(ib, contract, bracket, label=""):
    """
    Submit bracket orders and verify child exits were accepted.
    If children are missing after parent fills, submit fallback OCA TP/SL exits.
    """
    placed_trades = []
    for order in bracket:
        order.tif = 'DAY'
        placed_trades.append(ib.placeOrder(contract, order))

    parent = bracket[0]
    parent_id = int(getattr(parent, 'orderId', 0) or 0)

    # Poll for child orders to appear instead of relying on a fixed sleep.
    timeout_sec = 5.0
    poll_interval = 0.2
    start_time = time.monotonic()
    related_child_count = 0
    while time.monotonic() - start_time < timeout_sec:
        related_child_count = 0
        for t in ib.openTrades():
            c = t.contract
            o = t.order
            if getattr(c, 'conId', None) != getattr(contract, 'conId', None):
                continue
            parent_ref = int(getattr(o, 'parentId', 0) or 0)
            if parent_ref == parent_id:
                related_child_count += 1
        if related_child_count >= 2:
            break
        await asyncio.sleep(poll_interval)

    if related_child_count >= 2:
        return placed_trades

    status_parts = []
    for t in placed_trades:
        oid = getattr(getattr(t, 'order', None), 'orderId', '?')
        st = getattr(getattr(t, 'orderStatus', None), 'status', '?')
        status_parts.append(f"{oid}:{st}")
    print(
        f"⚠️ Bracket child verification warning for {label or getattr(contract, 'localSymbol', '?')}: "
        f"parentId={parent_id}, child_count={related_child_count}, statuses={'; '.join(status_parts)}"
    )

    try:
        parent_action = str(getattr(parent, 'action', 'BUY')).upper()
        fallback_exit_action = 'SELL' if parent_action == 'BUY' else 'BUY'

        pos_qty = 0
        for p in ib.positions():
            if getattr(p.contract, 'conId', None) != getattr(contract, 'conId', None):
                continue
            if p.position == 0:
                continue
            pos_qty = int(abs(p.position))
            fallback_exit_action = 'SELL' if p.position > 0 else 'BUY'
            break

        if pos_qty < 1:
            return placed_trades

        active_exits = 0
        for t in ib.openTrades():
            c = t.contract
            o = t.order
            if getattr(c, 'conId', None) != getattr(contract, 'conId', None):
                continue
            if str(getattr(o, 'action', '')).upper() != fallback_exit_action:
                continue
            status = str(getattr(t.orderStatus, 'status', '') or '').lower()
            if status in {'filled', 'cancelled', 'inactive'}:
                continue
            active_exits += 1

        if active_exits >= 2:
            return placed_trades

        tp_price = float(getattr(bracket[1], 'lmtPrice', 0) or 0)
        sl_price = float(
            getattr(bracket[2], 'auxPrice', 0) or getattr(bracket[2], 'lmtPrice', 0) or 0
        )
        if tp_price <= 0 or sl_price <= 0:
            print(
                f"⚠️ Cannot submit fallback OCA exits for {label}: "
                f"invalid TP/SL ({tp_price}, {sl_price})"
            )
            return placed_trades

        oca_group = f"OCA_{getattr(contract, 'conId', 'X')}_{int(time.time())}"
        tp_order = LimitOrder(fallback_exit_action, pos_qty, tp_price)
        sl_order = StopOrder(fallback_exit_action, pos_qty, sl_price)

        tp_order.tif = 'DAY'
        sl_order.tif = 'DAY'
        tp_order.ocaGroup = oca_group
        sl_order.ocaGroup = oca_group
        tp_order.ocaType = 1
        sl_order.ocaType = 1

        tp_trade = ib.placeOrder(contract, tp_order)
        sl_trade = ib.placeOrder(contract, sl_order)
        placed_trades.extend([tp_trade, sl_trade])

        tp_order_id = getattr(getattr(tp_trade, 'order', None), 'orderId', None)
        sl_order_id = getattr(getattr(sl_trade, 'order', None), 'orderId', None)
        print(
            f"🛡️ Fallback OCA exits submitted for {label or getattr(contract, 'localSymbol', '?')}: "
            f"Qty {pos_qty} | TP {tp_price} (orderId={tp_order_id}) | SL {sl_price} (orderId={sl_order_id})"
        )
    except Exception as e:
        print(f"⚠️ Fallback exit placement failed for {label or getattr(contract, 'localSymbol', '?')}: {e}")

    return placed_trades

# ================= SMC LOGIC (CAUSAL) =================

def detect_swing_highs_lows(df, window=3):
    """
    Identifies if a specific candle *was* a swing point.
    A swing high at index `i` is confirmed at index `i + window`.
    """
    # This function is kept for structural compatibility but the logic is moved to specific getters
    # or we can add columns that mark the 'confirmation' of a swing.
    return df

def get_last_swing_high(df, window=3):
    # Returns the price of the last confirmed swing high looking backwards from the end
    # A swing high at 'i' is confirmed when we are at 'i + window'
    # So we iterate backwards.
    
    # We need at least window*2 + 1 bars
    if len(df) < window * 2 + 1:
        return None
        
    # Iterate backwards from current candle. 
    # for a candidate candle at index `c`, we need data up to `c + window`.
    # so the latest candidate we can check is at index `len(df) - 1 - window`.
    
    for i in range(len(df) - 1 - window, window, -1):
        # check if df[i] is local max of [i-window, i+window]
        candidate_high = df['high'].iloc[i]
        
        # Check left
        left_max = df['high'].iloc[i-window:i].max()
        # Check right
        right_max = df['high'].iloc[i+1:i+window+1].max()
        
        if candidate_high > left_max and candidate_high > right_max:
            return candidate_high
            
    return None

def get_last_swing_low(df, window=3):
    if len(df) < window * 2 + 1:
        return None
        
    for i in range(len(df) - 1 - window, window, -1):
        candidate_low = df['low'].iloc[i]
        left_min = df['low'].iloc[i-window:i].min()
        right_min = df['low'].iloc[i+1:i+window+1].min()
        
        if candidate_low < left_min and candidate_low < right_min:
            return candidate_low
            
    return None

def detect_fvg(df):
    """
    Fair Value Gap:
    Bullish: High of candle[i-2] < Low of candle[i]. Gap is between them.
    Bearish: Low of candle[i-2] > High of candle[i].
    Confirmed at close of candle[i].
    """
    # Create FVG columns. 
    # 'bull_fvg' at row `i` means an FVG formed between i-2 and i.
    df['bull_fvg'] = (df['low'] > df['high'].shift(2))
    df['bull_fvg_top'] = df['low'] # The top of the gap is the low of candle i
    df['bull_fvg_bottom'] = df['high'].shift(2) # The bottom of the gap is high of candle i-2
    
    df['bear_fvg'] = (df['high'] < df['low'].shift(2))
    df['bear_fvg_top'] = df['low'].shift(2) # The top of the gap is the low of candle i-2
    df['bear_fvg_bottom'] = df['high'] # The bottom of the gap is high of candle i
    return df

def detect_order_block(df):
    """
    Order Block: 
    Bullish: The last down candle before a strong up move (impulse).
    Bearish: The last up candle before a strong down move.
    """
    # Identify impulse (large body candles)
    body_size = abs(df['close'] - df['open'])
    avg_body = body_size.rolling(20).mean()
    avg_vol = df['volume'].rolling(20).mean()
    
    df['body_size'] = body_size
    df['avg_body'] = avg_body
    df['avg_vol'] = avg_vol
    df['impulse'] = body_size > (avg_body * 1.5) # Arbitrary threshold for "strong" move
    
    # We are looking for where an OB *was* formed. 
    # A Bullish OB is confirmed when we have a down candle followed by an impulse up.
    # At index `i` (impulse candle), the OB is at `i-1`.
    
    # Logic: 
    # 1. Previous candle (i-1) was bearish (Close < Open)
    # 2. Current candle (i) is bullish (Close > Open)
    # 3. Current candle is 'impulse'
    # 4. Current Close > Previous High (optional structure break)
    
    df['is_bull_ob_candle'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                              (df['close'] > df['open']) & \
                              df['impulse'] & \
                              (df['close'] > df['high'].shift(1)) 
                              
    df['is_bear_ob_candle'] = (df['close'].shift(1) > df['open'].shift(1)) & \
                              (df['close'] < df['open']) & \
                              df['impulse'] & \
                              (df['close'] < df['low'].shift(1))

    return df

def detect_structure_shift(df):
    # Not fully implemented in causal way for this snippet, relying on simpler OB/Impulse logic
    return df

def calculate_confidence(ltf_row, last_htf, fvg_touch=False):
    """Calculates a confidence score between 0-100%."""
    score = 0
    
    # 1. Impulse Intensity & Volume (20%)
    if ltf_row.get('avg_vol', 0) > 0:
        vol_ratio = ltf_row['volume'] / ltf_row['avg_vol']
        vol_score = min(20, max(0, (vol_ratio - 1.0) / 1.0 * 20))
        score += vol_score
        
    # 2. FVG Pullback Confluence (30%)
    if fvg_touch:
        score += 30
        
    # 3. Trend Proximity (20%)
    price = ltf_row['close']
    ema = last_htf.get('ema50')
    if ema:
        dist_pct = abs(price - ema) / ema
        # Pullbacks often touch or get close to the EMA.
        if dist_pct < 0.003:
            score += 20
        elif dist_pct < 0.005:
            score += 10
            
    # 4. Trend Strength (ADX) (30%)
    adx = last_htf.get('adx', 0)
    if not pd.isna(adx):
        if adx >= 25:
            score += 30
        elif adx > 20:
            score += 15
        elif adx < 15:
            score -= 20 # Penalize very choppy markets

    return int(min(100, max(0, score)))

def get_confidence_label(score):
    if score >= 80: return "🔥🔥 High"
    if score >= 60: return "📈 Medium"
    return "⚖️ Low"

def liquidity_sweep(df):
    # Not fully implemented in causal way for this snippet
    return df

def send_discord_notification(signal, price, time_et, symbol, bias, confidence):
    webhook_url = getattr(config, 'DISCORD_WEBHOOK_URL', None)
    if not webhook_url:
        return
        
    label = get_confidence_label(confidence)
    color = 0x00ff00 if signal.lower() == "buy" else 0xff0000
    
    payload = {
        "embeds": [{
            "title": f"🚨 {signal.upper()} SIGNAL: {symbol}",
            "color": color,
            "fields": [
                {"name": "Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Time (ET)", "value": time_et, "inline": True},
                {"name": "HTF Bias", "value": bias.upper(), "inline": True},
                {"name": "Confidence", "value": f"{confidence}% [{label}]", "inline": True}
            ],
            "footer": {"text": f"SMC Bot Strategy Execution v{getattr(config, '__version__', '?.?.?')}"}
        }]
    }
    
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")

def send_discord_live_trading_notification(signal, symbol, order_details, confidence, strategy_bias):
    webhook_url = getattr(config, 'DISCORD_WEBHOOK_URL_LIVE_TRADING', None)
    if not webhook_url:
        return
        
    now_et = datetime.now(ZoneInfo("America/New_York"))
    time_et_str = now_et.strftime("%I:%M %p")
    
    label = get_confidence_label(confidence)
    # Color based on signal
    color = 0x00ff00 if signal.lower() == "buy" else 0xff0000
    if "close" in signal.lower() or "cleanup" in signal.lower() or "flip" in signal.lower():
        color = 0x0000ff # Blue for neutral/closure
    
    fields = [
        {"name": "Symbol", "value": symbol, "inline": True},
        {"name": "Time (ET)", "value": time_et_str, "inline": True},
        {"name": "HTF Bias", "value": strategy_bias.upper(), "inline": True},
        {"name": "Confidence", "value": f"{confidence}% [{label}]", "inline": True},
    ]
    
    # Add order details
    if isinstance(order_details, dict):
        for k, v in order_details.items():
            fields.append({"name": k, "value": str(v), "inline": True})
    else:
        fields.append({"name": "Order Details", "value": str(order_details), "inline": False})
    
    payload = {
        "embeds": [{
            "title": f"📈 LIVE ORDER: {signal.upper()}",
            "color": color,
            "fields": fields,
            "footer": {"text": f"SMC Bot Live Execution v{getattr(config, '__version__', '?.?.?')}"}
        }]
    }
    
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send live trading Discord notification: {e}")

# ================= STRATEGY =================

def get_strategy_signal(htf: pd.DataFrame, ltf: pd.DataFrame):
    """
    Pure strategy function.
    Takes historical dataframes and returns a (signal, confidence) tuple.
    """
    # 1. HTF Direction
    htf = htf.copy()
    
    # Only calculate EMA if it doesn't exist (prevents drift if already pre-calculated on full history)
    if 'ema50' not in htf.columns:
        if len(htf) < 50: return None
        htf['ema50'] = ta.ema(htf['close'], length=50)
    
    if htf['ema50'].isnull().all():
        return None
        
    last_htf = htf.iloc[-1]
    
    # ADX Filter: If ADX is below 25, the market is considered choppy.
    adx_val = last_htf.get('adx', 0)
    is_choppy = False
    if not pd.isna(adx_val) and adx_val < 25:
        is_choppy = True
        
    price = last_htf['close']
    ema = last_htf['ema50']
    atr = last_htf.get('atr', price * 0.001) 
    
    buffer = 0.1 * (atr / price)
    buffer = max(0.0002, min(0.001, buffer)) 

    if is_choppy:
        bias = "choppy"
    else:
        if price > (ema + buffer):
            bias = "bullish"
        elif price < (ema - buffer):
            bias = "bearish"
        else:
            bias = "choppy" 

        # --- BALANCED EXTENSION FILTER ---
        dist_from_ema = abs(price - ema)
        is_parabolic = adx_val > 40
        extension_limit = 3.0 * atr
        
        if dist_from_ema > extension_limit and not is_parabolic:
            return None, 0

    # --- RSI OVERBOUGHT / OVERSOLD FILTER ---
    # Avoid entering longs into overbought HTF conditions and shorts into oversold ones.
    rsi = last_htf.get('rsi14', None)
    if rsi is not None and not pd.isna(rsi):
        if bias == "bullish" and rsi > 77.5:
            return None, 0  # Overbought — risk of mean reversion against the trade
        elif bias == "bearish" and rsi < 22.5:
            return None, 0  # Oversold — risk of bounce against the trade

    # 2. LTF Analysis
    ltf = ltf.copy()
    if 'bull_fvg' not in ltf.columns:
        ltf = detect_fvg(ltf)
    if 'is_bull_ob_candle' not in ltf.columns:
        ltf = detect_order_block(ltf)
    
    if len(ltf) < 5: return None
    
    last_closed = ltf.iloc[-1]
    
    # --- FALLING KNIFE Protection ---
    is_opposite_impulse = False
    if last_closed.get('impulse', False):
        if bias == "bullish" and last_closed['close'] < last_closed['open']:
            is_opposite_impulse = True
        elif bias == "bearish" and last_closed['close'] > last_closed['open']:
            is_opposite_impulse = True
            
    vol_ratio = 1.0
    if last_closed.get('avg_vol', 0) > 0:
        vol_ratio = last_closed['volume'] / last_closed['avg_vol']
    
    signal = None
    confidence = 0
    fvg_touch = False

    # Block if high volume is moving AGAINST our bias (reversal threat).
    # High volume WITH our bias is fine — it signals strong directional interest on the pullback.
    vol_against_bias = False
    if vol_ratio > 2.0:
        if bias == "bullish" and last_closed['close'] < last_closed['open']:
            vol_against_bias = True  # Big red candle on high vol in a bull trend
        elif bias == "bearish" and last_closed['close'] > last_closed['open']:
            vol_against_bias = True  # Big green candle on high vol in a bear trend

    if is_opposite_impulse or vol_against_bias:
        return None, 0

    # We look back over the last N candles to find a recent valid FVG/OB structure
    lookback = 15
    recent_ltf = ltf.iloc[-lookback-1:-1]
    
    # Threshold for "close enough" (Scaled with volatility)
    atr_ltf = last_htf.get('atr', last_closed['close'] * 0.005)
    ltf_buffer = 0.1 * (atr_ltf / last_closed['close'])
    ltf_buffer = max(0.0002, min(0.001, ltf_buffer)) 
    
    if bias == "bullish":
        # Search for a recent Bullish FVG associated with an Impulse/OB
        for i in range(len(recent_ltf) - 1, -1, -1):
            row = recent_ltf.iloc[i]
            if row.get('bull_fvg', False):
                ob_nearby = any(recent_ltf.iloc[max(0, i-2):min(len(recent_ltf), i+1)]['is_bull_ob_candle'])
                if ob_nearby:
                    # Entry condition: Price pulls back to touch or get very close to FVG top
                    # But must stay above FVG bottom (structure preservation)
                    px_limit_top = row['bull_fvg_top'] * (1 + ltf_buffer)
                    px_limit_bot = row['bull_fvg_bottom'] * (1 - ltf_buffer)
                    
                    if last_closed['low'] <= px_limit_top and last_closed['close'] > px_limit_bot:
                        signal = "buy"
                        fvg_touch = True
                        break
             
    elif bias == "bearish":
        for i in range(len(recent_ltf) - 1, -1, -1):
            row = recent_ltf.iloc[i]
            if row.get('bear_fvg', False):
                ob_nearby = any(recent_ltf.iloc[max(0, i-2):min(len(recent_ltf), i+1)]['is_bear_ob_candle'])
                if ob_nearby:
                    px_limit_top = row['bear_fvg_top'] * (1 + ltf_buffer)
                    px_limit_bot = row['bear_fvg_bottom'] * (1 - ltf_buffer)
                    
                    if last_closed['high'] >= px_limit_bot and last_closed['close'] < px_limit_top:
                        signal = "sell"
                        fvg_touch = True
                        break

    if signal:
        confidence = calculate_confidence(last_closed, last_htf, fvg_touch=fvg_touch)

    return signal, confidence


def _normalize_timestamp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a UTC `timestamp` column exists for stable causal slicing."""
    out = df.copy()
    if 'timestamp' not in out.columns:
        out = out.reset_index()

    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True, errors='coerce')
    out = out.dropna(subset=['timestamp'])
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out


def precompute_strategy_features(htf_df: pd.DataFrame, ltf_df: pd.DataFrame):
    """Compute strategy features once on full history to avoid signal drift."""
    htf = _normalize_timestamp_frame(htf_df)
    ltf = _normalize_timestamp_frame(ltf_df)

    if 'ema50' not in htf.columns:
        htf['ema50'] = ta.ema(htf['close'], length=50)

    # Compute ADX for trend strength (using standard 14 period)
    if 'adx' not in htf.columns:
        # pandas_ta.adx returns a DataFrame with ADX_14, DMP_14, DMN_14
        adx_df = ta.adx(htf['high'], htf['low'], htf['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            # We just need the main ADX column. It's usually the first one 'ADX_14'
            adx_col = [c for c in adx_df.columns if c.startswith('ADX')][0]
            htf['adx'] = adx_df[adx_col]
        else:
            htf['adx'] = 0

    if 'atr' not in htf.columns:
        htf['atr'] = htf.ta.atr(length=14)

    if 'rsi14' not in htf.columns:
        htf['rsi14'] = ta.rsi(htf['close'], length=14)

    if 'bull_fvg' not in ltf.columns or 'bear_fvg' not in ltf.columns:
        ltf = detect_fvg(ltf)
    if 'is_bull_ob_candle' not in ltf.columns or 'is_bear_ob_candle' not in ltf.columns:
        ltf = detect_order_block(ltf)

    return htf, ltf


def get_causal_signal_from_precomputed(htf_df: pd.DataFrame, ltf_df: pd.DataFrame, evaluation_ts, ltf_window: int = 200):
    """
    Evaluate signal at a specific closed 1m candle timestamp.
    Uses only 15m candles that were already closed at that timestamp.
    """
    if evaluation_ts is None:
        return None

    ts = pd.Timestamp(evaluation_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    ltf_causal = ltf_df[ltf_df['timestamp'] <= ts]
    if ltf_causal.empty:
        return None

    htf_causal = htf_df[htf_df['timestamp'] <= (ts - timedelta(minutes=15))]
    if htf_causal.empty or len(htf_causal) < 50:
        return None

    return get_strategy_signal(htf_causal, ltf_causal.iloc[-ltf_window:])


async def generate_signal(symbol=None):
    if symbol is None:
        symbol = SYMBOL
        
    try:
        # Fetch bars for both timeframes
        # Increase limit to accommodate 14 days of data for EMA stability
        htf_df = await get_bars(symbol, config.TIMEFRAME_HTF, 2000)
        ltf_df = await get_bars(symbol, config.TIMEFRAME_LTF, 5000)
        
        if htf_df.empty or ltf_df.empty:
            print(f"Not enough data fetched for {symbol}.")
            return None, 0

        htf_df, ltf_df = precompute_strategy_features(htf_df, ltf_df)

        # Last fully closed LTF bar: drop newest potentially-forming row.
        if len(ltf_df) < 2:
            return None, 0

        evaluation_ts = ltf_df['timestamp'].iloc[-2]
        return get_causal_signal_from_precomputed(htf_df, ltf_df, evaluation_ts)

    except Exception as e:
        print(f"Error generating signal for {symbol}: {e}")
        return None, 0

# ================= RISK =================

from alpaca.trading.requests import StopLossRequest, TakeProfitRequest

async def calculate_smart_quantity(symbol, price, stop_loss_price, budget_override=None, risk_allocation_pct=None):
    """
    Calculates position size using IBKR account data.
    """
    try:
        # In ib_insync, account values are stored in 'accountValues'
        # NetLiquidation is usually the equity.
        acc_values = ibkr_mgr.ib.accountValues()
        equity = 10000.0
        buying_power = 10000.0
        
        for v in acc_values:
            if v.tag == 'NetLiquidation':
                equity = float(v.value)
            if v.tag == 'BuyingPower':
                buying_power = float(v.value)
    except Exception as e:
        print(f"Error fetching account info: {e}")
        equity = 10000.0
        buying_power = 10000.0
        
    alloc_risk_pct = 1.0 if risk_allocation_pct is None else float(risk_allocation_pct)
    alloc_risk_pct = min(1.0, max(0.0, alloc_risk_pct))
    risk_amount = equity * alloc_risk_pct * RISK_PER_TRADE
    stop_distance = abs(price - stop_loss_price)
    
    if stop_distance <= 0:
        print("Stop distance is 0 or negative! Defaulting to min qty.")
        return 0
        
    # 1. Risk-Based Qty
    qty_risk = risk_amount / stop_distance
    
    # 2. Ticker-Based Cap (Multi-Instance Safety)
    if budget_override is not None:
        max_ticker_pct = budget_override
    else:
        max_ticker_pct = getattr(config, 'STOCK_ALLOCATION_PCT', 0.80)
        
    max_ticker_amt = equity * max_ticker_pct
    
    # Calculate current exposure for this ticker (Stock Only)
    current_exposure = 0.0
    try:
        # ib.positions() returns a list of Position objects
        positions = ibkr_mgr.ib.positions()
        for p in positions:
            # We identify the ticker by checking contract.symbol
            if p.contract.symbol == symbol and isinstance(p.contract, Stock):
                current_exposure += abs(p.position * price)
    except Exception as e:
        print(f"Error calculating exposure for {symbol}: {e}")
        
    remaining_ticker_budget = max(0, max_ticker_amt - current_exposure)
    qty_ticker = remaining_ticker_budget / price
    
    # 3. Buying Power Cap
    max_cost = buying_power * 0.95
    qty_bp = max_cost / price
    
    # Final
    qty = int(min(qty_risk, qty_ticker, qty_bp))
    
    # Log
    print(f"DEBUG: Equity: ${equity:.2f} | RiskAlloc: {alloc_risk_pct:.2f} | Risk($): ${risk_amount:.2f} | Entry: {price} | SL: {stop_loss_price} | Dist: {stop_distance:.2f}")
    print(f"DEBUG: RiskQty: {int(qty_risk)} | BPQty: {int(qty_bp)} -> Final: {qty}")
    
    return max(0, qty)

# ================= HELPERS =================

from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType

async def get_best_option_contract(symbol, signal_type, known_price=None):
    ib = ibkr_mgr.ib
    """
    Finds the best option contract using dynamic expiry and strike discovery.
    """
    try:
        # 1. Get current price
        if known_price:
            current_price = known_price
        else:
            current_price_df = await get_bars(symbol, "1Min", 1)
            if current_price_df.empty:
                # Stock bar fallback
                print(f"ℹ️ Could not get bars for {symbol} to calculate strike. Using fallback price check.")
                current_price = await get_latest_price_fallback(symbol)
                if not current_price: return None
            else:
                current_price = current_price_df['close'].iloc[-1]

        # 2. Get Underlying Contract for conId
        underlying = Stock(symbol, 'SMART', 'USD')
        qualified_underlying = await ib.qualifyContractsAsync(underlying)
        if not qualified_underlying:
            print(f"❌ Could not qualify underlying stock for {symbol}")
            return None
        underlying_conId = qualified_underlying[0].conId

        # 3. Get Option Parameters
        print(f"🔍 Fetching option parameters for {symbol}...")
        chains = await ib.reqSecDefOptParamsAsync(underlying.symbol, '', underlying.secType, underlying_conId)
        if not chains:
            print(f"❌ No option chains found for {symbol}")
            return None
        
        # Filter chains: prefer those where tradingClass matches the symbol (Standard Options)
        # Some symbols have '2QQQ' or other classes for weeklies/non-standard variants.
        standard_chains = [c for c in chains if c.tradingClass == symbol and c.exchange == 'SMART']
        if not standard_chains:
            standard_chains = [c for c in chains if c.exchange == 'SMART']
        
        if not standard_chains:
            chain = chains[0]
        else:
            # Pick the chain with the most strikes (likely the main one)
            chain = max(standard_chains, key=lambda c: len(c.strikes))
        
        print(f"ℹ️ Selected Chain: exchange={chain.exchange}, tradingClass={chain.tradingClass}, mult={chain.multiplier}")

        # 4. Filter Expiries (1-14 days out)
        now = datetime.now()
        valid_expiries = []
        for exp in chain.expirations:
            try:
                clean_exp = exp.replace('-', '')
                dt_exp = datetime.strptime(clean_exp, "%Y%m%d")
                days_to_expiry = (dt_exp - now).days
                if 1 <= days_to_expiry <= 14:
                    valid_expiries.append(exp)
            except Exception: continue
        
        if not valid_expiries:
            # Fallback: pick the one closest to 5 days
            print("⚠️ No expiries in ideal 1-14d range. Searching closest...")
            try:
                sorted_exp = sorted(chain.expirations, key=lambda x: abs((datetime.strptime(x.replace('-', ''), "%Y%m%d") - now).days - 5))
                valid_expiries = sorted_exp[:1]
            except Exception:
                valid_expiries = chain.expirations[:1]
            
        target_expiry = valid_expiries[0]

        # 5. Pick ATM strike candidates and qualify defensively
        if not current_price or current_price <= 0:
            print(f"❌ Cannot determine ATM strike for {symbol} without valid price ({current_price})")
            return None

        right = 'C' if signal_type == "buy" else 'P'

        # Keep strikes exactly from IB chain, but normalize float artifacts.
        strike_candidates = []
        for s in chain.strikes:
            try:
                strike_candidates.append(round(float(s), 2))
            except Exception:
                continue
        strike_candidates = sorted(set(strike_candidates), key=lambda x: abs(x - current_price))
        if not strike_candidates:
            print(f"❌ No strike candidates returned by IB chain for {symbol}")
            return None

        # Prioritize intended expiry first, then nearby valid expiries as fallback.
        prioritized_expiries = [target_expiry] + [e for e in valid_expiries if e != target_expiry]
        # Keep search bounded for latency.
        prioritized_expiries = prioritized_expiries[:5]
        strike_candidates = strike_candidates[:40]

        print(
            f"🎯 Option search setup: symbol={symbol} right={right} "
            f"price={current_price:.2f} expiry_pref={target_expiry} "
            f"candidates(exp={len(prioritized_expiries)}, strikes={len(strike_candidates)})"
        )

        for exp in prioritized_expiries:
            for strike in strike_candidates:
                contract = Option(symbol, exp, strike, right, 'SMART')
                contract.tradingClass = chain.tradingClass
                try:
                    qualified = await ib.qualifyContractsAsync(contract)
                    if qualified:
                        res_contract = qualified[0]
                        res_contract.exchange = 'SMART'
                        print(
                            f"✅ Qualified option: {res_contract.localSymbol} "
                            f"(exp={exp}, strike={strike}, class={chain.tradingClass})"
                        )
                        return res_contract
                except Exception:
                    continue

        print(
            f"❌ Failed to qualify any contract for {symbol} "
            f"(right={right}, tradingClass={chain.tradingClass}, price={current_price:.2f})"
        )
        return None

    except Exception as e:
        print(f"Error getting option contract: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_current_position(symbol):
    """
    Returns the current position for the given symbol from the live list.
    """
    positions = ibkr_mgr.ib.positions()
    for p in positions:
        if p.contract.symbol == symbol:
            return p
    return None

async def cancel_all_orders_for_symbol(symbol):
    open_trades = ibkr_mgr.ib.openTrades()
    for t in open_trades:
        if t.contract.symbol == symbol:
            ibkr_mgr.ib.cancelOrder(t.order)
            print(f"Cancelled order for {symbol}")

def _cancel_ibkr_orders_by_conid(ib, con_id):
    """Cancel all IBKR open orders matching a given conId (options)."""
    try:
        for o in ib.openOrders():
            c = o.contract
            if hasattr(c, "conId") and c.conId == con_id:
                ib.cancelOrder(o)
                print(f"Cancelled stale IBKR order {getattr(o, 'orderId', '?')} for conId {con_id}")
    except Exception as e:
        print(f"⚠️ Error cancelling IBKR orders for conId {con_id}: {e}")

def _cleanup_stale_pending_entry(ib, symbol, right, state):
    """
    If a pending entry was left open (unfilled) too long, cancel it and clear state.
    Prevents a stale limit from blocking fresh signals all day.
    """
    ticker_state = state.get(symbol, {})
    pending = ticker_state.get("pending_entry")
    if not pending:
        return state
    try:
        submitted_at = pending.get("submitted_at")
        con_id = pending.get("con_id")
        if not submitted_at:
            ticker_state.pop("pending_entry", None)
            state[symbol] = ticker_state
            save_trade_state(state, symbol)
            return state
        age_min = (datetime.now(timezone.utc) - datetime.fromisoformat(submitted_at)).total_seconds() / 60
        max_age = getattr(config, "STALE_PENDING_ENTRY_MINUTES", 5)
        still_open = False
        for t in ib.openTrades():
            c = t.contract
            o = t.order
            status = str(getattr(t.orderStatus, "status", "") or "").lower()
            if not isinstance(c, Option):
                continue
            if con_id and c.conId != con_id:
                continue
            if c.symbol != symbol:
                continue
            if right and getattr(c, "right", "") != right:
                continue
            if status in {"filled", "cancelled", "inactive"}:
                continue
            still_open = True
            if age_min >= max_age:
                _cancel_ibkr_orders_by_conid(ib, c.conId)
        if (not still_open) or age_min >= max_age:
            ticker_state.pop("pending_entry", None)
            state[symbol] = ticker_state
            save_trade_state(state, symbol)
    except Exception as e:
        print(f"⚠️ Error cleaning stale pending entry: {e}")
    return state

async def place_trade(signal, confidence, symbol, use_daily_cap=True, daily_cap_value=None, option_allocation_override=None, max_option_contracts_override=None):
    ib = ibkr_mgr.ib
    # Determine bias for notifications
    bias = "bullish" if signal == "buy" else "bearish"

    # --- HARD FILTER CHECK ---
    if confidence <= 0:
        print(f"🛑 REJECTED: Confidence is 0 (Blocked by Strategy Filters) for {symbol}")
        return

    # --- GLOBAL SAFETY CHECKS ---
    safety = load_global_safety_state()
    if safety.get("halted", False):
        print("🚨 TRADING HALTED: Global Drawdown Circuit Breaker is ACTIVE. No new trades allowed.")
        return

    if safety.get("last_loss_time"):
        try:
            last_loss = datetime.fromisoformat(safety["last_loss_time"])
            wait_mins = getattr(config, 'COOL_DOWN_MINUTES', 60)
            time_diff = (datetime.now(timezone.utc) - last_loss).total_seconds() / 60
            if time_diff < wait_mins:
                print(f"🕒 COOL-DOWN: Last loss was {time_diff:.1f} mins ago. Waiting {wait_mins} total. Skipping.")
                return
        except Exception as e:
            print(f"Error checking cool-down: {e}")

    # --- TIME FILTER (09:40 AM - 3:55 PM ET) ---
    now_et = datetime.now(ZoneInfo("America/New_York"))
    current_time = now_et.time()
    start_time = datetime.strptime("09:40:00", "%H:%M:%S").time()
    end_time = datetime.strptime("15:55:00", "%H:%M:%S").time()
    
    if current_time < start_time or current_time > end_time:
        print(f"🕒 TIME FILTER: Current time {current_time} is outside the 09:40-15:55 window. Skipping.")
        return
    # --- DAILY TRADE CAP ---
    state = load_trade_state(symbol)
    today_str = now_et.strftime("%Y-%m-%d")
    
    ticker_state = state.get(symbol, {})
    last_date = ticker_state.get("last_trade_date")
    daily_count = ticker_state.get("daily_trade_count", 0)
    
    if last_date != today_str:
        daily_count = 0
        ticker_state["last_trade_date"] = today_str
    
    if daily_cap_value is None:
        cap_limit = getattr(config, 'DEFAULT_DAILY_CAP', 5)
    else:
        cap_limit = daily_cap_value
    
    if use_daily_cap and cap_limit >= 0 and daily_count >= cap_limit:
        print(f"🛑 DAILY CAP: Already placed {daily_count} trades for {symbol} today (limit: {cap_limit}). Skipping.")
        return

    # Get latest price
    price_df = await get_bars(symbol, "1Min", 100)
    if price_df.empty:
        print("Could not fetch price for placement.")
        return
    
    price = price_df['close'].iloc[-1]
    option_allocation_pct = option_allocation_override if option_allocation_override is not None else getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20)
    option_allocation_pct = min(1.0, max(0.0, float(option_allocation_pct)))
    stock_allocation_pct = 1.0 - option_allocation_pct
    max_option_contracts = max_option_contracts_override if max_option_contracts_override is not None else getattr(config, 'MAX_OPTION_CONTRACTS', -1)
    
    # 1. Fetch CURRENT positions
    positions = ib.positions()
    qty_held = 0
    side_held = None
    any_options_held = False
    
    for p in positions:
        if p.contract.symbol != symbol: continue
        if isinstance(p.contract, Stock):
            qty_held = p.position # Positive for long, negative for short in IBKR
            side_held = 'long' if qty_held > 0 else 'short' if qty_held < 0 else None
        elif isinstance(p.contract, Option):
            any_options_held = True

    print(f"DEBUG: Current Position for {symbol}: {qty_held} units (Side: {side_held}) | Options: {any_options_held}")

    # Discord Notification with final price
    # send_discord_notification(signal, price, time_et_str, symbol, bias, confidence)

    # 2. Global Position Cleanup (Cross-Asset Signal Flip)
    same_bias_held = False
    for p in positions:
        if p.contract.symbol != symbol: continue
        
        is_stock = isinstance(p.contract, Stock)
        is_option = isinstance(p.contract, Option)
        
        # Identify Bias
        is_bullish = False
        if is_stock:
            is_bullish = (p.position > 0)
        elif is_option:
            is_bullish = (p.contract.right == 'C')

        # Conflict Detection: Close if signal is opposite of current holding bias
        if (signal == "buy" and not is_bullish) or (signal == "sell" and is_bullish):
            print(f"🔄 CROSS-ASSET CLEANUP: Closing {p.contract.localSymbol} bias conflict with {signal.upper()} signal.")
            try:
                # Cancel open orders for this contract
                ib.reqGlobalCancel() # Quickest way in IBKR for all
                # Close position with a Market Order
                action = 'SELL' if p.position > 0 else 'BUY'
                close_order = MarketOrder(action, abs(p.position))
                ib.placeOrder(p.contract, close_order)
                
                # Notify
                send_discord_live_trading_notification(
                    signal=f"cleanup_close_{side_held}",
                    symbol=p.contract.localSymbol,
                    order_details={
                        "Action": action,
                        "Qty": abs(p.position),
                        "Price": price,
                        "Type": "Market (Signal Flip)"
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                
                # Update local state
                if is_stock:
                    qty_held = 0
                    side_held = None
                if is_option:
                    any_options_held = False
            except Exception as close_err:
                print(f"❌ Cleanup failed for {p.contract.localSymbol}: {close_err}")
        
        elif (signal == "buy" and is_bullish) or (signal == "sell" and not is_bullish):
            if is_option:
                same_bias_held = True



    # 3. Mix-up Guards (Prevent holding both Stock and Options for same symbol)
    if getattr(config, 'ENABLE_OPTIONS', False):
        if qty_held != 0:
             print(f"Warning: Holding underlying shares ({qty_held}) while in Options Mode. Skipping new option entries to avoid mix-up.")
             return
    else:
        if any_options_held:
             print(f"Warning: Holding option contracts while in Stock Mode. Skipping new stock entries to avoid mix-up.")
             return

    # ================= OPTIONS MODE =================
    if getattr(config, 'ENABLE_OPTIONS', False):
        desired_right = 'C' if signal == 'buy' else 'P'

        if same_bias_held:
            print(f"Existing {signal.upper()} option bias detected for {symbol}. Skipping redundant entry.")
            return

        # Guard: positions() may still be zero before a just-submitted order fills.
        # Check open/pending IBKR option orders to avoid duplicate entries.
        open_trades = ib.openTrades()
        for t in open_trades:
            try:
                c = t.contract
                o = t.order
                if not isinstance(c, Option):
                    continue
                if c.symbol != symbol:
                    continue
                if getattr(c, "right", "") != desired_right:
                    continue
                if getattr(o, "action", "") != "BUY":
                    continue
                status = str(getattr(t.orderStatus, "status", "") or "").lower()
                if status in {"filled", "cancelled", "inactive"}:
                    continue
                print(f"Pending {desired_right} option entry already exists for {symbol} ({c.localSymbol}). Skipping duplicate.")
                return
            except Exception:
                continue

        # Cleanup stale pending entries that block new signals
        state = _cleanup_stale_pending_entry(ib, symbol, desired_right, state)

        print(f"Options Trading Enabled. Searching for contract for {signal.upper()}...")
        opt_contract = await get_best_option_contract(symbol, signal)
        
        if opt_contract:
            # Check if we already hold THIS specific contract
            positions = ib.positions()
            for p in positions:
                if p.contract.conId == opt_contract.conId and p.position != 0:
                     print(f"Already hold {opt_contract.localSymbol}. Holding.")
                     return
            # Also block if this exact contract already has an open BUY entry order.
            for t in ib.openTrades():
                try:
                    c = t.contract
                    o = t.order
                    status = str(getattr(t.orderStatus, "status", "") or "").lower()
                    if not isinstance(c, Option):
                        continue
                    if c.conId != opt_contract.conId:
                        continue
                    if getattr(o, "action", "") != "BUY":
                        continue
                    if status in {"filled", "cancelled", "inactive"}:
                        continue
                    print(f"Pending entry already exists for {opt_contract.localSymbol}. Skipping duplicate.")
                    return
                except Exception:
                    continue

            try:
                # Fetch Current Option Price (Tick)
                ib.reqMktData(opt_contract)
                # Wait for ticker (Delayed data can take 2-5s to arrive initially)
                entry_est = None
                ticker = None
                for _ in range(10): # Max 5 seconds
                    await asyncio.sleep(0.5)
                    ticker = ib.ticker(opt_contract)
                    if ticker:
                        entry_est = ticker.ask if ticker.ask > 0 else ticker.last
                        if not entry_est or entry_est <= 0 or np.isnan(entry_est):
                             entry_est = ticker.marketPrice()
                    
                    if entry_est and entry_est > 0 and not np.isnan(entry_est):
                        break
                
                if not entry_est or entry_est <= 0 or np.isnan(entry_est):
                    # --- NEW FALLBACK: Historical Midpoint ---
                    print(f"ℹ️ Attempting historical fallback for {opt_contract.localSymbol}...")
                    entry_est = await get_latest_price_fallback(opt_contract)

                if not entry_est or entry_est <= 0 or np.isnan(entry_est):
                    print(f"Invalid option ask price for {opt_contract.localSymbol}. Skipping.")
                    return

                # IBKR may reject "too aggressive" option limits when stale/fallback prices drift.
                # Use live NBBO when available and normalize to contract minTick.
                bid = float(getattr(ticker, "bid", 0) or 0) if ticker else 0.0
                ask = float(getattr(ticker, "ask", 0) or 0) if ticker else 0.0
                last = float(getattr(ticker, "last", 0) or 0) if ticker else 0.0
                spread = (ask - bid) if (bid > 0 and ask > 0 and ask >= bid) else 0.0
                mid = ((bid + ask) / 2.0) if (bid > 0 and ask > 0 and ask >= bid) else 0.0
                spread_pct = (spread / mid) if mid > 0 else None

                max_spread_pct = float(getattr(config, 'OPTIONS_MAX_SPREAD_PCT', 0.20))
                if spread_pct is not None and max_spread_pct >= 0 and spread_pct > max_spread_pct:
                    print(
                        f"⚠️ Spread guard: skipping {opt_contract.localSymbol} | "
                        f"Bid {bid:.2f} Ask {ask:.2f} Spread {spread:.2f} "
                        f"({spread_pct*100:.1f}%) > Max {max_spread_pct*100:.1f}%"
                    )
                    return

                # Discover price increment from contract details (fallback 0.01).
                min_tick = 0.01
                try:
                    cds = await ib.reqContractDetailsAsync(opt_contract)
                    if cds and cds[0].minTick and cds[0].minTick > 0:
                        min_tick = float(cds[0].minTick)
                except Exception:
                    pass

                if bid > 0 and ask > 0 and ask >= bid:
                    # Place slightly above mid but never above ask to reduce overpay/rejections.
                    raw_entry = min(ask, ((bid + ask) / 2.0) + 0.25 * spread)
                elif ask > 0:
                    raw_entry = ask
                elif last > 0:
                    raw_entry = last
                else:
                    raw_entry = float(entry_est)

                entry_est = _round_to_tick(float(raw_entry), min_tick)
                if ask > 0:
                    # Hard guard against marketability-style rejection spikes.
                    entry_est = min(entry_est, _round_to_tick(ask, min_tick))
                if entry_est <= 0:
                    print(f"Invalid normalized entry price for {opt_contract.localSymbol}. Skipping.")
                    return

                entry_mode = str(getattr(config, 'BRACKET_ENTRY_MODE', 'market')).strip().lower()
                spread_text = f"{(spread_pct*100):.1f}%" if spread_pct is not None else "N/A"
                print(
                    f"Quote snapshot: Bid {bid:.2f} | Ask {ask:.2f} | Last {last:.2f} | "
                    f"Spread {spread_text} | Mode {entry_mode}"
                )
                if entry_mode in {'market', 'mkt'}:
                    print(
                        f"⚠️ Market entry can fill away from Est.Entry ({entry_est}) in fast/wide option books."
                    )

                sl_price = _round_to_tick(entry_est * 0.80, min_tick)
                tp_price = _round_to_tick(entry_est * 1.50, min_tick)
                
                print(f"Options Bracket: Est.Entry: {entry_est} | SL: {sl_price} | TP: {tp_price}")
                
                # --- GLOBAL OPTION EXPOSURE ---
                acc_values = ib.accountValues()
                equity = 10000.0
                for v in acc_values:
                    if v.tag == 'NetLiquidation': equity = float(v.value)
                budget_pct = option_allocation_pct
                total_budget = equity * budget_pct
                
                existing_option_exposure = 0.0
                ticker_option_exposure = 0.0
                for p in positions:
                    if isinstance(p.contract, Option):
                        existing_option_exposure += abs(p.position * entry_est * 100) # Rough estimate
                        if p.contract.symbol == symbol:
                            ticker_option_exposure += abs(p.position * entry_est * 100)
                
                available_budget = min(total_budget - existing_option_exposure, (equity * budget_pct) - ticker_option_exposure)
                cost_per_contract = entry_est * 100
                
                if available_budget < cost_per_contract:
                    print(f"⚠️ Insufficient Budget for {symbol} option.")
                    return
                
                current_risk_target = equity * option_allocation_pct * RISK_PER_TRADE
                risk_per_contract = entry_est * 0.20 * 100
                qty_risk = int(current_risk_target // risk_per_contract) if risk_per_contract > 0 else 0
                qty_cap = int(available_budget // cost_per_contract)
                qty = min(qty_risk, qty_cap)
                if max_option_contracts != -1 and qty > max_option_contracts:
                    print(f"DEBUG: Capping contracts from {qty} to {max_option_contracts}.")
                    qty = max_option_contracts
                if qty < 1:
                    print("⚠️ Calculated contracts < 1. Skipping.")
                    return

                print(f"Sizing: Equity ${equity:.2f} | Risk Target ${current_risk_target:.2f} | Risk/Ctr ${risk_per_contract:.2f} | Budget ${available_budget:.2f} | Qty: {qty}")
                # Place Bracket
                action = 'BUY'
                bracket = _build_fill_friendly_bracket(
                    ib,
                    action,
                    qty,
                    entry_est,
                    tp_price,
                    sl_price,
                    min_tick=min_tick,
                )
                placed_trades = await _submit_bracket_with_guard(
                    ib,
                    opt_contract,
                    bracket,
                    label=opt_contract.localSymbol,
                )
                parent_trade = placed_trades[0] if placed_trades else None
                filled = await _chase_or_cancel_unfilled_parent(
                    ib,
                    opt_contract,
                    parent_trade,
                    min_tick=min_tick,
                    label=opt_contract.localSymbol,
                )
                if not filled:
                    ticker_state.pop("pending_entry", None)
                    state[symbol] = ticker_state
                    save_trade_state(state, symbol)
                    return
                
                # Notify
                send_discord_live_trading_notification(
                    signal=f"buy_option_{opt_contract.right}",
                    symbol=opt_contract.localSymbol,
                    order_details={
                        "Action": action,
                        "Qty": qty,
                        "Entry": entry_est,
                        "SL": sl_price,
                        "TP": tp_price
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                
                print(f"✅ OPTION BRACKET SUBMITTED: {opt_contract.localSymbol}")
                # Update state
                ticker_state["daily_trade_count"] = daily_count + 1
                ticker_state.pop("pending_entry", None)
                state[symbol] = ticker_state
                save_trade_state(state, symbol)
                return

            except Exception as e:
                print(f"❌ Option Order failed: {e}")
        else:
             print("Could not find suitable option contract. Falling back to shares.")
        
    # --- STOCK TRADING LOGIC ---
    if signal == "buy":
        if qty_held == 0:
            price = float(price)
            swing_low = get_last_swing_low(price_df, window=5)
            if swing_low and swing_low < price:
                sl_price = float(round(swing_low, 2))
                print(f"Structure Stop: Swing Low at {sl_price}")
            else:
                sl_price = float(round(price * 0.995, 2))
                print(f"Fallback Stop: 0.5% at {sl_price}")

            risk_dist = price - sl_price
            tp_price = float(round(price + (risk_dist * 2.5), 2))
            
            qty = await calculate_smart_quantity(
                symbol,
                price,
                sl_price,
                budget_override=stock_allocation_pct,
                risk_allocation_pct=stock_allocation_pct
            )
            if qty <= 0: return

            print(f"Placing BUY Bracket: Entry ~{price} | SL {sl_price:.2f} | TP {tp_price:.2f} | Qty {qty}")
            
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                bracket = _build_fill_friendly_bracket(
                    ib,
                    'BUY',
                    qty,
                    price,
                    tp_price,
                    sl_price,
                    min_tick=0.01,
                )
                placed_trades = await _submit_bracket_with_guard(ib, contract, bracket, label=symbol)
                parent_trade = placed_trades[0] if placed_trades else None
                filled = await _chase_or_cancel_unfilled_parent(
                    ib,
                    contract,
                    parent_trade,
                    min_tick=0.01,
                    label=symbol,
                )
                if not filled:
                    return
                
                # Notify
                send_discord_live_trading_notification(
                    signal="buy_stock",
                    symbol=symbol,
                    order_details={
                        "Action": "BUY",
                        "Qty": qty,
                        "Entry": price,
                        "SL": sl_price,
                        "TP": tp_price
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                
                print("✅ BUY BRACKET ORDER SUBMITTED")
                ticker_state["daily_trade_count"] = daily_count + 1
                state[symbol] = ticker_state
                save_trade_state(state, symbol)
            except Exception as e:
                print(f"❌ Order failed: {e}")
        
        elif qty_held < 0: # We are Short in IBKR
             print(f"Closing Short Position ({qty_held}) due to BUY signal.")
             try:
                ib.reqGlobalCancel()
                close_order = MarketOrder('BUY', abs(qty_held))
                ib.placeOrder(Stock(symbol, 'SMART', 'USD'), close_order)
                
                # Notify
                send_discord_live_trading_notification(
                    signal="flip_close_short_stock",
                    symbol=symbol,
                    order_details={
                        "Action": "BUY (Market)",
                        "Qty": abs(qty_held),
                        "Price": price
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                print("✅ SHORT CLOSE SUBMITTED")
             except Exception as e:
                print(f"❌ Close Short failed: {e}")
        else:
            print(f"Already Long {qty_held}. Ignoring BUY signal.")

    elif signal == "sell":
        if qty_held == 0:
            print("SELL Signal detected but no position held. Skipping.")
            return
            
        elif qty_held > 0: # We are Long
            print(f"Closing Long Position ({qty_held}) due to SELL signal.")
            try:
                ib.reqGlobalCancel()
                close_order = MarketOrder('SELL', abs(qty_held))
                ib.placeOrder(Stock(symbol, 'SMART', 'USD'), close_order)
                
                # Notify
                send_discord_live_trading_notification(
                    signal="flip_close_long_stock",
                    symbol=symbol,
                    order_details={
                        "Action": "SELL (Market)",
                        "Qty": abs(qty_held),
                        "Price": price
                    },
                    confidence=confidence,
                    strategy_bias=bias
                )
                print("✅ LONG CLOSE SUBMITTED")
            except Exception as e:
                 print(f"❌ Close Long failed: {e}")
                 
        else:
            print(f"Already Short {qty_held} shares. Ignoring SELL signal.")



def parse_option_expiry(symbol):
    # Regex to capture YYMMDD from SPY251219C00500000
    # Format: Root(up to 6 chars) + YYMMDD + Type(C/P) + Strike
    match = re.search(r'[A-Z]+(\d{6})[CP]', symbol)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, "%y%m%d")
    return None

async def manage_option_expiry(target_symbol):
    ib = ibkr_mgr.ib
    """
    Checks for options expiring today and force closes them if RTH is open.
    """
    try:
        now = datetime.now()
        positions = ib.positions()
        for p in positions:
            if not isinstance(p.contract, Option): continue
            
            # Filter by target symbol
            if p.contract.symbol != target_symbol: continue

            contract = p.contract
            # Ensure contract is qualified (has exchange)
            if not contract.exchange:
                print(f"🔍 Qualifying contract for {contract.localSymbol}...")
                qualified = await ib.qualifyContractsAsync(contract)
                if qualified:
                    contract = qualified[0]
                    contract.exchange = 'SMART' # Force SMART
            
            expiry_str = contract.lastTradeDateOrContractMonth
            expiry = datetime.strptime(expiry_str, "%Y%m%d")
            
            is_expiry_day = (now.date() >= expiry.date())
            
            if is_expiry_day:
                print(f"⚠️ CRITICAL: {contract.localSymbol} expires TODAY! Force Closing.")
                # Close with Market Order
                action = 'SELL' if p.position > 0 else 'BUY'
                close_order = MarketOrder(action, abs(p.position))
                ib.placeOrder(contract, close_order)
                
                # Notify
                send_discord_live_trading_notification(
                    signal="option_expiry_close",
                    symbol=contract.localSymbol,
                    order_details={
                        "Action": action,
                        "Qty": abs(p.position),
                        "Reason": "Option Expiring Today"
                    },
                    confidence=0,
                    strategy_bias="neutral"
                )
            else:
                days_left = (expiry.date() - now.date()).days
                print(f"DEBUG: {contract.localSymbol} DTE: {days_left} days - Safe")

    except Exception as e:
        print(f"Error in manage_option_expiry: {e}")

from alpaca.trading.requests import ReplaceOrderRequest

# ================= STATE MANAGEMENT =================

# ================= STATE =================
STATE_FILE = "trade_state.json"
GLOBAL_SAFETY_FILE = "global_safety.json"

def load_global_safety_state():
    if os.path.exists(GLOBAL_SAFETY_FILE):
        try:
            with open(GLOBAL_SAFETY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"peak_equity": 0.0, "halted": False, "last_loss_time": None}

def save_global_safety_state(state):
    try:
        with open(GLOBAL_SAFETY_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error saving global safety state: {e}")

def update_peak_equity(current_equity):
    state = load_global_safety_state()
    if current_equity > state.get("peak_equity", 0.0):
        state["peak_equity"] = current_equity
        save_global_safety_state(state)
        print(f"📈 New Peak Equity: ${current_equity:.2f}")
    
    # Check for drawdown halt
    max_dd = getattr(config, 'MAX_GLOBAL_DRAWDOWN', 0.15)
    peak = state.get("peak_equity", 0.0)
    if peak > 0:
        dd = (peak - current_equity) / peak
        if dd >= max_dd and not state.get("halted", False):
            state["halted"] = True
            save_global_safety_state(state)
            print(f"🚨 GLOBAL DRAWDOWN CIRCUIT BREAKER HIT ({dd*100:.1f}%)! Trading Halted.")
    return state

def mark_loss():
    state = load_global_safety_state()
    state["last_loss_time"] = datetime.now(timezone.utc).isoformat()
    save_global_safety_state(state)
    print("🕒 Post-Loss Cool-down Triggered (60 min).")

def get_state_file_path(symbol=None):
    """Returns the state file path, prioritized by --state-file then symbol-specific."""
    if 'args' in globals() and hasattr(args, 'state_file') and args.state_file:
        return args.state_file
    if symbol:
        return f"trade_state_{symbol}.json"
    return STATE_FILE

def load_trade_state(symbol=None):
    path = get_state_file_path(symbol)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading state from {path}: {e}")
    return {}

def save_trade_state(state, symbol=None):
    path = get_state_file_path(symbol)
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error saving state to {path}: {e}")

async def manage_trade_updates(target_symbol):
    ib = ibkr_mgr.ib
    """
    Asynchronous trade updates for IBKR (Hybrid Trailing SL).
    """
    try:
        positions = ib.positions()
        for p in positions:
            symbol = p.contract.symbol
            if symbol != target_symbol: continue
            
            contract = p.contract
            # Ensure contract is qualified (has exchange)
            if not contract.exchange:
                print(f"🔍 Qualifying contract for updates on {contract.localSymbol}...")
                qualified = await ib.qualifyContractsAsync(contract)
                if qualified:
                    contract = qualified[0]
                    contract.exchange = 'SMART' # Force SMART

            ticker = ib.ticker(contract)
            # Only request if not already streaming (checking all tickers)
            if ticker not in ib.tickers():
                print(f"DEBUG: Starting data stream for {contract.localSymbol}...")
                ib.reqMktData(contract)
            
            if not ticker: continue
                
            curr_price = ticker.marketPrice()
            if not curr_price or curr_price <= 0 or np.isnan(curr_price):
                # Fallback for delayed data
                curr_price = ticker.last if ticker.last > 0 else ticker.close
            
            if not curr_price or curr_price <= 0 or np.isnan(curr_price):
                # Final fallback to historical MIDPOINT
                print(f"ℹ️ Market Data (Error 10089) - Falling back to Historical MIDPOINT for {symbol}...")
                curr_price = await get_latest_price_fallback(contract)
                
            if not curr_price or curr_price <= 0 or np.isnan(curr_price): continue
            
            curr_price = float(curr_price)
            
            # IBKR reports avgCost as total cost per contract (Price * 100)
            # We must normalize it to per-share to compare with curr_price
            multiplier = float(contract.multiplier) if contract.multiplier and float(contract.multiplier) > 1 else 1.0
            avg_cost = float(p.avgCost) / multiplier
            
            if avg_cost <= 0: continue
            
            is_long = (p.position > 0)
            pl_pct = (curr_price - avg_cost) / avg_cost if is_long else (avg_cost - curr_price) / avg_cost
 
            # --- OPTIONS RISK MANAGEMENT ---
            if isinstance(contract, Option):
                state = load_trade_state(symbol)
                symbol_state = state.get(symbol, {"virtual_stop": -0.20})
                
                # Exit Triggers
                if pl_pct <= virtual_stop:
                    print(f"🛑 OPTION STOP HIT: {contract.localSymbol} at {pl_pct*100:.1f}%. Closing.")
                    ib.reqGlobalCancel()
                    action = 'SELL' if p.position > 0 else 'BUY'
                    ib.placeOrder(contract, MarketOrder(action, abs(p.position)))
                    
                    # Notify
                    send_discord_live_trading_notification(
                        signal="stop_loss_close_option",
                        symbol=contract.localSymbol,
                        order_details={
                            "Action": action,
                            "Qty": abs(p.position),
                            "Price": curr_price,
                            "P&L%": f"{pl_pct*100:.2f}%"
                        },
                        confidence=0,
                        strategy_bias="neutral"
                    )
                    
                    if symbol in state: del state[symbol]; save_trade_state(state, symbol)
                    continue
                elif pl_pct >= 0.50: # TP 50%
                    print(f"🎯 OPTION TP HIT: {contract.localSymbol} at {pl_pct*100:.1f}%. Closing.")
                    ib.reqGlobalCancel()
                    action = 'SELL' if p.position > 0 else 'BUY'
                    ib.placeOrder(contract, MarketOrder(action, abs(p.position)))
                    
                    # Notify
                    send_discord_live_trading_notification(
                        signal="take_profit_close_option",
                        symbol=contract.localSymbol,
                        order_details={
                            "Action": action,
                            "Qty": abs(p.position),
                            "Price": curr_price,
                            "P&L%": f"{pl_pct*100:.2f}%"
                        },
                        confidence=0,
                        strategy_bias="neutral"
                    )
                    
                    if symbol in state: del state[symbol]; save_trade_state(state, symbol)
                    continue

                # Trailing logic
                updated = False
                
                # Hybrid Strategy: +10% BE, +20% -> +10%, +30% -> +20%, +40% -> +30%
                if pl_pct >= 0.40 and virtual_stop < 0.30:
                    virtual_stop = 0.30
                    updated = True
                    print(f"💰 OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to +30%.")
                elif pl_pct >= 0.30 and virtual_stop < 0.20:
                    virtual_stop = 0.20
                    updated = True
                    print(f"💰 OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to +20%.")
                elif pl_pct >= 0.20 and virtual_stop < 0.10:
                    virtual_stop = 0.10
                    updated = True
                    print(f"💰 OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to +10%.")
                elif pl_pct >= 0.10 and virtual_stop < 0.0:
                    virtual_stop = 0.0
                    updated = True
                    print(f"🛡️ OPTION TRAILING (HYBRID): {symbol} up {pl_pct*100:.1f}%. Virtual SL set to BE (0%).")
                
                if updated:
                    state[symbol] = {"virtual_stop": virtual_stop}
                    save_trade_state(state, symbol)
                continue

            # --- STOCK TRAILING ---
            if pl_pct < 0.15: continue
            
            open_trades = ib.openTrades()
            stop_order_trade = None
            for t in open_trades:
                if t.contract.conId == p.contract.conId and t.order.orderType in ['STP', 'STP LMT']:
                    stop_order_trade = t
                    break
            
            if not stop_order_trade: continue
            
            curr_stop = stop_order_trade.order.auxPrice
            new_stop = None
            
            if pl_pct > 0.09: # Lock 5%
                target_stop = avg_cost * 1.05 if is_long else avg_cost * 0.95
                if (is_long and curr_stop < target_stop) or (not is_long and curr_stop > target_stop):
                    new_stop = target_stop
            elif pl_pct > 0.05: # Break Even
                target_stop = avg_cost
                if (is_long and curr_stop < target_stop) or (not is_long and curr_stop > target_stop):
                    new_stop = target_stop
            
            if new_stop:
                stop_order_trade.order.auxPrice = round(new_stop, 2)
                ib.placeOrder(p.contract, stop_order_trade.order)
                print(f"✅ SL Updated for {symbol} to {new_stop:.2f}")

    except Exception as e:
        print(f"Error in manage_trade_updates: {e}")

    except Exception as e:
        print(f"Error in manage_trade_updates: {e}")
    
    # Final Cleanup: Remove state for any symbols we no longer hold
    try:
        current_positions = ib.positions()
        held_symbols = {p.contract.symbol for p in current_positions}
        state = load_trade_state(target_symbol)
        state_symbols = list(state.keys())
        cleaned = False
        for s in state_symbols:
            if s not in held_symbols:
                del state[s]
                cleaned = True
        if cleaned:
            save_trade_state(state, target_symbol)
    except Exception:
        pass

def is_market_open():
    """
    Checks if US markets are currently open (9:30 AM - 4:00 PM ET, Mon-Fri).
    Returns (bool, status_message)
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    
    # Weekend Check
    if now_et.weekday() >= 5:
        return False, "Market is CLOSED (Weekend)"
    
    # Time Check
    current_time = now_et.time()
    market_open = datetime.strptime("09:30:00", "%H:%M:%S").time()
    market_close = datetime.strptime("16:00:00", "%H:%M:%S").time()
    
    if current_time < market_open:
        return False, "Market is CLOSED (Pre-Market)"
    if current_time > market_close:
        return False, "Market is CLOSED (After-Hours)"
        
    return True, "Market is OPEN"

# ================= SESSION MANAGEMENT =================

class TradingSession:
    """Manages trading session state and statistics."""
    
    def __init__(self, duration_hours=None):
        ib = ibkr_mgr.ib
        self.start_time = datetime.now()
        self.duration_hours = duration_hours
        self.trades_executed = 0
        self.should_stop = False
        
        # Enhanced tracking
        self.opening_equity = None
        self.opening_positions = []
        self.closed_trades = []  # List of dicts with {symbol, pnl, win}
        self.day_starting_equity = None
        self.daily_loss_limit_hit = False
        
        # Capture opening state
        try:
            acc_values = ib.accountValues()
            for v in acc_values:
                if v.tag == 'NetLiquidation':
                    self.opening_equity = float(v.value)
                    break
            
            positions = ib.positions()
            for p in positions:
                multiplier = float(p.contract.multiplier) if p.contract.multiplier and float(p.contract.multiplier) > 1 else 1.0
                entry_price = float(p.avgCost) / multiplier
                
                self.opening_positions.append({
                    'symbol': p.contract.localSymbol,
                    'qty': float(p.position),
                    'side': 'long' if p.position > 0 else 'short',
                    'entry_price': entry_price,
                    'market_value': float(p.position * p.avgCost), # p.avgCost is already Price * Multiplier
                    'unrealized_pl': 0.0 # Requires ticker
                })
        except Exception as e:
            print(f"⚠️ Could not capture opening state: {e}")
            self.opening_equity = 0.0
        
    def record_trade(self):
        """Record that a trade was executed."""
        self.trades_executed += 1
    
    def record_closed_trade(self, symbol, pnl):
        """Record a closed trade with its P&L."""
        self.closed_trades.append({
            'symbol': symbol,
            'pnl': pnl,
            'win': pnl > 0
        })
        
    def should_continue(self):
        """Check if the session should continue running."""
        if self.should_stop:
            return False
            
        # Check duration limit
        if self.duration_hours is not None:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours >= self.duration_hours:
                return False
                
        return True
    
    def get_summary(self):
        ib = ibkr_mgr.ib
        """Get session summary statistics."""
        elapsed = datetime.now() - self.start_time
        hours = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)
        
        # Get current account state
        closing_equity = 0.0
        closing_positions = []
        try:
            acc_values = ib.accountValues()
            for v in acc_values:
                if v.tag == 'NetLiquidation':
                    closing_equity = float(v.value)
                    break
            
            positions = ib.positions()
            for p in positions:
                multiplier = float(p.contract.multiplier) if p.contract.multiplier and float(p.contract.multiplier) > 1 else 1.0
                entry_price = float(p.avgCost) / multiplier
                
                closing_positions.append({
                    'symbol': p.contract.localSymbol,
                    'qty': float(p.position),
                    'side': 'long' if p.position > 0 else 'short',
                    'entry_price': entry_price,
                    'current_price': 0.0, # Requires ticker
                    'market_value': float(p.position * p.avgCost),
                    'unrealized_pl': 0.0,
                    'unrealized_plpc': 0.0
                })
        except Exception as e:
            print(f"⚠️ Could not fetch closing state: {e}")
        
        # Calculate statistics
        total_pnl = closing_equity - self.opening_equity if self.opening_equity else 0.0
        pnl_pct = (total_pnl / self.opening_equity * 100) if self.opening_equity else 0.0
        
        # Win rate from closed trades
        wins = sum(1 for t in self.closed_trades if t['win'])
        losses = len(self.closed_trades) - wins
        win_rate = (wins / len(self.closed_trades) * 100) if self.closed_trades else 0.0
        
        # P&L stats from closed trades
        closed_pnl = sum(t['pnl'] for t in self.closed_trades)
        winning_trades = [t['pnl'] for t in self.closed_trades if t['win']]
        losing_trades = [t['pnl'] for t in self.closed_trades if not t['win']]
        
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        # Build summary
        summary = [
            "=" * 60,
            "SESSION SUMMARY",
            "=" * 60,
            f"Start Time:      {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"End Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration:        {hours}h {minutes}m",
            "",
            "ACCOUNT PERFORMANCE",
            "-" * 60,
            f"Opening Equity:  ${self.opening_equity:,.2f}",
            f"Closing Equity:  ${closing_equity:,.2f}",
            f"Total P&L:       ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)",
            "",
            "TRADING ACTIVITY",
            "-" * 60,
            f"Trades Executed: {self.trades_executed}",
            f"Trades Closed:   {len(self.closed_trades)}",
        ]
        
        if self.closed_trades:
            summary.extend([
                f"Win Rate:        {win_rate:.1f}% ({wins}W / {losses}L)",
                "",
                "CLOSED TRADES P&L",
                "-" * 60,
                f"Total Realized:  ${closed_pnl:+,.2f}",
                f"Average Win:     ${avg_win:+,.2f}",
                f"Average Loss:    ${avg_loss:+,.2f}",
                f"Largest Win:     ${largest_win:+,.2f}",
                f"Largest Loss:    ${largest_loss:+,.2f}",
            ])
        
        # Opening positions
        if self.opening_positions:
            summary.extend([
                "",
                "OPENING POSITIONS",
                "-" * 60,
            ])
            for pos in self.opening_positions:
                summary.append(
                    f"  {pos['symbol']:20s} {pos['side']:5s} {pos['qty']:>8.0f} @ ${pos['entry_price']:>8.2f} "
                    f"| Value: ${pos['market_value']:>10,.2f} | P&L: ${pos['unrealized_pl']:>+10,.2f}"
                )
        
        # Closing positions
        if closing_positions:
            summary.extend([
                "",
                "CLOSING POSITIONS",
                "-" * 60,
            ])
            for pos in closing_positions:
                summary.append(
                    f"  {pos['symbol']:20s} {pos['side']:5s} {pos['qty']:>8.0f} @ ${pos['entry_price']:>8.2f} "
                    f"| Current: ${pos['current_price']:>8.2f} | P&L: ${pos['unrealized_pl']:>+10,.2f} ({pos['unrealized_plpc']*100:+.2f}%)"
                )
        
        # Limits
        if self.duration_hours:
            summary.extend([
                "",
                "SESSION LIMITS",
                "-" * 60,
            ])
            if self.duration_hours:
                summary.append(f"Duration Limit:  {self.duration_hours} hours")
        
        summary.append("=" * 60)
        return "\n".join(summary)
    
    def request_stop(self):
        """Request session to stop gracefully."""
        self.should_stop = True
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print("\n\n🛑 Shutdown signal received. Stopping session gracefully...")
            self.request_stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def interruptible_sleep(seconds, session):
    """Sleeps for the specified time, but returns early if session should stop."""
    end_time = time.time() + seconds
    while time.time() < end_time and session.should_continue():
        # Sleep in small increments to stay responsive
        remaining = end_time - time.time()
        await asyncio.sleep(min(1.0, remaining))

async def main():
    parser = argparse.ArgumentParser(description="Run the SMC trading bot.")
    parser.add_argument("symbol", nargs="?", default=SYMBOL, help="Symbol to trade (default: SPY)")
    parser.add_argument("--options", action="store_true", help="Enable options trading (overrides config)")
    parser.add_argument("--cap", type=int, metavar="N", help="Daily trade cap: -1 for unlimited, 0 for no trades, positive for max trades per day (default: 5)")
    parser.add_argument("--session-duration", type=float, help="Session duration in hours (runs indefinitely if not specified)")
    parser.add_argument("--option-allocation", type=float, help="Fraction of equity allocated to options (0.0 to 1.0). Stock allocation is 1 - option-allocation.")
    parser.add_argument("--max-option-contracts", type=int, help="Maximum option contracts per trade (-1 for no limit)")
    parser.add_argument("--state-file", type=str, help="Custom path to the trade state JSON file")
    parser.add_argument("--min-conf", type=str, choices=['all', 'low', 'medium', 'high'], default='all', help="Minimum confidence level to display/alert (default: all)")
    
    global args
    args = parser.parse_args()
    target_symbol = args.symbol
    
    if args.options:
        print("Overriding ENABLE_OPTIONS to True from command line.")
        config.ENABLE_OPTIONS = True
    
    option_allocation = args.option_allocation if args.option_allocation is not None else getattr(config, 'OPTIONS_ALLOCATION_PCT', 0.20)
    option_allocation = float(option_allocation)
    if option_allocation < 0 or option_allocation > 1:
        print("❌ Error: --option-allocation must be within [0.0, 1.0].")
        sys.exit(1)
    max_option_contracts = args.max_option_contracts if args.max_option_contracts is not None else getattr(config, 'MAX_OPTION_CONTRACTS', -1)
    max_option_contracts = int(max_option_contracts)
    if max_option_contracts != -1 and max_option_contracts < 1:
        print("❌ Error: --max-option-contracts must be -1 (unlimited) or >= 1.")
        sys.exit(1)
    
    daily_cap = args.cap if args.cap is not None else getattr(config, 'DEFAULT_DAILY_CAP', 5)
    
    # Map confidence choices to numeric thresholds
    CONF_THRESHOLDS = {
        'all': 0,
        'low': 20,
        'medium': 60,
        'high': 80
    }
    min_conf_val = CONF_THRESHOLDS.get(args.min_conf, 0)

    # Connect to IBKR
    connected = await ibkr_mgr.connect()
    if not connected:
        print("❌ Failed to connect to IBKR. Exiting.")
        return

    # Initialize session
    session = TradingSession(duration_hours=args.session_duration)
    session.setup_signal_handlers()
    
    print(f"Starting SMC Bot for {target_symbol} (Options: {config.ENABLE_OPTIONS})...")
    
    try:
        while session.should_continue():
            try:
                now_et = datetime.now(ZoneInfo("America/New_York"))
                
                # 1. Market Hours Filter
                is_open, status_msg = is_market_open()
                if not is_open:
                    print(f"🕒 {status_msg}. Waiting 15 minutes...")
                    await interruptible_sleep(900, session)
                    continue

                # 2. Daily Loss Limit Check (3%)
                # Fetch equity from IBKR
                current_equity = 10000.0
                acc_values = ibkr_mgr.ib.accountValues()
                for v in acc_values:
                    if v.tag == 'NetLiquidation':
                        current_equity = float(v.value)
                        break
                        
                current_date = now_et.date()
                
                # Reset daily starting equity at start of new trading day
                if session.day_starting_equity is None or (hasattr(session, 'last_trading_date') and session.last_trading_date != current_date):
                    session.day_starting_equity = current_equity
                    session.daily_loss_limit_hit = False
                    session.last_trading_date = current_date
                    print(f"🔄 New trading day. Starting equity: ${session.day_starting_equity:,.2f}")
                
                # Check daily loss limit
                daily_pnl_pct = (current_equity - session.day_starting_equity) / session.day_starting_equity if session.day_starting_equity > 0 else 0
                if daily_pnl_pct <= -0.03 and not session.daily_loss_limit_hit:
                    print(f"🛑 DAILY LOSS LIMIT HIT (-3%). Current: ${current_equity:,.2f} | Start: ${session.day_starting_equity:,.2f} | Loss: {daily_pnl_pct*100:.2f}%")
                    print("🛑 Closing all positions and stopping trading for today.")
                    # Close all positions in IBKR
                    try:
                        ibkr_mgr.ib.reqGlobalCancel() # Cancel open orders
                        positions = ibkr_mgr.ib.positions()
                        for p in positions:
                            action = 'SELL' if p.position > 0 else 'BUY'
                            close_order = MarketOrder(action, abs(p.position))
                            ibkr_mgr.ib.placeOrder(p.contract, close_order)
                            print(f"Closed {p.contract.localSymbol}")
                    except Exception as e:
                        print(f"⚠️ Error closing positions: {e}")
                    session.daily_loss_limit_hit = True
                
                if session.daily_loss_limit_hit:
                    print("⏸️ Daily loss limit active. Waiting until next trading day...")
                    await interruptible_sleep(900, session)  # Wait 15 minutes
                    continue

                # 3. Active Trade Management (Trailing Stops & Option Expiry)
                await manage_trade_updates(target_symbol)
                await manage_option_expiry(target_symbol)

                # 2. Market is open, look for signals
                print(f"Analyzing {target_symbol} at {now_et}...")
                sig_res = await generate_signal(target_symbol)
                
                if sig_res:
                    sig, confidence = sig_res if isinstance(sig_res, tuple) else (sig_res, 0)
                    
                    if sig:
                        print(f"🚀 Signal detected: {sig.upper()}!({target_symbol}) | Confidence: {confidence}%")
                        
                        if confidence < min_conf_val:
                            print(f"⚠️ Filtering signal due to low confidence ({confidence}% < {min_conf_val}%)")
                            await interruptible_sleep(60, session)
                            continue

                        use_cap = (daily_cap != -1)
                        await place_trade(
                            sig,
                            confidence,
                            target_symbol, 
                            use_daily_cap=use_cap, 
                            daily_cap_value=daily_cap if use_cap else None,
                            option_allocation_override=option_allocation,
                            max_option_contracts_override=max_option_contracts
                        )
                        session.record_trade()
                        print("Trade placed. Cooling down for 5 minutes...")
                        await interruptible_sleep(300, session)
                else:
                    await interruptible_sleep(60, session)

            except Exception as e:
                print(f"An error occurred in the main loop: {e}")
                await interruptible_sleep(60, session)
    except KeyboardInterrupt:
        session.request_stop()
    finally:
        print("\n")
        print(session.get_summary())
        ibkr_mgr.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
