import logging
import asyncio
import time
from ib_insync import IB, util, Stock
try:
    from . import config
except ImportError:
    import config

# Setup ib_insync utility loop for async environments
util.patchAsyncio() 

class IBKRManager:
    _instance = None
    _ib = None
    _error_handler_attached = False
    _recent_error_keys = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IBKRManager, cls).__new__(cls)
            cls._ib = IB()
        return cls._instance

    @property
    def ib(self):
        if self._ib is None:
            self._ib = IB()
        return self._ib

    async def connect(self):
        """Establish connection to TWS/Gateway."""
        if self._ib is None:
            self._ib = IB()
            
        if self._ib.isConnected():
            return True
            
        try:
            print(f"Connecting to IBKR at {config.IBKR_HOST}:{config.IBKR_PORT}...")
            await self._ib.connectAsync(
                config.IBKR_HOST, 
                config.IBKR_PORT, 
                clientId=config.IBKR_CLIENT_ID
            )
            # Silence verbose raw ib_insync wrapper error logs; we print concise messages via errorEvent.
            logging.getLogger("ib_insync.wrapper").setLevel(logging.CRITICAL)
            if not self._error_handler_attached:
                self._ib.errorEvent += self._on_ib_error
                self._error_handler_attached = True
            # Use a single market data type. Calling 1/2/3/4 in sequence leaves IBKR on the last one.
            # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen
            market_data_type = int(getattr(config, "IBKR_MARKET_DATA_TYPE", 1))
            if market_data_type not in (1, 2, 3, 4):
                market_data_type = 1
            self._ib.reqMarketDataType(market_data_type)
            print(f"Successfully connected to IBKR (market data type={market_data_type}).")
            return True
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
            return False

    def _on_ib_error(self, req_id, code, message, contract):
        """Emit concise, deduplicated IBKR errors for known noisy cases."""
        symbol = getattr(contract, "localSymbol", None) or getattr(contract, "symbol", "?")
        code = int(code)

        if code == 300:
            # Cleanup noise ("Can't find EId with tickerId")
            return

        if code in (354, 10089, 10091, 10168):
            short = "Market data subscription missing"
            if code == 10168:
                short = "Market data not subscribed and delayed data not enabled"
            elif code == 10089:
                short = "Live market data requires additional API subscription (delayed may be available)"
            elif code == 10091:
                short = "Part of requested market data requires additional API subscription"
            detail = f"⚠️ {short} [{symbol}] (code {code})"
        elif code == 162:
            detail = f"⚠️ Historical data request failed/cancelled [{symbol}] (code 162)"
        else:
            # Keep unfamiliar errors visible but short.
            compact_msg = str(message).split(".")[0][:140]
            detail = f"⚠️ IBKR error {code} [{symbol}]: {compact_msg}"

        now = time.time()
        key = (code, symbol, detail)
        last = self._recent_error_keys.get(key, 0.0)
        if now - last < 8.0:
            return
        self._recent_error_keys[key] = now
        print(detail)

    async def historical_health_check(self, symbol="SPY", timeout_sec=15):
        """
        Lightweight historical data health probe.
        Returns (ok: bool, detail: str).
        """
        if not self._ib or not self._ib.isConnected():
            return False, "not connected"
        try:
            contract = await self._ib.qualifyContractsAsync(Stock(symbol, "SMART", "USD"))
            if not contract:
                return False, "contract qualification failed"
            bars = await asyncio.wait_for(
                self._ib.reqHistoricalDataAsync(
                    contract[0],
                    endDateTime="",
                    durationStr="2 D",
                    barSizeSetting="15 mins",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                    keepUpToDate=False,
                ),
                timeout=timeout_sec,
            )
            if not bars:
                return False, "no bars"
            return True, f"{len(bars)} bars"
        except asyncio.TimeoutError:
            return False, "timeout"
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            print("Disconnected from IBKR.")
        # Clear the reference to avoid __del__ errors after loop closure
        self._ib = None
        self._error_handler_attached = False
        self._recent_error_keys = {}

    def get_client(self):
        """Return the IB instance."""
        return self._ib

# Global instance
ibkr_mgr = IBKRManager()
