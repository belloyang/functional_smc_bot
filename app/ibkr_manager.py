import logging
import asyncio
from ib_insync import IB, util
try:
    from . import config
except ImportError:
    import config

# Setup ib_insync utility loop for async environments
# util.patchAsyncio() 

class IBKRManager:
    _instance = None
    _ib = None

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
            self._ib.reqMarketDataType(3) # Use delayed data if live is not available
            print("Successfully connected to IBKR (using Delayed Market Data).")
            return True
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            print("Disconnected from IBKR.")
        # Clear the reference to avoid __del__ errors after loop closure
        self._ib = None 

    def get_client(self):
        """Return the IB instance."""
        return self._ib

# Global instance
ibkr_mgr = IBKRManager()
