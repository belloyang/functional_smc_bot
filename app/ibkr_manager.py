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
    def ib((self)):
        return self._ib

    async def connect(self):
        """Establish connection to TWS/Gateway."""
        if not self._ib.isConnected():
            try:
                print(f"Connecting to IBKR at {config.IBKR_HOST}:{config.IBKR_PORT}...")
                await self._ib.connectAsync(
                    config.IBKR_HOST, 
                    config.IBKR_PORT, 
                    clientId=config.IBKR_CLIENT_ID
                )
                print("Successfully connected to IBKR.")
            except Exception as e:
                print(f"Failed to connect to IBKR: {e}")
                raise

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            print("Disconnected from IBKR.")

    def get_client(self):
        """Return the IB instance."""
        return self._ib

# Global instance
ibkr_mgr = IBKRManager()
