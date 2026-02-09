from ib_insync import *
import numpy as np

def test_mutability():
    # Mocking a position and contract
    contract = Stock('AAPL', 'SMART', 'USD')
    # Use a dummy position
    position = Position('account', contract, 100, 150.0)
    
    print(f"Testing Position mutability: {position}")
    try:
        position.contract = Stock('MSFT', 'SMART', 'USD')
        print("✅ Position.contract is mutable")
    except AttributeError as e:
        print(f"❌ Position.contract Error: {e}")
    except Exception as e:
        print(f"❌ Position.contract Random Error: {type(e).__name__}: {e}")

    print(f"Testing Contract mutability: {contract}")
    try:
        contract.exchange = 'ISLAND'
        print("✅ Contract.exchange is mutable")
    except AttributeError as e:
        print(f"❌ Contract.exchange Error: {e}")
    except Exception as e:
        print(f"❌ Contract.exchange Random Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_mutability()
