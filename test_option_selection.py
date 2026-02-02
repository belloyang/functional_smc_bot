from functional_smc_bot import get_best_option_contract
import alpaca_config as config

def test_selection():
    symbol = config.SYMBOL
    print(f"Testing Option Selection for {symbol}...")
    
    # Test Call
    print("\n--- Testing CALL Selection ---")
    call_contract = get_best_option_contract(symbol, "buy", known_price=553)
    if call_contract:
        print(f"PASS: Found Call: {call_contract.symbol} Exp: {call_contract.expiration_date} Strike: {call_contract.strike_price}")
    else:
        print("FAIL: No Call found.")

    # Test Put
    print("\n--- Testing PUT Selection ---")
    put_contract = get_best_option_contract(symbol, "sell", known_price=553)
    if put_contract:
        print(f"PASS: Found Put: {put_contract.symbol} Exp: {put_contract.expiration_date} Strike: {put_contract.strike_price}")
    else:
        print("FAIL: No Put found.")

if __name__ == "__main__":
    test_selection()
