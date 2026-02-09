import numpy as np
from ib_insync import Option, Position

def calculate_pl_pct_test(market_price, avg_cost_contract, multiplier, is_long=True):
    # This simulates the logic now in bot.py
    avg_price_share = avg_cost_contract / multiplier
    
    if is_long:
        pl_pct = (market_price - avg_price_share) / avg_price_share
    else:
        pl_pct = (avg_price_share - market_price) / avg_price_share
        
    return pl_pct

def test_multiplier_logic():
    # Scenario from user:
    # QQQ Opt at $5.63 Market Price
    # AvgCost reported as $563.70 (Price * 100)
    market_p = 5.63
    avg_c = 563.70
    mult = 100.0
    
    # Old logic would do (5.63 - 563.70) / 563.70 = ~-0.99
    # New logic:
    pl = calculate_pl_pct_test(market_p, avg_c, mult)
    
    print(f"Market Price: {market_p}")
    print(f"Avg Cost (Contract): {avg_c}")
    print(f"Multiplier: {mult}")
    print(f"Calculated PL%: {pl*100:.2f}%")
    
    assert abs(pl) < 0.01, f"Expected near 0% PL, got {pl*100:.2f}%"
    print("✅ Multiplier Logic Verified")

    # Test with stock (Multiplier 1)
    market_s = 150.0
    avg_s = 150.0
    mult_s = 1.0
    pl_s = calculate_pl_pct_test(market_s, avg_s, mult_s)
    print(f"Stock PL%: {pl_s*100:.2f}%")
    assert pl_s == 0.0
    print("✅ Stock Logic Verified")

if __name__ == "__main__":
    test_multiplier_logic()
