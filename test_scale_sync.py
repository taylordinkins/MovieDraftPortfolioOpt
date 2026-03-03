import pandas as pd
import numpy as np
import calculations

# 1. Setup mock data
# Imagine a movie worth $20 in HSX units.
# League total liquidity = $2000. Personal wallet = $100.
# The movie is 5% of the total available HSX value.
hsx_df = pd.DataFrame([{
    'ticker': 'TEST', 
    'current_price': 20.0, 
    'adjusted_expected': 20.0,
    'days_to_release': 14,
    'vol_30': 0.05,
    'drawdown_30': 0.02
}])

people_df = pd.DataFrame([
    {'name': 'Taylor', 'budget': 100.0},
    {'name': 'League', 'budget': 2000.0}
])

settings = {
    'strategy_risk_a_vol': 3.0,
    'strategy_risk_b_drawdown': 1.4,
    'strategy_risk_c_release': 0.45,
    'strategy_risk_max_penalty': 0.45,
    'strategy_bid_multiplier': 1.0,
    'strategy_drawdown_threshold': 0.15,
}

# 2. Test Personal Scale
print("--- TEST 1: Personal Scale ---")
# fair_budget_bid should be 5% of $100 = $5.00
# target_bid should be slightly less due to risk penalty.
df_p, _ = calculations.add_bid_guidance(hsx_df.copy(), people_df, settings, budget_mode='personal')
# Manually simulate P(edge) for personal
df_p = calculations.add_probability_of_edge(
    df_p, 
    bid_col='target_bid', 
    value_col='fair_budget_bid', 
    settings=settings,
    history_loader=lambda x: pd.DataFrame({'price': [20.0]*100}) # Flat history
)
print(f"Personal Anchor (fair_budget_bid): {df_p['fair_budget_bid'].iloc[0]}")
print(f"Personal Bid (target_bid): {df_p['target_bid'].iloc[0]}")
print(f"P(edge): {df_p['prob_positive_edge'].iloc[0]}")

# 3. Test Market Scale
print("\n--- TEST 2: Market Scale ---")
# market_fair_bid should be 5% of $2000 = $100.00
df_m, _ = calculations.add_bid_guidance(hsx_df.copy(), people_df, settings, budget_mode='personal')
# Manually simulate P(edge) for market
df_m = calculations.add_probability_of_edge(
    df_m, 
    bid_col='target_market_bid', 
    value_col='market_fair_bid', 
    settings=settings,
    history_loader=lambda x: pd.DataFrame({'price': [20.0]*100}) # Flat history
)
print(f"Market Anchor (market_fair_bid): {df_m['market_fair_bid'].iloc[0]}")
print(f"Market Bid (target_market_bid): {df_m['target_market_bid'].iloc[0]}")
print(f"P(edge): {df_m['prob_positive_edge'].iloc[0]}")

# 4. Test Mismatch (What we want to avoid)
print("\n--- TEST 3: SCALE MISMATCH (Anchor=Market, Bid=Personal) ---")
df_miss = calculations.add_probability_of_edge(
    df_m.copy(), 
    bid_col='target_bid', 
    value_col='market_fair_bid', 
    settings=settings,
    history_loader=lambda x: pd.DataFrame({'price': [20.0]*100}) 
)
print(f"Anchor (market_fair_bid): {df_miss['market_fair_bid'].iloc[0]}")
print(f"Bid (target_bid): {df_miss['target_bid'].iloc[0]}")
print(f"P(edge): {df_miss['prob_positive_edge'].iloc[0]}")
