import pandas as pd
import storage
import calculations
import numpy as np

# Load real data
movies_df = storage.load_movies()
people_df = storage.load_people()
cache_df = storage.load_cache()
settings = storage.load_settings()

print(f"Movies in pool: {len(movies_df)}")
print(f"People in list: {len(people_df)}")

# Run the dashboard logic
features = calculations.build_feature_table(cache_df, movies_df)
expected = calculations.add_adjusted_expected(features, settings=settings)
bids, budget_info = calculations.add_bid_guidance(expected, people_df, settings=settings, budget_mode='personal')

print("\n--- SCALE CHECK (Taylor) ---")
print(f"Basis Budget: {budget_info.get('basis_budget')}")
print(f"League Remaining: {budget_info.get('league_remaining')}")

# Check first few rows
check_cols = ['ticker', 'adjusted_expected', 'fair_budget_bid', 'market_fair_bid', 'target_bid', 'target_market_bid']
print(bids[check_cols].head())

# Check for NaNs or Zeros in anchors
print(f"\nNaNs in fair_budget_bid: {bids['fair_budget_bid'].isna().sum()}")
print(f"Zeros in fair_budget_bid: {(bids['fair_budget_bid'] == 0).sum()}")

# Check P(edge) with various anchors
# Case A: Market Scale
p_bid_m, p_val_m = calculations.resolve_probability_anchor_columns('market_fair_bid')
probs_m = calculations.add_probability_of_edge(bids, bid_col=p_bid_m, value_col=p_val_m, settings=settings)
print(f"\nMarket P(edge) Mean: {probs_m['prob_positive_edge'].mean():.2f}")
print(f"Market P(edge) Max: {probs_m['prob_positive_edge'].max():.2f}")

# Case B: Personal Scale
p_bid_p, p_val_p = calculations.resolve_probability_anchor_columns('target_bid')
probs_p = calculations.add_probability_of_edge(bids, bid_col=p_bid_p, value_col=p_val_p, settings=settings)
print(f"Personal P(edge) Mean: {probs_p['prob_positive_edge'].mean():.2f}")
print(f"Personal P(edge) Max: {probs_p['prob_positive_edge'].max():.2f}")
