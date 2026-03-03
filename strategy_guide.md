# Movie Draft Strategy Guide (Analyst Playbook)

## 1) Purpose

This guide explains how to use the current dashboard end-to-end to make draft decisions that maximize your final portfolio outcome under auction budget constraints.

It is designed for the codebase as implemented now (CLI + GUI), including:

- expected-gross optimization
- win-probability optimization
- integer bidding constraints
- draft-state tracking
- Monte Carlo risk diagnostics
- optional correlation/search/opponent simulation upgrades (Phase 4)

---

## 2) What The Engine Actually Optimizes

### Core objective layers

1. **Movie value estimate** (per movie)
   - `market_expected = current_price`
   - `adjusted_expected = market_expected * clip(1 + lambda * signal, low, high)`
   - `signal` is a weighted blend of momentum/trend/volatility/drawdown/release timing features.

2. **Bid recommendation** (auction-dollar scale)
   - `gross_share = adjusted_expected / sum(adjusted_expected over available pool)`
   - `fair_budget_bid = gross_share * budget_basis`
   - `market_fair_bid = gross_share * league_remaining_budget`
   - `risk_penalty = clip(a*vol_30 + b*drawdown_30 + c*release_risk, 0, r_max)`
   - `target_bid = fair_budget_bid * (1 - risk_penalty) * bid_multiplier`
   - `target_market_bid = market_fair_bid * (1 - risk_penalty) * bid_multiplier`
   - `max_bid` is auction-scale and capped by your personal executable budget.

3. **Portfolio optimization**
   - Expected-gross mode: knapsack-style maximize adjusted value under budget and constraints.
   - Win-probability mode: evaluate candidate portfolios via Monte Carlo vs simulated opponents.

4. **Risk diagnostics**
   - `P(Edge) = P(value_at_bid > bid)`
   - `P(DD) = P(value_at_bid < bid*(1-drawdown_threshold))`
   - Portfolio Monte Carlo: mean/p10/p50/p90, downside probability, concentration downside, aggression sensitivity.

### Important modeling note

This is a decision-support model, not an oracle. Opponent behavior and joint movie shocks are simulated assumptions. Use sensitivity analysis as part of decisions.

---

## 3) Auction Workflow Assumptions Supported By The Code

The system supports your sequential auction flow:

1. Movie appears.
2. Bidding runs upward.
3. Winner is assigned.
4. Winner budget decrements by final price (if provided).
5. Movie is removed from available pool.
6. Repeat in release-date order.

Draft state is persisted in `data/auction_state.json`, and assignments are synchronized with `movies.csv` and `people.csv`.

---

## 4) Pre-Draft Setup Checklist

Before strategy runs:

1. Update HSX snapshot (`Scrape latest prices`).
2. Update histories (`Scrape Options -> Fetch price history for all pool movies`).
3. Ensure `movies.csv` has all tickers and release dates.
4. Ensure `people.csv` has all players and current remaining money.
5. Run `Draft state / assignment history` and verify:
   - currently active movie and bid are correct
   - already won movies are assigned
   - no stale assignments for removed tickers

If these are wrong, portfolio recommendations will be wrong.

---

## 5) Dashboard Columns: Practical Meaning

- `Price`: current HSX price (market baseline proxy).
- `AdjExp`: model-adjusted expected value.
- `TgtBid` / `TgtInt`: recommended bid (continuous or integer-executable).
- `TgtMkt` / `TgtMktInt`: market-pressure-scaled target bid (same risk logic, league-liquidity scale).
- `MaxBid` / `MaxInt`: walk-away ceiling.
- `MinBid` (integer mode): minimum legal next bid (`previous_bid + 1`).
- `Risk%`: direct bid penalty from vol/drawdown/release.
- `P(Edge)`: probability value exceeds bid threshold (relative to your chosen bid anchor).
- `P(DD)`: probability of material downside vs bid.
- `MktVR`: `adjusted_expected / current_price`.
- `Score`: blended ranking score.
- `Opt`: selected in optimizer solution.
- `Quality filter behavior`: filters constrain optimizer selection; rows can still appear in ranking table even if filtered out from `Opt`.
- `TgtInt` and `TgtMktInt` are per-movie anchors, not a spend plan. Their column sums are not expected to equal your budget.

---

## 6) Knobs (Conceptual Meaning)

### A) Valuation and signal shape

- `strategy_lambda`: how strongly features move `AdjExp` away from market.
- `strategy_clip_low/high`: lower/upper bound on that movement.
- Feature weights (`w_mom_7`, `w_mom_14`, `w_mom_30`, `w_trend_30`, `w_vol_30`, `w_drawdown_30`, `w_days_to_release`):
  - positive weight: feature increases value
  - negative weight: feature decreases value

### B) Bid and risk controls

- `strategy_bid_multiplier`: global aggression scaler on target bids.
- `strategy_risk_a_vol`, `strategy_risk_b_drawdown`, `strategy_risk_c_release`: penalty sensitivity.
- `strategy_risk_release_window_days`: short-release penalty window.
- `strategy_risk_max_penalty`: cap on risk haircut.

### C) Budget and concentration

- `budget basis`:
  - personal (default actionable)
  - league (market pressure context)
  - custom (what-if)
- `strategy_max_budget_pct_per_film`: base per-film spend cap.
- `strategy_market_fair_stresstest_cap`: optional higher cap used for `market_fair_bid` stress mode.
- `strategy_diversification_penalty`, `strategy_correlation_penalty`: penalize crowded/correlated portfolios.

### D) Probability and quality filters

- `strategy_risk_model`: `bootstrap` or `student_t`.
- `strategy_bootstrap_samples`: scenario sample count for edge probabilities.
- `strategy_drawdown_threshold`: cutoff for `P(DD)`.
- `strategy_enable_quality_filters`: on/off hard filtering.
- `strategy_min_prob_positive_edge`, `strategy_max_prob_large_drawdown`: filter thresholds.

### E) Monte Carlo core

- `strategy_mc_samples`: simulation sample count.
- `strategy_mc_num_opponents`: number of simulated opponents.
- `strategy_mc_candidate_portfolios`: baseline candidate count.
- `strategy_mc_opponent_noise`, `strategy_mc_aggression_sd`: opponent randomness/aggression.
- `strategy_mc_concentration_threshold`: concentration downside trigger.
- `strategy_mc_seed_mode`: `fixed` vs `random`.
- `strategy_mc_random_seed`: fixed reproducible seed.

### F) Integer bidding

- `strategy_integer_bid_mode`: enforce whole-number executable bids.
- `strategy_integer_prev_bid`: current high bid on active movie.

### G) Phase 4 correlation simulation

- `strategy_corr_simulation_mode`: `independent`, `gaussian_copula`, `t_copula`.
- `strategy_corr_shrinkage`: stabilize sparse correlation estimates.
- `strategy_corr_min_history_points`: minimum data for pairwise correlation use.
- `strategy_corr_floor`: eigenvalue floor for PSD correction.
- `strategy_corr_t_df`: tail heaviness for t-copula.

### H) Phase 4 search controls

- `strategy_search_mode`: `current_sampled`, `random_multistart`, `local_search`, `genetic`.
- `strategy_search_candidates`: candidate-set breadth.
- `strategy_search_local_iters`: local swap refinements.
- `strategy_search_population`, `strategy_search_generations`, `strategy_search_elite_frac`, `strategy_search_mutation_rate`: genetic search depth/exploration.

### I) Phase 4 opponent presets

- `strategy_opponent_profile`: `passive_value`, `balanced_field`, `aggressive_bidup`.
- `strategy_opponent_bidup_strength`: price pressure on top movies.
- `strategy_opponent_cash_conservation`: tendency to save/spend cash.

---

## 6b) Quick-Start Config (Time-Constrained Prep)

If you only have 15-30 minutes before a draft, focus on these 10 parameters:

| Parameter | Recommended Default | Why It Matters |
|-----------|-------------------|----------------|
| Preset | `balanced` | Sensible starting point for all weights and penalties |
| `strategy_lambda` | 0.3-0.5 | Controls how far AdjExp deviates from market price |
| `strategy_bid_multiplier` | 0.9-1.0 | Global aggression; lower = more conservative bids |
| `strategy_max_budget_pct_per_film` | 0.25-0.35 | Prevents overconcentration on a single movie |
| `strategy_risk_b_drawdown` | 0.9-1.8 | Penalizes high-drawdown movies; start from preset and tune in this band |
| `strategy_mc_samples` | 1500-3000 | Fast and stable for prep; use 5000-10000 for final stress runs |
| `strategy_mc_seed_mode` | `fixed` | Reproducible results during prep |
| `strategy_integer_bid_mode` | `on` | Match your league rules |
| Budget basis | `personal` | Uses your remaining budget as actionable allocation basis |
| Optimizer cost basis | `target_bid` | Actionable auction-dollar execution basis |
| `strategy_corr_simulation_mode` | `independent` | Skip copula unless you have 30+ days of history for most movies |

For everything else, trust the `balanced` preset defaults. You can always refine later if time permits.

---

## 7) Recommended Step-By-Step Runbook

## Step 0: Fix your run mode first

Set:

- Preset: `balanced`
- Seed mode: `fixed`
- Integer mode: match league rules (usually `on`)
- Cost basis: start with `target_bid`
- Correlation mode: start `independent`
- Search mode: start `current_sampled`
- Opponent profile: `balanced_field`

Reason: reproducible baseline before sensitivity testing.

## Step 1: Baseline value map (expected-gross objective)

Set objective: `expected_gross`.

Inspect:

1. Top-ranked movies by `Score`, `AdjExp`, `P(Edge)`.
2. `Opt` selected portfolio.
3. Quality-filter retention count.
4. Per-film cap and leftover budget.

Interpretation:

- High `Score` + strong `P(Edge)` + acceptable `P(DD)` = core candidates.
- Large leftover budget usually means caps/filters are too tight for current pool.
- If selected set includes low `P(Edge)` fillers, tighten filter or lower aggression.

Action:

- Build first watchlist tiers:
  - Tier 1 anchors (core priorities)
  - Tier 2 substitutes
  - Tier 3 only if cheap/late

## Step 2: Market pressure sanity check

Switch optimizer cost basis for stress tests:

- `target_bid` (primary actionable)
- `market_fair_bid` (stress test for competitive pricing)
- `current_price` (diagnostic only; not auction-dollar scaled)

Inspect changes in:

- selected set
- spend/leftover
- win probability (if evaluated)

Interpretation:

- If picks collapse under `market_fair_bid`, your strategy depends on underpaying.
- Keep final operational plan tied to `target_bid`, use `market_fair_bid` as stress boundary.
- Use `TgtMkt`/`TgtMktInt` as the competitive-price context anchor; use `TgtBid`/`TgtInt` as your value-first execution anchor.

## Step 3: Risk calibration pass

Still in expected-gross mode, tune:

- `base cap`
- risk coefficients (`A/B/C`)
- quality filter thresholds

Use diagnostics:

- `P(gross_equiv < spend)`
- concentration downside
- selected portfolio composition

Action patterns:

- Downside too high: reduce bid multiplier, raise `B(drawdown)`, lower base cap.
- Too concentrated: lower cap, increase diversification/correlation penalties.
- Too conservative (high leftover): loosen filters or slightly raise cap.

## Step 4: Win-probability stress pass

Switch objective: `win_probability`.

Use:

- Search mode: `local_search` for routine runs; `genetic` for deeper periodic runs.
- Opponent profile: run all three presets.

Inspect:

- estimated win probability
- search diagnostics (`mode`, `candidates`, `runtime`)
- selected changes vs expected-gross mode

Interpretation:

- If win-probability portfolio is very different from EV portfolio, opponent pressure is material.
- If win probability is only strong in passive opponents, strategy is fragile.
- Compare win probability to the random baseline `1 / (opponents + 1)` (for 7 opponents, baseline is 12.5%).

**Opponent model caveat**: The three presets (`passive_value`, `balanced_field`, `aggressive_bidup`) are stylized archetypes, not calibrated to your specific league. Real opponents will have idiosyncratic patterns. If your league has run previous drafts, track actual bid histories and note which preset each player most resembles. Over multiple seasons, this lets you weight the opponent profiles more accurately rather than treating all three equally.

## Step 4b: Robustness Against Aggressive Bid-Up Opponents

If your win probability is strong in `passive_value` and `balanced_field` but drops materially in `aggressive_bidup`, treat this as a concentration fragility signal.

Use this robustness workflow:

1. Run the same setup across all three opponent profiles and record:
   - win probability
   - selected portfolio overlap
   - downside diagnostics
2. Apply a maximin rule:
   - prioritize setups that improve the *minimum* win probability across profiles, not only the best-case profile.
3. If using `market_fair_bid` with high stress cap (for example `0.70`) causes aggressive-profile collapse:
   - reduce `strategy_market_fair_stresstest_cap` toward `0.55-0.60`
   - keep `target_bid` as primary execution basis and `market_fair_bid` as stress test.
4. Increase anti-fragility penalties slightly:
   - `strategy_diversification_penalty` in `0.08-0.15`
   - `strategy_correlation_penalty` in `0.08-0.15`
5. Use deeper search for the final robustness pass:
   - `local_search` for routine speed
   - `genetic` for final check when time allows
   - increase candidate count and MC samples during this pass.
6. Draft execution rule for contested movies:
   - treat `TgtInt` as practical bid zone
   - use `TgtMktInt` as market-context checkpoint when pressure is high
   - reserve bidding up to `MaxInt` for true must-haves only
   - if current bid moves above `TgtInt` and aggressive-profile win probability drops sharply, pivot to alternates.

## Step 5: Correlation stress pass

Run same setup under:

- `independent`
- `gaussian_copula`
- `t_copula`

Inspect:

- `corr_mode_used`, fallback reason
- downside and concentration metrics
- winner set stability

Interpretation:

- If only independent mode looks good, portfolio likely underestimates clustered downside.
- Prefer lineups robust in copula stress modes.

## Step 6: Seed robustness pass

Run a small seed sweep:

- fixed seed baseline
- 3-5 random seed reruns

Inspect:

- overlap of top candidates
- stability of selected set
- variance in win probability

Interpretation:

- unstable outputs imply parameter fragility; simplify and reduce aggression.

## Step 7: Finalize draft operating bands

For each tier-1 and tier-2 movie, record:

- target bid
- max bid
- expected replacement if missed

This becomes your draft-day execution sheet.

---

## 8) Live Draft Execution Loop (Per Movie)

For each movie in sequence:

1. In `Draft state`, set **Current Movie** and **Current Bid**.
2. Ensure prior winners are assigned (so budgets/pool are current).
3. Run dashboard with integer mode on.
4. Read active movie row:
   - `MinBid` = legal next step
   - `TgtInt` = preferred execution level
   - `TgtMktInt` = competitive-market reference level
   - `MaxInt` = hard stop
   - `MinBid/TgtInt/TgtMktInt/MaxInt` are fully actionable for the active movie context; other rows are comparative until you switch context.
5. Decision:
   - if movie remains high priority and selected, bid within `[MinBid, MaxInt]`
   - if price exceeds `MaxInt`, stop and move to alternates
6. On win:
   - assign movie with final price
   - rerun dashboard immediately (budget and pool changed)
7. On loss:
   - rerun dashboard and follow top alternate path

Never carry previous bid across movies manually. Set draft context per active movie.

---

## 9) Portfolio Construction Heuristics Across Entire Draft

Use the model to maintain balance:

1. **Anchors**: high `P(Edge)`, low/moderate `P(DD)`, strong score.
2. **Upside adds**: acceptable `P(Edge)` with higher variance only when portfolio downside is still controlled.
3. **Avoid fillers**: cheap but low-edge/high-drawdown names that only consume leftover cash.

Suggested workflow:

1. Buy 2-3 anchors early at or below target/max discipline.
2. Re-optimize after each assignment.
3. Keep a "robust core" list: movies that remain selected under passive, balanced, and aggressive opponent profiles.
4. Use alternates table aggressively when contested movies exceed max.
5. Late draft: tighten quality filter less if stranded cash is large, but keep drawdown limits in view.

**Leftover budget decision rule**: If you've exhausted your anchor candidates and still have >15% of budget remaining, allocate to movies meeting *both* of these thresholds: `Score > median(pool)` and `P(Edge) > 0.45`. If no movies meet both criteria, hold the cash rather than buying low-edge fillers -- unspent budget is better than negative-edge exposure.

---

## 10) Reading Key Summary Metrics For Decisions

- `P(gross_equiv < spend)`:
  - high => portfolio likely negative surplus vs spend-equivalent baseline.
- `Concentration downside`:
  - high => too much bad-tail dependence on a few movies.
- `Aggression sensitivity`:
  - compare `x0.9 / x1.0 / x1.1` surplus and positive-surplus probability.
  - if `x1.1` collapses, your bidding room is tight; avoid chasing.
- `Estimated win probability`:
  - scenario estimate under opponent model assumptions.
  - compare across opponent profiles and seeds, not as a single absolute truth.

---

## 11) CLI and GUI: Operational Equivalence

Both interfaces support:

- strategy dashboard
- draft state tracking
- assignment and budget updates
- Monte Carlo diagnostics
- Phase 4 advanced controls

GUI adds easier repeated runs and side-by-side summary review.
CLI is faster for power users and explicit state updates.

---

## 12) Common Failure Modes And Fixes

1. **Optimizer selects too many/too few movies**
   - check cost basis and cap settings first.
   - `current_price` cost basis is diagnostic and can produce non-actionable selections.

2. **Very high `P(Edge)` across almost all rows**
   - this often means bids are conservative relative to the value anchor (`target_bid`/`fair_budget_bid` context).
   - tighten quality thresholds and review value anchor assumptions.
   - run Student-t and copula stress for realism.

3. **Win probability unstable across runs**
   - increase candidate count and MC samples.
   - use fixed seed for tuning; random seeds for robustness checks.

4. **No legal bids in integer mode**
   - previous bid may exceed your executable budget context.
   - verify current bid and personal remaining budget in draft state.

5. **Correlation mode requested but not used**
   - check `corr_fallback_reason` (usually insufficient data/overlap).

---

## 13) Recommended Default Protocol (Practical)

Use this as default operating rhythm:

1. Baseline expected-gross run (`balanced`, fixed seed, independent, target_bid).
2. Risk calibration (cap/filters/penalties) until downside metrics are acceptable.
3. Win-probability pass (`local_search`, balanced opponents).
4. Stress pass (`aggressive_bidup`, `t_copula`).
5. Build final tiers and max-bid sheet.
6. During draft, rerun after every assignment or major price jump.

---

## 13b) Game-Day Protocol (Condensed Live-Draft Checklist)

When the draft is starting and time is tight, use this 4-step loop instead of the full runbook:

1. **Load and verify** (5 min): Load saved settings. Confirm data freshness (prices scraped today, histories current). Verify draft state matches reality (assignments, budgets, current movie).

2. **Baseline snapshot** (5 min): Run one expected-gross dashboard pass. Note your top 5 target movies and their MaxInt walk-away prices. This is your operating sheet.

3. **Per-movie decision** (during draft): Set current movie + current bid -> run dashboard -> read TgtInt/TgtMktInt/MaxInt -> bid or pass. If price exceeds MaxInt, immediately check alternates column.

4. **Post-assignment rerun**: After winning a movie, rerun the dashboard (budget and pool changed). **Reoptimization priority**: always rerun after anchor picks (top 2-3 movies). For low-value late-round picks where the result won't change your remaining strategy, it's acceptable to skip the rerun and rely on your pre-draft tiers.

This protocol sacrifices the stress-testing depth of the full runbook but keeps you responsive during a live draft.

---

## 14) Notes On Settings Persistence

Strategy defaults reset per fresh app run for strategy-prefixed keys. This is intentional to avoid accidental carryover from last session.

Within a single run, manual advanced overrides take precedence over preset defaults.

---

## 15) Optional Draft-Rule Inputs To Further Refine This Playbook

If you want this guide tuned to your exact league, provide:

- exact tie/winner resolution rule
- nomination order constraints
- minimum increment rule (if any)
- whether every player must spend full budget
- any roster limits (min/max number of movies)

The current guide assumes no hard roster-size constraint and standard upward auction behavior.
