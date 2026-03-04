# Movie Draft Auction Engine - Updated Implementation Plan

## Objective

Maximize expected total gross of your drafted portfolio under a fixed personal budget.

This plan is constrained to use only data already in this codebase.

## Scope Constraints

- No new external data collection.
- No dependence on opponent-specific historical behavior for Phase 1 core logic.
- Add only functionality that materially improves movie selection or bid sizing.

## Available Data (Current)

- `data/hsx_cache.csv`: current HSX prices and daily changes.
- `data/<TICKER>/history.csv`: per-movie historical HSX price series.
- `data/movies.csv`: ticker pool, owner, release date.
- `data/people.csv`: starting and remaining budgets.

## Core Modeling Assumption

Without local realized-gross labels, current HSX price is the baseline market-implied expected gross proxy.

---

# Phase 1 - Practical Engine (Implement Now)

## 1) Unified Feature Table

Create one feature row per movie with:

- `current_price`, `price_change`, `pct_change`
- `days_to_release`
- `mom_7`, `mom_14`, `mom_30`
- `trend_30`
- `vol_30`
- `drawdown_30`
- `history_points`

Output must be stable for all pool movies with NaN-safe defaults.

## 2) Expected Gross Estimate (Data-Constrained)

Use:

- `market_expected = current_price`
- `adjustment_signal = weighted_z(momentum, trend, volatility, drawdown, days_to_release)`
- `adjusted_expected = market_expected * clip(1 + lambda * adjustment_signal, low, high)`

Defaults:

- conservative clamp (example `low=0.85`, `high=1.15`)
- tunable weights in `settings.json`

## 3) Explicit Probability-of-Edge Metrics

For each movie and bid candidate `B`, estimate local scenario outcomes from its own history:

- Phase 1 default: bootstrap local return windows
- Optional alternate method: Student-t fit to local returns

Compute and expose:

- `prob_positive_edge = P(value_at_bid > B)`
- `prob_large_drawdown = P(value_at_bid < B * (1 - drawdown_threshold))`

These are confidence metrics based on local HSX history behavior, not true gross probabilities.

## 4) Risk Layer and Risk Penalty

Compute:

- `risk_tier` from `vol_30` and `drawdown_30` bands
- `release_risk` as an increasing penalty when `days_to_release` is short

Define:

- `risk_penalty = clip(a * vol_30 + b * drawdown_30 + c * release_risk, 0, r_max)`

Risk must directly affect bid size.

## 5) Bid Guidance (Gross-Maximizing)

Primary budget basis:

- `fair_budget_bid = gross_share * your_remaining_budget`
- `gross_share = adjusted_expected / sum(adjusted_expected of available movies)`

If league budget is available, keep personal budget as the main basis and add:

- `market_pressure_factor = league_remaining_budget / your_remaining_budget`

Bid formulas:

- `target_bid = fair_budget_bid * (1 - risk_penalty) * preset_bid_multiplier`
- `max_rational_bid = market_fair_bid` (auction-dollar ceiling based on league proportional allocation)
- `value_ratio = adjusted_expected / target_bid`

Recommended outputs:

- `target_bid`, `max_bid`, `value_ratio`, `prob_positive_edge`, `prob_large_drawdown`

## 6) Portfolio Optimizer with Concentration Controls

Add budget-constrained optimizer (0/1 knapsack style):

- value: `adjusted_expected`
- cost: planned bid

Add constraints:

- `max_budget_pct_per_film`
- optional diversification penalty
- optional HSX co-movement proxy penalty using local history correlations

Deliverables:

- optimal watchlist under budget
- marginal value per dollar
- replacement suggestions if targets are lost

## 7) Strategy Presets (Runtime Selectable)

Add three presets:

- `Conservative`
- `Balanced`
- `Aggressive`

Each preset controls:

- clamp bounds (`low`, `high`)
- risk-penalty strength (`a`, `b`, `c`, `r_max`)
- bid multiplier
- concentration cap

## 8) CLI Integration

Add `Draft Strategy Dashboard` menu with:

- ranked available movies table
- key columns:
  - `market_expected`
  - `adjusted_expected`
  - `risk_tier`
  - `target_bid`
  - `max_bid`
  - `value_ratio`
  - `prob_positive_edge`
  - `prob_large_drawdown`
- preset selector at runtime
- optional CSV export

## 9) Validation and Guardrails

Because gross labels are unavailable locally, validate what can be validated:

- feature completeness checks
- sensitivity table for parameter and preset changes
- ranking stability under small perturbations
- simple forward check on HSX histories (signal vs next-window return direction)

## 10) Phase 1 Acceptance Criteria

- Feature table computes for all pool movies without manual cleanup.
- Dashboard runs in seconds and outputs bids plus probability metrics.
- Optimizer enforces concentration constraints and returns alternates.
- Presets are selectable at runtime and produce reproducible outputs.

## 11) Proposed Addition: Integer-Bid Constraint Toggle

Add a runtime toggle so all recommendations can respect auctions where bids must be whole-dollar (or fixed-step) integers.

Suggested settings:

- `integer_bid_mode`: `off` (default) | `on`
- `bid_step`: default `1.0` (can support `0.5`, `5`, etc. if rules differ)
- `integer_rounding_policy`: `nearest` (default) | `floor` | `ceil`

### Mathematical updates when `integer_bid_mode = on`

Let `s = bid_step`.
For any continuous bid estimate `x`:

- `round_to_step_nearest(x, s) = s * round(x / s)`
- `round_to_step_floor(x, s) = s * floor(x / s)`
- `round_to_step_ceil(x, s) = s * ceil(x / s)`

Apply to bid outputs:

- `target_bid_int = round_to_step(target_bid, s)` (policy-based)
- `max_bid_int = round_to_step_floor(max_bid, s)` (recommended hard cap behavior)
- enforce `target_bid_int <= max_bid_int` by clipping down if needed

### Probability metric updates (integer-aware)

Use integer bid values in edge/downside metrics:

- `prob_positive_edge = P(V >= target_bid_int)`
- `prob_large_drawdown = P(V < target_bid_int * (1 - drawdown_threshold))`

Distribution handling:

- Bootstrap model: unchanged sampling; compare simulated values to integer bids directly.
- Student-t model: keep continuous sampling; compare to integer thresholds directly.
  - Optional refinement: continuity correction at half-step boundaries for CDF-based checks.

### Optimizer updates (integer-aware)

- Use integerized cost inputs in optimizer:
  - `optimizer_cost = target_bid_int` or selected cost basis rounded to step `s`
- Keep knapsack/integer-program formulation in discrete units:
  - `cost_units = optimizer_cost / s`
  - `budget_units = budget / s`

This makes selected portfolios executable without manual post-processing.

### UI/Output updates

- Show both continuous and integer bids in dashboard when toggle is on:
  - `target_bid_raw`, `target_bid_int`
  - `max_bid_raw`, `max_bid_int`
- Add an `IntBid` column in main ranking table.
- Add warning if integer rounding materially changes rank/order.

### Validation additions for this toggle

- Report total rounding drift:
  - `sum(target_bid_int - target_bid_raw)`
- Report count of movies where clipping occurred (`target_bid_int > max_bid_int` before clip).
- Confirm optimizer portfolio is still budget-feasible in integer units.

---

# Phase 2 - Optional Upgrades

## A) Student-t Default Risk Engine

Promote Student-t from optional method to default tail model if it improves stability and calibration.

Add toggle for Gaussian or Student-t distribution

## B) Scenario Simulation

Run bootstrap or Student-t scenario simulations for portfolio stress testing:

- portfolio expected gross distribution
- concentration downside risk
- sensitivity to bidding aggressiveness

## C) Optional Win-Probability Optimization Mode

Add objective toggle:

- maximize expected total gross (default)
- maximize probability of finishing first (optional mode)

This mode should remain secondary unless opponent behavior data quality improves.

---

# Implementation Order (Recommended)

1. Feature table + data quality checks.
2. Adjusted expected + probability-of-edge metrics.
3. Risk-penalized bid guidance.
4. Optimizer with concentration controls.
5. Presets and CLI dashboard.
6. Validation diagnostics and parameter tuning.

This sequence keeps Phase 1 tightly aligned with expected-gross maximization using only current local data.

---

# Phase 1 Status (Completed)

Phase 1 steps 1-10 are now implemented in code.

High-level completion summary:

- Built unified per-movie feature table from current HSX snapshot + local history files.
- Added adjusted expected estimate anchored to HSX with bounded signal adjustment.
- Added bootstrap-based edge probabilities.
- Added explicit risk penalty and risk-tier logic tied to volatility, drawdown, and release timing.
- Added bid guidance (`fair_budget_bid`, `target_bid`, `max_bid`, `value_ratio`) with personal/league/custom budget basis support.
- Added concentration-aware portfolio optimizer with optional correlation and diversification penalties.
- Added runtime presets (Conservative/Balanced/Aggressive).
- Added integrated CLI `Draft strategy dashboard` with export support.
- Added integer bid constraint toggle in dashboard:
  - whole-dollar bids only
  - bids must be `> previous_bid`
  - bids are capped at personal remaining money
  - integer-aware probability/optimizer calculations
- Added Phase 1 validation outputs:
  - feature completeness diagnostics
  - preset sensitivity table
  - ranking stability under perturbation
  - forward-direction check on local histories
- Verified Phase 1 acceptance criteria operationally through smoke tests and CLI integration.

## Metric Quick Reference

- `market_expected`: baseline HSX-implied expected value proxy.
  - Use: anchor valuation before model adjustments.
- `adjusted_expected`: risk/signal-adjusted expected value.
  - Use: primary value input for ranking and optimization.
- `prob_positive_edge`: probability model value exceeds candidate bid.
  - Use: confidence that a bid is positive-edge.
- `prob_large_drawdown`: probability model value falls materially below bid.
  - Use: downside-risk filter.
- `risk_penalty`: bounded penalty from `vol_30`, `drawdown_30`, and short time-to-release.
  - Use: directly scales down target bids.
- `fair_budget_bid`: gross-share-based fair bid under selected budget basis.
  - Use: neutral allocation baseline.
- `target_bid`: actionable recommended bid after risk + preset adjustments.
  - Use: live bidding guide.
- `max_bid` / `max_rational_bid`: ceiling before negative standalone edge.
  - Use: hard stop in auction decisions.
- `value_ratio`: `adjusted_expected / target_bid`.
  - Use: efficiency measure (expected value per bid dollar).
- `priority_score`: blended ranking score from value, edge, and risk terms.
  - Use: ordered watchlist and triage.
- `mean_abs_corr`: HSX co-movement proxy.
  - Use: concentration/correlation control in optimization.
- `target_bid_raw` vs `target_bid`:
  - `target_bid_raw`: continuous recommendation before integer constraints.
  - `target_bid`: executable recommendation (integer-constrained when toggle is on).
- `min_legal_bid_int`:
  - Use: minimum allowed next bid under integer rules (`previous_bid + 1`).
- `can_bid_int`:
  - Use: quick indicator that a legal integer bid exists under current constraints.

## Quick Usage Guide

CLI usage:

1. Run `main.py`.
2. Scrape/update data:
   - `1` -> scrape latest HSX prices
   - `1` -> `2` (Scrape Options) to refresh movie price histories
3. Open `11` -> `Draft strategy dashboard`.
4. Choose:
   - preset (`Conservative`, `Balanced`, `Aggressive`)
   - budget basis (personal/league/custom)
   - integer bid mode toggle (`on` for whole-dollar legal bids, `off` for continuous planning)
   - optimizer cost basis (`current_price` recommended for stricter selection)
5. Review ranked output, optimizer picks, alternates, and diagnostics.
6. Optionally export the dashboard CSV for draft-day reference.

Practical bidding workflow (sequential auction, current framework):

1. Process movies in release-date order (closest -> farthest).
2. At the start of each movie:
   - set `previous_bid = 0` (or current high bid if bidding already started),
   - keep integer mode on when league requires whole-number bids.
3. During bidding on that active movie:
   - re-run dashboard with updated `previous_bid` and your current remaining budget,
   - use that movie's `MinBid`/`TgtInt`/`MaxInt` as your live decision band.
4. When a movie closes:
   - assign winner for that movie,
   - decrement winner's remaining budget by final price,
   - move to next movie in sequence.
5. Use `priority_score`, `prob_positive_edge`, `prob_large_drawdown`, and optimizer alternates for planning what to do next if you lose the current movie.

Important current limitation:

- `previous_bid` is global per dashboard run, not per-movie persistent state.
- This means `MinBid`/`TgtInt` are strictly reliable for the currently nominated movie in that run.
- For other movies shown in the same table, treat integer bid columns as planning context only.

Recommended live-draft operating pattern:

1. Keep one active nominated movie at a time.
2. Re-run dashboard after each meaningful bid increase on that movie.
3. Keep budgets synced immediately after each completed movie.
4. Use `expected_gross` as default objective during live bidding; use `win_probability` mainly for scenario checks.

## Budget Basis Clarification (Dashboard)

- `Personal budget` basis:
  - Uses your own remaining budget as the allocation pool.
  - `fair_budget_bid` and `target_bid` are scaled to what you can actually spend.
  - Best for real bid execution and staying budget-feasible.

- `League total budget` basis:
  - Uses all players' remaining budget as the allocation pool.
  - Produces higher fair/target bids because it reflects total market buying power.
  - Best for market-pressure context (how expensive movies may get overall), not your direct spend limit.

- Practical default:
  - Use `personal` for your actionable bidding plan.
  - Compare with `league` to gauge whether market competition may push prices above your personal fair levels.

## Phase 2 Status (Completed)

Phase 2 A/B/C are now implemented in the dashboard workflow.

- A) Risk-model toggle:
  - `bootstrap` (empirical sampling from local history windows)
  - `student_t` (fat-tail parametric sampling; better tail-risk representation)
- B) Monte Carlo portfolio simulation:
  - portfolio gross distribution (`mean`, `p10`, `p50`, `p90`)
  - `P(gross < spend)`
  - concentration downside probability
  - aggression sensitivity table (`spend multiplier` vs expected surplus / P(surplus > 0))
- C) Objective toggle:
  - `expected_gross` (deterministic knapsack objective)
  - `win_probability` (Monte Carlo objective vs simulated opponents)

## Updated Practical Recommendations

- Use `expected_gross` as default objective for stable, high-EV planning.
- Use `win_probability` when you want a tournament-style portfolio and can tolerate model variance.
- Use `bootstrap` when local history is sparse or you want strict empirical behavior.
- Use `student_t` when you want heavier tails and more conservative downside interpretation.
- Treat `win_probability` as scenario-based guidance (not ground truth), because opponent behavior is simulated.

## Phase 2 Practical Usage Examples

Example 1: Baseline draft prep

1. Open `Draft strategy dashboard`.
2. Preset: `Balanced`
3. Risk model: `bootstrap`
4. Objective: `expected_gross`
5. Budget basis: `personal`
6. Optimizer cost basis: `current_price`
7. Use output:
   - primary list from `priority_score`
   - bidding bands from `target_bid` to `max_bid`
   - alternates from optimizer section

Example 2: Stress-test downside before draft

1. Keep same preset/objective.
2. Switch risk model from `bootstrap` -> `student_t`.
3. Compare:
   - `P(gross < spend)`
   - concentration downside probability
   - aggression sensitivity (`x0.9`, `x1.0`, `x1.1`)
4. If downside worsens materially, reduce aggressiveness or move toward lower-correlation alternatives.

Example 3: Tournament mode (higher variance strategy)

1. Objective: `win_probability`
2. Preset: `Aggressive` (or Balanced if you want less variance)
3. Risk model: `student_t`
4. Review:
   - estimated win probability vs simulated opponents
   - selected portfolio differences vs expected-gross mode
5. Use this as an alternate plan only when your goal is rank/first-place probability over pure EV.

## Monte Carlo Usage Guide (Current)

What Monte Carlo is doing now:

- Samples movie-level outcome scenarios from local history (`bootstrap` or `student_t`).
- Aggregates outcomes for the selected portfolio.
- Reports distribution and downside diagnostics.
- In `win_probability` objective, compares your candidate portfolio against simulated opponents.

How to run it in CLI:

1. Open `Draft strategy dashboard`.
2. Select:
   - `Risk model`: `bootstrap` or `student_t`
   - `Objective`: `expected_gross` or `win_probability`
3. Keep budget inputs current (personal remaining budget, integer constraints if used).
4. Read Monte Carlo section printed under optimizer summary.

Key Monte Carlo outputs and interpretation:

- `gross mean/p10/p50/p90`:
  - `mean`: central EV estimate for selected portfolio
  - `p10`: conservative downside case
  - `p50`: median case
  - `p90`: upside case
- `P(gross < spend)`:
  - probability your simulated gross underperforms your spend level
  - lower is generally better for capital protection
- `Concentration downside`:
  - probability downside occurs while one movie dominates portfolio contribution
  - high values imply fragile, concentrated risk
- `Aggression sensitivity`:
  - compares expected surplus and `P(surplus > 0)` at spend multipliers (`x0.9`, `x1.0`, `x1.1`)
  - use to decide whether to bid tighter or push more aggressively

How to analyze for decision making:

1. Baseline with `expected_gross` + `bootstrap`.
2. Re-run with `student_t`; if downside metrics worsen materially, reduce aggression or diversify.
3. Use `win_probability` as a secondary scenario lens (especially for league-rank goals), not sole truth.
4. Prefer portfolios that keep `P(gross < spend)` and concentration downside acceptable while maintaining strong mean/p50.
5. During live bidding, re-run after each completed movie so Monte Carlo reflects updated budgets and assignments.

Practical caution:

- Current Monte Carlo is a portfolio outcome simulator, not a full round-by-round auction engine.
- It is most reliable for comparing candidate allocation profiles under consistent assumptions, not exact auction path prediction.

---

# Phase 3 - Optional GUI Application (PySide6)

## Objective

Add an optional desktop GUI mode (PySide6) that exposes all current CLI functionality with clean navigation, minimal visual design, and strong operational clarity.

GUI launch should be optional argument-based, so CLI remains fully supported.

## Scope and Principles

- Functional parity with CLI first; no feature loss.
- Minimal, clean, fast UI; avoid decorative complexity.
- Clear metric labeling and in-app glossary for decision support.
- Reuse existing calculations/storage modules; avoid duplicating business logic.
- Keep GUI code partitioned from CLI code to prevent coupling.

## Launch and Runtime Mode

Add entry mode options:

- `python main.py` -> existing CLI mode (default)
- `python main.py --gui` -> launch GUI mode

Optional future:

- separate entrypoint script (`python gui_app.py`) if preferred for packaging.

## Proposed Package Partitioning

Create a clean GUI package under `movie_scraping/`:

- `movie_scraping/gui/`
  - `__init__.py`
  - `app.py` (QApplication bootstrap, main window creation)
  - `main_window.py` (navigation shell, menus, status bar)
  - `controllers/`
    - `dashboard_controller.py`
    - `auction_controller.py`
    - `scrape_controller.py`
    - `people_controller.py`
    - `movies_controller.py`
  - `views/`
    - `dashboard_view.py`
    - `auction_view.py`
    - `scrape_view.py`
    - `people_view.py`
    - `movies_view.py`
    - `history_plot_view.py`
    - `summary_view.py`
    - `settings_view.py`
    - `glossary_view.py`
  - `widgets/`
    - reusable tables/forms/metric cards
  - `models/`
    - Qt table models/adapters for pandas DataFrames
  - `resources/`
    - styles/icons (minimal, utility-first)

Keep computation modules (`calculations.py`, `storage.py`, `scraper.py`) as shared backend services.

## CLI-to-GUI Feature Parity Matrix

Implement all existing CLI sections in GUI:

- Scraping
  - scrape latest cache
  - fetch all history
  - fetch release dates
- Movie pool management
  - list/add/remove/bulk add
  - release date editing
- People management
  - list/add/update/remove
- Assignment workflow
  - assign/unassign movies
- Dashboard workflow
  - presets, risk model, objective, budget basis
  - integer bid toggle and previous bid input
  - ranked metrics table, optimizer picks, alternates
  - Monte Carlo summary and diagnostics
- Price history plotting
  - ticker selection + interactive plot view
- Export
  - dashboard CSV export

## GUI Navigation Design (Minimal + Clear)

Use a left navigation rail or top tab bar with these sections:

1. Dashboard
2. Auction (live workflow)
3. Price History
4. Movies
5. People
6. Scraping
7. Settings
8. Glossary

### Dashboard View

- Inputs panel:
  - preset, risk model, objective
  - budget basis
  - integer mode + previous bid
  - optimizer cost basis
- Main results table:
  - sortable/filterable columns
  - selected-row highlight for optimizer picks
- Side/bottom panels:
  - optimizer summary
  - Monte Carlo summary
  - validation diagnostics
  - quick alternates list

### Price History View

- ticker dropdown/search
- embedded matplotlib canvas (Qt backend)
- optional multi-series compare mode (future)

### Monte Carlo Detail View (within Dashboard or separate tab)

- metric cards:
  - mean/p10/p50/p90
  - `P(gross < spend)`
  - concentration downside
  - win probability (when objective is `win_probability`)
- aggression sensitivity table/chart
- scenario assumptions block (samples, model type, opponent simulation params)

### Glossary View

- searchable metric glossary with:
  - metric name
  - plain-language purpose
  - formula (high-level)
  - interpretation guidance
- include all core fields used in dashboard tables and summaries.

## Core Draft State Tracking (CLI + GUI)

Draft occurs over multiple sessions/weeks, so state must persist cleanly between runs.

Required shared behavior (not GUI-specific):

- maintain durable draft state in storage (example: `data/auction_state.json`)
- persist assignment history (winner, final price, timestamp, movie)
- always exclude assigned movies from available draft recommendations
- keep player remaining budgets synchronized with recorded wins

Minimal workflow requirements:

- load queue sorted by release date (closest -> farthest)
- allow selecting/updating current movie context when drafting resumes
- on movie close:
  - assign owner
  - record final price/history
  - decrement winner budget
  - mark movie unavailable for future drafting and dashboard optimizer inputs

This should be implemented first in shared backend logic so both CLI and GUI use the same source of truth.

## Technical Integration Notes

- Use Qt model/view (`QAbstractTableModel`) for large table responsiveness.
- Run scrape/simulation tasks in worker threads (`QThread`/`QRunnable`) with progress and cancel support.
- Centralize all read/write operations through `storage.py` services.
- Keep GUI controllers thin; calculations remain in backend modules.

## Dependencies and Packaging

- Add dependency:
  - `PySide6` (preferred default for licensing simplicity)
- Optional extras:
  - `pyinstaller` packaging profile for one-file desktop executable.

Example dependency update:

- `requirements.txt` add `PySide6>=6.7.0`

## Phase 3 Implementation Milestones

1. Bootstrap GUI shell + mode toggle (`--gui`).
2. Build shared data adapters (pandas -> Qt models).
3. Implement Dashboard view with full existing controls/results.
4. Implement Price History plot view.
5. Implement Movies/People/Scraping management views.
6. Implement resumable draft workflow with state persistence and automatic budget decrement.
7. Add glossary view and metric tooltips.
8. Final QA for CLI parity, correctness, and performance.

## Phase 3 Acceptance Criteria

- GUI can run all core workflows currently available in CLI.
- Dashboard outputs match CLI for identical inputs/settings.
- History plots and Monte Carlo details are accessible and readable in GUI.
- Draft state persists across sessions and excludes already-assigned movies from remaining draft recommendations.
- CLI and GUI both read/write the same draft state and assignment history.
- Metric glossary is present and discoverable from dashboard.
- CLI remains fully functional and unchanged for non-GUI users.

## Phase 3 Progress Update (Current)

- Implemented a dedicated GUI package at `movie_scraping/gui/` with `PySide6` and `--gui` launch mode.
- Implemented shared draft state persistence via `data/auction_state.json` with:
  - current movie context (`current_movie`, `current_bid`)
  - assignment map (winner, final price, timestamp, budget-applied flag)
  - assignment event history
- Added shared storage APIs used by both CLI and GUI:
  - `assign_movie(...)`
  - `unassign_movie(...)`
  - `set_current_auction_context(...)`
  - `get_assigned_movies_df()` / `get_assignment_history_df()`
- Updated CLI assignment flow to use shared state APIs and optionally capture final winning price.
- Added CLI `Draft state / assignment history` menu for resumable multi-session workflow control.
- Added GUI tabs for:
  - Dashboard (strategy metrics + optimizer + Monte Carlo summary)
  - Draft State (context, assign/unassign, assignment tracking tables)
  - History (ticker-selectable price plots)
  - Data (movies, people, scraping actions)
- Added dependency entry `PySide6>=6.7.0` to `requirements.txt`.

---

# Phase 4 - Optional Robust Simulation + Search Upgrades

## Objective

Improve realism of portfolio outcome simulation and improve win-probability search quality, while preserving full backward compatibility with current independent-draw and current optimizer behavior.

## Design Constraint (Important)

All Phase 4 changes must be optional runtime settings.

Default behavior must remain functionally equivalent to the current system unless toggles are explicitly enabled.

## A) Correlated Returns via Copula Simulation (Optional)

### Current limitation

Movie returns are currently drawn independently in Monte Carlo, which can understate portfolio tail risk.

### Planned improvement

Add optional copula-based joint sampling using the existing local HSX co-movement information:

- `gaussian_copula` mode
- optional `t_copula` mode (heavier tails)

Reuse `add_correlation_proxy` inputs and build a stable correlation matrix for available movies.

### Optional settings

- `strategy_corr_simulation_mode`: `independent` (default) | `gaussian_copula` | `t_copula`
- `strategy_corr_shrinkage`: `0.0-1.0` (default `0.20`) for matrix stabilization
- `strategy_corr_min_history_points`: minimum overlap required to include pairwise correlation (default safe threshold)
- `strategy_corr_floor`: lower bound for eigenvalue stabilization / PSD correction

### Integration requirements

- If correlation data is insufficient, automatically fall back to `independent`.
- Keep existing bootstrap / Student-t marginal return models unchanged.
- Copula only changes the *joint dependence structure* of per-movie draws.

### Outputs

- Add simulation metadata:
  - `corr_mode_used`
  - `corr_effective_dim`
  - fallback reason (if reverted to independent)

## C) Stronger Global Search for Win-Probability Objective (Optional)

### Current limitation

Current win-probability optimizer evaluates a limited sampled candidate set.

### Planned improvement

Add optional search enhancements while keeping current method as default:

1. `random_multistart` (improved baseline): increase multi-start candidates + deterministic seed control.
2. `local_search` refinement: swap-based hill climbing from top candidate portfolios.
3. Optional `genetic` search mode for broader exploration on larger movie pools.

### Optional settings

- `strategy_search_mode`: `current_sampled` (default) | `random_multistart` | `local_search` | `genetic`
- `strategy_search_candidates`: base candidate count (extends current setting)
- `strategy_search_local_iters`: local-search swap iterations
- `strategy_search_population`: GA population size
- `strategy_search_generations`: GA generation count
- `strategy_search_elite_frac`: elite retention fraction
- `strategy_search_mutation_rate`: mutation probability

### Integration requirements

- Respect all existing constraints:
  - budget cap
  - per-film cap
  - integer bid legality
  - quality filters
- If a selected search mode fails or times out, fall back to `current_sampled`.

### Outputs

- Add optimizer metadata:
  - `search_mode_used`
  - `candidate_count_evaluated`
  - `search_runtime_ms`
  - fallback reason (if any)

## Opponent Presets (Lightweight Competitive Modeling)

### Goal

Improve opponent realism without requiring full sequential game-theory simulation.

### Preset profiles (optional)

- `passive_value`
  - lower aggression
  - lower noise
  - fewer bid-up events
- `balanced_field` (default)
  - current behavior equivalent
- `aggressive_bidup`
  - higher aggression
  - higher bid-up pressure on top-ranked movies
  - lower tolerance for leftover budget

### Optional settings

- `strategy_opponent_profile`: `balanced_field` (default) | `passive_value` | `aggressive_bidup`
- `strategy_opponent_bidup_strength`: intensity of bid-up behavior on top movies
- `strategy_opponent_cash_conservation`: tendency to preserve budget for later rounds

### Integration requirements

- Keep existing opponent model as default.
- Presets only adjust opponent sampling parameters (noise/aggression/bid pressure).
- No required changes to assignment/state persistence for this phase.

## CLI / GUI Integration (Phase 4)

Add advanced controls as optional toggles:

- Correlation simulation mode + stabilization knobs
- Search mode + search intensity knobs
- Opponent profile selector

Requirements:

- Defaults prefilled to backward-compatible values.
- Existing CLI/GUI flows run unchanged if user does not enable Phase 4 options.
- Summary sections show mode actually used and any fallback.

## Validation and Guardrails

- Add timing diagnostics for new simulation/search modes.
- Compare independent vs copula on:
  - `P(gross_equiv < spend)`
  - concentration downside
  - win-probability stability across seeds
- Add convergence sanity checks for search modes:
  - improvement over baseline sampled mode
  - variance across seed sweep

## Phase 4 Acceptance Criteria

- With all toggles off, outputs are materially consistent with current system.
- Enabling copula mode changes dependence structure without breaking marginal model selection.
- Enabling stronger search modes evaluates more portfolios and returns equal-or-better objective values than baseline in repeatable tests.
- Opponent presets are runtime-selectable and reflected in win-probability diagnostics.
- CLI and GUI both expose and report these options clearly.

## Phase 4 Status (Completed)

Phase 4 is now implemented with optional controls and backward-compatible defaults.

Implemented:

- Optional correlated simulation in Monte Carlo:
  - `independent` (default), `gaussian_copula`, `t_copula`
  - stabilized correlation matrix with shrinkage + PSD correction
  - automatic fallback to independent mode when data is insufficient
  - runtime metadata in summary (`corr_mode_requested`, `corr_mode_used`, dimension, fallback reason)
- Optional stronger search modes for win-probability optimization:
  - `current_sampled` (default), `random_multistart`, `local_search`, `genetic`
  - search diagnostics (`search_mode_used`, candidates evaluated, runtime)
- Optional opponent behavior presets:
  - `passive_value`, `balanced_field` (default), `aggressive_bidup`
  - tunable bid-up pressure and cash-conservation behavior
- CLI integration:
  - advanced prompt controls for all Phase 4 knobs
  - summary output now reports effective correlation/search/opponent settings used
- GUI integration:
  - dashboard controls for all Phase 4 knobs
  - summary output includes Phase 4 runtime metadata and fallback visibility
- Robustness fixes:
  - Monte Carlo empty-portfolio metadata bug fixed
  - copula correlation matrix construction fixed to avoid read-only array errors

## Phase 4 Usage Examples

### 1) Baseline (backward-compatible)

Use defaults:

- `strategy_corr_simulation_mode = independent`
- `strategy_search_mode = current_sampled`
- `strategy_opponent_profile = balanced_field`

Interpretation:

- Results are comparable to prior versions.
- Use this as control before enabling advanced options.

### 2) Correlation Stress Test

Set:

- `corr_mode = gaussian_copula` (or `t_copula` for heavier tails)
- keep same seed and same other settings

Compare vs baseline:

- `P(gross_equiv < spend)`
- concentration downside
- selected portfolio changes

Interpretation:

- If downside risk jumps materially under copula mode, your baseline portfolio likely relies on independence assumptions.

### 3) Search Quality Upgrade

Set:

- objective = `win_probability`
- search mode = `local_search` (faster refinement) or `genetic` (broader search)
- increase `strategy_search_candidates` for harder slates

Interpretation:

- Prefer settings that improve win probability with stable spend and reasonable runtime.
- If candidate count/runtime rises with no win-probability gain, revert to lighter mode.

### 4) Opponent Behavior Sensitivity

Run the same setup under:

- `passive_value`
- `balanced_field`
- `aggressive_bidup`

Interpretation:

- If portfolio win probability collapses only in `aggressive_bidup`, your strategy is vulnerable to price pressure on top targets.
- Use alternates and cap controls to reduce this fragility.

## Practical Interpretation Guide (Phase 4)

- `corr_mode_used`:
  - confirms the simulation actually used your requested dependence mode.
- `corr_fallback_reason`:
  - if populated, simulation reverted to independent mode; treat copula conclusions as unavailable for that run.
- `search_mode_used`, `candidate_count_evaluated`, `search_runtime_ms`:
  - indicate optimization depth vs speed trade-off.
- `opponent_profile_used`:
  - identifies the competitive assumption behind reported win probability.

Recommended workflow:

1. Start with baseline defaults.
2. Enable copula mode and compare downside metrics.
3. Enable stronger search mode only if it improves win probability consistently across a few seeds.
4. Test opponent presets to find picks that remain robust across field behavior assumptions.
