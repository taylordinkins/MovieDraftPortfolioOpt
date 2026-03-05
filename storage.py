import json
import os

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, 'data')

PEOPLE_CSV = os.path.join(DATA_DIR, 'people.csv')
MOVIES_CSV = os.path.join(DATA_DIR, 'movies.csv')
CACHE_CSV = os.path.join(DATA_DIR, 'hsx_cache.csv')
SETTINGS_JSON = os.path.join(DATA_DIR, 'settings.json')
AUCTION_STATE_JSON = os.path.join(DATA_DIR, 'auction_state.json')
STRATEGY_PROFILES_JSON = os.path.join(DATA_DIR, 'strategy_profiles.json')

_DEFAULT_SETTINGS = {
    'bid_ref_ticker': '',
    'bid_ref_price': 0.0,
    # Phase 1 strategy engine defaults (steps 1-3).
    'strategy_lambda': 0.35,
    'strategy_clip_low': 0.85,
    'strategy_clip_high': 1.15,
    'strategy_w_mom_7': 0.20,
    'strategy_w_mom_14': 0.20,
    'strategy_w_mom_30': 0.25,
    'strategy_w_trend_30': 0.20,
    'strategy_w_vol_30': -0.15,
    'strategy_w_drawdown_30': -0.20,
    'strategy_w_days_to_release': 0.00,
    'strategy_bootstrap_samples': 1000,
    'strategy_drawdown_threshold': 0.15,
    'strategy_default_horizon_days': 14,
    'strategy_min_horizon_days': 7,
    'strategy_max_horizon_days': 30,
    'strategy_validation_bootstrap_samples': 120,
    # Phase 1 strategy engine defaults (steps 4-8).
    'strategy_active_preset': 'balanced',
    'strategy_user_name': '',
    'strategy_bid_multiplier': 1.00,
    'strategy_risk_a_vol': 3.00,
    'strategy_risk_b_drawdown': 1.40,
    'strategy_risk_c_release': 0.45,
    'strategy_risk_release_window_days': 30,
    'strategy_risk_max_penalty': 0.45,
    'strategy_max_budget_pct_per_film': 0.22,
    'strategy_market_fair_stresstest_cap': 0.55,
    'strategy_diversification_penalty': 0.00,
    'strategy_correlation_penalty': 0.00,
    'strategy_enable_quality_filters': True,
    'strategy_min_prob_positive_edge': 0.20,
    'strategy_max_prob_large_drawdown': 0.20,
    # Phase 2 defaults.
    'strategy_risk_model': 'bootstrap',  # bootstrap | student_t
    'strategy_objective': 'expected_gross',  # expected_gross | win_probability
    'strategy_mc_samples': 1500,
    'strategy_mc_random_seed': 123,
    'strategy_mc_seed_mode': 'fixed',  # fixed | random
    'strategy_mc_num_opponents': 7,
    'strategy_mc_candidate_portfolios': 120,
    'strategy_mc_opponent_noise': 0.30,
    'strategy_mc_aggression_sd': 0.10,
    'strategy_mc_concentration_threshold': 0.40,
    'strategy_mc_opponent_universes': 10,
    'strategy_mc_aggression_grid': [0.9, 1.0, 1.1],
    # Phase 4 optional correlation simulation controls.
    'strategy_corr_simulation_mode': 'independent',  # independent | gaussian_copula | t_copula
    'strategy_corr_shrinkage': 0.20,
    'strategy_corr_min_history_points': 20,
    'strategy_corr_floor': 1e-4,
    'strategy_corr_t_df': 8,
    # Phase 4 optional search controls.
    'strategy_search_mode': 'current_sampled',  # current_sampled | random_multistart | local_search | genetic
    'strategy_search_candidates': 120,
    'strategy_search_local_iters': 25,
    'strategy_search_population': 80,
    'strategy_search_generations': 30,
    'strategy_search_elite_frac': 0.20,
    'strategy_search_mutation_rate': 0.08,
    # Phase 4 optional opponent profile controls.
    'strategy_opponent_profile': 'balanced_field',  # passive_value | balanced_field | aggressive_bidup
    'strategy_opponent_bidup_strength': 0.00,
    'strategy_opponent_cash_conservation': 0.00,
    # Runtime UI preferences (strategy dashboard).
    'strategy_budget_mode_preference': 'personal',  # personal | league | custom
    'strategy_custom_budget_amount': 200.0,
    'strategy_optimizer_cost_col': 'target_bid',  # target_bid | market_fair_bid | current_price
    'strategy_portfolio_eval_mode': 'optimizer_selected',  # optimizer_selected | fixed_active_paid
    # Integer bid constraint toggle.
    'strategy_integer_bid_mode': False,
    'strategy_integer_prev_bid': 0,
}
_STRATEGY_PREFIX = 'strategy_'
_STRATEGY_PROFILE_EXCLUDE_KEYS = {
    'strategy_runtime_seed',
    'strategy_personal_budget_cap',
    'strategy_league_remaining_budget',
}


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _default_auction_state() -> dict:
    return {
        'version': 1,
        'current_movie': '',
        'current_bid': 0,
        'updated_at': _utc_now_iso(),
        'assignments': {},
        'history': [],
    }


def _normalize_auction_state(state: dict | None) -> dict:
    base = _default_auction_state()
    if not isinstance(state, dict):
        return base

    out = dict(base)
    out.update(state)

    cur_movie = str(out.get('current_movie', '') or '').strip().upper()
    out['current_movie'] = cur_movie

    try:
        out['current_bid'] = max(int(float(out.get('current_bid', 0))), 0)
    except Exception:
        out['current_bid'] = 0

    assignments = out.get('assignments', {})
    if not isinstance(assignments, dict):
        assignments = {}
    norm_assignments = {}
    for raw_ticker, raw_entry in assignments.items():
        ticker = str(raw_ticker or '').strip().upper()
        if not ticker or not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        winner = str(entry.get('winner', '') or '').strip()
        if not winner:
            continue
        price = pd.to_numeric(pd.Series([entry.get('final_price', None)]), errors='coerce').iloc[0]
        norm_assignments[ticker] = {
            'winner': winner,
            'final_price': float(price) if pd.notna(price) and price > 0 else None,
            'assigned_at': str(entry.get('assigned_at', '') or _utc_now_iso()),
            'budget_applied': bool(entry.get('budget_applied', False)),
            'source': str(entry.get('source', 'unknown') or 'unknown'),
        }
    out['assignments'] = norm_assignments

    history = out.get('history', [])
    if not isinstance(history, list):
        history = []
    norm_history = []
    for h in history:
        if not isinstance(h, dict):
            continue
        evt = dict(h)
        evt['event_type'] = str(evt.get('event_type', '') or '').strip().lower()
        evt['ticker'] = str(evt.get('ticker', '') or '').strip().upper()
        if not evt['event_type'] or not evt['ticker']:
            continue
        evt['timestamp'] = str(evt.get('timestamp', '') or _utc_now_iso())
        norm_history.append(evt)
    out['history'] = norm_history
    out['updated_at'] = str(out.get('updated_at', '') or _utc_now_iso())
    return out


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_people() -> pd.DataFrame:
    if not os.path.exists(PEOPLE_CSV):
        return pd.DataFrame(columns=['name', 'starting_money', 'remaining_money'])
    return pd.read_csv(PEOPLE_CSV)


def save_people(df: pd.DataFrame):
    _ensure_dir()
    df.to_csv(PEOPLE_CSV, index=False)


def load_movies() -> pd.DataFrame:
    if not os.path.exists(MOVIES_CSV):
        return pd.DataFrame(columns=['ticker', 'owner', 'release_date'])
    df = pd.read_csv(MOVIES_CSV, dtype={'ticker': str, 'owner': str})
    df['owner'] = df['owner'].fillna('')
    if 'release_date' not in df.columns:
        df['release_date'] = ''
    else:
        df['release_date'] = df['release_date'].fillna('')
    return df


def save_movies(df: pd.DataFrame):
    _ensure_dir()
    df.to_csv(MOVIES_CSV, index=False)


def load_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_CSV):
        return pd.DataFrame(columns=['ticker', 'name', 'current_price', 'price_change', 'pct_change', 'scraped_at'])
    return pd.read_csv(CACHE_CSV)


def save_cache(df: pd.DataFrame):
    _ensure_dir()
    df.to_csv(CACHE_CSV, index=False)


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_JSON):
        return dict(_DEFAULT_SETTINGS)
    with open(SETTINGS_JSON) as f:
        return {**_DEFAULT_SETTINGS, **json.load(f)}


def load_strategy_runtime_defaults() -> dict:
    """
    Return settings for dashboard/runtime strategy use.
    Strategy keys always reset to _DEFAULT_SETTINGS each fresh app run.
    Non-strategy keys can still come from persisted settings.json.
    """
    persisted = load_settings()
    runtime = dict(_DEFAULT_SETTINGS)
    for key, value in persisted.items():
        if not str(key).startswith(_STRATEGY_PREFIX):
            runtime[key] = value
    return runtime


def save_settings(settings: dict):
    _ensure_dir()
    with open(SETTINGS_JSON, 'w') as f:
        json.dump(settings, f, indent=2)


def _extract_strategy_profile_payload(settings: dict | None) -> dict:
    src = settings or {}
    out: dict[str, object] = {}
    for key, value in src.items():
        k = str(key)
        if not k.startswith(_STRATEGY_PREFIX):
            continue
        if k in _STRATEGY_PROFILE_EXCLUDE_KEYS:
            continue
        out[k] = value
    return out


def load_strategy_profiles() -> dict:
    if not os.path.exists(STRATEGY_PROFILES_JSON):
        return {}
    try:
        with open(STRATEGY_PROFILES_JSON) as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict] = {}
    for name, payload in raw.items():
        profile_name = str(name or '').strip()
        if not profile_name or not isinstance(payload, dict):
            continue
        out[profile_name] = _extract_strategy_profile_payload(payload)
    return out


def save_strategy_profiles(profiles: dict):
    _ensure_dir()
    normalized: dict[str, dict] = {}
    if isinstance(profiles, dict):
        for name, payload in profiles.items():
            profile_name = str(name or '').strip()
            if not profile_name:
                continue
            normalized[profile_name] = _extract_strategy_profile_payload(payload if isinstance(payload, dict) else {})
    with open(STRATEGY_PROFILES_JSON, 'w') as f:
        json.dump(normalized, f, indent=2)


def list_strategy_profile_names() -> list[str]:
    names = list(load_strategy_profiles().keys())
    names.sort(key=lambda s: s.lower())
    return names


def load_strategy_profile(name: str) -> dict:
    key = str(name or '').strip()
    if not key:
        return {}
    profiles = load_strategy_profiles()
    payload = profiles.get(key, {})
    return _extract_strategy_profile_payload(payload)


def save_strategy_profile(name: str, settings: dict | None) -> tuple[bool, str]:
    profile_name = str(name or '').strip()
    if not profile_name:
        return False, 'Profile name is required.'
    payload = _extract_strategy_profile_payload(settings)
    if not payload:
        return False, 'No strategy settings found to save.'
    profiles = load_strategy_profiles()
    profiles[profile_name] = payload
    save_strategy_profiles(profiles)
    return True, f"Saved strategy profile '{profile_name}'."


def load_auction_state() -> dict:
    if not os.path.exists(AUCTION_STATE_JSON):
        return _default_auction_state()
    try:
        with open(AUCTION_STATE_JSON) as f:
            raw = json.load(f)
    except Exception:
        return _default_auction_state()
    return _normalize_auction_state(raw)


def save_auction_state(state: dict):
    _ensure_dir()
    normalized = _normalize_auction_state(state)
    normalized['updated_at'] = _utc_now_iso()
    with open(AUCTION_STATE_JSON, 'w') as f:
        json.dump(normalized, f, indent=2)


def history_path(ticker: str) -> str:
    """Returns data/<TICKER>/history.csv"""
    return os.path.join(DATA_DIR, ticker, 'history.csv')


def save_price_history(ticker: str, df: pd.DataFrame):
    path = history_path(ticker)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_price_history(ticker: str) -> pd.DataFrame | None:
    path = history_path(ticker)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _seed_auction_state_from_movies(state: dict, movies_df: pd.DataFrame) -> dict:
    """Populate assignment map from existing movie owners when state is empty."""
    out = _normalize_auction_state(state)
    if out['assignments']:
        return out
    if movies_df.empty or 'owner' not in movies_df.columns:
        return out

    changed = False
    now = _utc_now_iso()
    for _, row in movies_df.iterrows():
        ticker = str(row.get('ticker', '') or '').strip().upper()
        owner = str(row.get('owner', '') or '').strip()
        if not ticker or not owner:
            continue
        out['assignments'][ticker] = {
            'winner': owner,
            'final_price': None,
            'assigned_at': now,
            'budget_applied': False,
            'source': 'seed_from_movies',
        }
        changed = True
    if changed:
        out['history'].append({
            'event_type': 'seed',
            'ticker': '*',
            'winner': '',
            'final_price': None,
            'timestamp': now,
            'source': 'seed_from_movies',
            'notes': 'Initialized assignment map from movies.csv owners.',
        })
    return out


def _prune_auction_state_missing_movies(state: dict, movies_df: pd.DataFrame) -> dict:
    """Remove assignment entries for tickers no longer present in movies.csv."""
    out = _normalize_auction_state(state)
    if movies_df.empty:
        valid_tickers = set()
    else:
        valid_tickers = set(movies_df.get('ticker', pd.Series(dtype=object)).astype(str).str.upper().tolist())

    removed = [t for t in list(out.get('assignments', {}).keys()) if t not in valid_tickers]
    if not removed:
        cur = str(out.get('current_movie', '') or '').strip().upper()
        if cur and cur not in valid_tickers:
            out['current_movie'] = ''
            out['current_bid'] = 0
        return out

    for ticker in removed:
        out['assignments'].pop(ticker, None)
        out['history'].append({
            'event_type': 'prune_missing_movie',
            'ticker': ticker,
            'winner': '',
            'final_price': None,
            'timestamp': _utc_now_iso(),
            'source': 'state_reconcile',
            'notes': 'Removed assignment because ticker was removed from movies.csv.',
        })
    cur = str(out.get('current_movie', '') or '').strip().upper()
    if cur and cur not in valid_tickers:
        out['current_movie'] = ''
        out['current_bid'] = 0
    return out


def ensure_auction_state_seeded() -> dict:
    """Ensure auction state exists and captures existing assigned movie owners."""
    movies_df = load_movies()
    state = load_auction_state()
    seeded = _seed_auction_state_from_movies(state, movies_df)
    reconciled = _prune_auction_state_missing_movies(seeded, movies_df)
    if reconciled != _normalize_auction_state(state):
        save_auction_state(reconciled)
        return reconciled
    if seeded != _normalize_auction_state(state):
        save_auction_state(seeded)
        return seeded
    return reconciled


def set_current_auction_context(ticker: str = '', current_bid: int = 0) -> dict:
    state = ensure_auction_state_seeded()
    state['current_movie'] = str(ticker or '').strip().upper()
    try:
        state['current_bid'] = max(int(float(current_bid)), 0)
    except Exception:
        state['current_bid'] = 0
    save_auction_state(state)
    return state


def get_assigned_movies_df() -> pd.DataFrame:
    state = ensure_auction_state_seeded()
    rows = []
    for ticker, payload in state.get('assignments', {}).items():
        rows.append({
            'ticker': str(ticker).upper(),
            'winner': payload.get('winner', ''),
            'final_price': payload.get('final_price', np.nan),
            'assigned_at': payload.get('assigned_at', ''),
            'budget_applied': bool(payload.get('budget_applied', False)),
            'source': payload.get('source', 'unknown'),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['ticker', 'winner', 'final_price', 'assigned_at', 'budget_applied', 'source'])
    return df.sort_values(['assigned_at', 'ticker'], ascending=[False, True], na_position='last')


def get_assignment_history_df() -> pd.DataFrame:
    state = ensure_auction_state_seeded()
    history = state.get('history', [])
    if not history:
        return pd.DataFrame(columns=['timestamp', 'event_type', 'ticker', 'winner', 'final_price', 'source', 'notes'])
    df = pd.DataFrame(history)
    for col in ('timestamp', 'event_type', 'ticker', 'winner', 'source', 'notes'):
        if col not in df.columns:
            df[col] = ''
    if 'final_price' not in df.columns:
        df['final_price'] = np.nan
    return df.sort_values('timestamp', ascending=False, na_position='last')


def _apply_budget_change(people_df: pd.DataFrame, person: str, delta: float) -> tuple[pd.DataFrame, bool]:
    """Apply delta to remaining_money for person. Positive delta increases budget."""
    if people_df.empty or person not in people_df.get('name', pd.Series(dtype=object)).values:
        return people_df, False
    out = people_df.copy()
    cur = pd.to_numeric(out.loc[out['name'] == person, 'remaining_money'], errors='coerce')
    if cur.empty or pd.isna(cur.iloc[0]):
        return people_df, False
    out.loc[out['name'] == person, 'remaining_money'] = float(cur.iloc[0]) + float(delta)
    return out, True


def assign_movie(
    ticker: str,
    winner: str,
    final_price: float | None = None,
    source: str = 'cli',
    allow_reassign: bool = False,
) -> tuple[bool, str, dict]:
    """
    Assign a movie to a winner and persist assignment state/history.
    If final_price is provided (>0), winner remaining budget is decremented.
    """
    t = str(ticker or '').strip().upper()
    w = str(winner or '').strip()
    if not t:
        return False, 'Ticker is required.', {}
    if not w:
        return False, 'Winner is required.', {}

    movies_df = load_movies()
    people_df = load_people()
    if movies_df.empty or t not in movies_df['ticker'].astype(str).str.upper().values:
        return False, f"Ticker '{t}' is not in the movie pool.", {}
    if people_df.empty or w not in people_df['name'].astype(str).values:
        return False, f"Winner '{w}' is not in people.csv.", {}

    owners = movies_df.loc[movies_df['ticker'].astype(str).str.upper() == t, 'owner']
    current_owner = str(owners.iloc[0] or '').strip() if not owners.empty else ''
    if current_owner and not allow_reassign:
        return False, f'{t} is already assigned to {current_owner}. Unassign first.', {}

    state = ensure_auction_state_seeded()
    existing = state.get('assignments', {}).get(t)
    if existing and not allow_reassign:
        ex_winner = str(existing.get('winner', '') or '').strip()
        if ex_winner:
            return False, f'{t} is already assigned to {ex_winner}. Unassign first.', {}

    price_num = pd.to_numeric(pd.Series([final_price]), errors='coerce').iloc[0]
    price = float(price_num) if pd.notna(price_num) and price_num > 0 else None

    budget_applied = False
    if price is not None:
        rem = pd.to_numeric(people_df.loc[people_df['name'] == w, 'remaining_money'], errors='coerce')
        if rem.empty or pd.isna(rem.iloc[0]):
            return False, f'Unable to resolve remaining budget for {w}.', {}
        if float(rem.iloc[0]) + 1e-9 < price:
            return False, f"{w} has insufficient remaining budget for final price ${price:.2f}.", {}
        people_df, ok = _apply_budget_change(people_df, w, -price)
        if not ok:
            return False, f'Failed to decrement remaining budget for {w}.', {}
        budget_applied = True

    movies_df = movies_df.copy()
    movies_df.loc[movies_df['ticker'].astype(str).str.upper() == t, 'owner'] = w
    save_movies(movies_df)
    if budget_applied:
        save_people(people_df)

    now = _utc_now_iso()
    entry = {
        'winner': w,
        'final_price': price,
        'assigned_at': now,
        'budget_applied': bool(budget_applied),
        'source': str(source or 'unknown'),
    }
    state['assignments'][t] = entry
    if state.get('current_movie', '') == t:
        state['current_movie'] = ''
        state['current_bid'] = 0
    state['history'].append({
        'event_type': 'assign',
        'ticker': t,
        'winner': w,
        'final_price': price,
        'timestamp': now,
        'source': str(source or 'unknown'),
        'budget_applied': bool(budget_applied),
    })
    save_auction_state(state)

    msg = f'Assigned {t} -> {w}.'
    if price is not None:
        msg += f' Final price ${price:.2f}.'
    return True, msg, entry


def unassign_movie(
    ticker: str,
    source: str = 'cli',
    restore_budget: bool = True,
) -> tuple[bool, str, dict]:
    """
    Unassign a movie and optionally restore winner budget using recorded final_price.
    """
    t = str(ticker or '').strip().upper()
    if not t:
        return False, 'Ticker is required.', {}

    movies_df = load_movies()
    if movies_df.empty or t not in movies_df['ticker'].astype(str).str.upper().values:
        return False, f"Ticker '{t}' is not in the movie pool.", {}

    state = ensure_auction_state_seeded()
    existing = state.get('assignments', {}).get(t, {})
    prev_winner = str(existing.get('winner', '') or '').strip()
    prev_price_num = pd.to_numeric(pd.Series([existing.get('final_price', None)]), errors='coerce').iloc[0]
    prev_price = float(prev_price_num) if pd.notna(prev_price_num) and prev_price_num > 0 else None
    budget_applied = bool(existing.get('budget_applied', False))

    people_df = load_people()
    budget_restored = False
    if restore_budget and prev_winner and prev_price is not None and budget_applied:
        people_df, ok = _apply_budget_change(people_df, prev_winner, prev_price)
        if ok:
            save_people(people_df)
            budget_restored = True

    movies_df = movies_df.copy()
    movies_df.loc[movies_df['ticker'].astype(str).str.upper() == t, 'owner'] = ''
    save_movies(movies_df)

    if t in state.get('assignments', {}):
        del state['assignments'][t]
    now = _utc_now_iso()
    state['history'].append({
        'event_type': 'unassign',
        'ticker': t,
        'winner': prev_winner,
        'final_price': prev_price,
        'timestamp': now,
        'source': str(source or 'unknown'),
        'budget_restored': bool(budget_restored),
    })
    save_auction_state(state)

    msg = f'Unassigned {t}.'
    if budget_restored and prev_winner and prev_price is not None:
        msg += f' Restored ${prev_price:.2f} to {prev_winner}.'
    return True, msg, existing


def clear_all_assignments(
    source: str = 'cli',
    restore_budget: bool = True,
) -> tuple[bool, str, dict]:
    """
    Unassign all currently assigned movies.
    Optionally restores winner budgets for entries with applied final_price.
    Returns (success, message, metadata).
    """
    assigned = get_assigned_movies_df()
    if assigned.empty:
        return True, 'No assignments to clear.', {
            'total': 0,
            'cleared': 0,
            'failed': 0,
            'failures': [],
        }

    tickers = assigned.get('ticker', pd.Series(dtype=object)).astype(str).str.upper().tolist()
    cleared = 0
    failures: list[dict] = []
    for ticker in tickers:
        ok, msg, _ = unassign_movie(
            ticker=ticker,
            source=source,
            restore_budget=restore_budget,
        )
        if ok:
            cleared += 1
        else:
            failures.append({'ticker': ticker, 'message': msg})

    total = len(tickers)
    failed = len(failures)
    success = failed == 0
    status = f'Cleared {cleared}/{total} assignments.'
    if failed:
        status += f' Failed: {failed}.'
    return success, status, {
        'total': total,
        'cleared': cleared,
        'failed': failed,
        'failures': failures,
    }
