from collections.abc import Callable
import time

import numpy as np
import pandas as pd


def compute_pool_stats(
    hsx_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    people_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge HSX data with the movie pool and compute pricing columns.

    Columns added:
      full_prop_val     - movie's % share of the full pool's total HSX price
      break_even_full   - full_prop_val/100 * total_remaining_cash
      avail_prop_val    - movie's % share of *unassigned* movies only (NaN if assigned)
      break_even_avail  - avail_prop_val/100 * total_remaining_cash (NaN if assigned)
      cumsum            - cumulative full_prop_val when sorted ascending (useful for ranking)
    """
    pool = movies_df.copy()
    hsx_subset = hsx_df[['ticker', 'name', 'current_price', 'price_change', 'pct_change']].copy()
    pool = pool.merge(hsx_subset, on='ticker', how='left')

    valid = pool['current_price'].notna()
    total_remaining = float(people_df['remaining_money'].sum()) if not people_df.empty else 0.0

    # Full-pool proportional valuation
    total_price = pool.loc[valid, 'current_price'].sum()
    if total_price > 0:
        pool.loc[valid, 'full_prop_val'] = 100.0 * pool.loc[valid, 'current_price'] / total_price
        pool.loc[valid, 'break_even_full'] = pool.loc[valid, 'full_prop_val'] / 100.0 * total_remaining
    else:
        pool['full_prop_val'] = np.nan
        pool['break_even_full'] = np.nan

    # Cumulative sum (ascending sort) for ranking context
    sorted_idx = pool.loc[valid, 'full_prop_val'].sort_values().index
    pool.loc[sorted_idx, 'cumsum'] = (
        pool.loc[sorted_idx, 'full_prop_val'].sort_values().cumsum().values
    )

    # Available-only proportional valuation
    avail_mask = valid & (pool['owner'] == '')
    avail_total = pool.loc[avail_mask, 'current_price'].sum()
    if avail_total > 0:
        pool.loc[avail_mask, 'avail_prop_val'] = (
            100.0 * pool.loc[avail_mask, 'current_price'] / avail_total
        )
        pool.loc[avail_mask, 'break_even_avail'] = (
            pool.loc[avail_mask, 'avail_prop_val'] / 100.0 * total_remaining
        )
    else:
        pool['avail_prop_val'] = np.nan
        pool['break_even_avail'] = np.nan

    return pool


def add_bid_estimate(pool: pd.DataFrame, ref_ticker: str, ref_bid_price: float) -> pd.DataFrame:
    """
    Add a bid_estimate column anchored to a reference ticker.

    bid_estimate[movie] = ref_bid_price * (hsx_price[movie] / hsx_price[ref])

    This lets you set a known "fair bid" for one movie and scale all others
    proportionally by their relative HSX price.
    """
    valid = pool['current_price'].notna()
    ref_rows = pool.loc[pool['ticker'] == ref_ticker, 'current_price']
    if ref_rows.empty:
        raise ValueError(f"Reference ticker '{ref_ticker}' not found in pool.")
    ref_price = ref_rows.values[0]
    if pd.isna(ref_price) or ref_price == 0:
        raise ValueError(f"Reference ticker '{ref_ticker}' has no valid price.")
    pool = pool.copy()
    pool.loc[valid, 'bid_estimate'] = ref_bid_price * pool.loc[valid, 'current_price'] / ref_price
    return pool


def compute_person_summary(pool: pd.DataFrame, people_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each person return aggregate stats over their assigned movies:
      movie_count    - number of assigned movies
      total_prop_val - sum of full_prop_val for their movies (% of full pool)
      budget_equiv   - total_prop_val/100 * starting_money (estimated $ value)
    """
    rows = []
    for _, person in people_df.iterrows():
        name = person['name']
        owned = pool[pool['owner'] == name]
        total_pv = owned['full_prop_val'].sum() if 'full_prop_val' in pool.columns else 0.0
        budget_equiv = total_pv / 100.0 * person['starting_money']
        rows.append({
            'name': name,
            'starting_money': person['starting_money'],
            'remaining_money': person['remaining_money'],
            'movie_count': len(owned),
            'total_prop_val': round(total_pv, 2),
            'budget_equiv': round(budget_equiv, 2),
        })
    return pd.DataFrame(rows)


def _clean_history_df(history_df: pd.DataFrame | None) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=['date', 'price'])
    if 'date' not in history_df.columns or 'price' not in history_df.columns:
        return pd.DataFrame(columns=['date', 'price'])

    cleaned = history_df[['date', 'price']].copy()
    cleaned['date'] = pd.to_datetime(cleaned['date'], errors='coerce')
    cleaned['price'] = pd.to_numeric(cleaned['price'], errors='coerce')
    cleaned = cleaned.dropna(subset=['date', 'price'])
    cleaned = cleaned[cleaned['price'] > 0].sort_values('date').drop_duplicates('date', keep='last')
    return cleaned


def _window_return(prices: pd.Series, lookback_days: int) -> float:
    if len(prices) <= lookback_days:
        return np.nan
    start = float(prices.iloc[-(lookback_days + 1)])
    end = float(prices.iloc[-1])
    if start <= 0:
        return np.nan
    return end / start - 1.0


def _log_slope(prices: pd.Series) -> float:
    if len(prices) < 2:
        return np.nan
    log_prices = np.log(prices.to_numpy())
    x = np.arange(len(log_prices), dtype=float)
    slope, _ = np.polyfit(x, log_prices, 1)
    return float(slope)


def _max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return np.nan
    rolling_max = prices.cummax()
    drawdowns = prices / rolling_max - 1.0
    return float(abs(drawdowns.min()))


def _zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=series.index)
    std = float(valid.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    mean = float(valid.mean())
    z = (series - mean) / std
    return z.fillna(0.0)


def _resolve_as_of_date(hsx_df: pd.DataFrame, as_of_date: str | pd.Timestamp | None) -> pd.Timestamp:
    if as_of_date is not None:
        ts = pd.to_datetime(as_of_date, errors='coerce')
        if pd.notna(ts):
            return pd.Timestamp(ts).normalize()

    if 'scraped_at' in hsx_df.columns:
        scraped = pd.to_datetime(hsx_df['scraped_at'], errors='coerce').dropna()
        if not scraped.empty:
            return pd.Timestamp(scraped.max()).normalize()

    return pd.Timestamp.today().normalize()


def build_feature_table(
    hsx_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
) -> pd.DataFrame:
    """
    Phase 1 / Step 1:
    Build a unified feature table for pool movies using cache + local history files.
    """
    if history_loader is None:
        import storage
        history_loader = storage.load_price_history

    cols = [
        'ticker', 'owner', 'release_date', 'name',
        'current_price', 'price_change', 'pct_change', 'scraped_at',
        'days_to_release', 'mom_7', 'mom_14', 'mom_30',
        'trend_30', 'vol_30', 'drawdown_30', 'history_points',
    ]
    if movies_df.empty:
        return pd.DataFrame(columns=cols)

    base = movies_df.copy()
    base['ticker'] = base['ticker'].astype(str).str.upper()
    if 'owner' not in base.columns:
        base['owner'] = ''
    if 'release_date' not in base.columns:
        base['release_date'] = ''

    hsx_subset = hsx_df.copy()
    if not hsx_subset.empty:
        keep = [c for c in ['ticker', 'name', 'current_price', 'price_change', 'pct_change', 'scraped_at'] if c in hsx_subset.columns]
        hsx_subset = hsx_subset[keep].copy()
        hsx_subset['ticker'] = hsx_subset['ticker'].astype(str).str.upper()
        base = base.merge(hsx_subset, on='ticker', how='left')
    else:
        base['name'] = np.nan
        base['current_price'] = np.nan
        base['price_change'] = np.nan
        base['pct_change'] = np.nan
        base['scraped_at'] = np.nan

    as_of = _resolve_as_of_date(hsx_df, as_of_date)
    rel = pd.to_datetime(base['release_date'], errors='coerce')
    base['days_to_release'] = (rel - as_of).dt.days

    mom_7 = []
    mom_14 = []
    mom_30 = []
    trend_30 = []
    vol_30 = []
    drawdown_30 = []
    history_points = []

    for ticker in base['ticker']:
        history_df = _clean_history_df(history_loader(str(ticker)))
        history_points.append(int(len(history_df)))

        if history_df.empty:
            mom_7.append(np.nan)
            mom_14.append(np.nan)
            mom_30.append(np.nan)
            trend_30.append(np.nan)
            vol_30.append(np.nan)
            drawdown_30.append(np.nan)
            continue

        prices = history_df['price']
        last_31 = prices.tail(31)
        rets_30 = last_31.pct_change().dropna()

        mom_7.append(_window_return(prices, 7))
        mom_14.append(_window_return(prices, 14))
        mom_30.append(_window_return(prices, 30))
        trend_30.append(_log_slope(prices.tail(30)))
        vol_30.append(float(rets_30.std(ddof=0)) if not rets_30.empty else np.nan)
        drawdown_30.append(_max_drawdown(prices.tail(30)))

    base['mom_7'] = mom_7
    base['mom_14'] = mom_14
    base['mom_30'] = mom_30
    base['trend_30'] = trend_30
    base['vol_30'] = vol_30
    base['drawdown_30'] = drawdown_30
    base['history_points'] = history_points

    return base[cols]


def add_adjusted_expected(
    features_df: pd.DataFrame,
    settings: dict | None = None,
) -> pd.DataFrame:
    """
    Phase 1 / Step 2:
    Add market_expected, adjustment_signal, and adjusted_expected.
    """
    if settings is None:
        import storage
        settings = storage.load_settings()

    out = features_df.copy()
    if out.empty:
        return out

    lam = float(settings.get('strategy_lambda', 0.35))
    clip_low = float(settings.get('strategy_clip_low', 0.85))
    clip_high = float(settings.get('strategy_clip_high', 1.15))
    if clip_low > clip_high:
        clip_low, clip_high = clip_high, clip_low

    weights = {
        'mom_7': float(settings.get('strategy_w_mom_7', 0.20)),
        'mom_14': float(settings.get('strategy_w_mom_14', 0.20)),
        'mom_30': float(settings.get('strategy_w_mom_30', 0.25)),
        'trend_30': float(settings.get('strategy_w_trend_30', 0.20)),
        'vol_30': float(settings.get('strategy_w_vol_30', -0.15)),
        'drawdown_30': float(settings.get('strategy_w_drawdown_30', -0.20)),
        'days_to_release': float(settings.get('strategy_w_days_to_release', 0.00)),
    }

    for feature in weights:
        if feature not in out.columns:
            out[feature] = np.nan

    days = pd.to_numeric(out['days_to_release'], errors='coerce').clip(lower=0)
    z = {
        'mom_7': _zscore(pd.to_numeric(out['mom_7'], errors='coerce')),
        'mom_14': _zscore(pd.to_numeric(out['mom_14'], errors='coerce')),
        'mom_30': _zscore(pd.to_numeric(out['mom_30'], errors='coerce')),
        'trend_30': _zscore(pd.to_numeric(out['trend_30'], errors='coerce')),
        'vol_30': _zscore(pd.to_numeric(out['vol_30'], errors='coerce')),
        'drawdown_30': _zscore(pd.to_numeric(out['drawdown_30'], errors='coerce')),
        'days_to_release': _zscore(days),
    }

    signal = sum(weights[k] * z[k] for k in weights)
    multiplier = (1.0 + lam * signal).clip(lower=clip_low, upper=clip_high)
    market_expected = pd.to_numeric(out.get('current_price', np.nan), errors='coerce')

    out['market_expected'] = market_expected
    out['adjustment_signal'] = signal
    out['adjustment_multiplier'] = multiplier
    out['adjusted_expected'] = market_expected * multiplier

    # Expose standardized components for transparency/tuning.
    for k, series in z.items():
        out[f'z_{k}'] = series

    return out


def _compound_window_returns(daily_returns: np.ndarray, horizon_days: int) -> np.ndarray:
    if daily_returns.size == 0:
        return np.array([0.0], dtype=float)

    clipped = np.clip(daily_returns.astype(float), -0.999, None)
    if clipped.size >= horizon_days:
        log_r = np.log1p(clipped)
        csum = np.cumsum(log_r)
        starts = np.concatenate(([0.0], csum[:-horizon_days]))
        window_log = csum[horizon_days - 1:] - starts
        windows = np.expm1(window_log)
        windows = windows[np.isfinite(windows)]
        return windows if windows.size else np.array([0.0], dtype=float)

    # Fallback when history is short: scale one-day moves to horizon.
    scaled = np.clip(clipped * np.sqrt(max(horizon_days, 1)), -0.95, None)
    scaled = scaled[np.isfinite(scaled)]
    return scaled if scaled.size else np.array([0.0], dtype=float)


def _fit_student_t_params(samples: np.ndarray) -> tuple[float, float, float]:
    clean = samples[np.isfinite(samples)]
    if clean.size == 0:
        return 0.0, 0.0, 80.0
    mu = float(np.mean(clean))
    std = float(np.std(clean, ddof=0))
    if std <= 1e-12:
        return mu, 0.0, 80.0

    centered = clean - mu
    m4 = float(np.mean(centered ** 4))
    kurt_excess = m4 / (std ** 4) - 3.0 if std > 0 else 0.0
    if kurt_excess > 1e-6:
        df = 6.0 / kurt_excess + 4.0
    else:
        df = 80.0
    df = float(np.clip(df, 5.0, 80.0))

    # t variance is df/(df-2) * scale^2, so back out scale from target std.
    scale = std * np.sqrt((df - 2.0) / df)
    return mu, float(max(scale, 0.0)), df


def _draw_returns_from_model(
    base_returns: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    risk_model: str,
) -> np.ndarray:
    model = (risk_model or 'bootstrap').strip().lower()
    base = base_returns[np.isfinite(base_returns)]
    if base.size == 0:
        return np.zeros(samples, dtype=float)

    if model == 'student_t':
        mu, scale, df = _fit_student_t_params(base)
        if scale <= 1e-12:
            draws = np.full(samples, mu, dtype=float)
        else:
            draws = mu + scale * rng.standard_t(df, size=samples)
        return np.clip(draws, -0.95, 5.0)

    return rng.choice(base, size=samples, replace=True)


def _resolve_runtime_seed(settings: dict | None, random_seed: int | None = None) -> tuple[int, str]:
    """Resolve runtime seed from explicit value or settings seed mode."""
    if random_seed is not None:
        return int(random_seed), 'fixed'

    s = settings or {}
    mode = str(s.get('strategy_mc_seed_mode', 'fixed') or 'fixed').strip().lower()
    runtime = pd.to_numeric(pd.Series([s.get('strategy_runtime_seed', np.nan)]), errors='coerce').iloc[0]
    if pd.notna(runtime):
        return int(runtime), ('random' if mode == 'random' else 'fixed')

    if mode == 'random':
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        return seed, 'random'

    fixed = pd.to_numeric(pd.Series([s.get('strategy_mc_random_seed', 123)]), errors='coerce').iloc[0]
    if pd.isna(fixed):
        fixed = 123
    return int(fixed), 'fixed'


def _nearest_psd_correlation(corr: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Project a symmetric matrix onto a numerically stable PSD correlation matrix."""
    c = np.array(corr, dtype=float, copy=True)
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    vals = np.clip(vals, floor, None)
    c_psd = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(c_psd), 1e-12, None))
    c_corr = c_psd / np.outer(d, d)
    c_corr = 0.5 * (c_corr + c_corr.T)
    c_corr = np.clip(c_corr, -0.999, 0.999)
    np.fill_diagonal(c_corr, 1.0)
    return c_corr


def _copula_correlation_matrix(
    tickers: list[str],
    history_loader: Callable[[str], pd.DataFrame | None],
    settings: dict,
) -> tuple[np.ndarray | None, dict]:
    """
    Build a stabilized correlation matrix for copula simulation from local history.
    Returns (corr_matrix_or_none, meta).
    """
    min_points = max(int(settings.get('strategy_corr_min_history_points', 20)), 5)
    shrink = float(np.clip(settings.get('strategy_corr_shrinkage', 0.20), 0.0, 1.0))
    eig_floor = float(np.clip(settings.get('strategy_corr_floor', 1e-4), 1e-9, 0.1))

    series_map: dict[str, pd.Series] = {}
    for t in tickers:
        h = _clean_history_df(history_loader(t))
        if h.empty:
            continue
        s = h.set_index('date')['price'].pct_change().dropna()
        s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) >= min_points:
            series_map[t] = s

    if len(series_map) < 2:
        return None, {
            'corr_mode_requested': str(settings.get('strategy_corr_simulation_mode', 'independent')),
            'corr_mode_used': 'independent',
            'corr_effective_dim': int(len(series_map)),
            'corr_fallback_reason': 'insufficient_history',
        }

    joined = pd.concat(series_map, axis=1)
    if isinstance(joined.columns, pd.MultiIndex):
        joined.columns = [str(c[0]) for c in joined.columns]
    joined = joined.sort_index()
    corr_df = joined.corr(min_periods=min_points)
    corr_df = corr_df.reindex(index=tickers, columns=tickers)
    corr_df = corr_df.fillna(0.0)
    c = corr_df.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(c, 1.0)
    # Shrink toward identity for stability on sparse overlaps.
    c = (1.0 - shrink) * c + shrink * np.eye(c.shape[0], dtype=float)
    c = _nearest_psd_correlation(c, floor=eig_floor)

    return c, {
        'corr_mode_requested': str(settings.get('strategy_corr_simulation_mode', 'independent')),
        'corr_mode_used': str(settings.get('strategy_corr_simulation_mode', 'independent')),
        'corr_effective_dim': int(c.shape[0]),
        'corr_fallback_reason': '',
    }


def _rank_reorder_to_copula_marginals(marginals: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """
    Reorder each marginal draw column by latent copula ranks (Iman-Conover style).
    Preserves per-column marginal distributions while introducing dependence.
    """
    n_sims, n_dim = marginals.shape
    out = np.empty_like(marginals)
    for j in range(n_dim):
        order = np.argsort(latent[:, j], kind='mergesort')
        sorted_vals = np.sort(marginals[:, j])
        col = np.empty(n_sims, dtype=float)
        col[order] = sorted_vals
        out[:, j] = col
    return out


def add_probability_of_edge(
    metrics_df: pd.DataFrame,
    bid_col: str = 'target_bid',
    value_col: str | None = None,
    settings: dict | None = None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    random_seed: int | None = None,
    risk_model: str | None = None,
) -> pd.DataFrame:
    """
    Phase 1 / Step 3:
    Add probability-of-edge metrics using local-history bootstrap scenarios.
    """
    if history_loader is None:
        import storage
        history_loader = storage.load_price_history
    if settings is None:
        import storage
        settings = storage.load_settings()

    out = metrics_df.copy()
    if out.empty:
        return out

    samples = max(int(settings.get('strategy_bootstrap_samples', 1000)), 1)
    seed, seed_mode = _resolve_runtime_seed(settings, random_seed=random_seed)
    model = (risk_model or settings.get('strategy_risk_model', 'bootstrap') or 'bootstrap').strip().lower()
    threshold = float(settings.get('strategy_drawdown_threshold', 0.15))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    default_horizon = max(int(settings.get('strategy_default_horizon_days', 14)), 1)
    min_horizon = max(int(settings.get('strategy_min_horizon_days', 7)), 1)
    max_horizon = max(int(settings.get('strategy_max_horizon_days', 30)), min_horizon)

    if 'adjusted_expected' not in out.columns:
        out['adjusted_expected'] = pd.to_numeric(out.get('current_price', np.nan), errors='coerce')
    if bid_col not in out.columns:
        out[bid_col] = pd.to_numeric(out.get('adjusted_expected', np.nan), errors='coerce')
    if value_col is None:
        value_col = 'fair_budget_bid' if 'fair_budget_bid' in out.columns else 'adjusted_expected'
    if value_col not in out.columns:
        value_col = 'adjusted_expected'

    pos_probs = []
    dd_probs = []
    horizons = []
    scenario_counts = []

    for row_num, (_, row) in enumerate(out.iterrows()):
        adj = pd.to_numeric(pd.Series([row.get('adjusted_expected')]), errors='coerce').iloc[0]
        anchor_value = pd.to_numeric(pd.Series([row.get(value_col)]), errors='coerce').iloc[0]
        bid = pd.to_numeric(pd.Series([row.get(bid_col)]), errors='coerce').iloc[0]
        can_bid_int = row.get('can_bid_int', True)
        integer_mode_row = bool(row.get('integer_bid_mode', False))

        if pd.isna(adj):
            pos_probs.append(np.nan)
            dd_probs.append(np.nan)
            horizons.append(np.nan)
            scenario_counts.append(0)
            continue

        # If integer mode says there is no legal bid for this row, suppress edge probabilities.
        if integer_mode_row and (can_bid_int is False or (isinstance(can_bid_int, (int, float)) and can_bid_int == 0)):
            pos_probs.append(np.nan)
            dd_probs.append(np.nan)
            horizons.append(np.nan)
            scenario_counts.append(0)
            continue

        if pd.isna(anchor_value) or anchor_value <= 0:
            anchor_value = float(adj)
        if pd.isna(bid) or bid <= 0:
            if integer_mode_row:
                pos_probs.append(np.nan)
                dd_probs.append(np.nan)
                horizons.append(np.nan)
                scenario_counts.append(0)
                continue
            bid = float(anchor_value)

        raw_days = pd.to_numeric(pd.Series([row.get('days_to_release')]), errors='coerce').iloc[0]
        if pd.isna(raw_days):
            horizon = default_horizon
        else:
            horizon = int(round(float(raw_days)))
            horizon = min(max(horizon, min_horizon), max_horizon)
        if horizon <= 0:
            horizon = min_horizon

        history_df = _clean_history_df(history_loader(str(row.get('ticker', '')).upper()))
        daily_returns = history_df['price'].pct_change().dropna().to_numpy(dtype=float)
        base_returns = _compound_window_returns(daily_returns, horizon)

        rng = np.random.default_rng(seed + row_num)
        draw_returns = _draw_returns_from_model(
            base_returns=base_returns,
            samples=samples,
            rng=rng,
            risk_model=model,
        )
        projected = np.maximum(0.0, float(anchor_value) * (1.0 + draw_returns))

        pos_probs.append(float(np.mean(projected > bid)))
        dd_floor = bid * (1.0 - threshold)
        dd_probs.append(float(np.mean(projected < dd_floor)))
        horizons.append(horizon)
        scenario_counts.append(int(samples))

    out['prob_positive_edge'] = pos_probs
    out['prob_large_drawdown'] = dd_probs
    out['scenario_horizon_days'] = horizons
    out['scenario_samples'] = scenario_counts
    out['risk_model'] = model
    out['prob_bid_anchor_col'] = bid_col
    out['prob_value_anchor_col'] = value_col
    out['prob_seed_mode'] = seed_mode
    out['prob_seed_used'] = int(seed)
    return out


def compute_phase1_step1_to_3(
    hsx_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    settings: dict | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    bid_col: str = 'current_price',
    risk_model: str | None = None,
    value_col: str | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for Phase 1 steps 1-3.
    """
    features = build_feature_table(
        hsx_df=hsx_df,
        movies_df=movies_df,
        as_of_date=as_of_date,
        history_loader=history_loader,
    )
    with_expected = add_adjusted_expected(features, settings=settings)
    return add_probability_of_edge(
        with_expected,
        bid_col=bid_col,
        value_col=value_col,
        settings=settings,
        history_loader=history_loader,
        risk_model=risk_model,
    )


def get_strategy_presets() -> dict[str, dict]:
    """Return runtime-selectable strategy preset parameter overrides."""
    return {
        'conservative': {
            'strategy_lambda': 0.28,
            'strategy_clip_low': 0.90,
            'strategy_clip_high': 1.08,
            'strategy_bid_multiplier': 0.92,
            'strategy_risk_a_vol': 3.80,
            'strategy_risk_b_drawdown': 1.80,
            'strategy_risk_c_release': 0.60,
            'strategy_risk_max_penalty': 0.55,
            'strategy_max_budget_pct_per_film': 0.18,
            'strategy_diversification_penalty': 0.05,
            'strategy_correlation_penalty': 0.05,
        },
        'balanced': {
            'strategy_lambda': 0.35,
            'strategy_clip_low': 0.85,
            'strategy_clip_high': 1.15,
            'strategy_bid_multiplier': 1.00,
            'strategy_risk_a_vol': 3.00,
            'strategy_risk_b_drawdown': 1.40,
            'strategy_risk_c_release': 0.45,
            'strategy_risk_max_penalty': 0.45,
            'strategy_max_budget_pct_per_film': 0.22,
            'strategy_diversification_penalty': 0.00,
            'strategy_correlation_penalty': 0.00,
        },
        'aggressive': {
            'strategy_lambda': 0.45,
            'strategy_clip_low': 0.80,
            'strategy_clip_high': 1.20,
            'strategy_bid_multiplier': 1.08,
            'strategy_risk_a_vol': 2.00,
            'strategy_risk_b_drawdown': 0.90,
            'strategy_risk_c_release': 0.25,
            'strategy_risk_max_penalty': 0.35,
            'strategy_max_budget_pct_per_film': 0.30,
            'strategy_diversification_penalty': 0.00,
            'strategy_correlation_penalty': 0.00,
        },
    }


def apply_strategy_preset(settings: dict | None, preset: str | None) -> tuple[dict, str]:
    """
    Merge base settings with a preset profile.
    Advanced/runtime override keys (if provided via `strategy_override_keys`)
    are applied after the preset so manual knob edits take precedence.
    Returns (resolved_settings, resolved_preset_name).
    """
    base = dict(settings or {})
    merged = dict(base)
    presets = get_strategy_presets()
    requested = (preset or merged.get('strategy_active_preset', 'balanced') or 'balanced').strip().lower()
    resolved = requested if requested in presets else 'balanced'
    merged.update(presets[resolved])

    override_keys = base.get('strategy_override_keys', [])
    if isinstance(override_keys, (list, tuple, set)):
        for key in override_keys:
            k = str(key)
            if k in base:
                merged[k] = base[k]
        merged['strategy_override_keys'] = [str(k) for k in override_keys]

    merged['strategy_active_preset'] = resolved
    return merged, resolved


def resolve_budget_basis(
    people_df: pd.DataFrame,
    settings: dict | None,
    budget_mode: str = 'personal',
    custom_budget: float | None = None,
) -> dict:
    """Resolve budget basis used for bid sizing."""
    settings = settings or {}
    league_remaining = float(pd.to_numeric(people_df.get('remaining_money', pd.Series(dtype=float)), errors='coerce').sum()) if not people_df.empty else 0.0

    personal_name = str(settings.get('strategy_user_name', '')).strip()
    personal_remaining = np.nan
    if custom_budget is not None and custom_budget > 0:
        personal_remaining = float(custom_budget)
    elif personal_name and not people_df.empty and 'name' in people_df.columns and 'remaining_money' in people_df.columns:
        row = people_df.loc[people_df['name'] == personal_name, 'remaining_money']
        if not row.empty:
            personal_remaining = float(pd.to_numeric(row.iloc[0], errors='coerce'))

    mode = (budget_mode or 'personal').strip().lower()
    if mode == 'league':
        basis = league_remaining
    else:
        basis = personal_remaining if pd.notna(personal_remaining) and personal_remaining > 0 else league_remaining
        mode = 'personal'

    pressure = np.nan
    if pd.notna(personal_remaining) and personal_remaining > 0 and league_remaining > 0:
        pressure = league_remaining / personal_remaining

    return {
        'budget_mode': mode,
        'personal_name': personal_name,
        'personal_remaining': float(personal_remaining) if pd.notna(personal_remaining) else np.nan,
        'league_remaining': league_remaining,
        'basis_budget': float(max(basis, 0.0)),
        'market_pressure_factor': pressure,
    }


def _risk_tier_from_penalty(risk_penalty: pd.Series) -> pd.Series:
    tier = pd.Series('unknown', index=risk_penalty.index, dtype='object')
    tier.loc[risk_penalty <= 0.15] = 'low'
    tier.loc[(risk_penalty > 0.15) & (risk_penalty <= 0.30)] = 'medium'
    tier.loc[risk_penalty > 0.30] = 'high'
    tier.loc[risk_penalty.isna()] = 'unknown'
    return tier


def add_bid_guidance(
    metrics_df: pd.DataFrame,
    people_df: pd.DataFrame,
    settings: dict | None = None,
    budget_mode: str = 'personal',
    custom_budget: float | None = None,
    integer_bid_mode: bool = False,
    previous_bid: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Phase 1 / Steps 4-5:
    Add risk-penalized target bid and related gross-maximization columns.
    """
    settings = settings or {}
    out = metrics_df.copy()
    if out.empty:
        return out, resolve_budget_basis(people_df, settings, budget_mode=budget_mode, custom_budget=custom_budget)

    budgets = resolve_budget_basis(
        people_df=people_df,
        settings=settings,
        budget_mode=budget_mode,
        custom_budget=custom_budget,
    )

    a = float(settings.get('strategy_risk_a_vol', 3.0))
    b = float(settings.get('strategy_risk_b_drawdown', 1.4))
    c = float(settings.get('strategy_risk_c_release', 0.45))
    release_window = max(float(settings.get('strategy_risk_release_window_days', 30)), 1.0)
    r_max = float(np.clip(settings.get('strategy_risk_max_penalty', 0.45), 0.0, 0.95))
    bid_mult = float(settings.get('strategy_bid_multiplier', 1.0))

    vol = pd.to_numeric(out.get('vol_30', np.nan), errors='coerce').fillna(0.0).clip(lower=0.0)
    dd = pd.to_numeric(out.get('drawdown_30', np.nan), errors='coerce').fillna(0.0).clip(lower=0.0)
    days = pd.to_numeric(out.get('days_to_release', np.nan), errors='coerce')

    release_risk = ((release_window - days) / release_window).clip(lower=0.0, upper=1.0)
    release_risk = release_risk.fillna(0.5)
    release_risk = release_risk.where(days >= 0, 1.0)

    risk_penalty = (a * vol + b * dd + c * release_risk).clip(lower=0.0, upper=r_max)

    owner = out.get('owner', pd.Series('', index=out.index)).fillna('')
    avail_mask = owner == ''

    adj = pd.to_numeric(out.get('adjusted_expected', np.nan), errors='coerce')
    avail_total = float(adj.where(avail_mask).clip(lower=0.0).sum())
    if avail_total > 0:
        gross_share = adj.where(avail_mask).clip(lower=0.0) / avail_total
    else:
        gross_share = pd.Series(np.nan, index=out.index)

    basis_budget = float(budgets.get('basis_budget', 0.0))
    league_budget = float(budgets.get('league_remaining', 0.0))
    exec_budget = budgets.get('personal_remaining', np.nan)
    has_personal_cap = pd.notna(exec_budget) and exec_budget > 0

    fair_budget_bid = gross_share * basis_budget
    if league_budget > 0:
        market_fair_bid = gross_share * league_budget
    else:
        market_fair_bid = pd.Series(np.nan, index=out.index)

    target_bid_raw = (fair_budget_bid * (1.0 - risk_penalty) * bid_mult).clip(lower=0.0)
    market_target_base = market_fair_bid.where(pd.notna(market_fair_bid), fair_budget_bid)
    target_market_bid_raw = (market_target_base * (1.0 - risk_penalty) * bid_mult).clip(lower=0.0)
    # Max walk-away bid should be auction-scale, not HSX-gross-scale.
    max_rational_bid = market_fair_bid.where(pd.notna(market_fair_bid), fair_budget_bid).clip(lower=0.0)
    if has_personal_cap:
        max_bid_raw = pd.Series(
            np.minimum(pd.to_numeric(max_rational_bid, errors='coerce').fillna(0.0).to_numpy(dtype=float), float(exec_budget)),
            index=out.index,
        )
    else:
        max_bid_raw = max_rational_bid

    if integer_bid_mode:
        prev = max(int(previous_bid), 0)
        min_legal = prev + 1
        exec_cap = int(np.floor(exec_budget)) if has_personal_cap else 0

        target_floor = np.floor(pd.to_numeric(target_bid_raw, errors='coerce').fillna(0.0).to_numpy(dtype=float)).astype(int)
        target_market_floor = np.floor(
            pd.to_numeric(target_market_bid_raw, errors='coerce').fillna(0.0).to_numpy(dtype=float)
        ).astype(int)
        max_floor = np.floor(pd.to_numeric(max_bid_raw, errors='coerce').fillna(0.0).to_numpy(dtype=float)).astype(int)
        upper = np.minimum(max_floor, exec_cap)
        can_bid = upper >= min_legal

        target_int = np.maximum(target_floor, min_legal)
        target_int = np.where(can_bid, np.minimum(target_int, upper), 0)
        target_market_int = np.maximum(target_market_floor, min_legal)
        target_market_int = np.where(can_bid, np.minimum(target_market_int, upper), 0)
        max_int = upper

        target_bid = pd.Series(target_int.astype(float), index=out.index)
        target_market_bid = pd.Series(target_market_int.astype(float), index=out.index)
        max_bid = pd.Series(max_int.astype(float), index=out.index)
        min_legal_series = pd.Series(float(min_legal), index=out.index)
        can_bid_series = pd.Series(can_bid.astype(bool), index=out.index)
    else:
        target_bid = target_bid_raw
        target_market_bid = target_market_bid_raw
        max_bid = max_bid_raw
        min_legal_series = pd.Series(np.nan, index=out.index)
        can_bid_series = pd.Series(True, index=out.index)

    bid_edge_ratio = np.where(target_bid > 0, adj / target_bid, np.nan)
    market_expected = pd.to_numeric(out.get('market_expected', out.get('current_price', np.nan)), errors='coerce')
    market_value_ratio = np.where(market_expected > 0, adj / market_expected, np.nan)

    out['release_risk'] = release_risk
    out['risk_penalty'] = risk_penalty
    out['risk_tier'] = _risk_tier_from_penalty(risk_penalty)
    out['gross_share'] = gross_share
    out['fair_budget_bid'] = fair_budget_bid
    out['market_fair_bid'] = market_fair_bid
    out['target_bid_raw'] = target_bid_raw
    out['target_market_bid_raw'] = target_market_bid_raw
    out['max_bid_raw'] = max_bid_raw
    out['target_bid'] = target_bid
    out['target_market_bid'] = target_market_bid
    out['max_rational_bid'] = max_rational_bid
    out['max_bid'] = max_bid
    out['target_bid_int'] = target_bid if integer_bid_mode else np.floor(target_bid_raw)
    out['target_market_bid_int'] = target_market_bid if integer_bid_mode else np.floor(target_market_bid_raw)
    out['max_bid_int'] = max_bid if integer_bid_mode else np.floor(max_bid_raw)
    out['min_legal_bid_int'] = min_legal_series
    out['can_bid_int'] = can_bid_series
    out['integer_cap_source'] = 'personal_remaining' if has_personal_cap else 'unavailable'
    out['integer_rounding_drift'] = out['target_bid'] - out['target_bid_raw']
    out['integer_bid_mode'] = bool(integer_bid_mode)
    out['bid_edge_ratio'] = bid_edge_ratio
    out['market_value_ratio'] = market_value_ratio
    out['value_ratio'] = bid_edge_ratio
    out['market_pressure_factor'] = budgets.get('market_pressure_factor', np.nan)

    # Break-even bid based on raw HSX price share of the available pool.
    # This is the price-proportional anchor (vs market_fair_bid which uses adjusted_expected shares).
    raw_price = pd.to_numeric(out.get('current_price', np.nan), errors='coerce')
    avail_price_total = float(raw_price.where(avail_mask).clip(lower=0.0).sum())
    if avail_price_total > 0 and league_budget > 0:
        price_share = raw_price.where(avail_mask).clip(lower=0.0) / avail_price_total
        out['break_even_avail'] = (price_share * league_budget).where(avail_mask, np.nan)
    else:
        out['break_even_avail'] = np.nan

    return out, budgets


def add_correlation_proxy(
    metrics_df: pd.DataFrame,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
) -> pd.DataFrame:
    """
    Add mean absolute correlation of daily returns as a co-movement proxy.
    """
    if history_loader is None:
        import storage
        history_loader = storage.load_price_history

    out = metrics_df.copy()
    if out.empty:
        out['mean_abs_corr'] = []
        return out

    tickers = out['ticker'].astype(str).str.upper().tolist()
    rets_by_ticker = {}
    for ticker in tickers:
        history_df = _clean_history_df(history_loader(ticker))
        if history_df.empty:
            continue
        s = history_df.set_index('date')['price'].pct_change().dropna()
        if len(s) >= 5:
            rets_by_ticker[ticker] = s

    if not rets_by_ticker:
        out['mean_abs_corr'] = 0.0
        return out

    returns_wide = pd.concat(rets_by_ticker, axis=1, join='inner')
    if returns_wide.shape[0] < 5 or returns_wide.shape[1] < 2:
        out['mean_abs_corr'] = 0.0
        return out

    corr = returns_wide.corr().abs()
    mean_corr = {}
    for t in corr.columns:
        others = corr.loc[t].drop(labels=[t], errors='ignore')
        mean_corr[t] = float(others.mean()) if not others.empty else 0.0

    out['mean_abs_corr'] = out['ticker'].astype(str).str.upper().map(mean_corr).fillna(0.0)
    return out


def _resolve_per_film_cap_pct(settings: dict, cost_col: str) -> float:
    base = float(np.clip(settings.get('strategy_max_budget_pct_per_film', 0.22), 0.01, 1.0))
    if str(cost_col or '').strip().lower() == 'market_fair_bid':
        stress_cap = float(np.clip(settings.get('strategy_market_fair_stresstest_cap', 0.55), 0.01, 1.0))
        return max(base, stress_cap)
    return base


def resolve_probability_anchor_columns(cost_col: str) -> tuple[str, str]:
    """
    Resolve probability metric anchor columns for a selected optimizer cost basis.
    """
    mode = str(cost_col or '').strip().lower()
    if mode == 'market_fair_bid':
        return 'target_market_bid', 'market_fair_bid'
    return 'target_bid', 'fair_budget_bid'


def resolve_portfolio_eval_mode(mode: str | None) -> str:
    """Resolve runtime portfolio evaluation mode."""
    m = str(mode or '').strip().lower()
    if m in ('fixed_active_paid', 'fixed_paid', 'fixed'):
        return 'fixed_active_paid'
    return 'optimizer_selected'


def build_fixed_paid_portfolio(
    strategy_df: pd.DataFrame,
    assigned_df: pd.DataFrame,
    active_user: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Build a fixed portfolio for a user from assigned movies with recorded paid prices.
    Returns (portfolio_df, meta).
    """
    user = str(active_user or '').strip()
    out_meta = {
        'mode': 'fixed_active_paid',
        'active_user': user,
        'rows_total_assigned': 0,
        'rows_user_assigned': 0,
        'rows_with_paid_price': 0,
        'rows_with_metrics': 0,
        'rows_missing_metrics': 0,
    }
    if not user or assigned_df.empty:
        return pd.DataFrame(), out_meta

    assigns = assigned_df.copy()
    if 'ticker' not in assigns.columns:
        return pd.DataFrame(), out_meta
    assigns['ticker'] = assigns['ticker'].astype(str).str.upper()
    assigns['winner'] = assigns.get('winner', pd.Series('', index=assigns.index)).astype(str).str.strip()
    assigns['final_price'] = pd.to_numeric(assigns.get('final_price', np.nan), errors='coerce')
    out_meta['rows_total_assigned'] = int(len(assigns))

    mine = assigns[assigns['winner'] == user].copy()
    out_meta['rows_user_assigned'] = int(len(mine))
    if mine.empty:
        return pd.DataFrame(), out_meta

    mine = mine[mine['final_price'].notna() & (mine['final_price'] > 0)].copy()
    out_meta['rows_with_paid_price'] = int(len(mine))
    if mine.empty:
        return pd.DataFrame(), out_meta

    metrics = strategy_df.copy()
    metrics['ticker'] = metrics['ticker'].astype(str).str.upper()
    keep_cols = [c for c in (
        'ticker', 'adjusted_expected', 'target_bid', 'target_bid_int', 'target_market_bid',
        'target_market_bid_int', 'max_bid', 'max_bid_int', 'prob_positive_edge',
        'prob_large_drawdown', 'priority_score'
    ) if c in metrics.columns]
    merged = mine.merge(metrics[keep_cols], on='ticker', how='left')
    merged = merged.rename(columns={'final_price': 'paid_price'})
    merged['optimizer_cost'] = pd.to_numeric(merged['paid_price'], errors='coerce')
    adj = pd.to_numeric(merged.get('adjusted_expected', np.nan), errors='coerce')
    cst = pd.to_numeric(merged.get('optimizer_cost', np.nan), errors='coerce')
    merged['eff_per_dollar'] = np.where(cst > 0, adj / cst, np.nan)
    out_meta['rows_with_metrics'] = int(pd.to_numeric(merged.get('adjusted_expected', np.nan), errors='coerce').notna().sum())
    out_meta['rows_missing_metrics'] = int(len(merged) - out_meta['rows_with_metrics'])
    return merged, out_meta


def _apply_quality_filters(df: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, dict]:
    info = {
        'enabled': bool(settings.get('strategy_enable_quality_filters', True)),
        'min_prob_positive_edge': float(np.clip(settings.get('strategy_min_prob_positive_edge', 0.20), 0.0, 1.0)),
        'max_prob_large_drawdown': float(np.clip(settings.get('strategy_max_prob_large_drawdown', 0.20), 0.0, 1.0)),
        'rows_before': int(len(df)),
        'rows_after': int(len(df)),
    }
    if df.empty or not info['enabled']:
        return df, info

    mask = pd.Series(True, index=df.index)
    if 'prob_positive_edge' in df.columns:
        p_edge = pd.to_numeric(df['prob_positive_edge'], errors='coerce')
        mask &= p_edge.fillna(0.0) >= info['min_prob_positive_edge']
    if 'prob_large_drawdown' in df.columns:
        p_dd = pd.to_numeric(df['prob_large_drawdown'], errors='coerce')
        mask &= p_dd.fillna(1.0) <= info['max_prob_large_drawdown']

    out = df[mask].copy()
    info['rows_after'] = int(len(out))
    return out, info


def optimize_portfolio(
    strategy_df: pd.DataFrame,
    budget: float,
    settings: dict | None = None,
    cost_col: str = 'target_bid',
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Phase 1 / Step 6:
    Budget-constrained portfolio selection with concentration controls.
    """
    settings = settings or {}
    if strategy_df.empty or budget <= 0:
        return {
            'selected': pd.DataFrame(),
            'alternates': pd.DataFrame(),
            'total_spend': 0.0,
            'total_adjusted_expected': 0.0,
            'total_effective_value': 0.0,
            'budget': float(max(budget, 0.0)),
            'leftover': float(max(budget, 0.0)),
            'num_selected': 0,
        }

    max_pct = _resolve_per_film_cap_pct(settings, cost_col)
    div_pen = float(np.clip(settings.get('strategy_diversification_penalty', 0.0), 0.0, 1.0))
    corr_pen = float(np.clip(settings.get('strategy_correlation_penalty', 0.0), 0.0, 1.0))
    integer_mode = bool(settings.get('strategy_integer_bid_mode', False))
    prev_bid = max(int(settings.get('strategy_integer_prev_bid', 0)), 0)
    min_legal = prev_bid + 1
    personal_cap = pd.to_numeric(pd.Series([settings.get('strategy_personal_budget_cap', np.nan)]), errors='coerce').iloc[0]
    if integer_mode and (pd.isna(personal_cap) or personal_cap <= 0):
        return {
            'selected': pd.DataFrame(),
            'alternates': pd.DataFrame(),
            'total_spend': 0.0,
            'total_adjusted_expected': 0.0,
            'total_effective_value': 0.0,
            'budget': float(budget),
            'leftover': float(budget),
            'num_selected': 0,
            'max_budget_pct_per_film': max_pct,
            'cost_col': cost_col,
        }
    if pd.isna(personal_cap) or personal_cap <= 0:
        personal_cap = budget
    integer_cap = int(np.floor(min(personal_cap, budget)))
    if integer_mode and integer_cap < min_legal:
        return {
            'selected': pd.DataFrame(),
            'alternates': pd.DataFrame(),
            'total_spend': 0.0,
            'total_adjusted_expected': 0.0,
            'total_effective_value': 0.0,
            'budget': float(budget),
            'leftover': float(budget),
            'num_selected': 0,
            'max_budget_pct_per_film': max_pct,
            'cost_col': cost_col,
        }

    df = strategy_df.copy()
    if 'owner' in df.columns:
        df = df[df['owner'].fillna('') == ''].copy()

    if cost_col not in df.columns:
        cost_col = 'target_bid'
    df['optimizer_cost'] = pd.to_numeric(df.get(cost_col, np.nan), errors='coerce')
    if integer_mode:
        # Enforce executable integer bidding costs under the active previous-bid and personal-cap context.
        if cost_col in ('target_bid', 'target_bid_int'):
            if 'target_bid_int' in df.columns:
                df['optimizer_cost'] = pd.to_numeric(df['target_bid_int'], errors='coerce')
            else:
                df['optimizer_cost'] = np.floor(df['optimizer_cost'])
        else:
            df['optimizer_cost'] = np.ceil(df['optimizer_cost'])
        df['optimizer_cost'] = df['optimizer_cost'].clip(lower=min_legal, upper=integer_cap)

    df['target_bid'] = pd.to_numeric(df.get('target_bid', np.nan), errors='coerce')
    df['adjusted_expected'] = pd.to_numeric(df.get('adjusted_expected', np.nan), errors='coerce')
    df['mean_abs_corr'] = pd.to_numeric(df.get('mean_abs_corr', 0.0), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    df = df.dropna(subset=['optimizer_cost', 'adjusted_expected'])
    df = df[(df['optimizer_cost'] > 0) & (df['adjusted_expected'] > 0)]
    df, quality_info = _apply_quality_filters(df, settings)

    if df.empty:
        return {
            'selected': pd.DataFrame(),
            'alternates': pd.DataFrame(),
            'total_spend': 0.0,
            'total_adjusted_expected': 0.0,
            'total_effective_value': 0.0,
            'budget': float(budget),
            'leftover': float(budget),
            'num_selected': 0,
            'quality_filter_info': quality_info,
        }

    per_film_cap = budget * max_pct
    if integer_mode:
        per_film_cap = min(per_film_cap, float(integer_cap))
    df = df[df['optimizer_cost'] <= per_film_cap].copy()
    if df.empty:
        return {
            'selected': pd.DataFrame(),
            'alternates': pd.DataFrame(),
            'total_spend': 0.0,
            'total_adjusted_expected': 0.0,
            'total_effective_value': 0.0,
            'budget': float(budget),
            'leftover': float(budget),
            'num_selected': 0,
            'quality_filter_info': quality_info,
        }

    cost_ratio = (df['optimizer_cost'] / budget).clip(lower=0.0, upper=1.0)
    effective_multiplier = (1.0 - div_pen * cost_ratio - corr_pen * df['mean_abs_corr']).clip(lower=0.0)
    df['effective_value'] = df['adjusted_expected'] * effective_multiplier
    df['eff_per_dollar'] = np.where(df['optimizer_cost'] > 0, df['effective_value'] / df['optimizer_cost'], 0.0)

    costs = np.maximum(1, np.round(df['optimizer_cost'].to_numpy() * 100).astype(int))
    values = df['effective_value'].to_numpy(dtype=float)
    budget_cents = max(1, int(round(budget * 100)))
    n = len(df)

    dp = np.zeros(budget_cents + 1, dtype=float)
    take = np.zeros((n, budget_cents + 1), dtype=bool)

    for i in range(n):
        c = costs[i]
        v = values[i]
        if c > budget_cents or v <= 0:
            continue
        for b in range(budget_cents, c - 1, -1):
            cand = dp[b - c] + v
            if cand > dp[b] + 1e-12:
                dp[b] = cand
                take[i, b] = True

    best_b = int(np.argmax(dp))
    selected_idx = []
    b = best_b
    for i in range(n - 1, -1, -1):
        if take[i, b]:
            selected_idx.append(i)
            b -= costs[i]
            if b <= 0:
                break
    selected_idx = list(reversed(selected_idx))

    selected = df.iloc[selected_idx].copy() if selected_idx else df.iloc[[]].copy()
    selected_tickers = set(selected['ticker'].tolist())
    alternates = df[~df['ticker'].isin(selected_tickers)].copy()
    alternates = alternates.sort_values('eff_per_dollar', ascending=False).head(10)

    total_spend = float(selected['optimizer_cost'].sum()) if not selected.empty else 0.0
    total_adj = float(selected['adjusted_expected'].sum()) if not selected.empty else 0.0
    total_eff = float(selected['effective_value'].sum()) if not selected.empty else 0.0

    return {
        'selected': selected.sort_values('eff_per_dollar', ascending=False),
        'alternates': alternates,
        'total_spend': total_spend,
        'total_adjusted_expected': total_adj,
        'total_effective_value': total_eff,
        'budget': float(budget),
        'leftover': float(max(budget - total_spend, 0.0)),
        'num_selected': int(len(selected)),
        'max_budget_pct_per_film': max_pct,
        'cost_col': cost_col,
        'quality_filter_info': quality_info,
    }


def _build_return_scenarios(
    strategy_df: pd.DataFrame,
    settings: dict,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    risk_model: str = 'bootstrap',
    samples: int | None = None,
    random_seed: int | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[dict[str, np.ndarray], int, str, dict]:
    if history_loader is None:
        import storage
        history_loader = storage.load_price_history

    n_sims = int(samples or settings.get('strategy_mc_samples', 1500))
    n_sims = max(n_sims, 100)
    seed, seed_mode = _resolve_runtime_seed(settings, random_seed=random_seed)
    corr_mode = str(settings.get('strategy_corr_simulation_mode', 'independent') or 'independent').strip().lower()
    if corr_mode not in ('independent', 'gaussian_copula', 't_copula'):
        corr_mode = 'independent'

    min_h = max(int(settings.get('strategy_min_horizon_days', 7)), 1)
    max_h = max(int(settings.get('strategy_max_horizon_days', 30)), min_h)
    default_h = max(int(settings.get('strategy_default_horizon_days', 14)), min_h)
    # Minimum return volatility floor: pre-release HSX prices are stable (~1-2% std),
    # but actual box office uncertainty is far higher. This floor prevents the MC from
    # collapsing to a near-deterministic comparison when history shows very low variance.
    # Set to 0 to disable. Default 0.15 ≈ 40% annualized uncertainty for a 14-day window.
    min_return_vol = float(np.clip(settings.get('strategy_min_return_volatility', 0.15), 0.0, 1.0))

    scenarios: dict[str, np.ndarray] = {}
    ticker_order: list[str] = []
    marginal_cols: list[np.ndarray] = []
    for row_num, (_, row) in enumerate(strategy_df.iterrows()):
        ticker = str(row.get('ticker', '')).upper()
        if not ticker:
            continue

        if progress_callback and (row_num % 5 == 0 or row_num == len(strategy_df) - 1):
            pct = (row_num / len(strategy_df)) * 90.0 
            progress_callback(pct, f"Scenarios: {ticker} ({row_num+1}/{len(strategy_df)})")

        raw_days = pd.to_numeric(pd.Series([row.get('scenario_horizon_days', row.get('days_to_release'))]), errors='coerce').iloc[0]
        if pd.isna(raw_days):
            horizon = default_h
        else:
            horizon = int(round(float(raw_days)))
            horizon = min(max(horizon, min_h), max_h)

        history_df = _clean_history_df(history_loader(ticker))
        daily_returns = history_df['price'].pct_change().dropna().to_numpy(dtype=float)
        base = _compound_window_returns(daily_returns, horizon)

        rng = np.random.default_rng(seed + row_num)
        draws = _draw_returns_from_model(
            base_returns=base,
            samples=n_sims,
            rng=rng,
            risk_model=risk_model,
        )
        if min_return_vol > 0:
            hist_std = float(np.std(draws)) if draws.size > 1 else 0.0
            if hist_std < min_return_vol:
                extra_sd = float(np.sqrt(max(min_return_vol ** 2 - hist_std ** 2, 0.0)))
                draws = np.clip(draws + rng.normal(0.0, extra_sd, size=n_sims), -0.95, 5.0)
        scenarios[ticker] = draws
        ticker_order.append(ticker)
        marginal_cols.append(draws.astype(float))

    corr_meta = {
        'corr_mode_requested': corr_mode,
        'corr_mode_used': 'independent',
        'corr_effective_dim': int(len(ticker_order)),
        'corr_fallback_reason': '',
    }

    if corr_mode == 'independent' or len(ticker_order) < 2:
        if len(ticker_order) < 2 and corr_mode != 'independent':
            corr_meta['corr_fallback_reason'] = 'insufficient_movies'
        return scenarios, int(seed), seed_mode, corr_meta

    corr_matrix, corr_build_meta = _copula_correlation_matrix(
        tickers=ticker_order,
        history_loader=history_loader,
        settings=settings,
    )
    if corr_matrix is None:
        corr_meta.update(corr_build_meta)
        return scenarios, int(seed), seed_mode, corr_meta

    corr_meta.update(corr_build_meta)
    rng_c = np.random.default_rng(seed + 100003)
    n_dim = len(ticker_order)
    try:
        latent_mean = np.zeros(n_dim, dtype=float)
        if corr_mode == 'gaussian_copula':
            latent = rng_c.multivariate_normal(
                mean=latent_mean,
                cov=corr_matrix,
                size=n_sims,
                method='svd',
            )
            corr_meta['corr_mode_used'] = 'gaussian_copula'
        else:
            t_df = float(np.clip(settings.get('strategy_corr_t_df', 8), 3.0, 60.0))
            z = rng_c.multivariate_normal(
                mean=latent_mean,
                cov=corr_matrix,
                size=n_sims,
                method='svd',
            )
            chi = rng_c.chisquare(df=t_df, size=n_sims)
            scale = np.sqrt(t_df / np.clip(chi, 1e-12, None))
            latent = z * scale[:, None]
            corr_meta['corr_mode_used'] = 't_copula'
            corr_meta['corr_t_df_used'] = float(t_df)

        marginals = np.column_stack(marginal_cols)
        correlated = _rank_reorder_to_copula_marginals(marginals, latent)
        for j, ticker in enumerate(ticker_order):
            scenarios[ticker] = correlated[:, j]
    except Exception:
        corr_meta['corr_mode_used'] = 'independent'
        corr_meta['corr_fallback_reason'] = 'copula_generation_failed'

    return scenarios, int(seed), seed_mode, corr_meta


def simulate_portfolio_monte_carlo(
    strategy_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    settings: dict | None = None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    risk_model: str = 'bootstrap',
    cost_col: str = 'optimizer_cost',
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Phase 2 / B:
    Simulate portfolio gross distribution and risk diagnostics.
    """
    settings = settings or {}
    corr_mode_requested = str(settings.get('strategy_corr_simulation_mode', 'independent') or 'independent').strip().lower()
    if corr_mode_requested not in ('independent', 'gaussian_copula', 't_copula'):
        corr_mode_requested = 'independent'

    league_budget = pd.to_numeric(pd.Series([settings.get('strategy_league_remaining_budget', np.nan)]), errors='coerce').iloc[0]
    avail = strategy_df.copy()
    if 'owner' in avail.columns:
        avail = avail[avail['owner'].fillna('') == ''].copy()
    avail_adj_total = float(pd.to_numeric(avail.get('adjusted_expected', np.nan), errors='coerce').fillna(0.0).clip(lower=0.0).sum())
    if pd.notna(league_budget) and float(league_budget) > 0 and avail_adj_total > 0:
        exchange_rate = float(avail_adj_total / float(league_budget))
    else:
        exchange_rate = np.nan

    if portfolio_df.empty:
        seed_used, seed_mode = _resolve_runtime_seed(settings, random_seed=None)
        return {
            'samples': 0,
            'gross_mean': 0.0,
            'gross_p10': 0.0,
            'gross_p50': 0.0,
            'gross_p90': 0.0,
            'prob_gross_below_spend': np.nan,
            'concentration_downside_prob': np.nan,
            'aggression_sensitivity': pd.DataFrame(),
            'exchange_rate_million_per_auction_dollar': exchange_rate,
            'gross_budget_equiv_mean': np.nan,
            'gross_budget_equiv_p10': np.nan,
            'gross_budget_equiv_p50': np.nan,
            'gross_budget_equiv_p90': np.nan,
            'seed_used': int(seed_used),
            'seed_mode': seed_mode,
            'corr_mode_requested': corr_mode_requested,
            'corr_mode_used': 'independent',
            'corr_effective_dim': 0,
            'corr_fallback_reason': 'empty_portfolio',
        }

    scenarios, seed_used, seed_mode, corr_meta = _build_return_scenarios(
        strategy_df=strategy_df,
        settings=settings,
        history_loader=history_loader,
        risk_model=risk_model,
        progress_callback=progress_callback, # Pass callback
    )

    mc_samples = max(int(settings.get('strategy_mc_samples', 1500)), 100)
    threshold = float(np.clip(settings.get('strategy_mc_concentration_threshold', 0.40), 0.0, 1.0))

    strategy_map = strategy_df.copy()
    strategy_map['ticker'] = strategy_map['ticker'].astype(str).str.upper()
    strategy_map = strategy_map.set_index('ticker')
    tickers = portfolio_df['ticker'].astype(str).str.upper().tolist()

    # Prepare row_vals for efficient access
    row_vals = {}
    for ticker in tickers:
        if ticker not in strategy_map.index:
            continue
        adj = float(pd.to_numeric(pd.Series([strategy_map.at[ticker, 'adjusted_expected']]), errors='coerce').iloc[0] or 0.0)
        if adj > 0:
            row_vals[ticker] = adj

    # Run MC samples.
    # To support progress updates for a single matrix-multiply, we split into chunks.
    chunk_size = 1000
    n_chunks = (mc_samples + chunk_size - 1) // chunk_size
    all_gross = []
    all_top_share = []
    
    for i in range(n_chunks):
        if progress_callback:
            pct = 50.0 + (i / n_chunks) * 50.0  # Start from 50% after scenario building
            progress_callback(pct, f"MC Simulation: Chunk {i+1}/{n_chunks}")
        start = i * chunk_size
        end = min((i + 1) * chunk_size, mc_samples)
        
        chunk_scenarios = {k: v[start:end] for k, v in scenarios.items()}
        chunk_adj_gross = np.zeros(end - start, dtype=float)
        chunk_max_gross = np.zeros(end - start, dtype=float)
        
        for t, adj in row_vals.items():
            draws = chunk_scenarios.get(t, np.zeros(end - start, dtype=float))
            gross_t = adj * (1.0 + draws)
            chunk_adj_gross += gross_t
            chunk_max_gross = np.maximum(chunk_max_gross, gross_t)
        
        # Clip to 0 after summing
        chunk_adj_gross_clipped = np.maximum(0.0, chunk_adj_gross)
        all_gross.append(chunk_adj_gross_clipped)
        
        # Calculate share for concentration metric
        share_u = np.where(chunk_adj_gross_clipped > 0, chunk_max_gross / chunk_adj_gross_clipped, 0.0)
        all_top_share.append(share_u)

    portfolio_gross = np.concatenate(all_gross)
    top_share = np.concatenate(all_top_share)

    spend = 0.0
    if cost_col in portfolio_df.columns:
        spend = float(pd.to_numeric(portfolio_df[cost_col], errors='coerce').sum())
    elif 'optimizer_cost' in portfolio_df.columns:
        spend = float(pd.to_numeric(portfolio_df['optimizer_cost'], errors='coerce').sum())
    elif 'target_bid' in portfolio_df.columns:
        spend = float(pd.to_numeric(portfolio_df['target_bid'], errors='coerce').sum())

    if not row_vals: # Check if any valid movies were added to row_vals
        return {
            'samples': mc_samples,
            'gross_mean': 0.0,
            'gross_p10': 0.0,
            'gross_p50': 0.0,
            'gross_p90': 0.0,
            'prob_gross_below_spend': 1.0 if spend > 0 and pd.notna(exchange_rate) else np.nan,
            'concentration_downside_prob': np.nan,
            'aggression_sensitivity': pd.DataFrame(),
            'exchange_rate_million_per_auction_dollar': exchange_rate,
            'gross_budget_equiv_mean': np.nan,
            'gross_budget_equiv_p10': np.nan,
            'gross_budget_equiv_p50': np.nan,
            'gross_budget_equiv_p90': np.nan,
            'seed_used': int(seed_used),
            'seed_mode': seed_mode,
            'corr_mode_requested': corr_meta.get('corr_mode_requested', 'independent'),
            'corr_mode_used': corr_meta.get('corr_mode_used', 'independent'),
            'corr_effective_dim': int(corr_meta.get('corr_effective_dim', 0)),
            'corr_fallback_reason': corr_meta.get('corr_fallback_reason', ''),
        }

    total = portfolio_gross 
    threshold = float(settings.get('strategy_mc_concentration_threshold', 0.40))
    concentration_downside_prob = np.nan

    if pd.notna(exchange_rate) and exchange_rate > 0:
        total_budget_equiv = total / float(exchange_rate)
        downside_mask = total_budget_equiv < spend
        concentration_downside_prob = float(np.mean(downside_mask & (top_share > threshold)))
        prob_below_spend = float(np.mean(downside_mask)) if spend > 0 else 0.0
        budget_equiv_mean = float(np.mean(total_budget_equiv))
        budget_equiv_p10 = float(np.percentile(total_budget_equiv, 10))
        budget_equiv_p50 = float(np.percentile(total_budget_equiv, 50))
        budget_equiv_p90 = float(np.percentile(total_budget_equiv, 90))
    else:
        total_budget_equiv = np.full_like(total, np.nan, dtype=float)
        # conc_downside = np.nan # Removed
        prob_below_spend = np.nan if spend > 0 else 0.0
        budget_equiv_mean = np.nan
        budget_equiv_p10 = np.nan
        budget_equiv_p50 = np.nan
        budget_equiv_p90 = np.nan

    grid = settings.get('strategy_mc_aggression_grid', [0.9, 1.0, 1.1])
    if not isinstance(grid, list) or not grid:
        grid = [0.9, 1.0, 1.1]
    agg_rows = []
    for g in grid:
        try:
            factor = float(g)
        except Exception:
            continue
        eff_spend = spend * factor
        if pd.notna(exchange_rate) and exchange_rate > 0:
            surplus = total_budget_equiv - eff_spend
            expected_surplus = float(np.mean(surplus))
            prob_pos = float(np.mean(surplus > 0))
        else:
            expected_surplus = np.nan
            prob_pos = np.nan
        agg_rows.append({
            'aggression_factor': factor,
            'expected_surplus': expected_surplus,
            'prob_positive_surplus': prob_pos,
        })

    return {
        'samples': mc_samples,
        'gross_mean': float(np.mean(total)),
        'gross_p10': float(np.percentile(total, 10)),
        'gross_p50': float(np.percentile(total, 50)),
        'gross_p90': float(np.percentile(total, 90)),
        'prob_gross_below_spend': prob_below_spend,
        'concentration_downside_prob': concentration_downside_prob,
        'aggression_sensitivity': pd.DataFrame(agg_rows),
        'exchange_rate_million_per_auction_dollar': exchange_rate,
        'gross_budget_equiv_mean': budget_equiv_mean,
        'gross_budget_equiv_p10': budget_equiv_p10,
        'gross_budget_equiv_p50': budget_equiv_p50,
        'gross_budget_equiv_p90': budget_equiv_p90,
        'seed_used': int(seed_used),
        'seed_mode': seed_mode,
        'corr_mode_requested': corr_meta.get('corr_mode_requested', 'independent'),
        'corr_mode_used': corr_meta.get('corr_mode_used', 'independent'),
        'corr_effective_dim': int(corr_meta.get('corr_effective_dim', 0)),
        'corr_fallback_reason': corr_meta.get('corr_fallback_reason', ''),
    }


def _random_greedy_tickers(
    strategy_df: pd.DataFrame,
    budget: float,
    cost_col: str,
    max_pct: float,
    rng: np.random.Generator,
    score_noise: float,
    aggression: float,
    integer_mode: bool = False,
    min_legal_bid: int = 1,
    integer_cap: int | None = None,
    bidup_strength: float = 0.0,
    cash_conservation: float = 0.0,
) -> list[str]:
    df = strategy_df.copy()
    if 'owner' in df.columns:
        df = df[df['owner'].fillna('') == '']
    if df.empty:
        return []

    # Base ranking driver for random-greedy and bid-up pressure.
    df['base_score'] = pd.to_numeric(df.get('priority_score', np.nan), errors='coerce')
    fallback = pd.to_numeric(df.get('eff_per_dollar', np.nan), errors='coerce')
    df['base_score'] = df['base_score'].fillna(fallback).fillna(0.0)

    df['raw_cost'] = pd.to_numeric(df.get(cost_col, np.nan), errors='coerce')
    df['adj_cost'] = (df['raw_cost'] * aggression).clip(lower=0.0)
    bidup = float(np.clip(bidup_strength, 0.0, 1.0))
    if bidup > 0 and len(df) > 1:
        rank = df['base_score'].rank(method='first', ascending=False)
        top_weight = 1.0 - (rank - 1.0) / max(len(df) - 1, 1)
        df['adj_cost'] = df['adj_cost'] * (1.0 + bidup * top_weight)

    if integer_mode:
        if cost_col in ('target_bid', 'target_bid_int') and 'target_bid_int' in df.columns:
            df['adj_cost'] = pd.to_numeric(df['target_bid_int'], errors='coerce')
        else:
            df['adj_cost'] = np.ceil(df['adj_cost'])
        cap = integer_cap if integer_cap is not None else int(np.floor(budget))
        if cap < min_legal_bid:
            return []
        df['adj_cost'] = df['adj_cost'].clip(lower=min_legal_bid, upper=cap)

    df = df.dropna(subset=['adj_cost'])
    per_film_cap = budget * max_pct
    if integer_mode and integer_cap is not None:
        per_film_cap = min(per_film_cap, float(integer_cap))
    df = df[(df['adj_cost'] > 0) & (df['adj_cost'] <= per_film_cap)]
    if df.empty:
        return []

    noise = rng.lognormal(mean=0.0, sigma=max(score_noise, 1e-4), size=len(df))
    df['rand_score'] = df['base_score'].to_numpy(dtype=float) * noise
    df = df.sort_values('rand_score', ascending=False)

    spend_cap = float(np.clip(1.0 - float(cash_conservation), 0.50, 1.20)) * float(budget)
    chosen = []
    spend = 0.0
    for _, row in df.iterrows():
        c = float(row['adj_cost'])
        if spend + c <= spend_cap:
            chosen.append(str(row['ticker']).upper())
            spend += c
    return chosen


def _resolve_opponent_profile_params(settings: dict) -> dict:
    """Resolve effective opponent behavior parameters from profile + knobs."""
    profile = str(settings.get('strategy_opponent_profile', 'balanced_field') or 'balanced_field').strip().lower()
    if profile not in ('passive_value', 'balanced_field', 'aggressive_bidup'):
        profile = 'balanced_field'

    noise = float(np.clip(settings.get('strategy_mc_opponent_noise', 0.30), 0.01, 1.0))
    aggr_sd = float(np.clip(settings.get('strategy_mc_aggression_sd', 0.10), 0.0, 1.0))
    bidup = float(np.clip(settings.get('strategy_opponent_bidup_strength', 0.0), 0.0, 1.0))
    conserve = float(np.clip(settings.get('strategy_opponent_cash_conservation', 0.0), -0.50, 0.80))

    if profile == 'passive_value':
        noise *= 0.80
        aggr_sd *= 0.75
        bidup = max(bidup, 0.05)
        conserve = max(conserve, 0.15)
    elif profile == 'aggressive_bidup':
        noise = min(1.0, noise * 1.20)
        aggr_sd = min(1.0, aggr_sd * 1.25)
        bidup = max(bidup, 0.25)
        conserve = min(conserve, -0.05)

    return {
        'opponent_profile_used': profile,
        'noise_sigma': float(np.clip(noise, 0.01, 1.0)),
        'aggr_sd': float(np.clip(aggr_sd, 0.0, 1.0)),
        'bidup_strength': float(np.clip(bidup, 0.0, 1.0)),
        'cash_conservation': float(np.clip(conserve, -0.50, 0.80)),
    }


def _build_exclusive_opponent_sets(
    strategy_df: pd.DataFrame,
    excluded_tickers: frozenset,
    n_opp: int,
    rng: np.random.Generator,
    cost_col: str,
    max_pct: float,
    budget: float,
    opp_params: dict,
    integer_mode: bool,
    min_legal: int,
    integer_cap: int,
) -> list[list[str]]:
    """
    Generate n_opp mutually exclusive opponent portfolios from a shrinking pool.

    Each opponent picks from the movies not yet taken by Taylor (excluded_tickers)
    or any previously constructed opponent. This guarantees that every movie
    appears in at most one portfolio across all players in the simulation.
    """
    taken = set(excluded_tickers)
    opponent_portfolios: list[list[str]] = []
    for _ in range(n_opp):
        avail_df = strategy_df[
            ~strategy_df['ticker'].astype(str).str.upper().isin(taken)
        ].copy()
        if avail_df.empty:
            break
        aggr = max(0.6, rng.normal(1.0, opp_params['aggr_sd']))
        ticks = _random_greedy_tickers(
            strategy_df=avail_df,
            budget=budget,
            cost_col=cost_col,
            max_pct=max_pct,
            rng=rng,
            score_noise=opp_params['noise_sigma'],
            aggression=aggr,
            integer_mode=integer_mode,
            min_legal_bid=min_legal,
            integer_cap=integer_cap,
            bidup_strength=opp_params['bidup_strength'],
            cash_conservation=opp_params['cash_conservation'],
        )
        if ticks:
            opponent_portfolios.append(ticks)
            taken.update(str(t).upper() for t in ticks)
    return opponent_portfolios


def _resolve_search_mode(settings: dict) -> str:
    mode = str(settings.get('strategy_search_mode', 'current_sampled') or 'current_sampled').strip().lower()
    allowed = {'current_sampled', 'random_multistart', 'local_search', 'genetic'}
    return mode if mode in allowed else 'current_sampled'


def _local_search_refine(
    ticks: list[str],
    cost_map: dict[str, float],
    value_map: dict[str, float],
    budget: float,
    iterations: int,
) -> list[str]:
    cur = sorted(set(str(t).upper() for t in ticks if str(t)))
    if not cur:
        return cur
    spend = float(sum(cost_map.get(t, 0.0) for t in cur))
    universe = list(cost_map.keys())
    iters = max(int(iterations), 0)
    for _ in range(iters):
        best = None
        best_gain = 0.0
        cur_set = set(cur)
        for out_t in cur:
            out_cost = cost_map.get(out_t, 0.0)
            out_val = value_map.get(out_t, 0.0)
            base_spend = spend - out_cost
            for in_t in universe:
                if in_t in cur_set:
                    continue
                new_spend = base_spend + cost_map.get(in_t, 0.0)
                if new_spend > budget + 1e-9:
                    continue
                gain = value_map.get(in_t, 0.0) - out_val
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (out_t, in_t, new_spend)
        if best is None:
            break
        out_t, in_t, new_spend = best
        cur.remove(out_t)
        cur.append(in_t)
        cur = sorted(set(cur))
        spend = float(new_spend)
    return cur


def _genetic_candidate_sets(
    base_candidates: list[list[str]],
    cost_map: dict[str, float],
    value_map: dict[str, float],
    budget: float,
    rng: np.random.Generator,
    population_size: int,
    generations: int,
    elite_frac: float,
    mutation_rate: float,
    target_sets: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[list[str]]:
    universe = list(cost_map.keys())
    if not universe:
        return []

    pop_size = max(int(population_size), 20)
    gens = max(int(generations), 1)
    elite_n = max(1, int(round(float(np.clip(elite_frac, 0.05, 0.90)) * pop_size)))
    mut = float(np.clip(mutation_rate, 0.0, 0.8))

    ratio = {t: value_map.get(t, 0.0) / max(cost_map.get(t, 1e-9), 1e-9) for t in universe}

    def _repair(candidate: set[str]) -> set[str]:
        spend = float(sum(cost_map.get(t, 0.0) for t in candidate))
        if spend > budget:
            ordered = sorted(candidate, key=lambda t: ratio.get(t, 0.0))
            for t in ordered:
                if spend <= budget + 1e-9:
                    break
                candidate.remove(t)
                spend -= cost_map.get(t, 0.0)
        # Greedy fill.
        for t in sorted(universe, key=lambda x: ratio.get(x, 0.0), reverse=True):
            if t in candidate:
                continue
            c = cost_map.get(t, 0.0)
            if c <= 0:
                continue
            if spend + c <= budget + 1e-9:
                candidate.add(t)
                spend += c
        return candidate

    def _fitness(candidate: set[str]) -> float:
        spend = float(sum(cost_map.get(t, 0.0) for t in candidate))
        if spend > budget + 1e-9:
            return -1e12
        return float(sum(value_map.get(t, 0.0) for t in candidate))

    # Seed initial population.
    population: list[set[str]] = []
    for cand in base_candidates:
        s = _repair(set(cand))
        if s:
            population.append(s)
    while len(population) < pop_size:
        s = set()
        spend = 0.0
        perm = list(universe)
        rng.shuffle(perm)
        perm = sorted(perm, key=lambda t: ratio.get(t, 0.0) * rng.uniform(0.8, 1.2), reverse=True)
        for t in perm:
            c = cost_map.get(t, 0.0)
            if c <= 0:
                continue
            if spend + c <= budget + 1e-9:
                s.add(t)
                spend += c
        if s:
            population.append(s)

    for gen_num in range(gens):
        if progress_callback:
            pct = (gen_num / gens) * 100.0
            progress_callback(pct, f"Genetic Search: Generation {gen_num+1}/{gens}")
        scored = sorted(population, key=_fitness, reverse=True)
        elites = scored[:elite_n]
        new_pop = [set(e) for e in elites]
        pool = scored[: max(elite_n * 2, 4)]
        while len(new_pop) < pop_size:
            p1 = set(pool[int(rng.integers(0, len(pool)))])
            p2 = set(pool[int(rng.integers(0, len(pool)))])
            child = set()
            for t in universe:
                in1 = t in p1
                in2 = t in p2
                if in1 and in2:
                    child.add(t)
                elif in1 != in2 and rng.uniform() < 0.5:
                    child.add(t)
            if rng.uniform() < mut:
                if child and rng.uniform() < 0.5:
                    rem = list(child)[int(rng.integers(0, len(child)))]
                    child.remove(rem)
                else:
                    add_pool = [t for t in universe if t not in child]
                    if add_pool:
                        add_t = add_pool[int(rng.integers(0, len(add_pool)))]
                        child.add(add_t)
            child = _repair(child)
            if child:
                new_pop.append(child)
        population = new_pop
    if progress_callback:
        progress_callback(100.0, "Genetic Search: Complete")

    final_scored = sorted(population, key=_fitness, reverse=True)
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for cand in final_scored:
        key = tuple(sorted(cand))
        if key and key not in seen:
            seen.add(key)
            out.append(list(key))
        if len(out) >= max(int(target_sets), 1):
            break
    return out


def optimize_portfolio_by_win_probability(
    strategy_df: pd.DataFrame,
    budget: float,
    settings: dict | None = None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    risk_model: str = 'bootstrap',
    cost_col: str = 'current_price',
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Phase 2 / C:
    Optimize by estimated probability of finishing first vs simulated opponents.
    """
    settings = settings or {}
    if budget <= 0 or strategy_df.empty:
        base = optimize_portfolio(strategy_df, budget, settings=settings, cost_col=cost_col, progress_callback=progress_callback)
        base.update({
            'objective': 'win_probability',
            'win_probability': np.nan,
            'opponent_count': 0,
        })
        return base

    # Base deterministic portfolio included as a candidate.
    deterministic = optimize_portfolio(strategy_df, budget, settings=settings, cost_col=cost_col, progress_callback=progress_callback)
    max_pct = _resolve_per_film_cap_pct(settings, cost_col)
    search_mode = _resolve_search_mode(settings)
    candidate_count = max(
        int(settings.get('strategy_search_candidates', settings.get('strategy_mc_candidate_portfolios', 120))),
        20,
    )
    if search_mode == 'current_sampled':
        candidate_count = max(int(settings.get('strategy_mc_candidate_portfolios', 120)), 20)
    n_opp = max(int(settings.get('strategy_mc_num_opponents', 7)), 1)
    opp_params = _resolve_opponent_profile_params(settings)
    noise_sigma = opp_params['noise_sigma']
    aggr_sd = opp_params['aggr_sd']
    seed, seed_mode = _resolve_runtime_seed(settings, random_seed=None)
    sims = max(int(settings.get('strategy_mc_samples', 1500)), 200)
    search_t0 = time.perf_counter()
    integer_mode = bool(settings.get('strategy_integer_bid_mode', False))
    prev_bid = max(int(settings.get('strategy_integer_prev_bid', 0)), 0)
    min_legal = prev_bid + 1
    personal_cap = pd.to_numeric(pd.Series([settings.get('strategy_personal_budget_cap', np.nan)]), errors='coerce').iloc[0]
    if integer_mode and (pd.isna(personal_cap) or personal_cap <= 0):
        base = optimize_portfolio(strategy_df, budget, settings=settings, cost_col=cost_col, progress_callback=progress_callback)
        base.update({
            'objective': 'win_probability',
            'win_probability': np.nan,
            'opponent_count': 0,
        })
        return base
    if pd.isna(personal_cap) or personal_cap <= 0:
        personal_cap = budget
    integer_cap = int(np.floor(min(personal_cap, budget)))

    # Build the executable cost frame used to enforce hard budget feasibility.
    cost_df = strategy_df.copy()
    if 'owner' in cost_df.columns:
        cost_df = cost_df[cost_df['owner'].fillna('') == ''].copy()
    cost_df['ticker'] = cost_df['ticker'].astype(str).str.upper()
    cost_df['optimizer_cost'] = pd.to_numeric(cost_df.get(cost_col, np.nan), errors='coerce')
    if integer_mode:
        if cost_col in ('target_bid', 'target_bid_int') and 'target_bid_int' in cost_df.columns:
            cost_df['optimizer_cost'] = pd.to_numeric(cost_df['target_bid_int'], errors='coerce')
        else:
            cost_df['optimizer_cost'] = np.ceil(cost_df['optimizer_cost'])
        if integer_cap < min_legal:
            cost_df['optimizer_cost'] = np.nan
        else:
            cost_df['optimizer_cost'] = cost_df['optimizer_cost'].clip(lower=min_legal, upper=integer_cap)
    cost_df['adjusted_expected'] = pd.to_numeric(cost_df.get('adjusted_expected', np.nan), errors='coerce')
    cost_df = cost_df.dropna(subset=['optimizer_cost', 'adjusted_expected'])
    cost_df = cost_df[(cost_df['optimizer_cost'] > 0) & (cost_df['adjusted_expected'] > 0)]
    cost_df, quality_info = _apply_quality_filters(cost_df, settings)
    per_film_cap = budget * max_pct
    if integer_mode:
        per_film_cap = min(per_film_cap, float(integer_cap))
    cost_df = cost_df[cost_df['optimizer_cost'] <= per_film_cap].copy()
    if cost_df.empty:
        base = optimize_portfolio(strategy_df, budget, settings=settings, cost_col=cost_col, progress_callback=progress_callback)
        base.update({
            'objective': 'win_probability',
            'win_probability': np.nan,
            'opponent_count': 0,
            'quality_filter_info': quality_info,
        })
        return base
    cost_df['eff_per_dollar'] = np.where(
        cost_df['optimizer_cost'] > 0,
        cost_df['adjusted_expected'] / cost_df['optimizer_cost'],
        np.nan,
    )
    cost_map = dict(zip(cost_df['ticker'].tolist(), cost_df['optimizer_cost'].tolist()))
    valid_tickers = set(cost_map.keys())

    def _is_budget_feasible(ticks: list[str]) -> bool:
        if not ticks:
            return False
        uniq = sorted(set(str(t).upper() for t in ticks if str(t)))
        if not uniq:
            return False
        if any(t not in valid_tickers for t in uniq):
            return False
        spend = float(sum(float(cost_map[t]) for t in uniq))
        return spend <= float(budget) + 1e-9

    rng = np.random.default_rng(seed)

    # Candidate portfolios.
    candidate_sets: dict[tuple[str, ...], list[str]] = {}
    if not deterministic['selected'].empty:
        base_ticks = sorted(deterministic['selected']['ticker'].astype(str).str.upper().tolist())
        if _is_budget_feasible(base_ticks):
            candidate_sets[tuple(base_ticks)] = base_ticks

    attempts = 0
    attempt_mult = 6 if search_mode == 'current_sampled' else 10
    if progress_callback:
        progress_callback(10.0, "Generating initial candidate portfolios...")
    while len(candidate_sets) < candidate_count and attempts < candidate_count * attempt_mult:
        attempts += 1
        aggr = max(0.6, rng.normal(1.0, aggr_sd))
        ticks = _random_greedy_tickers(
            strategy_df=strategy_df,
            budget=budget,
            cost_col=cost_col,
            max_pct=max_pct,
            rng=rng,
            score_noise=noise_sigma,
            aggression=aggr,
            integer_mode=integer_mode,
            min_legal_bid=min_legal,
            integer_cap=integer_cap,
            bidup_strength=opp_params['bidup_strength'],
            cash_conservation=opp_params['cash_conservation'],
        )
        if not ticks:
            continue
        key = tuple(sorted(set(ticks)))
        if key and _is_budget_feasible(list(key)):
            candidate_sets[key] = list(key)
    if progress_callback:
        progress_callback(20.0, f"Generated {len(candidate_sets)} initial candidates.")

    # Optional local refinement layer.
    if search_mode == 'local_search' and candidate_sets:
        if progress_callback:
            progress_callback(25.0, "Performing local search refinement...")
        local_iters = max(int(settings.get('strategy_search_local_iters', 25)), 0)
        value_map = dict(zip(
            cost_df['ticker'].astype(str).str.upper().tolist(),
            pd.to_numeric(cost_df['adjusted_expected'], errors='coerce').fillna(0.0).tolist(),
        ))
        initial_candidate_keys = list(candidate_sets.keys())
        for i, key in enumerate(initial_candidate_keys):
            if progress_callback:
                pct = 25.0 + (i / len(initial_candidate_keys)) * 10.0
                progress_callback(pct, f"Local Search: Refining candidate {i+1}/{len(initial_candidate_keys)}")
            ticks = candidate_sets[key]
            refined = _local_search_refine(
                ticks=ticks,
                cost_map=cost_map,
                value_map=value_map,
                budget=float(budget),
                iterations=local_iters,
            )
            new_key = tuple(sorted(set(refined)))
            if new_key and _is_budget_feasible(list(new_key)):
                candidate_sets[new_key] = list(new_key)
        if progress_callback:
            progress_callback(35.0, "Local search complete.")

    # Optional genetic search augmentation.
    if search_mode == 'genetic' and candidate_sets:
        if progress_callback:
            progress_callback(35.0, "Performing genetic search augmentation...")
        value_map = dict(zip(
            cost_df['ticker'].astype(str).str.upper().tolist(),
            pd.to_numeric(cost_df['adjusted_expected'], errors='coerce').fillna(0.0).tolist(),
        ))
        ga_candidates = _genetic_candidate_sets(
            base_candidates=list(candidate_sets.values()),
            cost_map=cost_map,
            value_map=value_map,
            budget=float(budget),
            rng=rng,
            population_size=max(int(settings.get('strategy_search_population', 80)), 20),
            generations=max(int(settings.get('strategy_search_generations', 30)), 1),
            elite_frac=float(np.clip(settings.get('strategy_search_elite_frac', 0.20), 0.05, 0.90)),
            mutation_rate=float(np.clip(settings.get('strategy_search_mutation_rate', 0.08), 0.0, 0.80)),
            target_sets=max(candidate_count, 20),
            progress_callback=lambda p, msg: progress_callback(35.0 + p * 0.15, f"Genetic Search: {msg}") if progress_callback else None,
        )
        for ticks in ga_candidates:
            key = tuple(sorted(set(ticks)))
            if key and _is_budget_feasible(list(key)):
                candidate_sets[key] = list(key)
        if progress_callback:
            progress_callback(50.0, "Genetic search complete.")

    if not candidate_sets:
        base = optimize_portfolio(strategy_df, budget, settings=settings, cost_col=cost_col, progress_callback=progress_callback)
        base.update({
            'objective': 'win_probability',
            'win_probability': np.nan,
            'opponent_count': n_opp,
            'quality_filter_info': quality_info,
        })
        return base

    strategy_map = strategy_df.set_index('ticker')

    scenarios, _, _, _ = _build_return_scenarios(
        strategy_df=strategy_df,
        settings={**settings, 'strategy_mc_samples': sims},
        history_loader=history_loader,
        risk_model=risk_model,
        samples=sims,
        random_seed=seed,
        progress_callback=lambda p, msg: progress_callback(50.0 + p * 0.10, f"Building Scenarios: {msg}") if progress_callback else None,
    )
    # Precompute per-movie gross arrays for fast portfolio evaluation.
    per_movie_gross: dict[str, np.ndarray] = {}
    for t in list(strategy_map.index):
        t_upper = str(t).upper()
        adj = float(pd.to_numeric(pd.Series([strategy_map.at[t, 'adjusted_expected']]), errors='coerce').iloc[0] or 0.0)
        draws = scenarios.get(t, np.zeros(sims, dtype=float))
        per_movie_gross[t_upper] = np.maximum(0.0, adj * (1.0 + draws))

    def _gross_from_cache(tickers: list[str]) -> np.ndarray:
        arrays = [per_movie_gross[str(t).upper()] for t in tickers if str(t).upper() in per_movie_gross]
        return np.sum(arrays, axis=0).astype(float) if arrays else np.zeros(sims, dtype=float)

    n_univ = max(int(settings.get('strategy_mc_opponent_universes', 25)), 1)
    total_steps = len(candidate_sets) * n_univ
    current_step = 0

    # Evaluate each candidate portfolio against globally exclusive opponent sets.
    best = None
    for cand_idx, ticks in enumerate(candidate_sets.values()):
        ticks_upper = frozenset(str(t).upper() for t in ticks)
        gross = _gross_from_cache(list(ticks_upper))

        cand_univ_win_probs = []
        cand_univ_opp_counts = []

        for u_idx in range(n_univ):
            current_step += 1
            if progress_callback:
                pct = 60.0 + (current_step / total_steps) * 40.0
                progress_callback(pct, f"Optimizing: Evaluating candidate {cand_idx+1}/{len(candidate_sets)}, universe {u_idx+1}/{n_univ}")

            # Consistent seed for this universe across all candidates
            # so they face the same baseline opponent scenarios.
            u_rng = np.random.default_rng([seed, 1000 + u_idx])

            opp_sets = _build_exclusive_opponent_sets(
                strategy_df=strategy_df,
                excluded_tickers=ticks_upper,
                n_opp=n_opp,
                rng=u_rng,
                cost_col=cost_col,
                max_pct=max_pct,
                budget=budget,
                opp_params=opp_params,
                integer_mode=integer_mode,
                min_legal=1,
                integer_cap=integer_cap,
            )

            if not opp_sets:
                cand_univ_win_probs.append(1.0)
                cand_univ_opp_counts.append(0)
                continue

            opp_gross_list = [_gross_from_cache(o) for o in opp_sets]
            excl_opp_max = np.maximum.reduce(opp_gross_list)

            wins = gross > (excl_opp_max + 1e-9)
            ties = np.abs(gross - excl_opp_max) <= 1e-9
            win_prob_u = float(np.mean(wins.astype(float) + 0.5 * ties.astype(float)))

            cand_univ_win_probs.append(win_prob_u)
            cand_univ_opp_counts.append(len(opp_sets))

        avg_wp = float(np.mean(cand_univ_win_probs))
        win_prob = avg_wp
        exp_gross = float(np.mean(gross))
        key = (win_prob, exp_gross)

        if best is None or key > best['key']:
            best = {
                'ticks': list(ticks),
                'gross': gross,
                'win_prob': win_prob,
                'exp_gross': exp_gross,
                'key': key,
                'opp_count': float(np.mean(cand_univ_opp_counts)),
                'opponent_universes': n_univ,
            }
    if progress_callback:
        progress_callback(100.0, "Optimization complete.")

    if best is None:
        base = optimize_portfolio(strategy_df, budget, settings=settings, cost_col=cost_col, progress_callback=progress_callback)
        base.update({
            'objective': 'win_probability',
            'win_probability': np.nan,
            'opponent_count': n_opp,
            'quality_filter_info': quality_info,
        })
        return base

    selected = cost_df[cost_df['ticker'].astype(str).str.upper().isin(best['ticks'])].copy()
    alternates = cost_df[~cost_df['ticker'].astype(str).str.upper().isin(best['ticks'])].copy()
    alternates = alternates.sort_values('eff_per_dollar', ascending=False).head(10)
    total_spend = float(pd.to_numeric(selected['optimizer_cost'], errors='coerce').sum())
    total_adj = float(pd.to_numeric(selected['adjusted_expected'], errors='coerce').sum())

    result = {
        'selected': selected.sort_values('priority_score', ascending=False),
        'alternates': alternates,
        'total_spend': total_spend,
        'total_adjusted_expected': total_adj,
        'total_effective_value': float(total_adj),
        'budget': float(budget),
        'leftover': float(max(budget - total_spend, 0.0)),
        'num_selected': int(len(selected)),
        'max_budget_pct_per_film': max_pct,
        'cost_col': cost_col,
        'objective': 'win_probability',
        'win_probability': best['win_prob'],
        'opponent_count': int(round(best.get('opp_count', n_opp))),
        'n_opp_requested': int(n_opp),
        'opponent_universes': int(best.get('opponent_universes', 1)),
        'expected_gross_mc': best['exp_gross'],
        'quality_filter_info': quality_info,
        'seed_mode': seed_mode,
        'seed_used': int(seed),
        'search_mode_used': search_mode,
        'candidate_count_evaluated': int(len(candidate_sets)),
        'search_runtime_ms': float(max((time.perf_counter() - search_t0) * 1000.0, 0.0)),
        'opponent_profile_used': opp_params.get('opponent_profile_used', 'balanced_field'),
    }
    return result


def estimate_portfolio_win_probability(
    strategy_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    budget: float,
    settings: dict | None = None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    risk_model: str = 'bootstrap',
    cost_col: str = 'current_price',
    portfolio_cost_col: str = 'optimizer_cost',
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """
    Estimate win probability for a *given* portfolio against simulated opponents.
    Useful when objective is expected_gross but a win-probability diagnostic is desired.
    """
    settings = settings or {}
    search_mode = _resolve_search_mode(settings)
    opp_params = _resolve_opponent_profile_params(settings)
    search_t0 = time.perf_counter()
    if budget <= 0 or strategy_df.empty or portfolio_df.empty:
        return {
            'win_probability': np.nan,
            'opponent_count': 0,
            'samples': 0,
            'seed_mode': str(settings.get('strategy_mc_seed_mode', 'fixed')),
            'seed_used': pd.to_numeric(pd.Series([settings.get('strategy_mc_random_seed', 123)]), errors='coerce').fillna(123).iloc[0],
            'search_mode_used': search_mode,
            'candidate_count_evaluated': 0,
            'search_runtime_ms': float(max((time.perf_counter() - search_t0) * 1000.0, 0.0)),
            'opponent_profile_used': opp_params.get('opponent_profile_used', 'balanced_field'),
        }

    max_pct = _resolve_per_film_cap_pct(settings, cost_col)
    n_opp = max(int(settings.get('strategy_mc_num_opponents', 7)), 1)
    seed, seed_mode = _resolve_runtime_seed(settings, random_seed=None)
    sims = max(int(settings.get('strategy_mc_samples', 1500)), 200)
    integer_mode = bool(settings.get('strategy_integer_bid_mode', False))
    prev_bid = max(int(settings.get('strategy_integer_prev_bid', 0)), 0)
    min_legal = prev_bid + 1
    personal_cap = pd.to_numeric(pd.Series([settings.get('strategy_personal_budget_cap', np.nan)]), errors='coerce').iloc[0]
    if pd.isna(personal_cap) or personal_cap <= 0:
        personal_cap = budget
    integer_cap = int(np.floor(min(personal_cap, budget)))

    if integer_mode and integer_cap < min_legal:
        return {
            'win_probability': np.nan,
            'opponent_count': 0,
            'samples': sims,
            'seed_mode': seed_mode,
            'seed_used': int(seed),
            'search_mode_used': search_mode,
            'candidate_count_evaluated': 0,
            'search_runtime_ms': float(max((time.perf_counter() - search_t0) * 1000.0, 0.0)),
            'opponent_profile_used': opp_params.get('opponent_profile_used', 'balanced_field'),
            'portfolio_budget_feasible': np.nan,
            'selected_portfolio_spend': np.nan,
        }

    strategy_map = strategy_df.set_index('ticker')
    strategy_tickers = set(strategy_map.index.astype(str).str.upper().tolist())

    selected_ticks = (
        portfolio_df.get('ticker', pd.Series(dtype=object))
        .astype(str).str.upper().tolist()
    )
    selected_ticks = sorted(set([t for t in selected_ticks if t in strategy_tickers]))
    if not selected_ticks:
        return {
            'win_probability': np.nan,
            'opponent_count': 0,
            'samples': sims,
            'seed_mode': seed_mode,
            'seed_used': int(seed),
            'search_mode_used': search_mode,
            'candidate_count_evaluated': 0,
            'search_runtime_ms': float(max((time.perf_counter() - search_t0) * 1000.0, 0.0)),
            'opponent_profile_used': opp_params.get('opponent_profile_used', 'balanced_field'),
            'portfolio_budget_feasible': np.nan,
            'selected_portfolio_spend': np.nan,
        }

    sel_spend = np.nan
    candidate_cost_cols = [portfolio_cost_col, cost_col, 'optimizer_cost', 'paid_price', 'final_price', 'target_bid']
    for c_name in candidate_cost_cols:
        if c_name in portfolio_df.columns:
            sel_spend = float(pd.to_numeric(portfolio_df[c_name], errors='coerce').fillna(0.0).sum())
            break
    portfolio_budget_feasible = True if pd.isna(sel_spend) else (float(sel_spend) <= float(budget) + 1e-9)

    scenarios, _, _, _ = _build_return_scenarios(
        strategy_df=strategy_df,
        settings={**settings, 'strategy_mc_samples': sims},
        history_loader=history_loader,
        risk_model=risk_model,
        samples=sims,
        random_seed=seed,
        progress_callback=lambda p, msg: progress_callback(p * 0.4, f"Building Scenarios: {msg}") if progress_callback else None,
    )

    # Precompute per-movie gross arrays.
    per_movie_gross_ep: dict[str, np.ndarray] = {}
    for t in list(strategy_map.index):
        t_upper = str(t).upper()
        adj = float(pd.to_numeric(pd.Series([strategy_map.at[t, 'adjusted_expected']]), errors='coerce').iloc[0] or 0.0)
        draws = scenarios.get(t, np.zeros(sims, dtype=float))
        per_movie_gross_ep[t_upper] = np.maximum(0.0, adj * (1.0 + draws))

    def _gross_from_cache_ep(tickers: list[str]) -> np.ndarray:
        arrays = [per_movie_gross_ep[str(t).upper()] for t in tickers if str(t).upper() in per_movie_gross_ep]
        return np.sum(arrays, axis=0).astype(float) if arrays else np.zeros(sims, dtype=float)

    selected_upper = frozenset(str(t).upper() for t in selected_ticks)
    gross = _gross_from_cache_ep(list(selected_upper))

    # Average results across multiple Draft Universes to ensure the estimate
    # is robust against different opponent drafting outcomes.
    n_univ = max(int(settings.get('strategy_mc_opponent_universes', 25)), 1)
    univ_win_probs = []
    univ_opp_counts = []

    for u_idx in range(n_univ):
        if progress_callback:
            pct = 40.0 + (u_idx / n_univ) * 60.0
            progress_callback(pct, f"Estimating: Universe {u_idx+1}/{n_univ}")
        # Fresh RNG sequence for each draft universe
        u_rng = np.random.default_rng([seed, 1000 + u_idx])
        
        opp_sets_u = _build_exclusive_opponent_sets(
            strategy_df=strategy_df,
            excluded_tickers=selected_upper,
            n_opp=n_opp,
            rng=u_rng,
            cost_col=cost_col,
            max_pct=max_pct,
            budget=budget,
            opp_params=opp_params,
            integer_mode=integer_mode,
            min_legal=1,
            integer_cap=integer_cap,
        )
        
        if not opp_sets_u:
            univ_win_probs.append(1.0) # No opponents = 100% win
            univ_opp_counts.append(0)
            continue
            
        opp_gross_list = [_gross_from_cache_ep(o) for o in opp_sets_u]
        opp_max = np.maximum.reduce(opp_gross_list)
        
        wins = gross > (opp_max + 1e-9)
        ties = np.abs(gross - opp_max) <= 1e-9
        win_prob_u = float(np.mean(wins.astype(float) + 0.5 * ties.astype(float)))
        
        univ_win_probs.append(win_prob_u)
        univ_opp_counts.append(len(opp_sets_u))

    win_prob = float(np.mean(univ_win_probs))
    avg_opp_count = float(np.mean(univ_opp_counts))
    if progress_callback:
        progress_callback(100.0, "Estimation complete.")

    return {
        'win_probability': win_prob,
        'opponent_count': int(round(avg_opp_count)),
        'n_opp_requested': int(n_opp),
        'opponent_universes': n_univ,
        'samples': int(sims),
        'seed_mode': seed_mode,
        'seed_used': int(seed),
        'search_mode_used': search_mode,
        'candidate_count_evaluated': 1,
        'search_runtime_ms': float(max((time.perf_counter() - search_t0) * 1000.0, 0.0)),
        'opponent_profile_used': opp_params.get('opponent_profile_used', 'balanced_field'),
        'portfolio_budget_feasible': bool(portfolio_budget_feasible) if pd.notna(portfolio_budget_feasible) else np.nan,
        'selected_portfolio_spend': float(sel_spend) if pd.notna(sel_spend) else np.nan,
    }


def add_priority_score(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ranking score that blends edge confidence with value efficiency,
    and a deal_quality metric that measures confidence-weighted fair value
    per dollar of market-competitive bid.
    """
    out = strategy_df.copy()
    if out.empty:
        return out
    adj = pd.to_numeric(out.get('adjusted_expected', np.nan), errors='coerce').fillna(0.0).clip(lower=0.0)
    p_edge = pd.to_numeric(out.get('prob_positive_edge', np.nan), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    p_dd = pd.to_numeric(out.get('prob_large_drawdown', np.nan), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    risk = pd.to_numeric(out.get('risk_penalty', np.nan), errors='coerce').fillna(0.0).clip(0.0, 1.0)

    z_adj = _zscore(np.log1p(adj))
    z_edge = _zscore(p_edge)
    z_dd = _zscore(p_dd)
    z_risk = _zscore(risk)

    out['priority_score'] = (
        0.50 * z_adj
        + 0.25 * z_edge
        - 0.10 * z_dd
        - 0.15 * z_risk
    )

    # Deal Quality: confidence-weighted fair value per market-competitive dollar.
    # Uses market_fair_bid (fair auction-$ value) / target_market_bid (risk-
    # adjusted market cost), weighted by empirical P(Edge) and (1-P(DD)).
    # In integer mode, uses the integerized market bid for genuine per-movie
    # variation from rounding/clipping.
    # DealQ > 1 = underpriced (confidence exceeds cost); < 1 = overpriced.
    integer_mode = bool(out.get('integer_bid_mode', pd.Series(False)).iloc[0]) if not out.empty else False
    if integer_mode and 'target_market_bid_int' in out.columns:
        mkt_cost = pd.to_numeric(out['target_market_bid_int'], errors='coerce')
    else:
        mkt_cost = pd.to_numeric(out.get('target_market_bid', np.nan), errors='coerce')
    mkt_fair = pd.to_numeric(out.get('market_fair_bid', np.nan), errors='coerce')
    p_edge_raw = pd.to_numeric(out.get('prob_positive_edge', np.nan), errors='coerce')

    confidence = p_edge * (1.0 - p_dd)
    fair_ratio = np.where(mkt_cost > 0, mkt_fair / mkt_cost, np.nan)
    raw_deal_q = pd.Series(fair_ratio, index=out.index) * confidence
    # Preserve NaN when probability data or market bid is missing.
    raw_deal_q = raw_deal_q.where(p_edge_raw.notna() & (mkt_cost > 0) & mkt_fair.notna(), np.nan)
    out['deal_quality'] = raw_deal_q

    return out


def strategy_diagnostics(strategy_df: pd.DataFrame) -> dict:
    """Basic Phase 1 validation diagnostics."""
    if strategy_df.empty:
        return {
            'rows': 0,
            'missing_current_price': 0,
            'missing_history': 0,
            'missing_probs': 0,
        }
    return {
        'rows': int(len(strategy_df)),
        'missing_current_price': int(pd.to_numeric(strategy_df.get('current_price', np.nan), errors='coerce').isna().sum()),
        'missing_history': int((pd.to_numeric(strategy_df.get('history_points', 0), errors='coerce').fillna(0) <= 1).sum()),
        'missing_probs': int(pd.to_numeric(strategy_df.get('prob_positive_edge', np.nan), errors='coerce').isna().sum()),
    }


def _top_tickers_by_score(strategy_df: pd.DataFrame, top_n: int = 10) -> list[str]:
    avail = strategy_df.copy()
    if 'owner' in avail.columns:
        avail = avail[avail['owner'].fillna('') == '']
    if avail.empty:
        return []
    avail['priority_score'] = pd.to_numeric(avail.get('priority_score', np.nan), errors='coerce')
    avail = avail.dropna(subset=['priority_score']).sort_values('priority_score', ascending=False)
    return avail['ticker'].astype(str).head(top_n).tolist()


def _top_overlap(base: list[str], other: list[str], top_n: int = 10) -> float:
    if top_n <= 0:
        return np.nan
    if not base or not other:
        return 0.0
    b = set(base[:top_n])
    o = set(other[:top_n])
    return float(len(b & o) / top_n)


def strategy_ranking_stability(
    strategy_df: pd.DataFrame,
    top_n: int = 10,
    perturb_std: float = 0.03,
    trials: int = 100,
    random_seed: int = 17,
) -> dict:
    """
    Estimate ranking stability under small random perturbations of priority score.
    Returns mean/min/max top-N overlap vs base ranking.
    """
    avail = strategy_df.copy()
    if 'owner' in avail.columns:
        avail = avail[avail['owner'].fillna('') == '']
    if avail.empty or 'priority_score' not in avail.columns:
        return {'top_n': top_n, 'trials': 0, 'mean_top_overlap': np.nan, 'min_top_overlap': np.nan, 'max_top_overlap': np.nan}

    scores = pd.to_numeric(avail['priority_score'], errors='coerce')
    avail = avail.loc[scores.notna()].copy()
    if avail.empty:
        return {'top_n': top_n, 'trials': 0, 'mean_top_overlap': np.nan, 'min_top_overlap': np.nan, 'max_top_overlap': np.nan}

    base_top = _top_tickers_by_score(avail, top_n=top_n)
    if not base_top:
        return {'top_n': top_n, 'trials': 0, 'mean_top_overlap': np.nan, 'min_top_overlap': np.nan, 'max_top_overlap': np.nan}

    score_std = float(pd.to_numeric(avail['priority_score'], errors='coerce').std(ddof=0))
    noise_sd = max(score_std * perturb_std, 1e-9)
    rng = np.random.default_rng(random_seed)

    overlaps = []
    for _ in range(max(trials, 1)):
        perturbed = avail.copy()
        noise = rng.normal(0.0, noise_sd, size=len(perturbed))
        perturbed['priority_score'] = pd.to_numeric(perturbed['priority_score'], errors='coerce') + noise
        top = _top_tickers_by_score(perturbed, top_n=top_n)
        overlaps.append(_top_overlap(base_top, top, top_n=top_n))

    return {
        'top_n': top_n,
        'trials': int(max(trials, 1)),
        'mean_top_overlap': float(np.mean(overlaps)),
        'min_top_overlap': float(np.min(overlaps)),
        'max_top_overlap': float(np.max(overlaps)),
    }


def _extract_features_at_time(prices: pd.Series) -> dict[str, float]:
    """Extract raw features from a price series ending at a given time step."""
    if len(prices) < 31:
        return {}
    return {
        'mom_7': _window_return(prices, 7),
        'mom_14': _window_return(prices, 14),
        'mom_30': _window_return(prices, 30),
        'trend_30': _log_slope(prices.tail(30)),
        'vol_30': float(prices.tail(31).pct_change().dropna().std(ddof=0)),
        'drawdown_30': _max_drawdown(prices.tail(30)),
    }


def _cross_sectional_zscore_signal(
    feature_rows: dict[str, dict[str, float]],
    settings: dict,
) -> dict[str, float]:
    """
    Compute z-scored weighted signal for each ticker across a cross-section,
    matching the dashboard's add_adjusted_expected logic.
    """
    feature_names = ['mom_7', 'mom_14', 'mom_30', 'trend_30', 'vol_30', 'drawdown_30']
    weights = {
        'mom_7': float(settings.get('strategy_w_mom_7', 0.20)),
        'mom_14': float(settings.get('strategy_w_mom_14', 0.20)),
        'mom_30': float(settings.get('strategy_w_mom_30', 0.25)),
        'trend_30': float(settings.get('strategy_w_trend_30', 0.20)),
        'vol_30': float(settings.get('strategy_w_vol_30', -0.15)),
        'drawdown_30': float(settings.get('strategy_w_drawdown_30', -0.20)),
    }

    tickers = list(feature_rows.keys())
    if len(tickers) < 2:
        return {t: 0.0 for t in tickers}

    # Build a DataFrame of raw features across the cross-section.
    raw = pd.DataFrame.from_dict(feature_rows, orient='index')
    signal = pd.Series(0.0, index=raw.index)
    for feat in feature_names:
        if feat not in raw.columns:
            continue
        col = pd.to_numeric(raw[feat], errors='coerce')
        z = _zscore(col)
        signal += weights.get(feat, 0.0) * z

    return {t: float(signal.get(t, 0.0)) for t in tickers}


def strategy_forward_check(
    movies_df: pd.DataFrame,
    settings: dict,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
) -> dict:
    """
    Forward check: cross-sectionally z-scored signal direction vs next-window
    return direction.  Matches the dashboard's add_adjusted_expected z-score
    methodology so the validation reflects the model actually in use.
    """
    if history_loader is None:
        import storage
        history_loader = storage.load_price_history

    horizon = max(int(settings.get('strategy_min_horizon_days', 7)), 1)
    tickers = movies_df['ticker'].dropna().astype(str).str.upper().unique().tolist() if not movies_df.empty else []

    # Pre-load all price histories.
    histories: dict[str, pd.Series] = {}
    for ticker in tickers:
        history_df = _clean_history_df(history_loader(ticker))
        if history_df.empty or len(history_df) < (31 + horizon + 1):
            continue
        histories[ticker] = history_df['price'].reset_index(drop=True)

    if not histories:
        return {
            'samples': 0,
            'tickers_used': 0,
            'horizon_days': int(horizon),
            'directional_accuracy': np.nan,
            'information_coefficient': np.nan,
        }

    # Find the common time-step range across all tickers.
    min_len = min(len(p) for p in histories.values())
    signals = []
    forwards = []

    for t in range(30, min_len - horizon):
        # Compute raw features for every ticker at this time step.
        feature_rows: dict[str, dict[str, float]] = {}
        for ticker, prices in histories.items():
            if t + 1 > len(prices):
                continue
            window = prices.iloc[:t + 1]
            feats = _extract_features_at_time(window)
            if feats:
                feature_rows[ticker] = feats

        if len(feature_rows) < 2:
            continue

        # Z-score across the cross-section and compute weighted signal.
        ticker_signals = _cross_sectional_zscore_signal(feature_rows, settings)

        # Record (signal, forward_return) pairs for each ticker at this step.
        for ticker, sig in ticker_signals.items():
            prices = histories[ticker]
            if t + horizon >= len(prices):
                continue
            p0 = float(prices.iloc[t])
            p1 = float(prices.iloc[t + horizon])
            if p0 <= 0:
                continue
            fwd = p1 / p0 - 1.0
            if pd.notna(sig) and pd.notna(fwd):
                signals.append(float(sig))
                forwards.append(float(fwd))

    if not signals:
        return {
            'samples': 0,
            'tickers_used': int(len(histories)),
            'horizon_days': int(horizon),
            'directional_accuracy': np.nan,
            'information_coefficient': np.nan,
        }

    s = np.array(signals, dtype=float)
    f = np.array(forwards, dtype=float)
    nonzero = (np.abs(s) > 1e-12) & (np.abs(f) > 1e-12)
    if nonzero.any():
        direction_match = (np.sign(s[nonzero]) == np.sign(f[nonzero]))
        directional_accuracy = float(direction_match.mean())
    else:
        directional_accuracy = np.nan

    if len(s) >= 2 and np.std(s) > 1e-12 and np.std(f) > 1e-12:
        ic = float(np.corrcoef(s, f)[0, 1])
    else:
        ic = np.nan

    return {
        'samples': int(len(s)),
        'tickers_used': int(len(histories)),
        'horizon_days': int(horizon),
        'directional_accuracy': directional_accuracy,
        'information_coefficient': ic,
    }


def strategy_preset_sensitivity(
    hsx_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    people_df: pd.DataFrame,
    settings: dict,
    budget_mode: str,
    custom_budget: float | None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    top_n: int = 10,
    risk_model: str = 'bootstrap',
    integer_bid_mode: bool = False,
    previous_bid: int = 0,
) -> pd.DataFrame:
    """
    Build a compact sensitivity table across presets.
    """
    rows = []
    presets = list(get_strategy_presets().keys())
    base_preset = str(settings.get('strategy_active_preset', 'balanced')).strip().lower()
    if base_preset not in presets:
        base_preset = 'balanced'

    validation_samples = max(int(settings.get('strategy_validation_bootstrap_samples', 120)), 30)

    top_map: dict[str, list[str]] = {}
    aggregate_map: dict[str, dict] = {}

    for p in presets:
        s = dict(settings)
        s['strategy_bootstrap_samples'] = validation_samples
        s['strategy_active_preset'] = p
        df_p, _, _ = build_strategy_dashboard(
            hsx_df=hsx_df,
            movies_df=movies_df,
            people_df=people_df,
            settings=s,
            preset=p,
            budget_mode=budget_mode,
            custom_budget=custom_budget,
            history_loader=history_loader,
            include_validation=False,
            risk_model=risk_model,
            integer_bid_mode=integer_bid_mode,
            previous_bid=previous_bid,
        )
        if not df_p.empty and 'owner' in df_p.columns:
            avail = df_p[df_p['owner'].fillna('') == ''].copy()
        else:
            avail = df_p.copy()
        top = _top_tickers_by_score(avail, top_n=top_n)
        top_map[p] = top
        aggregate_map[p] = {
            'avg_target_bid': float(pd.to_numeric(avail.get('target_bid', np.nan), errors='coerce').mean()) if not avail.empty else np.nan,
            'avg_prob_edge': float(pd.to_numeric(avail.get('prob_positive_edge', np.nan), errors='coerce').mean()) if not avail.empty else np.nan,
        }

    base_top = top_map.get(base_preset, [])
    for p in presets:
        rows.append({
            'preset': p,
            'top_overlap_vs_base': _top_overlap(base_top, top_map.get(p, []), top_n=top_n),
            'avg_target_bid': aggregate_map[p]['avg_target_bid'],
            'avg_prob_positive_edge': aggregate_map[p]['avg_prob_edge'],
        })
    return pd.DataFrame(rows)


def build_phase1_validation_report(
    strategy_df: pd.DataFrame,
    hsx_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    people_df: pd.DataFrame,
    settings: dict,
    budget_mode: str,
    custom_budget: float | None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    risk_model: str = 'bootstrap',
    integer_bid_mode: bool = False,
    previous_bid: int = 0,
) -> dict:
    """
    Phase 1 / Step 9 validation report.
    """
    return {
        'feature_completeness': strategy_diagnostics(strategy_df),
        'ranking_stability': strategy_ranking_stability(strategy_df, top_n=10, perturb_std=0.03, trials=60),
        'forward_check': strategy_forward_check(movies_df, settings, history_loader=history_loader),
        'preset_sensitivity': strategy_preset_sensitivity(
            hsx_df=hsx_df,
            movies_df=movies_df,
            people_df=people_df,
            settings=settings,
            budget_mode=budget_mode,
            custom_budget=custom_budget,
            history_loader=history_loader,
            top_n=10,
            risk_model=risk_model,
            integer_bid_mode=integer_bid_mode,
            previous_bid=previous_bid,
        ),
    }


def build_strategy_dashboard(
    hsx_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    people_df: pd.DataFrame,
    settings: dict | None = None,
    preset: str | None = None,
    budget_mode: str = 'personal',
    custom_budget: float | None = None,
    history_loader: Callable[[str], pd.DataFrame | None] | None = None,
    include_validation: bool = True,
    risk_model: str | None = None,
    integer_bid_mode: bool = False,
    previous_bid: int = 0,
    prob_bid_col: str | None = None,
    prob_value_col: str | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    End-to-end Phase 1 pipeline for dashboard + optimizer inputs.
    """
    base_settings = dict(settings or {})
    tuned_settings, resolved_preset = apply_strategy_preset(base_settings, preset)
    resolved_risk_model = (risk_model or tuned_settings.get('strategy_risk_model', 'bootstrap') or 'bootstrap').strip().lower()
    if resolved_risk_model not in ('bootstrap', 'student_t'):
        resolved_risk_model = 'bootstrap'
    tuned_settings['strategy_risk_model'] = resolved_risk_model
    tuned_settings['strategy_integer_bid_mode'] = bool(integer_bid_mode)
    tuned_settings['strategy_integer_prev_bid'] = int(max(previous_bid, 0))
    default_prob_bid_col, default_prob_value_col = resolve_probability_anchor_columns(
        tuned_settings.get('strategy_optimizer_cost_col', 'target_bid')
    )
    resolved_prob_bid_col = str(prob_bid_col or default_prob_bid_col).strip() or default_prob_bid_col
    resolved_prob_value_col = str(prob_value_col or default_prob_value_col).strip() or default_prob_value_col

    features = build_feature_table(
        hsx_df=hsx_df,
        movies_df=movies_df,
        history_loader=history_loader,
    )
    expected = add_adjusted_expected(features, settings=tuned_settings)
    bids, budget_info = add_bid_guidance(
        expected,
        people_df=people_df,
        settings=tuned_settings,
        budget_mode=budget_mode,
        custom_budget=custom_budget,
        integer_bid_mode=integer_bid_mode,
        previous_bid=previous_bid,
    )
    personal_cap = budget_info.get('personal_remaining', np.nan)
    tuned_settings['strategy_personal_budget_cap'] = (
        float(personal_cap) if pd.notna(personal_cap) and personal_cap > 0 else np.nan
    )
    league_cap = budget_info.get('league_remaining', np.nan)
    tuned_settings['strategy_league_remaining_budget'] = (
        float(league_cap) if pd.notna(league_cap) and league_cap > 0 else np.nan
    )
    probs = add_probability_of_edge(
        bids,
        bid_col=resolved_prob_bid_col,
        value_col=resolved_prob_value_col,
        settings=tuned_settings,
        history_loader=history_loader,
        risk_model=resolved_risk_model,
    )
    with_corr = add_correlation_proxy(probs, history_loader=history_loader)
    scored = add_priority_score(with_corr)

    meta = {
        'preset': resolved_preset,
        'risk_model': resolved_risk_model,
        'settings': tuned_settings,
        'budget_info': budget_info,
        'diagnostics': strategy_diagnostics(scored),
        'integer_bid_mode': bool(integer_bid_mode),
        'previous_bid': int(max(previous_bid, 0)),
        'prob_bid_col': resolved_prob_bid_col,
        'prob_value_col': resolved_prob_value_col,
    }
    if include_validation:
        meta['validation'] = build_phase1_validation_report(
            strategy_df=scored,
            hsx_df=hsx_df,
            movies_df=movies_df,
            people_df=people_df,
            settings=tuned_settings,
            budget_mode=budget_mode,
            custom_budget=custom_budget,
            history_loader=history_loader,
            risk_model=resolved_risk_model,
            integer_bid_mode=integer_bid_mode,
            previous_bid=previous_bid,
        )
    return scored, meta, tuned_settings
