import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure sibling modules are importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import storage
import calculations
from history_viewer import show_price_history_viewer
from scraper import scrape_hsx, scrape_release_dates, scrape_price_history


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prompt(msg='> '):
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        print('\nExiting.')
        sys.exit(0)


def _pause():
    _prompt('\nPress Enter to continue...')


def _menu(title, options):
    print(f'\n=== {title} ===')
    for key, desc in options.items():
        print(f'  [{key}] {desc}')
    return _prompt('Choice: ')


def _clean_name(name, ticker):
    """Strip 'TICKER: ' prefix from HSX movie names for cleaner display."""
    if not isinstance(name, str) or not name:
        return '(not found in HSX)'
    prefix = f'{ticker}: '
    return name[len(prefix):] if name.startswith(prefix) else name


def _fmt(val, width=7, decimals=2):
    if pd.isna(val):
        return ' ' * (width - 3) + 'N/A'
    return f'{val:{width}.{decimals}f}'


def _fmt_delta(val, width=7):
    if pd.isna(val):
        return ' ' * (width - 3) + 'N/A'
    return f'{val:+{width}.2f}'


def _fmt_prob(val, width=7):
    """Format a probability as a percentage string."""
    if pd.isna(val):
        return ' ' * (width - 3) + 'N/A'
    return f'{100.0 * val:{width - 1}.1f}%'


def _fmt_date(val) -> str:
    """Format a YYYY-MM-DD string as MM/DD/YY for display (8 chars)."""
    if not val or not isinstance(val, str) or not val.strip():
        return '       -'
    try:
        return pd.to_datetime(val).strftime('%m/%d/%y')
    except Exception:
        return '       -'


def _load_pool():
    """Load and compute pool stats. Returns (pool_df, people_df) or (None, people_df) on failure."""
    movies_df = storage.load_movies()
    people_df = storage.load_people()
    cache = storage.load_cache()
    if movies_df.empty:
        print('No movies in pool. Add tickers first (Manage Movie Pool).')
        return None, people_df
    if cache.empty:
        print('No HSX data cached. Please scrape first (option 1).')
        return None, people_df
    pool = calculations.compute_pool_stats(cache, movies_df, people_df)
    return pool, people_df


def _ask_sort_order() -> str:
    """Prompt for sort preference. Returns 'propval' or 'release_date'."""
    print('\nSort by:')
    print('  [1] Proportional value, descending (default)')
    print('  [2] Release date, ascending')
    choice = _prompt('Choice (Enter for default): ')
    return 'release_date' if choice == '2' else 'propval'


def _apply_bid_estimate(pool):
    """Attempt to apply bid_estimate column from settings. Returns (pool, has_bid)."""
    settings = storage.load_settings()
    ref_ticker = settings.get('bid_ref_ticker', '')
    ref_bid_price = settings.get('bid_ref_price', 0.0)
    if ref_ticker and ref_bid_price:
        try:
            pool = calculations.add_bid_estimate(pool, ref_ticker, ref_bid_price)
            return pool, True
        except ValueError as e:
            print(f'  [Note] BidEstimate unavailable: {e}')
    return pool, False


# ── Scraping ──────────────────────────────────────────────────────────────────

def _fetch_price_history():
    """Fetch and save price history CSVs for all pool movies."""
    movies_df = storage.load_movies()
    if movies_df.empty:
        print('No movies in pool.')
        return

    tickers = movies_df['ticker'].tolist()
    total = len(tickers)
    print(f'\nFetching price history for {total} movies...')
    print('Results are saved as each movie completes.\n')

    succeeded = []
    failed = {}

    for i, ticker in enumerate(tickers, 1):
        print(f'  [{i}/{total}] {ticker:<10} ...', end='', flush=True)
        df, status = scrape_price_history(ticker)
        if df is not None:
            storage.save_price_history(ticker, df)
            succeeded.append(ticker)
            print(f'  OK  ({len(df)} rows, via {status})')
        else:
            failed[ticker] = status
            print(f'  FAIL  {status}')

    print(f'\nDone. Saved history for {len(succeeded)}/{total} movie(s).')
    if failed:
        print('\nFailed tickers:')
        for t, status in failed.items():
            print(f'  {t:<10}  {status}')
        print('\nCheck that the ticker is valid and the HSX page loads correctly.')


def menu_scrape():
    cache = storage.load_cache()
    if not cache.empty:
        ts = cache['scraped_at'].iloc[0]
        print(f'Current cache: {len(cache)} movies, last scraped {ts}')
    else:
        print('No cache found.')

    choice = _menu('Scrape Options', {
        '1': 'Scrape fresh prices from HSX (names, current price)',
        '2': 'Fetch price history for all pool movies',
        '0': 'Back',
    })
    if choice == '1':
        print('Scraping HSX (this takes ~30–60 seconds)...')
        df = scrape_hsx()
        storage.save_cache(df)
        print(f'Done. Scraped {len(df)} movies.')
    elif choice == '2':
        _fetch_price_history()
    _pause()


# ── Movie Views ───────────────────────────────────────────────────────────────

def menu_all_movies():
    pool, _ = _load_pool()
    if pool is None:
        _pause()
        return

    sort_order = _ask_sort_order()
    pool, has_bid = _apply_bid_estimate(pool)

    if sort_order == 'release_date':
        pool['_date_sort'] = pd.to_datetime(pool.get('release_date', ''), errors='coerce')
        df = pool.sort_values('_date_sort', ascending=True, na_position='last')
        show_date = True
    else:
        df = pool.sort_values('full_prop_val', ascending=False, na_position='last')
        show_date = False

    header = (
        f"{'Ticker':<8}  {'Movie':<38}  {'Price':>7}  {'Δ':>7}  "
        f"{'FullPV%':>7}  {'Owner':<14}"
    )
    if show_date:
        header += f"  {'Release':>8}"
    if has_bid:
        header += f"  {'BidEst':>7}"

    print(f'\n=== All Pool Movies ({len(df)}) ===')
    print(header)
    print('─' * len(header))

    for _, row in df.iterrows():
        movie = _clean_name(row.get('name'), row['ticker'])[:38]
        owner = str(row['owner']) if row.get('owner') else '-'
        line = (
            f"{row['ticker']:<8}  {movie:<38}  "
            f"{_fmt(row.get('current_price'))}  "
            f"{_fmt_delta(row.get('price_change'))}  "
            f"{_fmt(row.get('full_prop_val'), 7)}  "
            f"{owner:<14}"
        )
        if show_date:
            line += f"  {_fmt_date(row.get('release_date', ''))}"
        if has_bid:
            line += f"  {_fmt(row.get('bid_estimate'))}"
        print(line)

    _pause()


def menu_available_movies():
    pool, people_df = _load_pool()
    if pool is None:
        _pause()
        return

    sort_order = _ask_sort_order()
    pool, has_bid = _apply_bid_estimate(pool)
    avail = pool[pool['owner'] == ''].copy()
    total_remaining = float(people_df['remaining_money'].sum()) if not people_df.empty else 0.0

    if sort_order == 'release_date':
        avail['_date_sort'] = pd.to_datetime(avail.get('release_date', ''), errors='coerce')
        avail = avail.sort_values('_date_sort', ascending=True, na_position='last')
        show_date = True
    else:
        avail = avail.sort_values('avail_prop_val', ascending=False, na_position='last')
        show_date = False

    header = (
        f"{'Ticker':<8}  {'Movie':<38}  {'Price':>7}  {'Δ':>7}  "
        f"{'AvailPV%':>8}  {'BrkEvAvail':>10}  {'FullPV%':>7}  {'BrkEvFull':>9}"
    )
    if show_date:
        header += f"  {'Release':>8}"
    if has_bid:
        header += f"  {'BidEst':>7}"

    print(f'\n=== Available Movies ({len(avail)}) — Total remaining cash: ${total_remaining:.2f} ===')
    print(header)
    print('─' * len(header))

    for _, row in avail.iterrows():
        movie = _clean_name(row.get('name'), row['ticker'])[:38]
        line = (
            f"{row['ticker']:<8}  {movie:<38}  "
            f"{_fmt(row.get('current_price'))}  "
            f"{_fmt_delta(row.get('price_change'))}  "
            f"{_fmt(row.get('avail_prop_val'), 8)}  "
            f"{_fmt(row.get('break_even_avail'), 10)}  "
            f"{_fmt(row.get('full_prop_val'), 7)}  "
            f"{_fmt(row.get('break_even_full'), 9)}"
        )
        if show_date:
            line += f"  {_fmt_date(row.get('release_date', ''))}"
        if has_bid:
            line += f"  {_fmt(row.get('bid_estimate'))}"
        print(line)

    _pause()


# ── Assign / Unassign ─────────────────────────────────────────────────────────

def menu_assign():
    movies_df = storage.load_movies()
    people_df = storage.load_people()
    cache = storage.load_cache()

    if movies_df.empty:
        print('No movies in pool.')
        _pause()
        return
    if people_df.empty:
        print('No people configured. Add people first (Manage People).')
        _pause()
        return

    unassigned = movies_df[movies_df['owner'] == ''].copy()
    if unassigned.empty:
        print('All movies are already assigned.')
        _pause()
        return
    unassigned['_date_sort'] = pd.to_datetime(unassigned.get('release_date', ''), errors='coerce')
    unassigned = unassigned.sort_values('_date_sort', ascending=True, na_position='last')
    unassigned = unassigned.drop(columns=['_date_sort'])

    if not cache.empty:
        unassigned = unassigned.merge(
            cache[['ticker', 'name', 'current_price']], on='ticker', how='left'
        )

    print('\n=== Unassigned Movies (release-date order) ===')
    for idx, row in enumerate(unassigned.itertuples(index=False), start=1):
        ticker = str(getattr(row, 'ticker', ''))
        name = getattr(row, 'name', '')
        current_price = getattr(row, 'current_price', np.nan)
        release_date = getattr(row, 'release_date', '')
        movie = _clean_name(name, ticker)
        price = f"  ${current_price:.2f}" if pd.notna(current_price) else ''
        rel = _fmt_date(release_date)
        print(f"  [{idx:>2}] {ticker:<8}  {movie}{price}  rel:{rel}")

    raw_pick = _prompt('\nTicker to assign [# or ticker, blank to cancel]: ').strip()
    if not raw_pick:
        return
    if raw_pick.isdigit():
        pick = int(raw_pick)
        if 1 <= pick <= len(unassigned):
            ticker = str(unassigned.iloc[pick - 1]['ticker']).upper()
        else:
            print('Invalid selection index.')
            _pause()
            return
    else:
        ticker = raw_pick.upper()

    valid_unassigned = set(unassigned['ticker'].astype(str).str.upper().tolist())
    if ticker not in valid_unassigned:
        print(f"'{ticker}' is not currently in the remaining unassigned pool.")
        _pause()
        return

    print('\n=== People ===')
    for _, p in people_df.iterrows():
        print(f"  {p['name']}  (remaining: ${p['remaining_money']:.2f})")

    person = _prompt('\nPerson name (blank to cancel): ')
    if not person:
        return
    if person not in people_df['name'].values:
        print(f"'{person}' not found. Check spelling.")
        _pause()
        return

    price_raw = _prompt('Final winning price (optional, Enter to skip): $').strip()
    final_price = None
    if price_raw:
        try:
            final_price = float(price_raw)
            if final_price <= 0:
                print('Final price must be > 0 when provided.')
                _pause()
                return
        except ValueError:
            print('Invalid final price.')
            _pause()
            return

    ok, msg, _ = storage.assign_movie(
        ticker=ticker,
        winner=person,
        final_price=final_price,
        source='cli_assign',
    )
    print(msg)
    _pause()


def menu_unassign():
    movies_df = storage.load_movies()
    assigned = movies_df[movies_df['owner'] != ''].copy()

    if assigned.empty:
        print('No movies are currently assigned.')
        _pause()
        return

    cache = storage.load_cache()
    if not cache.empty:
        assigned = assigned.merge(cache[['ticker', 'name']], on='ticker', how='left')

    print('\n=== Assigned Movies ===')
    for _, row in assigned.iterrows():
        movie = _clean_name(row.get('name'), row['ticker'])
        print(f"  {row['ticker']:<8}  {movie:<38}  → {row['owner']}")

    ticker = _prompt('\nTicker to unassign (blank to cancel): ').upper()
    if not ticker:
        return
    if ticker not in movies_df['ticker'].values:
        print(f"'{ticker}' not in pool.")
        _pause()
        return

    ok, msg, _ = storage.unassign_movie(
        ticker=ticker,
        source='cli_unassign',
        restore_budget=True,
    )
    print(msg)
    _pause()


# ── Person Summary ────────────────────────────────────────────────────────────

def menu_person_summary():
    pool, people_df = _load_pool()
    if pool is None:
        _pause()
        return
    if people_df.empty:
        print('No people configured.')
        _pause()
        return

    summary = calculations.compute_person_summary(pool, people_df)

    header = (
        f"{'Name':<14}  {'Start$':>7}  {'Remain$':>8}  "
        f"{'Movies':>6}  {'PropVal%':>8}  {'BudgetEq$':>9}"
    )
    print('\n=== Person Summary ===')
    print(header)
    print('─' * len(header))
    for _, row in summary.iterrows():
        print(
            f"{row['name']:<14}  {row['starting_money']:>7.2f}  "
            f"{row['remaining_money']:>8.2f}  {int(row['movie_count']):>6}  "
            f"{row['total_prop_val']:>8.2f}  {row['budget_equiv']:>9.2f}"
        )

    # Drill-down
    name = _prompt('\nEnter a name for movie detail (blank to go back): ')
    if not name:
        return
    if name not in people_df['name'].values:
        print(f"'{name}' not found.")
        _pause()
        return

    owned = pool[pool['owner'] == name].sort_values('full_prop_val', ascending=False, na_position='last')
    person = people_df[people_df['name'] == name].iloc[0]

    detail_header = (
        f"{'Ticker':<8}  {'Movie':<38}  {'Price':>7}  {'FullPV%':>7}  {'BudgetEq$':>9}"
    )
    print(f"\n=== {name}'s Movies ({len(owned)}) ===")
    print(detail_header)
    print('─' * len(detail_header))
    for _, row in owned.iterrows():
        movie = _clean_name(row.get('name'), row['ticker'])[:38]
        pv = row.get('full_prop_val', np.nan)
        equiv = (pv / 100.0 * person['starting_money']) if pd.notna(pv) else np.nan
        print(
            f"{row['ticker']:<8}  {movie:<38}  "
            f"{_fmt(row.get('current_price'))}  "
            f"{_fmt(pv, 7)}  "
            f"{_fmt(equiv, 9)}"
        )

    _pause()


# ── Manage People ─────────────────────────────────────────────────────────────

def menu_people():
    while True:
        choice = _menu('Manage People', {
            '1': 'List people',
            '2': 'Add person',
            '3': 'Update remaining money',
            '4': 'Remove person',
            '0': 'Back',
        })
        if choice == '0':
            break

        elif choice == '1':
            df = storage.load_people()
            if df.empty:
                print('No people configured.')
            else:
                print('\n' + df.to_string(index=False))
            _pause()

        elif choice == '2':
            name = _prompt('Name: ')
            if not name:
                continue
            df = storage.load_people()
            if name in df['name'].values:
                print(f"'{name}' already exists.")
                _pause()
                continue
            try:
                starting = float(_prompt('Starting money: $'))
            except ValueError:
                print('Invalid amount.')
                _pause()
                continue
            rem_input = _prompt(f'Remaining money (press Enter to default to {starting}): $')
            if rem_input == '':
                remaining = starting
            else:
                try:
                    remaining = float(rem_input)
                except ValueError:
                    print('Invalid amount.')
                    _pause()
                    continue
            new_row = pd.DataFrame([{
                'name': name,
                'starting_money': starting,
                'remaining_money': remaining,
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            storage.save_people(df)
            print(f"Added {name} (start: ${starting:.2f}, remaining: ${remaining:.2f}).")
            _pause()

        elif choice == '3':
            df = storage.load_people()
            if df.empty:
                print('No people configured.')
                _pause()
                continue
            print('\n' + df[['name', 'remaining_money']].to_string(index=False))
            name = _prompt('\nName to update: ')
            if name not in df['name'].values:
                print(f"'{name}' not found.")
                _pause()
                continue
            try:
                amount = float(_prompt('New remaining money: $'))
            except ValueError:
                print('Invalid amount.')
                _pause()
                continue
            df.loc[df['name'] == name, 'remaining_money'] = amount
            storage.save_people(df)
            print(f"Updated {name}'s remaining money to ${amount:.2f}.")
            _pause()

        elif choice == '4':
            df = storage.load_people()
            if df.empty:
                print('No people configured.')
                _pause()
                continue
            print('\n' + df['name'].to_string(index=False))
            name = _prompt('\nName to remove: ')
            if name not in df['name'].values:
                print(f"'{name}' not found.")
                _pause()
                continue
            df = df[df['name'] != name]
            storage.save_people(df)
            print(f"Removed {name}.")
            _pause()


# ── Manage Movie Pool ─────────────────────────────────────────────────────────

def menu_movies():
    while True:
        choice = _menu('Manage Movie Pool', {
            '1': 'List pool',
            '2': 'Add ticker',
            '3': 'Remove ticker',
            '4': 'Bulk add tickers (comma-separated)',
            '5': 'Set release dates manually',
            '6': 'Fetch release dates from HSX',
            '0': 'Back',
        })
        if choice == '0':
            break

        elif choice == '1':
            movies_df = storage.load_movies()
            cache = storage.load_cache()
            if movies_df.empty:
                print('Pool is empty.')
            else:
                display = movies_df.copy()
                if not cache.empty:
                    display = display.merge(
                        cache[['ticker', 'name', 'current_price']], on='ticker', how='left'
                    )
                print()
                for _, row in display.iterrows():
                    movie = _clean_name(row.get('name'), row['ticker'])
                    price = f"  ${row['current_price']:.2f}" if pd.notna(row.get('current_price')) else ''
                    owner = f"  → {row['owner']}" if row.get('owner') else ''
                    print(f"  {row['ticker']:<8}  {movie}{price}{owner}")
            _pause()

        elif choice == '2':
            ticker = _prompt('Ticker to add: ').upper()
            if not ticker:
                continue
            movies_df = storage.load_movies()
            if ticker in movies_df['ticker'].values:
                print(f"'{ticker}' already in pool.")
                _pause()
                continue
            new_row = pd.DataFrame([{'ticker': ticker, 'owner': ''}])
            movies_df = pd.concat([movies_df, new_row], ignore_index=True)
            storage.save_movies(movies_df)
            print(f"Added {ticker}.")
            _pause()

        elif choice == '3':
            movies_df = storage.load_movies()
            if movies_df.empty:
                print('Pool is empty.')
                _pause()
                continue
            ticker = _prompt('Ticker to remove: ').upper()
            if ticker not in movies_df['ticker'].values:
                print(f"'{ticker}' not in pool.")
                _pause()
                continue
            movies_df = movies_df[movies_df['ticker'] != ticker]
            storage.save_movies(movies_df)
            print(f"Removed {ticker}.")
            _pause()

        elif choice == '4':
            raw = _prompt('Tickers (comma-separated): ')
            tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
            if not tickers:
                continue
            movies_df = storage.load_movies()
            existing = set(movies_df['ticker'].values)
            new_tickers = [t for t in tickers if t not in existing]
            dupes = [t for t in tickers if t in existing]
            if dupes:
                print(f"Already in pool (skipped): {', '.join(dupes)}")
            if new_tickers:
                new_rows = pd.DataFrame([{'ticker': t, 'owner': ''} for t in new_tickers])
                movies_df = pd.concat([movies_df, new_rows], ignore_index=True)
                storage.save_movies(movies_df)
                print(f"Added: {', '.join(new_tickers)}")
            _pause()

        elif choice == '5':
            menu_set_release_dates()

        elif choice == '6':
            menu_fetch_release_dates()


# ── Release Dates ─────────────────────────────────────────────────────────────

def menu_fetch_release_dates():
    """Scrape release dates from HSX detail pages for all pool movies."""
    movies_df = storage.load_movies()
    if movies_df.empty:
        print('Pool is empty.')
        _pause()
        return

    tickers = movies_df['ticker'].tolist()
    print(f'\nFetching release dates for {len(tickers)} movies from HSX...')
    dates = scrape_release_dates(tickers)

    if not dates:
        print('No release dates found.')
        _pause()
        return

    for ticker, date in dates.items():
        movies_df.loc[movies_df['ticker'] == ticker, 'release_date'] = date

    storage.save_movies(movies_df)
    missing = [t for t in tickers if t not in dates]
    print(f'Updated {len(dates)} date(s).')
    if missing:
        print(f'No date found for: {", ".join(missing)}')
    _pause()


def menu_set_release_dates():
    """Bulk-enter release dates for pool movies."""
    print('\nEnter one date per line in the format:  TICKER YYYY-MM-DD')
    print('Example:  AVRND 2025-05-02')
    print('Press Enter on a blank line when done.\n')

    movies_df = storage.load_movies()
    if movies_df.empty:
        print('Pool is empty.')
        _pause()
        return

    changes = 0
    while True:
        line = _prompt()
        if not line:
            break
        parts = line.split()
        if len(parts) != 2:
            print('  Expected: TICKER YYYY-MM-DD')
            continue
        ticker, date_str = parts[0].upper(), parts[1]
        if ticker not in movies_df['ticker'].values:
            print(f"  '{ticker}' not in pool — skipped.")
            continue
        try:
            pd.to_datetime(date_str)
        except Exception:
            print(f"  Invalid date '{date_str}'. Use YYYY-MM-DD (e.g. 2025-07-04).")
            continue
        movies_df.loc[movies_df['ticker'] == ticker, 'release_date'] = date_str
        changes += 1
        print(f'  Set {ticker} → {date_str}')

    if changes:
        storage.save_movies(movies_df)
        print(f'\nSaved {changes} date(s).')
    else:
        print('No changes made.')
    _pause()


# ── Bid Reference ─────────────────────────────────────────────────────────────

def menu_bid_reference():
    settings = storage.load_settings()
    cur_ticker = settings.get('bid_ref_ticker', '')
    cur_price = settings.get('bid_ref_price', 0.0)

    if cur_ticker:
        print(f'\nCurrent reference: {cur_ticker} @ ${cur_price:.2f}')
    else:
        print('\nNo bid reference set.')

    print('Set a reference movie and your intended bid for it.')
    print('All other movies will scale proportionally by HSX price.')

    ticker = _prompt('Reference ticker (blank to clear): ').upper()
    if not ticker:
        settings.update({'bid_ref_ticker': '', 'bid_ref_price': 0.0})
        storage.save_settings(settings)
        print('Bid reference cleared.')
        _pause()
        return

    try:
        price = float(_prompt(f'Your bid price for {ticker}: $'))
    except ValueError:
        print('Invalid price.')
        _pause()
        return

    settings.update({'bid_ref_ticker': ticker, 'bid_ref_price': price})
    storage.save_settings(settings)
    print(f'Set bid reference: {ticker} @ ${price:.2f}')
    _pause()


# ── Price History Viewer ───────────────────────────────────────────────────────

def menu_view_price_history():
    shown = show_price_history_viewer()
    if shown:
        print('Closed history viewer.')
    _pause()


def _remaining_tickers_by_release() -> pd.DataFrame:
    """Return unassigned movies ordered by release date (nearest first, missing last)."""
    movies_df = storage.load_movies()
    if movies_df.empty:
        return pd.DataFrame(columns=['ticker', 'release_date'])
    out = movies_df[movies_df['owner'].fillna('') == ''][['ticker', 'release_date']].copy()
    out['ticker'] = out['ticker'].astype(str).str.upper()
    out['_release_sort'] = pd.to_datetime(out['release_date'], errors='coerce')
    out = out.sort_values('_release_sort', ascending=True, na_position='last')
    return out.drop(columns=['_release_sort'])


def menu_draft_state():
    """Manage persistent draft context and assignment history across sessions."""
    storage.ensure_auction_state_seeded()

    while True:
        state = storage.load_auction_state()
        cur_movie = str(state.get('current_movie', '') or '')
        cur_bid = int(max(state.get('current_bid', 0), 0))
        assigned_count = len(state.get('assignments', {}))

        title = (
            f"Draft State  [current={cur_movie or '-'}  bid={cur_bid}  "
            f"assigned={assigned_count}]"
        )
        choice = _menu(title, {
            '1': 'View current summary',
            '2': 'Set current movie + current bid',
            '3': 'Clear current movie context',
            '4': 'List assigned movies',
            '5': 'View assignment history',
            '0': 'Back',
        })
        if choice == '0':
            break

        if choice == '1':
            print('\n=== Draft State Summary ===')
            print(f"Current movie: {cur_movie or '-'}")
            print(f'Current bid: {cur_bid}')
            print(f'Assigned movies tracked: {assigned_count}')
            print(f"Last updated: {state.get('updated_at', '-')}")
            _pause()
            continue

        if choice == '2':
            remaining = _remaining_tickers_by_release()
            ticker = ''
            if remaining.empty:
                print('No remaining unassigned movies in pool.')
            else:
                print('\nRemaining movies (ordered by release date):')
                for idx, row in enumerate(remaining.itertuples(index=False), start=1):
                    rel = _fmt_date(getattr(row, 'release_date', ''))
                    print(f'  [{idx:>2}] {row.ticker:<8}  {rel}')
                sel = _prompt('\nSelect current movie [# or ticker, Enter to clear]: ').strip()
                if sel:
                    if sel.isdigit():
                        pick = int(sel)
                        if 1 <= pick <= len(remaining):
                            ticker = str(remaining.iloc[pick - 1]['ticker']).upper()
                        else:
                            print('Invalid selection index; leaving movie context unchanged.')
                            ticker = cur_movie
                    else:
                        raw_ticker = sel.upper()
                        valid = set(remaining['ticker'].astype(str).str.upper().tolist())
                        if raw_ticker in valid:
                            ticker = raw_ticker
                        else:
                            print(f"'{raw_ticker}' is not in remaining pool; leaving movie context unchanged.")
                            ticker = cur_movie
            raw_bid = _prompt(f'Current bid integer (default {cur_bid}): ').strip()
            next_bid = cur_bid
            if raw_bid:
                try:
                    next_bid = max(int(float(raw_bid)), 0)
                except ValueError:
                    print('Invalid bid; keeping previous value.')
            storage.set_current_auction_context(ticker=ticker, current_bid=next_bid)
            print(f'Updated context: movie={ticker or "-"} bid={next_bid}')
            _pause()
            continue

        if choice == '3':
            storage.set_current_auction_context('', 0)
            print('Cleared current movie context.')
            _pause()
            continue

        if choice == '4':
            assigned = storage.get_assigned_movies_df()
            if assigned.empty:
                print('No assigned movies tracked yet.')
                _pause()
                continue
            cache = storage.load_cache()
            if not cache.empty:
                assigned = assigned.merge(cache[['ticker', 'name']], on='ticker', how='left')
            print('\n=== Assigned Movies (Tracked) ===')
            for _, row in assigned.iterrows():
                movie = _clean_name(row.get('name'), row['ticker'])[:38]
                p = pd.to_numeric(pd.Series([row.get('final_price')]), errors='coerce').iloc[0]
                p_txt = f"  ${float(p):.2f}" if pd.notna(p) else '  $-'
                print(
                    f"{row['ticker']:<8}  {movie:<38}  -> {row.get('winner', ''):<12}"
                    f"{p_txt}  {row.get('assigned_at', '')}"
                )
            _pause()
            continue

        if choice == '5':
            hist = storage.get_assignment_history_df()
            if hist.empty:
                print('No assignment history available.')
                _pause()
                continue
            print('\n=== Assignment History (latest first) ===')
            for _, row in hist.head(50).iterrows():
                p = pd.to_numeric(pd.Series([row.get('final_price')]), errors='coerce').iloc[0]
                p_txt = f"${float(p):.2f}" if pd.notna(p) else '-'
                print(
                    f"{row.get('timestamp', '')}  "
                    f"{str(row.get('event_type', '')).upper():<8}  "
                    f"{row.get('ticker', ''):<8}  "
                    f"{row.get('winner', ''):<12}  "
                    f"{p_txt:<8}  "
                    f"{row.get('source', '')}"
                )
            _pause()
            continue


def _prompt_strategy_preset(settings: dict) -> str:
    """Prompt for strategy preset selection."""
    current = str(settings.get('strategy_active_preset', 'balanced')).strip().lower()
    print('\nStrategy preset:')
    print('  [1] Conservative')
    print('  [2] Balanced (default)')
    print('  [3] Aggressive')
    print(f'Current/default: {current or "balanced"}')
    choice = _prompt('Preset choice (Enter to keep default): ').strip()
    mapping = {'1': 'conservative', '2': 'balanced', '3': 'aggressive'}
    return mapping.get(choice, current if current in mapping.values() else 'balanced')


def _prompt_load_strategy_profile(settings: dict) -> tuple[dict, str]:
    """Optional load of a named saved strategy profile."""
    profiles = storage.list_strategy_profile_names()
    if not profiles:
        print('\nNo saved strategy profiles found.')
        return settings, ''

    print('\nSaved strategy profiles:')
    for idx, name in enumerate(profiles, start=1):
        print(f'  [{idx}] {name}')
    raw = _prompt('Load profile [# or name, Enter to skip]: ').strip()
    if not raw:
        return settings, ''

    selected = ''
    if raw.isdigit():
        pick = int(raw)
        if 1 <= pick <= len(profiles):
            selected = profiles[pick - 1]
    else:
        selected = raw

    if not selected:
        print('Invalid profile selection. Continuing without loading.')
        return settings, ''

    payload = storage.load_strategy_profile(selected)
    if not payload:
        print(f"Profile '{selected}' was not found or is empty.")
        return settings, ''

    merged = dict(settings)
    merged.update(payload)
    print(f"Loaded strategy profile: {selected}")
    return merged, selected


def _prompt_save_strategy_profile(settings: dict):
    """Optional save of current strategy settings as a named profile."""
    choice = _prompt('\nSave current strategy settings as profile? [y/N]: ').strip().lower()
    if choice != 'y':
        return
    name = _prompt('Profile name: ').strip()
    ok, msg = storage.save_strategy_profile(name, settings)
    print(msg)


def _prompt_budget_inputs(people_df: pd.DataFrame, settings: dict) -> tuple[str, float | None]:
    """Prompt for budget basis inputs used by the strategy dashboard."""
    cur_mode = str(settings.get('strategy_budget_mode_preference', 'personal')).strip().lower()
    if cur_mode not in ('personal', 'league', 'custom'):
        cur_mode = 'personal'
    cur_custom = pd.to_numeric(pd.Series([settings.get('strategy_custom_budget_amount', 200.0)]), errors='coerce').iloc[0]
    if pd.isna(cur_custom) or float(cur_custom) <= 0:
        cur_custom = 200.0

    print('\nBudget basis:')
    print('  [1] Personal remaining budget')
    print('  [2] Total league remaining budget')
    print('  [3] Custom budget amount')
    print(f'Current/default: {cur_mode}')
    mode_choice = _prompt('Choice (Enter to keep default): ').strip()

    if mode_choice == '':
        if cur_mode == 'league':
            settings['strategy_budget_mode_preference'] = 'league'
            return 'league', None
        if cur_mode == 'custom':
            settings['strategy_budget_mode_preference'] = 'custom'
            settings['strategy_custom_budget_amount'] = float(cur_custom)
            return 'personal', float(cur_custom)
        settings['strategy_budget_mode_preference'] = 'personal'
        return 'personal', None

    if mode_choice == '2':
        settings['strategy_budget_mode_preference'] = 'league'
        return 'league', None

    if mode_choice == '3':
        raw = _prompt(f'Custom budget amount [{float(cur_custom):.2f}]: $').strip()
        try:
            amount = float(cur_custom) if raw == '' else float(raw)
            if amount > 0:
                settings['strategy_budget_mode_preference'] = 'custom'
                settings['strategy_custom_budget_amount'] = float(amount)
                return 'personal', amount
        except ValueError:
            pass
        print('Invalid custom budget. Falling back to automatic personal/league basis.')
        settings['strategy_budget_mode_preference'] = 'personal'
        return 'personal', None

    # Personal mode
    settings['strategy_budget_mode_preference'] = 'personal'
    if people_df.empty:
        raw = _prompt(f'No people configured. Enter your budget amount [{float(cur_custom):.2f}]: $').strip()
        try:
            amount = float(cur_custom) if raw == '' else float(raw)
            if amount > 0:
                settings['strategy_custom_budget_amount'] = float(amount)
                return 'personal', amount
        except ValueError:
            pass
        print('Invalid budget. Dashboard will use league fallback if available.')
        return 'personal', None

    print('\nPeople and remaining budgets:')
    for _, p in people_df.iterrows():
        print(f"  {p['name']:<14}  ${float(p['remaining_money']):.2f}")

    cur_name = str(settings.get('strategy_user_name', '')).strip()
    if cur_name:
        print(f'Current personal budget owner: {cur_name}')
    name = _prompt('Use which person as your budget basis? (Enter to keep current): ').strip()
    if name:
        if name in people_df['name'].values:
            settings['strategy_user_name'] = name
        else:
            print(f"'{name}' not found. Keeping existing/default basis.")

    return 'personal', None


def _prompt_optimizer_cost_col(settings: dict) -> str:
    """Choose which cost estimate the optimizer should use."""
    current = str(settings.get('strategy_optimizer_cost_col', 'target_bid')).strip().lower()
    if current not in ('target_bid', 'market_fair_bid', 'current_price'):
        current = 'target_bid'
    print('\nOptimizer cost basis:')
    print('  [1] Target bid (recommended for auction-scale optimization)')
    print('  [2] Market fair bid (league-scale diagnostic)')
    print('  [3] Current HSX price (diagnostic only; not auction-scale)')
    print(f'Current/default: {current}')
    choice = _prompt('Choice (Enter to keep default): ').strip()
    if choice == '':
        settings['strategy_optimizer_cost_col'] = current
        return current
    if choice == '2':
        settings['strategy_optimizer_cost_col'] = 'market_fair_bid'
        return 'market_fair_bid'
    if choice == '3':
        settings['strategy_optimizer_cost_col'] = 'current_price'
        return 'current_price'
    settings['strategy_optimizer_cost_col'] = 'target_bid'
    return 'target_bid'


def _prompt_risk_model(settings: dict) -> str:
    """Choose risk model for probability and simulation layers."""
    current = str(settings.get('strategy_risk_model', 'bootstrap')).strip().lower()
    print('\nRisk model:')
    print('  [1] Bootstrap (default)')
    print('  [2] Student-t')
    print(f'Current/default: {current or "bootstrap"}')
    choice = _prompt('Choice (Enter to keep default): ').strip()
    mapping = {'1': 'bootstrap', '2': 'student_t'}
    return mapping.get(choice, current if current in mapping.values() else 'bootstrap')


def _prompt_objective(settings: dict) -> str:
    """Choose optimization objective."""
    current = str(settings.get('strategy_objective', 'expected_gross')).strip().lower()
    print('\nOptimization objective:')
    print('  [1] Maximize expected gross (default)')
    print('  [2] Maximize probability of finishing 1st (Monte Carlo)')
    print(f'Current/default: {current or "expected_gross"}')
    choice = _prompt('Choice (Enter to keep default): ').strip()
    mapping = {'1': 'expected_gross', '2': 'win_probability'}
    return mapping.get(choice, current if current in mapping.values() else 'expected_gross')


def _prompt_integer_bid_constraints(settings: dict) -> tuple[bool, int]:
    """Toggle integer bid mode and use draft-state bid context as default when available."""
    cur_mode = bool(settings.get('strategy_integer_bid_mode', False))
    state = storage.ensure_auction_state_seeded()
    state_prev = int(max(state.get('current_bid', 0), 0))
    cur_prev = state_prev if state_prev > 0 else int(max(settings.get('strategy_integer_prev_bid', 0), 0))
    cur_movie = str(state.get('current_movie', '') or '').strip().upper()
    print('\nInteger bid constraint:')
    print(f"Current/default: {'on' if cur_mode else 'off'}")
    if cur_movie:
        print(f'Current draft context: {cur_movie} with previous bid {cur_prev}')
    choice = _prompt('Enable integer-only bid mode? [y/N]: ').strip().lower()
    mode = cur_mode if choice == '' else (choice == 'y')
    prev_bid = cur_prev
    if mode:
        raw = _prompt(f'Current previous bid (integer, default {cur_prev}): ').strip()
        if raw:
            try:
                prev_bid = max(int(float(raw)), 0)
            except ValueError:
                prev_bid = cur_prev
        if cur_movie:
            storage.set_current_auction_context(cur_movie, prev_bid)
    return mode, prev_bid


def _prompt_strategy_advanced_controls(settings: dict, optimizer_cost_col: str) -> None:
    """Optional advanced tuning for optimization/scenario controls."""
    print('\nAdvanced tuning controls:')
    print('  - valuation signal strength (lambda)')
    print('  - bid multiplier')
    print('  - per-film cap % (base + market-fair stress cap)')
    print('  - risk penalty shape + portfolio penalties')
    print('  - quality filters for low-edge/high-drawdown fillers')
    print('  - probability and Monte Carlo settings')
    choice = _prompt('Adjust advanced controls now? [y/N]: ').strip().lower()
    if choice != 'y':
        return

    override_keys = set(settings.get('strategy_override_keys', []))

    def _mark_override(key: str):
        override_keys.add(key)
        settings['strategy_override_keys'] = sorted(str(k) for k in override_keys)

    def _ask_float(key: str, label: str, low: float, high: float):
        cur = float(settings.get(key, low))
        raw = _prompt(f'{label} [{cur}]: ').strip()
        if not raw:
            return
        try:
            val = float(raw)
            settings[key] = float(np.clip(val, low, high))
            _mark_override(key)
        except ValueError:
            pass

    def _ask_int(key: str, label: str, low: int, high: int):
        cur = int(settings.get(key, low))
        raw = _prompt(f'{label} [{cur}]: ').strip()
        if not raw:
            return
        try:
            val = int(float(raw))
            settings[key] = int(np.clip(val, low, high))
            _mark_override(key)
        except ValueError:
            pass

    _ask_float('strategy_lambda', 'Valuation lambda', 0.0, 2.0)
    _ask_float('strategy_bid_multiplier', 'Bid multiplier', 0.50, 2.00)
    _ask_float('strategy_max_budget_pct_per_film', 'Base per-film cap (0.01-1.00)', 0.01, 1.00)
    _ask_float('strategy_market_fair_stresstest_cap', 'Market-fair stress cap (0.01-1.00)', 0.01, 1.00)

    cur_filter = bool(settings.get('strategy_enable_quality_filters', True))
    raw_filter = _prompt(f"Enable quality filters? [{'Y' if cur_filter else 'N'}]: ").strip().lower()
    if raw_filter in ('y', 'n'):
        settings['strategy_enable_quality_filters'] = (raw_filter == 'y')
        _mark_override('strategy_enable_quality_filters')

    _ask_float('strategy_min_prob_positive_edge', 'Min P(Edge) filter (0-1)', 0.0, 1.0)
    _ask_float('strategy_max_prob_large_drawdown', 'Max P(DD) filter (0-1)', 0.0, 1.0)
    _ask_float('strategy_drawdown_threshold', 'Large drawdown threshold for P(DD) (0-1)', 0.0, 1.0)
    _ask_int('strategy_bootstrap_samples', 'Probability samples (bootstrap/student-t)', 100, 20000)

    _ask_float('strategy_risk_a_vol', 'Risk penalty coeff A (vol)', 0.0, 10.0)
    _ask_float('strategy_risk_b_drawdown', 'Risk penalty coeff B (drawdown)', 0.0, 10.0)
    _ask_float('strategy_risk_c_release', 'Risk penalty coeff C (days-to-release)', 0.0, 5.0)
    _ask_int('strategy_risk_release_window_days', 'Risk release window days', 1, 180)
    _ask_float('strategy_risk_max_penalty', 'Risk max penalty cap (0-0.95)', 0.0, 0.95)

    _ask_float('strategy_diversification_penalty', 'Diversification penalty (0-1)', 0.0, 1.0)
    _ask_float('strategy_correlation_penalty', 'Correlation penalty (0-1)', 0.0, 1.0)

    _ask_int('strategy_mc_samples', 'Monte Carlo samples', 100, 20000)
    _ask_int('strategy_mc_num_opponents', 'Monte Carlo opponents', 1, 30)
    _ask_int('strategy_mc_candidate_portfolios', 'MC candidate portfolios', 20, 2000)
    _ask_int('strategy_search_candidates', 'Search candidates (Phase 4)', 20, 5000)
    _ask_float('strategy_mc_opponent_noise', 'MC opponent noise (0.01-1.00)', 0.01, 1.00)
    _ask_float('strategy_mc_aggression_sd', 'MC opponent aggression stdev (0-1)', 0.0, 1.0)
    _ask_float('strategy_mc_concentration_threshold', 'MC concentration downside threshold (0-1)', 0.0, 1.0)
    cur_seed_mode = str(settings.get('strategy_mc_seed_mode', 'fixed')).strip().lower()
    print(f"Seed mode: current={cur_seed_mode}  [1] fixed  [2] random")
    raw_seed_mode = _prompt('Seed mode (Enter to keep): ').strip()
    if raw_seed_mode == '1':
        settings['strategy_mc_seed_mode'] = 'fixed'
        _mark_override('strategy_mc_seed_mode')
    elif raw_seed_mode == '2':
        settings['strategy_mc_seed_mode'] = 'random'
        _mark_override('strategy_mc_seed_mode')
    if str(settings.get('strategy_mc_seed_mode', 'fixed')).strip().lower() == 'fixed':
        _ask_int('strategy_mc_random_seed', 'Fixed MC seed', 0, 2_147_483_647)

    cur_corr_mode = str(settings.get('strategy_corr_simulation_mode', 'independent')).strip().lower()
    print(f"Correlation simulation mode: current={cur_corr_mode}  [1] independent  [2] gaussian_copula  [3] t_copula")
    raw_corr_mode = _prompt('Correlation mode (Enter to keep): ').strip()
    corr_map = {'1': 'independent', '2': 'gaussian_copula', '3': 't_copula'}
    if raw_corr_mode in corr_map:
        settings['strategy_corr_simulation_mode'] = corr_map[raw_corr_mode]
        _mark_override('strategy_corr_simulation_mode')
    _ask_float('strategy_corr_shrinkage', 'Copula corr shrinkage (0-1)', 0.0, 1.0)
    _ask_int('strategy_corr_min_history_points', 'Copula min history points', 5, 400)
    _ask_float('strategy_corr_floor', 'Copula eigen floor (1e-9 to 0.1)', 1e-9, 0.1)
    _ask_int('strategy_corr_t_df', 't-copula df (3-60)', 3, 60)

    cur_search_mode = str(settings.get('strategy_search_mode', 'current_sampled')).strip().lower()
    print(
        f"Search mode: current={cur_search_mode}  [1] current_sampled  "
        f"[2] random_multistart  [3] local_search  [4] genetic"
    )
    raw_search_mode = _prompt('Search mode (Enter to keep): ').strip()
    search_map = {'1': 'current_sampled', '2': 'random_multistart', '3': 'local_search', '4': 'genetic'}
    if raw_search_mode in search_map:
        settings['strategy_search_mode'] = search_map[raw_search_mode]
        _mark_override('strategy_search_mode')
    _ask_int('strategy_search_local_iters', 'Local search iterations', 0, 500)
    _ask_int('strategy_search_population', 'Genetic population', 20, 500)
    _ask_int('strategy_search_generations', 'Genetic generations', 1, 500)
    _ask_float('strategy_search_elite_frac', 'Genetic elite fraction (0.05-0.90)', 0.05, 0.90)
    _ask_float('strategy_search_mutation_rate', 'Genetic mutation rate (0-0.80)', 0.0, 0.80)

    cur_opp = str(settings.get('strategy_opponent_profile', 'balanced_field')).strip().lower()
    print(f"Opponent profile: current={cur_opp}  [1] passive_value  [2] balanced_field  [3] aggressive_bidup")
    raw_opp = _prompt('Opponent profile (Enter to keep): ').strip()
    opp_map = {'1': 'passive_value', '2': 'balanced_field', '3': 'aggressive_bidup'}
    if raw_opp in opp_map:
        settings['strategy_opponent_profile'] = opp_map[raw_opp]
        _mark_override('strategy_opponent_profile')
    _ask_float('strategy_opponent_bidup_strength', 'Opponent bid-up strength (0-1)', 0.0, 1.0)
    _ask_float('strategy_opponent_cash_conservation', 'Opponent cash conservation (-0.50 to 0.80)', -0.50, 0.80)

    if optimizer_cost_col == 'market_fair_bid':
        print(
            f"  [Info] market_fair_bid mode active. Effective per-film cap will use "
            f"max(base cap, stress cap)."
        )


def _resolve_runtime_seed_for_run(settings: dict) -> tuple[int, str]:
    """Resolve one seed per dashboard run to keep all stochastic layers aligned."""
    mode = str(settings.get('strategy_mc_seed_mode', 'fixed') or 'fixed').strip().lower()
    fixed_seed = int(pd.to_numeric(pd.Series([settings.get('strategy_mc_random_seed', 123)]), errors='coerce').fillna(123).iloc[0])
    if mode == 'random':
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        return seed, 'random'
    return fixed_seed, 'fixed'


def menu_draft_strategy_dashboard():
    """Phase 1 dashboard for strategy metrics, bids, and optimizer selection."""
    movies_df = storage.load_movies()
    people_df = storage.load_people()
    cache = storage.load_cache()
    settings = storage.load_strategy_runtime_defaults()
    settings, _ = _prompt_load_strategy_profile(settings)

    if movies_df.empty:
        print('No movies in pool. Add tickers first.')
        _pause()
        return
    if cache.empty:
        print('No HSX cache found. Scrape first (option 1).')
        _pause()
        return

    preset = _prompt_strategy_preset(settings)
    settings['strategy_active_preset'] = preset
    settings, _ = calculations.apply_strategy_preset(settings, preset)
    risk_model = _prompt_risk_model(settings)
    objective = _prompt_objective(settings)
    integer_bid_mode, previous_bid = _prompt_integer_bid_constraints(settings)
    settings['strategy_risk_model'] = risk_model
    settings['strategy_objective'] = objective
    settings['strategy_integer_bid_mode'] = bool(integer_bid_mode)
    settings['strategy_integer_prev_bid'] = int(previous_bid)

    budget_mode, custom_budget = _prompt_budget_inputs(people_df, settings)
    optimizer_cost_col = _prompt_optimizer_cost_col(settings)
    prob_bid_col, prob_value_col = calculations.resolve_probability_anchor_columns(optimizer_cost_col)
    _prompt_strategy_advanced_controls(settings, optimizer_cost_col)
    run_seed, run_seed_mode = _resolve_runtime_seed_for_run(settings)
    run_settings = dict(settings)
    run_settings['strategy_runtime_seed'] = int(run_seed)

    dashboard, meta, tuned_settings = calculations.build_strategy_dashboard(
        hsx_df=cache,
        movies_df=movies_df,
        people_df=people_df,
        settings=run_settings,
        preset=preset,
        budget_mode=budget_mode,
        custom_budget=custom_budget,
        risk_model=risk_model,
        integer_bid_mode=integer_bid_mode,
        previous_bid=previous_bid,
        prob_bid_col=prob_bid_col,
        prob_value_col=prob_value_col,
    )
    if dashboard.empty:
        print('No strategy metrics available.')
        _pause()
        return

    budget_info = meta.get('budget_info', {})
    basis_budget = float(budget_info.get('basis_budget', 0.0))
    if basis_budget <= 0:
        print('Budget basis resolved to $0.00. Configure people/custom budget first.')
        _pause()
        return
    if optimizer_cost_col == 'current_price':
        print('[Note] Optimizer cost basis = current HSX price (diagnostic mode).')
        print('       This is not auction-scale and can distort draft recommendations.')

    if objective == 'win_probability':
        opt = calculations.optimize_portfolio_by_win_probability(
            strategy_df=dashboard,
            budget=basis_budget,
            settings=tuned_settings,
            cost_col=optimizer_cost_col,
            risk_model=risk_model,
        )
    else:
        opt = calculations.optimize_portfolio(
            strategy_df=dashboard,
            budget=basis_budget,
            settings=tuned_settings,
            cost_col=optimizer_cost_col,
        )
    effective_cost_col = optimizer_cost_col
    if opt.get('num_selected', 0) == 0 and optimizer_cost_col == 'market_fair_bid':
        if objective == 'win_probability':
            opt = calculations.optimize_portfolio_by_win_probability(
                strategy_df=dashboard,
                budget=basis_budget,
                settings=tuned_settings,
                cost_col='target_bid',
                risk_model=risk_model,
            )
        else:
            opt = calculations.optimize_portfolio(
                strategy_df=dashboard,
                budget=basis_budget,
                settings=tuned_settings,
                cost_col='target_bid',
            )
        effective_cost_col = 'target_bid'
        print('\n[Note] `market_fair_bid` unavailable/insufficient. Fell back to `target_bid` for optimizer.')

    win_eval = None
    if objective != 'win_probability':
        try:
            win_eval = calculations.estimate_portfolio_win_probability(
                strategy_df=dashboard,
                portfolio_df=opt.get('selected', pd.DataFrame()),
                budget=basis_budget,
                settings=tuned_settings,
                risk_model=risk_model,
                cost_col=opt.get('cost_col', effective_cost_col),
            )
        except Exception:
            win_eval = None

    sim = calculations.simulate_portfolio_monte_carlo(
        strategy_df=dashboard,
        portfolio_df=opt['selected'],
        settings=tuned_settings,
        risk_model=risk_model,
    )

    selected_tickers = set(opt['selected']['ticker'].tolist()) if not opt['selected'].empty else set()
    dashboard = dashboard.copy()
    dashboard['optimizer_selected'] = dashboard['ticker'].isin(selected_tickers)

    avail = dashboard[dashboard['owner'] == ''].copy()
    if avail.empty:
        print('No unassigned movies available.')
        _pause()
        return

    avail = avail.sort_values(
        ['priority_score', 'prob_positive_edge', 'adjusted_expected'],
        ascending=[False, False, False],
        na_position='last',
    )

    print('\n=== Draft Strategy Dashboard ===')
    print(
        f"Preset: {meta.get('preset', 'balanced')}  "
        f"| Budget basis: ${basis_budget:.2f} ({budget_info.get('budget_mode', 'personal')})  "
        f"| Risk model: {risk_model}  | Objective: {objective}  "
        f"| Integer bids: {'on' if integer_bid_mode else 'off'}"
    )
    print(f"Seed mode: {run_seed_mode}  | Seed used: {run_seed}")
    if integer_bid_mode:
        print(f'Integer bid rules in effect: bid is whole number, > {previous_bid}, <= remaining budget.')
        cap_sources = dashboard.get('integer_cap_source')
        if cap_sources is not None:
            unique_sources = pd.Series(cap_sources).dropna().astype(str).unique().tolist()
            if unique_sources == ['unavailable']:
                print('  [Note] Personal remaining budget is not configured; no legal integer bids can be formed.')
    if pd.notna(budget_info.get('market_pressure_factor', np.nan)):
        print(f"Market pressure factor (league/personal): {budget_info['market_pressure_factor']:.2f}")

    if integer_bid_mode:
        header = (
            f"{'Ticker':<8}  {'Price':>7}  {'AdjExp':>7}  {'TgtInt':>7}  {'TgtMktInt':>9}  "
            f"{'MaxInt':>7}  {'MinBid':>7}  {'Risk%':>7}  "
            f"{'P(Edge)':>8}  {'P(DD)':>8}  {'MktVR':>7}  {'Score':>7}  {'Opt':>3}"
        )
    else:
        header = (
            f"{'Ticker':<8}  {'Price':>7}  {'AdjExp':>7}  {'TgtBid':>7}  {'TgtMkt':>7}  "
            f"{'MaxBid':>7}  {'Risk%':>7}  {'P(Edge)':>8}  {'P(DD)':>8}  "
            f"{'MktVR':>7}  {'Score':>7}  {'Opt':>3}"
        )
    print(header)
    print('─' * len(header))

    for _, row in avail.iterrows():
        if integer_bid_mode:
            print(
                f"{row['ticker']:<8}  "
                f"{_fmt(row.get('current_price'))}  "
                f"{_fmt(row.get('adjusted_expected'))}  "
                f"{_fmt(row.get('target_bid_int'))}  "
                f"{_fmt(row.get('target_market_bid_int'))}  "
                f"{_fmt(row.get('max_bid_int'))}  "
                f"{_fmt(row.get('min_legal_bid_int'))}  "
                f"{_fmt_prob(row.get('risk_penalty'), 7)}  "
                f"{_fmt_prob(row.get('prob_positive_edge'), 8)}  "
                f"{_fmt_prob(row.get('prob_large_drawdown'), 8)}  "
                f"{_fmt(row.get('market_value_ratio'))}  "
                f"{_fmt(row.get('priority_score'))}  "
                f"{'*' if row.get('optimizer_selected') else '':>3}"
            )
        else:
            print(
                f"{row['ticker']:<8}  "
                f"{_fmt(row.get('current_price'))}  "
                f"{_fmt(row.get('adjusted_expected'))}  "
                f"{_fmt(row.get('target_bid'))}  "
                f"{_fmt(row.get('target_market_bid'))}  "
                f"{_fmt(row.get('max_bid'))}  "
                f"{_fmt_prob(row.get('risk_penalty'), 7)}  "
                f"{_fmt_prob(row.get('prob_positive_edge'), 8)}  "
                f"{_fmt_prob(row.get('prob_large_drawdown'), 8)}  "
                f"{_fmt(row.get('market_value_ratio'))}  "
                f"{_fmt(row.get('priority_score'))}  "
                f"{'*' if row.get('optimizer_selected') else '':>3}"
            )

    print('\nOptimizer summary:')
    print(
        f"  Selected: {opt['num_selected']} movie(s) | "
        f"Spend: ${opt['total_spend']:.2f} / ${opt['budget']:.2f} | "
        f"Left: ${opt['leftover']:.2f} | "
        f"AdjExp total: {opt['total_adjusted_expected']:.2f} | "
        f"Cost basis: {opt.get('cost_col', effective_cost_col)}"
    )
    print(
        f"  Probability basis: bid={meta.get('prob_bid_col', 'target_bid')}  "
        f"value={meta.get('prob_value_col', 'fair_budget_bid')}"
    )
    cap_pct = pd.to_numeric(pd.Series([opt.get('max_budget_pct_per_film', np.nan)]), errors='coerce').iloc[0]
    if pd.notna(cap_pct):
        print(f"  Per-film cap: {100.0 * float(cap_pct):.1f}% of budget")
    qf = opt.get('quality_filter_info', {})
    if isinstance(qf, dict) and qf.get('enabled'):
        print(
            f"  Quality filter: P(Edge)>={100.0*float(qf.get('min_prob_positive_edge', 0.0)):.0f}% "
            f"and P(DD)<={100.0*float(qf.get('max_prob_large_drawdown', 1.0)):.0f}%  "
            f"(rows kept: {qf.get('rows_after', 0)}/{qf.get('rows_before', 0)})"
        )
    print(
        f"  Valuation controls: lambda={float(tuned_settings.get('strategy_lambda', 0.35)):.2f}  "
        f"clip=[{float(tuned_settings.get('strategy_clip_low', 0.85)):.2f}, {float(tuned_settings.get('strategy_clip_high', 1.15)):.2f}]"
    )
    print(
        f"  Risk penalty coeffs: A={float(tuned_settings.get('strategy_risk_a_vol', 3.0)):.2f}  "
        f"B={float(tuned_settings.get('strategy_risk_b_drawdown', 1.4)):.2f}  "
        f"C={float(tuned_settings.get('strategy_risk_c_release', 0.45)):.2f}  "
        f"max={100.0 * float(tuned_settings.get('strategy_risk_max_penalty', 0.45)):.1f}%  "
        f"window={int(tuned_settings.get('strategy_risk_release_window_days', 30))}d"
    )
    print(
        f"  Portfolio penalties: div={float(tuned_settings.get('strategy_diversification_penalty', 0.0)):.2f}  "
        f"corr={float(tuned_settings.get('strategy_correlation_penalty', 0.0)):.2f}"
    )
    if integer_bid_mode:
        rounding_drift = float(pd.to_numeric(dashboard.get('integer_rounding_drift', np.nan), errors='coerce').sum())
        clipped_count = int((pd.to_numeric(dashboard.get('can_bid_int', True), errors='coerce') == 0).sum())
        print(f"  Integer mode: total rounding drift={rounding_drift:+.2f}  no-legal-bid rows={clipped_count}")
    if objective == 'win_probability':
        print(
            f"  Estimated win probability: {_fmt_prob(opt.get('win_probability'), 8)}  "
            f"(vs {opt.get('opponent_count', 0)} simulated opponents)"
        )
        if pd.notna(opt.get('seed_used', np.nan)):
            print(f"  Win-prob seed: {opt.get('seed_mode', run_seed_mode)} ({int(opt.get('seed_used'))})")
    elif isinstance(win_eval, dict):
        print(
            f"  Estimated win probability (selected portfolio): {_fmt_prob(win_eval.get('win_probability'), 8)}  "
            f"(vs {win_eval.get('opponent_count', 0)} simulated opponents)"
        )
        if pd.notna(win_eval.get('seed_used', np.nan)):
            print(f"  Win-prob seed: {win_eval.get('seed_mode', run_seed_mode)} ({int(win_eval.get('seed_used'))})")
    eval_meta = opt if objective == 'win_probability' else (win_eval if isinstance(win_eval, dict) else {})
    if isinstance(eval_meta, dict):
        search_mode_used = str(eval_meta.get('search_mode_used', tuned_settings.get('strategy_search_mode', 'current_sampled')))
        cand_eval = pd.to_numeric(pd.Series([eval_meta.get('candidate_count_evaluated', np.nan)]), errors='coerce').iloc[0]
        search_ms = pd.to_numeric(pd.Series([eval_meta.get('search_runtime_ms', np.nan)]), errors='coerce').iloc[0]
        opp_profile_used = str(eval_meta.get('opponent_profile_used', tuned_settings.get('strategy_opponent_profile', 'balanced_field')))
        search_parts = [f"  Search mode: {search_mode_used}"]
        if pd.notna(cand_eval):
            search_parts.append(f"candidates={int(cand_eval)}")
        if pd.notna(search_ms):
            search_parts.append(f"runtime_ms={float(search_ms):.1f}")
        print('  ' + '  '.join(search_parts).strip())
        print(
            f"  Opponent profile: {opp_profile_used}  "
            f"bidup={float(tuned_settings.get('strategy_opponent_bidup_strength', 0.0)):.2f}  "
            f"cash_conserve={float(tuned_settings.get('strategy_opponent_cash_conservation', 0.0)):.2f}"
        )

    print('\nMonte Carlo portfolio simulation:')
    print(
        f"  samples={sim.get('samples', 0)}  "
        f"gross mean/p10/p50/p90="
        f"{_fmt(sim.get('gross_mean'))} / {_fmt(sim.get('gross_p10'))} / "
        f"{_fmt(sim.get('gross_p50'))} / {_fmt(sim.get('gross_p90'))}"
    )
    if pd.notna(sim.get('seed_used', np.nan)):
        print(f"  MC seed: {sim.get('seed_mode', run_seed_mode)} ({int(sim.get('seed_used'))})")
    print(
        f"  Correlation simulation: requested={sim.get('corr_mode_requested', 'independent')}  "
        f"used={sim.get('corr_mode_used', 'independent')}  "
        f"dim={int(sim.get('corr_effective_dim', 0))}"
    )
    corr_fallback_reason = str(sim.get('corr_fallback_reason', '') or '').strip()
    if corr_fallback_reason:
        print(f"  Correlation fallback: {corr_fallback_reason}")
    if pd.notna(sim.get('exchange_rate_million_per_auction_dollar', np.nan)):
        print(
            f"  exchange_rate={sim.get('exchange_rate_million_per_auction_dollar', np.nan):.4f} "
            f"(gross-units per $1 auction)"
        )
        print(
            f"  budget-equiv gross mean/p10/p50/p90="
            f"{_fmt(sim.get('gross_budget_equiv_mean'))} / {_fmt(sim.get('gross_budget_equiv_p10'))} / "
            f"{_fmt(sim.get('gross_budget_equiv_p50'))} / {_fmt(sim.get('gross_budget_equiv_p90'))}"
        )
    print(
        f"  P(gross_equiv < spend)={_fmt_prob(sim.get('prob_gross_below_spend'), 8)}  "
        f"Concentration downside={_fmt_prob(sim.get('concentration_downside_prob'), 8)}"
    )
    print(
        f"  prob_samples={int(tuned_settings.get('strategy_bootstrap_samples', 1000))}  "
        f"opp_noise={float(tuned_settings.get('strategy_mc_opponent_noise', 0.30)):.2f}  "
        f"opp_aggr_sd={float(tuned_settings.get('strategy_mc_aggression_sd', 0.10)):.2f}  "
        f"conc_threshold={_fmt_prob(tuned_settings.get('strategy_mc_concentration_threshold', 0.40), 7)}"
    )
    agg = sim.get('aggression_sensitivity')
    if isinstance(agg, pd.DataFrame) and not agg.empty:
        print('  Aggression sensitivity (spend multiplier):')
        for _, row in agg.iterrows():
            print(
                f"    x{row.get('aggression_factor', 1.0):.2f}  "
                f"E[surplus]={_fmt(row.get('expected_surplus'))}  "
                f"P(surplus>0)={_fmt_prob(row.get('prob_positive_surplus'), 8)}"
            )

    if not opt['selected'].empty:
        print('\nSelected portfolio:')
        for _, row in opt['selected'].iterrows():
            eff = pd.to_numeric(pd.Series([row.get('eff_per_dollar', np.nan)]), errors='coerce').iloc[0]
            if pd.isna(eff):
                cost = pd.to_numeric(pd.Series([row.get('optimizer_cost', np.nan)]), errors='coerce').iloc[0]
                adj = pd.to_numeric(pd.Series([row.get('adjusted_expected', np.nan)]), errors='coerce').iloc[0]
                eff = (adj / cost) if pd.notna(cost) and cost > 0 and pd.notna(adj) else np.nan
            print(
                f"  {row['ticker']:<8}  bid=${row['target_bid']:.2f}  "
                f"cost=${row['optimizer_cost']:.2f}  "
                f"adj={row['adjusted_expected']:.2f}  eff/$={float(eff):.2f}"
            )

    if not opt['alternates'].empty:
        print('\nTop alternates:')
        for _, row in opt['alternates'].head(5).iterrows():
            eff = pd.to_numeric(pd.Series([row.get('eff_per_dollar', np.nan)]), errors='coerce').iloc[0]
            if pd.isna(eff):
                cost = pd.to_numeric(pd.Series([row.get('optimizer_cost', np.nan)]), errors='coerce').iloc[0]
                adj = pd.to_numeric(pd.Series([row.get('adjusted_expected', np.nan)]), errors='coerce').iloc[0]
                eff = (adj / cost) if pd.notna(cost) and cost > 0 and pd.notna(adj) else np.nan
            print(
                f"  {row['ticker']:<8}  bid=${row['target_bid']:.2f}  "
                f"cost=${row['optimizer_cost']:.2f}  "
                f"adj={row['adjusted_expected']:.2f}  eff/$={float(eff):.2f}"
            )

    diag = meta.get('diagnostics', {})
    print('\nDiagnostics:')
    print(
        f"  rows={diag.get('rows', 0)}  "
        f"missing_price={diag.get('missing_current_price', 0)}  "
        f"missing_history={diag.get('missing_history', 0)}  "
        f"missing_probs={diag.get('missing_probs', 0)}"
    )

    validation = meta.get('validation', {})
    stability = validation.get('ranking_stability', {})
    forward = validation.get('forward_check', {})
    if stability:
        print(
            f"  ranking_stability(top{stability.get('top_n', 10)} overlap mean/min/max): "
            f"{_fmt_prob(stability.get('mean_top_overlap'), 8)} / "
            f"{_fmt_prob(stability.get('min_top_overlap'), 8)} / "
            f"{_fmt_prob(stability.get('max_top_overlap'), 8)}"
        )
    if forward:
        print(
            f"  forward_check(h={forward.get('horizon_days', 0)}d, samples={forward.get('samples', 0)}): "
            f"dir_acc={_fmt_prob(forward.get('directional_accuracy'), 8)}  "
            f"IC={_fmt(forward.get('information_coefficient'), 7, 3)}"
        )

    sensitivity = validation.get('preset_sensitivity')
    if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
        print('\nPreset sensitivity (vs active preset top-10):')
        for _, row in sensitivity.iterrows():
            print(
                f"  {str(row.get('preset', '')).capitalize():<12}  "
                f"overlap={_fmt_prob(row.get('top_overlap_vs_base'), 8)}  "
                f"avg_bid={_fmt(row.get('avg_target_bid'))}  "
                f"avg_p_edge={_fmt_prob(row.get('avg_prob_positive_edge'), 8)}"
            )

    export = _prompt('\nExport dashboard CSV to data/? [y/N]: ').lower()
    if export == 'y':
        stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        export_name = f'strategy_dashboard_{stamp}.csv'
        export_path = os.path.join(storage.DATA_DIR, export_name)
        avail.to_csv(export_path, index=False)
        print(f'Exported: {export_path}')

    _prompt_save_strategy_profile(settings)
    _pause()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('\nHSX Movie Draft Tool')
    storage.ensure_auction_state_seeded()

    while True:
        cache = storage.load_cache()
        cache_info = (
            f'  [cache: {cache["scraped_at"].iloc[0]}]'
            if not cache.empty else '  [no cache — scrape first]'
        )

        choice = _menu(f'Main Menu{cache_info}', {
            '1': 'Scrape latest prices from HSX',
            '2': 'View all pool movies',
            '3': 'View available (unassigned) movies',
            '4': 'Assign movie to person',
            '5': 'Unassign movie',
            '6': 'Person summary',
            '7': 'Manage people',
            '8': 'Manage movie pool',
            '9': 'Set bid reference (anchor for BidEstimate column)',
            '10': 'Interactive price history viewer',
            '11': 'Draft strategy dashboard',
            '12': 'Draft state / assignment history',
            '0': 'Exit',
        })

        if choice == '0':
            print('Goodbye.')
            break
        elif choice == '1':
            menu_scrape()
        elif choice == '2':
            menu_all_movies()
        elif choice == '3':
            menu_available_movies()
        elif choice == '4':
            menu_assign()
        elif choice == '5':
            menu_unassign()
        elif choice == '6':
            menu_person_summary()
        elif choice == '7':
            menu_people()
        elif choice == '8':
            menu_movies()
        elif choice == '9':
            menu_bid_reference()
        elif choice == '10':
            menu_view_price_history()
        elif choice == '11':
            menu_draft_strategy_dashboard()
        elif choice == '12':
            menu_draft_state()


def launch_gui():
    try:
        from gui.app import run_gui
    except Exception as exc:
        print('GUI dependencies are missing or failed to load.')
        print('Install requirements (including PySide6) and retry.')
        print(f'Error: {exc}')
        return
    run_gui()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSX Movie Draft Tool')
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch PySide6 GUI mode instead of CLI.',
    )
    args = parser.parse_args()
    if args.gui:
        launch_gui()
    else:
        main()
