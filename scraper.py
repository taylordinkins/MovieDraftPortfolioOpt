import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

HSX_LIST_URL = 'https://www.hsx.com/security/list.php?id=1'


def _get_page_count() -> int:
    resp = requests.get(HSX_LIST_URL, timeout=15)
    soup = BeautifulSoup(resp.content, 'html.parser')
    return int(soup.find_all('p')[4].find('span').text.split()[-1]) + 1


def scrape_hsx() -> pd.DataFrame:
    """Scrape all movie tickers from HSX and return as a DataFrame."""
    range_cap = _get_page_count()
    urls = [
        f'{HSX_LIST_URL}&sfield=name&sdir=asc&page={i}'
        for i in range(1, range_cap)
    ]

    tickers, names, prices, pr_changes, per_changes = [], [], [], [], []

    for i, url in enumerate(urls, 1):
        print(f'  Fetching page {i}/{len(urls)}...', end='\r', flush=True)
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.content, 'html.parser')
        tbody = soup.find('tbody')
        rows = tbody.find_all('tr')

        for row in rows:
            prices.append(row.find('td', class_='right').text.strip()[2:])
            pr_change, per_change = row.find_all('td', class_='right')[1].text.strip()[1:].split()
            pr_changes.append(pr_change[1:])
            per_changes.append(per_change[1:-2])

        for link in tbody.find_all('a', class_='trade_link'):
            names.append(link['title'])
            tickers.append(link.text.strip())

    print()

    df = pd.DataFrame({
        'ticker': tickers,
        'name': names,
        'current_price': np.array(prices, dtype=float),
        'price_change': np.array(pr_changes, dtype=float),
        'pct_change': np.array(per_changes, dtype=float),
    })

    # Correct sign of price_change using pct_change direction
    mask = df['pct_change'] < 0
    df.loc[mask, 'price_change'] = -df.loc[mask, 'price_change'].abs()

    df['scraped_at'] = datetime.now().isoformat(timespec='seconds')
    return df


HSX_DETAIL_URL = 'https://www.hsx.com/security/view/{ticker}'


def scrape_release_date(ticker: str) -> str | None:
    """
    Fetch the release date for a single movie from its HSX detail page.
    Returns a 'YYYY-MM-DD' string, or None if not found or on error.
    """
    url = HSX_DETAIL_URL.format(ticker=ticker)
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.content, 'html.parser')
        # Find the label cell — its text is "Release\xa0Date:" (non-breaking space)
        label = soup.find(
            'td', class_='label',
            string=lambda t: t and 'Release' in t and 'Date' in t
        )
        if not label:
            return None
        date_td = label.find_next_sibling('td')
        if not date_td:
            return None
        # Normalize whitespace ("Mar  6, 2026" → "Mar 6, 2026")
        date_text = ' '.join(date_td.get_text(strip=True).split())
        dt = datetime.strptime(date_text, '%b %d, %Y')
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return None


def scrape_release_dates(tickers: list) -> dict:
    """
    Fetch release dates for a list of HSX tickers.
    Returns {ticker: 'YYYY-MM-DD'} for each successfully fetched date.
    Tickers with no date on HSX (or fetch errors) are omitted.
    """
    results = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f'  [{i}/{total}] {ticker:<10}', end='\r', flush=True)
        date = scrape_release_date(ticker)
        if date:
            results[ticker] = date
    print()
    return results


# ── Price History ─────────────────────────────────────────────────────────────
#
# HSX uses amCharts v4. The chart data is served as CSV from:
#   /chart/detail_chart_data.php?id=<security_id>
# The security_id is embedded in the raw page HTML inside a
# window.priceHistoryQueue.push({..., params: {id: <security_id>}}) block.
# Columns in the CSV response are: Date, Close
#
# This means we only need two plain requests.get() calls — no headless browser.


def _chart_id_from_scripts(soup) -> str | None:
    """
    Find the numeric security ID embedded in the HSX page's priceHistoryQueue
    block, e.g.:  window.priceHistoryQueue.push({..., params: {id: 45074}})
    Returns the ID as a string, or None if not found.
    """
    for script in soup.find_all('script'):
        text = script.string or ''
        if 'priceHistoryQueue' not in text:
            continue
        m = re.search(r'params\s*:\s*\{\s*id\s*:\s*(\d+)', text)
        if m:
            return m.group(1)
    return None


def _parse_csv_history(text: str) -> pd.DataFrame | None:
    """
    Parse the CSV returned by detail_chart_data.php.
    Expected columns: Date, Close  (as revealed by security_price_history.js)
    Returns a DataFrame with columns 'date' (YYYY-MM-DD) and 'price' (float).
    """
    import io
    try:
        df = pd.read_csv(io.StringIO(text.strip()))
        df.columns = [c.strip() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        price_col = next(
            (c for c in df.columns if c.lower() in ('close', 'price', 'value')), None
        )
        if not date_col or not price_col:
            return None
        result = pd.DataFrame({
            'date': pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d'),
            'price': pd.to_numeric(df[price_col], errors='coerce'),
        }).dropna()
        return result if len(result) >= 2 else None
    except Exception:
        return None


def scrape_price_history(ticker: str) -> tuple[pd.DataFrame | None, str]:
    """
    Fetch historical price data for a movie from HSX.

    Fetches the detail page, extracts the security ID from the
    priceHistoryQueue script block, then fetches the CSV endpoint:
      /chart/detail_chart_data.php?id=<security_id>

    Returns (DataFrame, source_label) where DataFrame has columns:
        date  (str YYYY-MM-DD)
        price (float)
    or (None, reason_string) if no data could be extracted.
    """
    url = HSX_DETAIL_URL.format(ticker=ticker)
    try:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.content, 'html.parser')
    except Exception as e:
        return None, f'page fetch error: {e}'

    chart_id = _chart_id_from_scripts(soup)
    if not chart_id:
        return None, 'security ID not found in page (priceHistoryQueue block missing)'

    data_url = f'https://www.hsx.com/chart/detail_chart_data.php?id={chart_id}'
    try:
        r = requests.get(data_url, timeout=15)
        if not r.ok:
            return None, f'CSV endpoint returned HTTP {r.status_code}'
        df = _parse_csv_history(r.text)
        if df is not None:
            return df, data_url
        return None, f'CSV parse failed (first 200 chars: {r.text[:200]!r})'
    except Exception as e:
        return None, f'CSV fetch error: {e}'


def scrape_all_price_histories(tickers: list) -> tuple[dict, dict]:
    """
    Fetch price histories for a list of tickers in a single pass.

    Returns:
        results  — {ticker: DataFrame} for each ticker where data was found
        statuses — {ticker: source_label_or_error} for every ticker
    """
    results = {}
    statuses = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f'  [{i}/{total}] {ticker:<10}', end='\r', flush=True)
        df, status = scrape_price_history(ticker)
        statuses[ticker] = status
        if df is not None:
            results[ticker] = df
    print()
    return results, statuses
