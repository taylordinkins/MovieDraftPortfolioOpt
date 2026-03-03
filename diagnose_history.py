"""
diagnose_history.py — Debug price history scraping for a single ticker.
Run: python diagnose_history.py DUNE3
"""
import json
import re
import sys
import os

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper import HSX_DETAIL_URL

ticker = sys.argv[1] if len(sys.argv) > 1 else 'DUNE3'
url = HSX_DETAIL_URL.format(ticker=ticker)

print(f'Diagnosing: {url}\n')

resp = requests.get(url, timeout=15)
soup = BeautifulSoup(resp.content, 'html.parser')

# ── Step 1: dump all inline script tag contents ───────────────────────────────
print('=== Step 1: inline script tag contents ===')
for i, script in enumerate(soup.find_all('script')):
    src = script.get('src', '')
    text = (script.string or '').strip()
    if text:
        print(f'\n  [script #{i} inline, {len(text)} chars]:')
        print('  ' + text[:600].replace('\n', '\n  '))
    elif src:
        print(f'\n  [script #{i} external]: {src}')

# ── Step 2: look for data-* attrs and any element containing a 5-digit number ─
print('\n=== Step 2: elements with data-* attributes ===')
for tag in soup.find_all(True):
    attrs = {k: v for k, v in tag.attrs.items() if k.startswith('data-')}
    if attrs:
        print(f'  <{tag.name}> {attrs}')

# ── Step 3: fetch security_price_history.js ───────────────────────────────────
print('\n=== Step 3: security_price_history.js contents ===')
js_url = None
for script in soup.find_all('script', src=True):
    if 'security_price_history' in script['src']:
        js_url = script['src']
        if not js_url.startswith('http'):
            js_url = 'https://www.hsx.com' + js_url
        break

if js_url:
    print(f'  Fetching: {js_url}')
    r = requests.get(js_url, timeout=15)
    print(f'  Status: {r.status_code}, length: {len(r.text)} chars')
    print(f'  Full contents:')
    print(r.text)
else:
    print('  security_price_history.js not found in page scripts.')
