"""
find_tickers.py — Match a list of movie titles to HSX tickers.

Reads from the local HSX cache (run the main tool and scrape first).
Shows the top 3 HSX candidates for each target title so you can confirm
the right ticker, then prints a final summary list ready to bulk-paste
into the main tool.
"""

import os
import re
import sys
from difflib import SequenceMatcher

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import storage

# ── Movie titles to find ──────────────────────────────────────────────────────

TARGETS = [
    "Hoppers",
    "The Bride",
    "Project Hail Mary",
    "Ready or Not: Here I Come",
    "Super Mario Galaxy",
    "The Drama",
    "Michael",
    "The Devil Wears Prada 2",
    "Mortal Kombat II",
    "The Sheep Detectives",
    "The Mandalorian and Grogu",
    "Masters of the Universe",
    "Animal Friends",
    "Disclosure Day",
    "Toy Story 5",
    "The Death of Robin Hood",
    "Supergirl",
    "Minions & Monsters",
    "Moana",
    "The Odyssey",
    "Spider-Man Brand New Day",
    "Flowervale Street",
    "Paw Patrol 3",
    "The Dog Stars",
    "Coyote vs ACME",
    "How to Rob a Bank",
    "Clayface",
    "Practical Magic 2",
    "Resident Evil",
    "Forgotten Island",
    "Digger",
    "The Social Reckoning",
    "Street Fighter",
    "The Cat in the Hat",
    "The Great Beyond",
    "Ebenezer: A Christmas Carol",
    "The Hunger Games: Sunrise on the Reaping",
    "Hexed",
    "Focker in Law",
    "Jumanji 3",
    "Avengers Doomsday",
    "Dune: Part Three",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_ROMAN = {'ii': '2', 'iii': '3', 'iv': '4', 'vi': '6', 'vii': '7'}

_STOPWORDS = {'the', 'a', 'an', 'of', 'and', 'in', 'on', 'at', 'to', 'vs', 'aka'}


def _clean(s: str) -> str:
    """Lowercase, expand roman numerals, strip punctuation."""
    s = s.lower()
    # Normalize roman numerals as whole words
    for roman, digit in _ROMAN.items():
        s = re.sub(rf'\b{roman}\b', digit, s)
    s = re.sub(r'[^\w\s]', ' ', s)
    return s.strip()


def _sig_words(cleaned: str) -> set:
    """Significant (non-stopword) words from a cleaned string."""
    return {w for w in cleaned.split() if w not in _STOPWORDS}


def _score(target_clean: str, target_sig: set, candidate_clean: str) -> float:
    """
    Blend of two signals:
      - Sequence match ratio (character-level similarity)
      - Word coverage: fraction of target's significant words found in candidate
    """
    seq = SequenceMatcher(None, target_clean, candidate_clean).ratio()

    cand_words = set(candidate_clean.split())
    if target_sig:
        coverage = len(target_sig & cand_words) / len(target_sig)
    else:
        coverage = seq  # title is all stopwords, fall back to seq

    return 0.35 * seq + 0.65 * coverage


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cache = storage.load_cache()
    if cache.empty:
        print('No HSX cache found. Open the main tool and scrape first (option 1).')
        sys.exit(1)

    # Pre-compute cleaned names (strip "TICKER: " prefix)
    def _strip_prefix(name, ticker):
        if not isinstance(name, str):
            return ''
        prefix = f'{ticker}: '
        return name[len(prefix):] if name.startswith(prefix) else name

    cache = cache.copy()
    cache['clean_name'] = cache.apply(
        lambda r: _clean(_strip_prefix(r['name'], r['ticker'])), axis=1
    )

    top_picks = []   # (target, ticker, hsx_name)
    ambiguous = []   # targets where best score < threshold

    CONFIDENT = 0.75  # auto-accept threshold

    for target in TARGETS:
        t_clean = _clean(target)
        t_sig = _sig_words(t_clean)

        cache['_s'] = cache['clean_name'].apply(lambda c: _score(t_clean, t_sig, c))
        top3 = cache.nlargest(3, '_s')[['ticker', 'name', '_s']]

        best_score = top3.iloc[0]['_s']
        confident = best_score >= CONFIDENT

        marker = '  ✓' if confident else '  ?'
        print(f'\nTarget: "{target}"{marker}')
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            flag = ' ◀' if rank == 1 else ''
            print(f'  [{rank}] {row["ticker"]:<8}  {row["name"]:<52}  {row["_s"]:.2f}{flag}')

        top_picks.append((target, top3.iloc[0]['ticker'], top3.iloc[0]['name']))
        if not confident:
            ambiguous.append(target)

    # Summary
    print('\n' + '=' * 70)
    print('TOP-PICK TICKERS (review ? entries above before using):')
    print()
    tickers = [t for _, t, _ in top_picks]
    print(', '.join(tickers))
    print()

    if ambiguous:
        print(f'Low-confidence matches ({len(ambiguous)}) — double-check these:')
        for a in ambiguous:
            match = next((t for tgt, t, _ in top_picks if tgt == a), None)
            print(f'  {a!r:45s}  → {match}')

    print('\nTo bulk-add to the pool, paste the ticker list into main.py')
    print("option 8 → 4 (Bulk add tickers).")


if __name__ == '__main__':
    main()
