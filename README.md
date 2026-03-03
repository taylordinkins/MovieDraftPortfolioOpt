# HSX Movie Draft Tool

A Python toolkit for managing and optimizing a movie auction draft using HSX data.

The project supports:

- data scraping and local caching
- movie pool and player budget management
- assignment and draft-state tracking
- interactive CLI and optional PySide6 GUI
- strategy dashboard with optimization and Monte Carlo diagnostics

Vibe-coded with various AI tools, sanity checked (somewhat) by self.

## Table of Contents

- Overview
- Features
- Project Structure
- Data Files
- Requirements
- Installation
- Quick Start
- CLI Workflow
- GUI Workflow
- Strategy Dashboard Concepts
- Saved Strategy Profiles
- Draft-State Workflow
- Troubleshooting
- Notes and Limitations
- Related Docs

## Overview

This tool is designed for repeated movie draft analysis over time, not only a single live session.
It keeps your state (assigned movies, budgets, active movie/bid context) and recomputes strategy as the draft evolves.

The dashboard provides:

- ranking and bid guidance (`target_bid`, `max_bid`, integer variants)
- expected-gross and win-probability optimization modes
- bootstrap or Student-t risk modeling
- optional copula correlation simulation and enhanced search modes

## Features

- **Data ingestion**
  - scrape latest HSX snapshot (price, change, metadata)
  - scrape per-ticker historical price series
  - scrape release dates
- **Pool management**
  - add/remove tickers
  - assign/unassign winners
  - update people and budgets
- **State tracking**
  - persistent current movie + current bid
  - assignment history log
  - automatic budget decrement on assignment price
- **Dashboard analytics**
  - feature table from local history
  - adjusted expected value, risk penalty, probability-of-edge metrics
  - portfolio optimization (expected gross / win probability)
  - Monte Carlo risk diagnostics and sensitivity views
- **Interfaces**
  - full CLI menu flow
  - optional GUI (`--gui`) with dashboard and draft-state tabs
- **Profile iteration**
  - save/load named strategy parameter profiles
  - reset-to-default runtime behavior still preserved

## Project Structure

```text
movie_scraping/
  main.py                  # CLI entrypoint (+ --gui switch)
  gui/app.py               # PySide6 GUI application
  scraper.py               # HSX scraping functions
  storage.py               # local persistence APIs
  calculations.py          # strategy metrics, optimization, simulation
  history_viewer.py        # interactive matplotlib history viewer
  implementation_plan.md   # implementation roadmap and status
  strategy_guide.md        # analyst playbook for dashboard use
  requirements.txt
  data/
    hsx_cache.csv
    movies.csv
    people.csv
    settings.json
    auction_state.json
    strategy_profiles.json
    <TICKER>/history.csv
```

## Data Files

- `data/hsx_cache.csv`: latest scraped HSX snapshot
- `data/movies.csv`: draft pool (`ticker`, `owner`, `release_date`)
- `data/people.csv`: participants and budgets
- `data/settings.json`: persisted non-strategy app settings
- `data/auction_state.json`: current movie/bid and assignment history
- `data/strategy_profiles.json`: named saved dashboard parameter sets
- `data/<TICKER>/history.csv`: historical price series per movie

## Requirements

Python 3.10+ recommended.

Dependencies (see `requirements.txt`):

- `requests`
- `beautifulsoup4`
- `pandas`
- `numpy`
- `matplotlib`
- `PySide6` (GUI mode)

## Installation

From the project root:

```bash
python -m venv .venv
```

Activate venv:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r movie_scraping/requirements.txt
```

## Quick Start

Run CLI:

```bash
python movie_scraping/main.py
```

Run GUI:

```bash
python movie_scraping/main.py --gui
```

## CLI Workflow

Main menu highlights:

- `1` Scrape latest prices and history
- `7` Manage people and budgets
- `8` Manage movie pool and release dates
- `11` Draft strategy dashboard
- `12` Draft state / assignment history

Recommended first run:

1. Scrape latest HSX data.
2. Add/verify people and remaining budgets.
3. Add/verify movies and release dates.
4. Open strategy dashboard and run baseline analysis.

## GUI Workflow

Dashboard tab includes:

- strategy controls (preset, risk model, objective, optimizer cost basis)
- advanced knobs (risk penalties, filters, simulation/search/correlation options)
- strategy profile dropdown + save/load buttons
- ranked table, selected portfolio, summary diagnostics

Draft State tab includes:

- current movie and bid context
- assign/unassign actions
- assigned movies and event history tables

## Strategy Dashboard Concepts

Core outputs:

- `adjusted_expected`: model-adjusted value proxy
- `target_bid` / `target_bid_int`: recommended bid level
- `max_bid` / `max_bid_int`: walk-away threshold
- `prob_positive_edge (P(Edge))`: confidence value exceeds bid threshold
- `prob_large_drawdown (P(DD))`: downside risk estimate
- `priority_score`: blended ranking score

Optimization modes:

- `expected_gross`: maximize expected portfolio value
- `win_probability`: maximize first-place probability vs simulated opponents

Monte Carlo supports:

- independent returns
- optional Gaussian/t-copula dependence
- opponent profile presets
- candidate search modes (`current_sampled`, `local_search`, `genetic`, etc.)

## Saved Strategy Profiles

Both CLI and GUI support named strategy profiles:

- save current knob settings under a custom profile name
- load a profile later for repeatable what-if runs

Notes:

- profiles are stored in `data/strategy_profiles.json`
- strategy runtime defaults still reset each fresh app run
- loading a profile is explicit and optional

## Draft-State Workflow

Use `Draft state / assignment history` to keep the model aligned with real draft progress:

1. Set current movie + current bid before bidding.
2. Assign winner with final price when auction closes.
3. Re-run dashboard after each assignment.

Assignment logic:

- assigned movie removed from available pool
- winner budget decremented by final price (if provided)
- action recorded in history

## Troubleshooting

- **GUI fails to launch**
  - install `PySide6` and dependencies
  - run `python movie_scraping/main.py --gui` again
- **No dashboard output**
  - verify `movies.csv` and `hsx_cache.csv` are populated
  - ensure basis budget resolves above zero
- **No legal bids in integer mode**
  - check current bid context and personal remaining budget
- **Correlation mode falls back to independent**
  - inspect summary `corr_fallback_reason`
  - history overlap may be insufficient
- **Different outcomes between runs**
  - use fixed seed for reproducibility
  - use random seed mode for robustness testing

## Notes and Limitations

- HSX prices are used as value proxies; results are decision support, not guaranteed outcomes.
- Opponent and correlation models are synthetic approximations.
- Output quality depends on freshness/completeness of local data and draft-state updates.

## Related Docs

- [strategy_guide.md](./strategy_guide.md): detailed analyst playbook
- [implementation_plan.md](./implementation_plan.md): phased implementation and status
