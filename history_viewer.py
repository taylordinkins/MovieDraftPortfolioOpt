import tkinter as tk
from tkinter import ttk

import pandas as pd

import storage


def _ticker_label_map() -> dict[str, str]:
    """Return ticker -> cleaned movie display name from HSX cache when available."""
    cache = storage.load_cache()
    labels = {}
    if cache.empty:
        return labels

    for _, row in cache.iterrows():
        ticker = str(row.get('ticker', '')).strip().upper()
        name = row.get('name', '')
        if not ticker:
            continue
        if isinstance(name, str) and name.startswith(f'{ticker}: '):
            name = name[len(f'{ticker}: '):]
        labels[ticker] = name if isinstance(name, str) and name else ticker
    return labels


def _load_histories() -> dict[str, pd.DataFrame]:
    """Load all non-empty history files for movies in the pool."""
    movies_df = storage.load_movies()
    histories: dict[str, pd.DataFrame] = {}

    if movies_df.empty:
        return histories

    for ticker in movies_df['ticker'].dropna().astype(str).str.upper().unique():
        df = storage.load_price_history(ticker)
        if df is None or df.empty:
            continue

        if 'date' not in df.columns or 'price' not in df.columns:
            continue

        parsed = df.copy()
        parsed['date'] = pd.to_datetime(parsed['date'], errors='coerce')
        parsed['price'] = pd.to_numeric(parsed['price'], errors='coerce')
        parsed = parsed.dropna(subset=['date', 'price']).sort_values('date')
        if parsed.empty:
            continue

        histories[ticker] = parsed

    return histories


def show_price_history_viewer() -> bool:
    """
    Open an interactive window with a ticker dropdown for price history plots.

    Returns True if the window was shown, False if there was no data to show.
    """
    histories = _load_histories()
    if not histories:
        print('No saved history data found. Fetch price history first (Scrape Options -> 2).')
        return False

    # Local imports so the rest of the CLI still works even if matplotlib is missing.
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
    except Exception:
        print('matplotlib is required for plotting. Install dependencies from requirements.txt.')
        return False

    labels = _ticker_label_map()
    tickers = sorted(histories.keys())

    root = tk.Tk()
    root.title('HSX Price History Viewer')
    root.geometry('980x640')

    top = ttk.Frame(root, padding=(10, 10, 10, 5))
    top.pack(fill='x')
    ttk.Label(top, text='Ticker:').pack(side='left')

    selected = tk.StringVar(value=tickers[0])
    combo = ttk.Combobox(top, textvariable=selected, values=tickers, state='readonly', width=16)
    combo.pack(side='left', padx=(8, 0))

    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.92, left=0.08, right=0.98, bottom=0.11)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill='both', expand=True, padx=10, pady=(0, 6))

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    toolbar.pack(fill='x', padx=10, pady=(0, 10))

    def redraw_plot(ticker: str) -> None:
        df = histories[ticker]
        label = labels.get(ticker, ticker)

        ax.clear()
        ax.plot(df['date'], df['price'], linewidth=2.0, color='#1f77b4')
        ax.set_title(f'{ticker} - {label}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        canvas.draw_idle()

    def on_select(_event=None) -> None:
        redraw_plot(selected.get())

    combo.bind('<<ComboboxSelected>>', on_select)
    redraw_plot(selected.get())
    root.mainloop()
    return True
