# Value Stock Screener

A working Python screener that ranks stocks by a composite of valuation, quality, and growth metrics.

## Setup

```bash
pip install yfinance pandas numpy
python value_screener.py <url_or_path>
```

The argument is an HTTP(S) URL or a local file path to a plain-text file with **one ticker symbol per line**. Blank lines and lines starting with `#` are ignored.

```
# my_tickers.txt
AAPL
MSFT
GOOGL
# add more below
JPM
```

Output prints to console and saves two files:
- `screener_results.html` — styled HTML report of the top 15 (open in any browser)
- `screener_results.csv` — full ranked list of all stocks that passed filters

## Running tests

```bash
pip install pytest
pytest test_screener.py -v
```

Tests use injected data only — no live API calls, no network required.

## What it does

1. **Fetches** fundamentals for every ticker in the provided list via Yahoo Finance (free, no API key)
2. **Filters** obvious traps: negative earnings, extreme leverage, market cap under $1B, P/E over 100
3. **Scores** each surviving stock on three dimensions (percentile-ranked within the universe):
   - Valuation (50% weight): P/E trailing & forward, P/B, P/S, EV/EBITDA, PEG
   - Quality (35% weight): ROE, ROA, gross/operating/profit margins, FCF yield
   - Growth (15% weight): revenue growth, earnings growth
4. **Ranks** by composite score and outputs the top 15

## Customizing

- **Universe:** point the script at any ticker file — a curated watchlist, a sector slice, or a full index. Keep one symbol per line; `#` comments are allowed.
- **Weights:** edit the composite formula in `score()`. Pure deep-value? Bump valuation to 0.7.
- **Filters:** edit the "trap filter" block in `score()`. Tighter debt thresholds, minimum margin requirements, sector exclusions, etc.

## Important caveats

- **Yahoo data is free and occasionally wrong.** Restatements, missing values, sector mislabels. For real money, pay for a real data source (Financial Modeling Prep, EOD Historical, Polygon).
- **Trailing metrics lag reality.** A company can have a great trailing P/E because earnings cratered last quarter. Cross-check with forward P/E and recent news.
- **Composite ranking is sector-blind.** A bank's P/B of 1.2 means something different from a software company's. For better results, rank *within* sectors — group by `sector` and percentile-rank each group separately.
- **No catalyst detection.** This finds statistically cheap + high-quality names. Whether the market will *re-rate* them is a separate question that requires reading filings, news, and earnings calls.
- **Survivorship bias.** Your ticker list only contains companies that exist today. Backtesting this strategy properly requires point-in-time data including delisted names.

## Sensible next steps

1. Run it, look at the top 15
2. For each surviving name, read the latest 10-K and recent earnings call transcript
3. Build a DCF on the 3-5 most interesting names
4. Check insider transactions (Form 4 filings on SEC EDGAR or openinsider.com)
5. Only then consider position sizing

Screening is the first 5% of the work, not the whole job.
