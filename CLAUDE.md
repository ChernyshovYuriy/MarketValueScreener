# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the screener (fetches live data, prints top 15, saves screener_results.csv)
python value_screener.py <url_or_path>   # one ticker per line; # lines ignored
```

No test suite, linter config, or build step exists.

## Architecture

Single-file script (`value_screener.py`) with five sequential stages:

1. **Universe** — `load_tickers(source)` fetches a plain-text file (HTTP(S) URL or local path) and returns a list of symbols; one per line, `#` comments skipped.
2. **Fetch** — `fetch_metrics()` pulls one ticker via `yf.Ticker(ticker).info`; `build_dataframe()` iterates the list with a 0.3s sleep between calls.
3. **Filter** — `score()` drops rows before ranking: negative trailing P/E, P/E > 100, D/E ≥ 300, market cap < $1B.
4. **Score** — percentile-ranks each metric within the surviving universe, then combines three sub-scores: valuation (50%), quality (35%), growth (15%). NaN metrics are filled to 0.5 (neutral) in the composite.
5. **Output** — `display()` prints top N to console; full ranked DataFrame writes to `screener_results.csv`.

### Scoring columns

| Direction | Columns |
|-----------|---------|
| Lower is better | `pe_trailing`, `pe_forward`, `pb`, `ps`, `ev_ebitda`, `peg`, `debt_to_equity` |
| Higher is better | `fcf_yield`, `roe`, `roa`, `gross_margin`, `operating_margin`, `profit_margin`, `current_ratio`, `revenue_growth`, `earnings_growth` |

### Key customization points

- **Universe**: pass any ticker file URL or local path — curated watchlist, sector slice, full index.
- **Weights**: edit the composite formula in `score()` — currently `0.50 * val + 0.35 * qual + 0.15 * growth`.
- **Filters**: edit the trap-filter block inside `score()`.
- **Sector-aware ranking**: the current approach ranks all tickers together; for sector-neutral scoring, group by `sector` and rank within groups.

### Data notes

- Data source is Yahoo Finance (free, no API key). Fields can be missing or stale — `None` values are handled via `fillna` throughout.
- `screener_results.csv` contains all post-filter rows, not just the top 15 displayed.
