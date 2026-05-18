"""
Value Stock Screener
--------------------
Pulls fundamentals for a list of tickers, computes valuation + quality metrics,
and outputs a ranked shortlist of potentially undervalued, high-quality stocks.

Approach:
  1. Valuation score: cheaper on P/E, P/B, P/S, EV/EBITDA, PEG (lower = better)
  2. Quality score: higher ROE, ROA, margins; lower debt; positive FCF
  3. Composite rank = average of valuation and quality percentile ranks
  4. Filter out obvious traps (negative earnings, extreme leverage, etc.)

Requirements:
    pip install yfinance pandas numpy

Usage:
    python value_screener.py <url_or_path>

    The argument may be an HTTP(S) URL or a local file path pointing to a
    plain-text file with one ticker symbol per line. Blank lines and lines
    starting with '#' are ignored.
"""

import argparse
import datetime
import time
import urllib.request
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. UNIVERSE — loaded at runtime from a URL or local file.
# ---------------------------------------------------------------------------
def load_tickers(source: str) -> list[str]:
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            text = resp.read().decode()
    else:
        with open(source) as f:
            text = f.read()

    tickers = [
        line.strip().upper()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not tickers:
        raise ValueError(f"No tickers found in '{source}'.")
    return tickers


# ---------------------------------------------------------------------------
# 2. METRIC EXTRACTION
# ---------------------------------------------------------------------------
def fetch_metrics(ticker: str) -> dict | None:
    """Pull all the fields we need for one ticker. Returns None on failure."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or "marketCap" not in info:
            return None

        return {
            "ticker": ticker,
            "name": info.get("shortName"),
            "sector": info.get("sector"),
            "market_cap": info.get("marketCap"),
            "price": info.get("currentPrice"),

            # --- Valuation ---
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "pb": info.get("priceToBook"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "peg": info.get("pegRatio"),
            "fcf_yield": (info.get("freeCashflow") or 0) / info["marketCap"]
            if info.get("marketCap") else None,

            # --- Quality ---
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),

            # --- Financial health ---
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),

            # --- Growth ---
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),

            # --- Sentiment / extras ---
            "dividend_yield": info.get("dividendYield"),
            "held_by_insiders": info.get("heldPercentInsiders"),
            "short_ratio": info.get("shortRatio"),
        }
    except Exception as e:
        print(f"  ! {ticker}: {e}")
        return None


def build_dataframe(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}")
        data = fetch_metrics(t)
        if data:
            rows.append(data)
        time.sleep(0.3)  # polite to the API
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. SCORING
# ---------------------------------------------------------------------------
# Columns where LOWER is better (cheap valuation = good)
LOWER_IS_BETTER = ["pe_trailing", "pe_forward", "pb", "ps", "ev_ebitda", "peg",
                   "debt_to_equity"]

# Columns where HIGHER is better (quality + growth)
HIGHER_IS_BETTER = ["fcf_yield", "roe", "roa", "gross_margin", "operating_margin",
                    "profit_margin", "current_ratio", "revenue_growth",
                    "earnings_growth"]


def percentile_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Rank each value as a percentile (0-1). NaNs stay NaN."""
    ranks = series.rank(pct=True, na_option="keep")
    return ranks if higher_is_better else 1 - ranks


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Trap filters: drop obvious garbage before ranking ---
    df = df[df["pe_trailing"].fillna(-1) > 0]  # no losses
    df = df[df["pe_trailing"].fillna(9999) < 100]  # no froth
    df = df[df["debt_to_equity"].fillna(0) < 300]  # not over-levered
    df = df[df["market_cap"].fillna(0) > 1e9]  # >$1B market cap

    # --- Valuation score (avg percentile across cheap-is-good metrics) ---
    val_cols = ["pe_trailing", "pe_forward", "pb", "ps", "ev_ebitda", "peg"]
    val_ranks = pd.DataFrame({c: percentile_rank(df[c], False) for c in val_cols})
    df["valuation_score"] = val_ranks.mean(axis=1, skipna=True)

    # --- Quality score ---
    qual_cols = ["roe", "roa", "gross_margin", "operating_margin",
                 "profit_margin", "fcf_yield"]
    qual_ranks = pd.DataFrame({c: percentile_rank(df[c], True) for c in qual_cols})
    df["quality_score"] = qual_ranks.mean(axis=1, skipna=True)

    # --- Growth score (lighter weight) ---
    growth_cols = ["revenue_growth", "earnings_growth"]
    growth_ranks = pd.DataFrame({c: percentile_rank(df[c], True) for c in growth_cols})
    df["growth_score"] = growth_ranks.mean(axis=1, skipna=True)

    # --- Composite: 50% valuation, 35% quality, 15% growth ---
    df["composite_score"] = (
            0.50 * df["valuation_score"].fillna(0.5) +
            0.35 * df["quality_score"].fillna(0.5) +
            0.15 * df["growth_score"].fillna(0.5)
    )

    return df.sort_values("composite_score", ascending=False)


# ---------------------------------------------------------------------------
# 4. OUTPUT
# ---------------------------------------------------------------------------
def display(df: pd.DataFrame, top_n: int = 15):
    cols = ["ticker", "name", "sector", "pe_trailing", "pb", "ev_ebitda",
            "peg", "roe", "debt_to_equity", "fcf_yield",
            "valuation_score", "quality_score", "composite_score"]
    out = df[cols].head(top_n).copy()

    # Pretty formatting
    for c in ["pe_trailing", "pb", "ev_ebitda", "peg", "debt_to_equity"]:
        out[c] = out[c].round(2)
    for c in ["roe", "fcf_yield"]:
        out[c] = (out[c] * 100).round(2).astype(str) + "%"
    for c in ["valuation_score", "quality_score", "composite_score"]:
        out[c] = out[c].round(3)

    print("\n" + "=" * 100)
    print(f"TOP {top_n} CANDIDATES — RANKED BY COMPOSITE SCORE")
    print("=" * 100)
    print(out.to_string(index=False))
    print("\nNote: this is a screening starting point, NOT a buy list.")
    print("Do qualitative due diligence on each name before any decision.")


def save_html(df: pd.DataFrame, top_n: int = 15, filename: str = "screener_results.html"):
    cols = ["ticker", "name", "sector", "pe_trailing", "pb", "ev_ebitda",
            "peg", "roe", "debt_to_equity", "fcf_yield",
            "valuation_score", "quality_score", "composite_score"]
    out = df[cols].head(top_n).copy()

    for c in ["pe_trailing", "pb", "ev_ebitda", "peg", "debt_to_equity"]:
        out[c] = out[c].apply(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    for c in ["roe", "fcf_yield"]:
        out[c] = out[c].apply(lambda x: "—" if pd.isna(x) else f"{x * 100:.2f}%")
    for c in ["valuation_score", "quality_score", "composite_score"]:
        out[c] = out[c].apply(lambda x: "—" if pd.isna(x) else f"{x:.3f}")

    out.columns = ["Ticker", "Name", "Sector", "P/E (TTM)", "P/B", "EV/EBITDA",
                   "PEG", "ROE", "D/E", "FCF Yield", "Val. Score", "Qual. Score", "Composite"]

    table_html = out.to_html(index=False, classes="results-table", border=0)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Value Screener — {timestamp}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f2f5;
    color: #1a1a1a;
    margin: 0;
    padding: 32px 24px;
  }}
  h1 {{ margin: 0 0 4px; font-size: 1.4rem; letter-spacing: -.3px; }}
  .meta {{ color: #888; font-size: 0.82rem; margin-bottom: 24px; }}
  .wrapper {{ overflow-x: auto; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .results-table {{
    border-collapse: collapse;
    width: 100%;
    background: #fff;
    font-size: 0.875rem;
  }}
  .results-table th {{
    background: #0f172a;
    color: #e2e8f0;
    padding: 11px 16px;
    text-align: left;
    white-space: nowrap;
    font-weight: 600;
  }}
  .results-table th:first-child {{ border-radius: 10px 0 0 0; }}
  .results-table th:last-child  {{ border-radius: 0 10px 0 0; }}
  .results-table td {{
    padding: 9px 16px;
    border-bottom: 1px solid #f1f5f9;
    white-space: nowrap;
  }}
  .results-table tr:last-child td {{ border-bottom: none; }}
  .results-table tr:nth-child(even) td {{ background: #fafbfc; }}
  .results-table tr:hover td {{ background: #eff6ff; }}
  .note {{ margin-top: 14px; font-size: 0.78rem; color: #94a3b8; }}
</style>
</head>
<body>
<h1>Value Screener — Top {top_n}</h1>
<p class="meta">Generated {timestamp} &nbsp;·&nbsp; {len(df)} stocks passed filters</p>
<div class="wrapper">
{table_html}
</div>
<p class="note">Screening starting point only — not a buy list. Do qualitative due diligence before any decision.</p>
</body>
</html>"""

    with open(filename, "w") as f:
        f.write(html)
    print(f"HTML report saved to {filename}")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Value stock screener — ranks tickers by valuation, quality, and growth."
    )
    parser.add_argument(
        "url",
        help="HTTP(S) URL or local file path to a plain-text ticker list (one symbol per line).",
    )
    args = parser.parse_args()

    print(f"Load from '{args.url}'.")
    tickers = load_tickers(args.url)
    print(f"Loaded {len(tickers)} tickers from '{args.url}'.")
    print(f"Fetching data...\n")
    df = build_dataframe(tickers)
    print(f"\nSuccessfully pulled {len(df)} tickers.")

    ranked = score(df)
    print(f"After trap filters: {len(ranked)} candidates remain.\n")

    display(ranked, top_n=15)

    ranked.to_csv("screener_results.csv", index=False)
    save_html(ranked, top_n=15)
    print("Full results also saved to screener_results.csv")
