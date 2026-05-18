"""
Tests for value_screener.py — zero live API calls, injected data only.

Acts as both financial advisor and Python engineer: each section covers code
correctness, filter edge cases, and real financial edge cases.  Advisory
findings are tagged [BUG], [DESIGN], or [BEHAVIOR] and explain the practical
implication.

Run with:  pytest test_screener.py -v
"""

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from value_screener import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    load_tickers,
    percentile_rank,
    score,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Sane baseline that passes every filter.  Override individual fields in tests.
_DEFAULTS = dict(
    ticker="X",
    name="Test Co",
    sector="Technology",
    pe_trailing=15.0,
    pe_forward=12.0,
    pb=2.0,
    ps=2.0,
    ev_ebitda=10.0,
    peg=1.0,
    fcf_yield=0.05,
    roe=0.15,
    roa=0.08,
    gross_margin=0.40,
    operating_margin=0.20,
    profit_margin=0.12,
    debt_to_equity=50.0,
    current_ratio=2.0,
    revenue_growth=0.10,
    earnings_growth=0.12,
    dividend_yield=0.02,
    held_by_insiders=0.05,
    short_ratio=2.0,
    market_cap=10e9,
    price=100.0,
)


def mk(**kwargs) -> dict:
    return {**_DEFAULTS, **kwargs}


def mkdf(*stocks) -> pd.DataFrame:
    return pd.DataFrame(list(stocks))


# ---------------------------------------------------------------------------
# 1. load_tickers
# ---------------------------------------------------------------------------

class TestLoadTickers:

    def test_parses_basic_file(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("AAPL\nMSFT\nGOOGL\n")
        assert load_tickers(str(f)) == ["AAPL", "MSFT", "GOOGL"]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("\nAAPL\n\nMSFT\n")
        assert load_tickers(str(f)) == ["AAPL", "MSFT"]

    def test_skips_comment_lines(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("# Tech\nAAPL\n  # inline\nMSFT\n")
        assert load_tickers(str(f)) == ["AAPL", "MSFT"]

    def test_uppercases_symbols(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("aapl\nMsft\n")
        assert load_tickers(str(f)) == ["AAPL", "MSFT"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("  AAPL  \n\tMSFT\t\n")
        assert load_tickers(str(f)) == ["AAPL", "MSFT"]

    def test_raises_on_empty_or_all_comments(self, tmp_path):
        f = tmp_path / "t.txt"
        f.write_text("# comment only\n\n")
        with pytest.raises(ValueError, match="No tickers found"):
            load_tickers(str(f))

    def test_http_url_is_fetched_not_opened_as_file(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b"AAPL\nMSFT\n"
        with patch("urllib.request.urlopen", return_value=mock_resp) as m:
            result = load_tickers("https://example.com/tickers.txt")
        m.assert_called_once_with("https://example.com/tickers.txt")
        assert result == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# 2. percentile_rank
# ---------------------------------------------------------------------------

class TestPercentileRank:

    def test_higher_is_better_ascending(self):
        r = percentile_rank(pd.Series([1.0, 2.0, 3.0]), higher_is_better=True)
        assert r.iloc[0] < r.iloc[1] < r.iloc[2]

    def test_lower_is_better_descending(self):
        r = percentile_rank(pd.Series([1.0, 2.0, 3.0]), higher_is_better=False)
        assert r.iloc[0] > r.iloc[1] > r.iloc[2]

    def test_directions_sum_to_one_per_element(self):
        s = pd.Series([10.0, 20.0, 30.0])
        hi = percentile_rank(s, higher_is_better=True)
        lo = percentile_rank(s, higher_is_better=False)
        for h, l in zip(hi, lo):
            assert h + l == pytest.approx(1.0)

    def test_nan_stays_nan(self):
        r = percentile_rank(pd.Series([1.0, float("nan"), 3.0]), higher_is_better=True)
        assert math.isnan(r.iloc[1])

    def test_nan_does_not_shift_other_ranks(self):
        # With na_option="keep", non-NaN values rank as if NaN doesn't exist
        r = percentile_rank(pd.Series([10.0, float("nan"), 30.0]), higher_is_better=True)
        assert r.iloc[0] < r.iloc[2]
        assert math.isnan(r.iloc[1])

    def test_identical_values_receive_equal_rank(self):
        r = percentile_rank(pd.Series([5.0, 5.0, 5.0]), higher_is_better=True)
        assert r.nunique() == 1

    def test_single_value_lower_is_better_ranks_zero(self):
        """
        [BEHAVIOR] The lone stock in a universe gets percentile rank 1.0 (highest),
        which inverts to 0.0 for valuation (lower-is-better metrics).
        That means a single surviving stock gets valuation_score = 0 — the worst
        possible valuation score — regardless of how cheap it actually is.
        """
        r = percentile_rank(pd.Series([42.0]), higher_is_better=False)
        assert r.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Filters — every boundary tested individually
# ---------------------------------------------------------------------------

class TestFilters:

    def _passes(self, **kwargs) -> bool:
        return len(score(mkdf(mk(**kwargs)))) == 1

    # --- P/E ---

    def test_negative_pe_filtered(self):
        assert not self._passes(pe_trailing=-1.0)

    def test_zero_pe_filtered(self):
        assert not self._passes(pe_trailing=0.0)

    def test_pe_exactly_100_filtered(self):
        # filter is `< 100`, so 100 itself must not pass
        assert not self._passes(pe_trailing=100.0)

    def test_pe_just_below_100_passes(self):
        assert self._passes(pe_trailing=99.99)

    def test_nan_pe_filtered(self):
        # fillna(-1) → -1 > 0 fails
        assert not self._passes(pe_trailing=float("nan"))

    # --- D/E ---

    def test_de_exactly_300_filtered(self):
        # filter is `< 300`
        assert not self._passes(debt_to_equity=300.0)

    def test_de_just_below_300_passes(self):
        assert self._passes(debt_to_equity=299.9)

    def test_nan_de_passes_as_zero(self):
        """
        [BUG] Missing D/E is treated as if the company has zero debt.
        A company that does not report leverage silently passes the filter,
        potentially hiding highly leveraged structures.
        """
        assert self._passes(debt_to_equity=float("nan"))

    # --- Market cap ---

    def test_market_cap_exactly_1b_filtered(self):
        # filter is `> 1e9`, so exactly 1 B is excluded
        assert not self._passes(market_cap=1e9)

    def test_market_cap_above_1b_passes(self):
        assert self._passes(market_cap=1e9 + 1)

    def test_nan_market_cap_filtered(self):
        # fillna(0) → 0 > 1e9 fails
        assert not self._passes(market_cap=float("nan"))

    def test_all_filters_independent(self):
        # Good P/E but over-levered
        assert not self._passes(pe_trailing=15.0, debt_to_equity=300.0)
        # Good D/E but negative P/E
        assert not self._passes(pe_trailing=-1.0, debt_to_equity=50.0)
        # Good P/E and D/E but micro-cap
        assert not self._passes(pe_trailing=15.0, debt_to_equity=50.0, market_cap=5e8)


# ---------------------------------------------------------------------------
# 4. Scoring logic
# ---------------------------------------------------------------------------

class TestScoring:

    def test_ideal_stock_ranks_first(self):
        """A stock dominating on every axis should unconditionally rank #1."""
        ideal = mk(
            ticker="IDEAL",
            pe_trailing=4.0, pe_forward=3.5, pb=0.4, ps=0.4,
            ev_ebitda=3.0, peg=0.3,
            roe=0.55, roa=0.28, gross_margin=0.72, operating_margin=0.44,
            profit_margin=0.33, fcf_yield=0.15,
            revenue_growth=0.42, earnings_growth=0.52,
        )
        ranked = score(mkdf(ideal, mk(ticker="AVG1"), mk(ticker="AVG2")))
        assert ranked.iloc[0]["ticker"] == "IDEAL"

    def test_cheap_and_quality_beats_pure_value_trap(self):
        """
        A stock that is BOTH cheap AND high-quality outranks a value trap only
        when a 4th stock breaks the perfect inverse correlation between valuation
        and quality.  With only 3 stocks in perfect inverse order all composites
        tie at 0.5, but adding an expensive+great-quality stock lifts CQ above TRAP.
        """
        # 4-stock universe: vary only pe_trailing and roe to isolate the effect.
        trap = mk(ticker="TRAP", pe_trailing=5.0, roe=0.02)          # cheapest, worst quality
        cheap_quality = mk(ticker="CQ", pe_trailing=8.0, roe=0.35)   # cheap AND good quality
        neutral = mk(ticker="NEUT")                                   # average (all defaults)
        expensive_quality = mk(ticker="EQ", pe_trailing=45.0, roe=0.40)  # expensive, best quality

        ranked = score(mkdf(trap, cheap_quality, neutral, expensive_quality))
        order = list(ranked["ticker"])
        assert order.index("CQ") < order.index("TRAP"), (
            "Cheap + quality should rank above a pure value trap once a 4th "
            "stock breaks the perfect val/qual inverse correlation."
        )

    def test_value_trap_floats_above_average_in_large_universe(self):
        """
        [BEHAVIOR/WARNING] A stock with the lowest P/E but otherwise AVERAGE
        quality ranks #1 in a universe of fairly-valued peers.  This is the
        screener's blind spot: a deteriorating business that still looks cheap
        on trailing P/E, with average-looking quality metrics, will dominate
        the output before those metrics turn negative.
        """
        # TRAP is cheapest on P/E; quality metrics are identical to peers (default).
        trap = mk(ticker="TRAP", pe_trailing=5.0)   # only P/E differs — quality is average
        avg1 = mk(ticker="A1", pe_trailing=15.0)
        avg2 = mk(ticker="A2", pe_trailing=20.0)
        avg3 = mk(ticker="A3", pe_trailing=25.0)

        ranked = score(mkdf(trap, avg1, avg2, avg3))
        order = list(ranked["ticker"])
        # Cheapest stock wins when quality is identical across the universe
        assert order[0] == "TRAP", (
            "The cheapest stock ranks #1 when quality is uniform — the 50% "
            "valuation weight is decisive."
        )

    def test_negative_fcf_ranks_last_on_quality(self):
        """A cash-burning company (negative FCF yield) should score lowest on quality."""
        burner = mk(ticker="BURN", fcf_yield=-0.15)
        generator = mk(ticker="GEN", fcf_yield=0.12)
        neutral = mk(ticker="N")
        ranked = score(mkdf(burner, generator, neutral))
        gen_q = ranked.loc[ranked["ticker"] == "GEN", "quality_score"].iloc[0]
        burn_q = ranked.loc[ranked["ticker"] == "BURN", "quality_score"].iloc[0]
        assert gen_q > burn_q

    def test_all_filtered_returns_empty_dataframe(self):
        bad = [mk(ticker=f"X{i}", pe_trailing=-1.0) for i in range(4)]
        assert len(score(mkdf(*bad))) == 0

    def test_scores_bounded_0_to_1(self):
        stocks = [mk(ticker=f"S{i}", pe_trailing=5.0 + i * 8, roe=0.05 + i * 0.06)
                  for i in range(6)]
        ranked = score(mkdf(*stocks))
        assert (ranked["composite_score"] >= 0.0).all()
        assert (ranked["composite_score"] <= 1.0).all()

    def test_ranking_is_input_order_independent(self):
        a = mk(ticker="A", pe_trailing=8.0, roe=0.30)
        b = mk(ticker="B", pe_trailing=20.0, roe=0.15)
        c = mk(ticker="C", pe_trailing=35.0, roe=0.05)
        assert (list(score(mkdf(a, b, c))["ticker"]) ==
                list(score(mkdf(c, a, b))["ticker"]))

    def test_partial_nan_metrics_do_not_crash(self):
        s = mk(pe_trailing=15.0,
               pe_forward=float("nan"), pb=float("nan"),
               roe=float("nan"), revenue_growth=float("nan"))
        result = score(mkdf(s))
        assert len(result) == 1
        assert not math.isnan(result.iloc[0]["composite_score"])

    def test_all_nan_quality_and_growth_fills_to_neutral_in_composite(self):
        """
        When quality and growth sub-scores are entirely NaN, they fill to 0.5
        in the composite.  The stock still gets a valid composite score.
        """
        s = mk(
            pe_trailing=15.0,
            roe=float("nan"), roa=float("nan"), gross_margin=float("nan"),
            operating_margin=float("nan"), profit_margin=float("nan"),
            fcf_yield=float("nan"),
            revenue_growth=float("nan"), earnings_growth=float("nan"),
        )
        result = score(mkdf(s))
        assert math.isnan(result.iloc[0]["quality_score"])
        assert math.isnan(result.iloc[0]["growth_score"])
        assert not math.isnan(result.iloc[0]["composite_score"])

    def test_universe_contamination_displaces_rank_position(self):
        """
        [BEHAVIOR] Scores are purely relative.  Adding one very cheap outlier
        displaces A from #1 to #2 even though A's fundamentals haven't changed.
        Screener results from different universes or dates are not comparable —
        a stock can "get worse" overnight simply because a cheaper peer was added.
        Note: the absolute valuation_score may actually RISE when a new stock
        enters (tied metrics' baseline shifts), so rank position — not score
        magnitude — is the right signal to watch.
        """
        a = mk(ticker="A", pe_trailing=15.0)
        b = mk(ticker="B", pe_trailing=18.0)

        rank_a_small = list(score(mkdf(a, b))["ticker"]).index("A")         # 0 (first)

        outlier = mk(ticker="CHEAP", pe_trailing=2.0)
        rank_a_large = list(score(mkdf(a, b, outlier))["ticker"]).index("A")  # 1 (second)

        assert rank_a_large > rank_a_small, (
            "Adding a cheaper outlier should push A down the ranking even "
            "though A's fundamentals are unchanged."
        )


# ---------------------------------------------------------------------------
# 5. Financial advisory findings
# ---------------------------------------------------------------------------

class TestAdvisoryFindings:
    """
    Each test documents a concrete flaw or surprising behavior.
    Tests assert the ACTUAL (possibly undesirable) behavior so it is
    visible and can be intentionally fixed — or intentionally accepted.
    """

    def test_BUG_current_ratio_collected_but_never_scored(self):
        """
        [BUG] current_ratio appears in HIGHER_IS_BETTER and is fetched from Yahoo,
        but is absent from qual_cols inside score().  A company on the verge of
        insolvency (current_ratio = 0.2) scores identically to a very liquid one
        (current_ratio = 5.0).  Liquidity is completely invisible to the screener.
        """
        liquid = mk(ticker="LIQ", current_ratio=5.0)
        illiquid = mk(ticker="ILL", current_ratio=0.2)
        neutral = mk(ticker="N")

        ranked = score(mkdf(liquid, illiquid, neutral))
        liq_q = ranked.loc[ranked["ticker"] == "LIQ", "quality_score"].iloc[0]
        ill_q = ranked.loc[ranked["ticker"] == "ILL", "quality_score"].iloc[0]

        assert liq_q == pytest.approx(ill_q), (
            "current_ratio is excluded from qual_cols — liquidity has zero effect."
        )

    def test_BUG_module_constants_LOWER_HIGHER_IS_BETTER_are_dead_code(self):
        """
        [BUG] LOWER_IS_BETTER and HIGHER_IS_BETTER are defined at module level but
        score() hard-codes its own column lists and never references them.
        Editing the constants has no effect whatsoever on scoring.
        Notable victims: current_ratio (in HIGHER_IS_BETTER, not scored) and
        debt_to_equity (in LOWER_IS_BETTER, only used as a filter).
        """
        assert "current_ratio" in HIGHER_IS_BETTER   # listed but not scored
        assert "debt_to_equity" in LOWER_IS_BETTER   # listed but not scored

    def test_DESIGN_debt_to_equity_is_a_gate_not_a_score(self):
        """
        [DESIGN] D/E only acts as a binary pass/fail gate (< 300).
        A near-debt-free company (D/E = 5) scores identically on quality to a
        highly leveraged one (D/E = 299).  Prudent capital structure is not rewarded.
        """
        low_debt = mk(ticker="LD", debt_to_equity=5.0)
        high_debt = mk(ticker="HD", debt_to_equity=299.0)
        neutral = mk(ticker="N")

        ranked = score(mkdf(low_debt, high_debt, neutral))
        ld_q = ranked.loc[ranked["ticker"] == "LD", "quality_score"].iloc[0]
        hd_q = ranked.loc[ranked["ticker"] == "HD", "quality_score"].iloc[0]

        assert ld_q == pytest.approx(hd_q), (
            "D/E is not in qual_cols — leverage level has zero effect on quality_score."
        )

    def test_DESIGN_banks_systematically_excluded(self):
        """
        [DESIGN] Banks are structurally leveraged: D/E of 800-1200 is normal
        because customer deposits count as liabilities.  The blanket D/E < 300
        cutoff eliminates ALL typical banks — including healthy, profitable ones.
        This screener is effectively inapplicable to the Financials sector.
        """
        healthy_bank = mk(
            ticker="BNK",
            sector="Financial Services",
            pe_trailing=11.0,
            roe=0.13,
            debt_to_equity=900.0,   # normal for a large bank
        )
        assert len(score(mkdf(healthy_bank))) == 0, (
            "A healthy bank with D/E=900 is incorrectly excluded."
        )

    def test_BUG_valuation_score_fillna_is_unreachable(self):
        """
        [BUG] The composite formula uses valuation_score.fillna(0.5), but
        valuation_score can NEVER be NaN in practice: pe_trailing must be a valid
        positive number to survive the first two filters, so val_ranks.mean()
        always has at least one non-NaN input.  The fillna(0.5) branch for
        valuation_score is dead code that creates a false sense of safety.
        """
        # All val metrics except pe_trailing are NaN
        s = mk(
            pe_trailing=15.0,
            pe_forward=float("nan"), pb=float("nan"), ps=float("nan"),
            ev_ebitda=float("nan"), peg=float("nan"),
        )
        result = score(mkdf(s))
        # valuation_score is derived from pe_trailing alone — NOT NaN, NOT 0.5
        val_score = result.iloc[0]["valuation_score"]
        assert not math.isnan(val_score)
        assert val_score != pytest.approx(0.5)  # it is 0.0 (single stock, lowest rank inverted)

    def test_DESIGN_peg_undefined_for_mature_low_growth_companies(self):
        """
        [DESIGN] PEG is undefined (NaN) for companies with zero or negative expected
        growth — precisely the mature, capital-light businesses that deep-value
        investors target.  Their PEG contribution is skipped (skipna=True), but
        peers with a low PEG get a full bonus on that column, giving growth-oriented
        cheap stocks a systematic edge over classic value names.
        """
        no_peg = mk(ticker="NPEG", pe_trailing=8.0, peg=float("nan"))
        low_peg = mk(ticker="LPEG", pe_trailing=8.0, peg=0.4)   # same P/E, has PEG
        neutral = mk(ticker="N")

        ranked = score(mkdf(no_peg, low_peg, neutral))
        npeg_val = ranked.loc[ranked["ticker"] == "NPEG", "valuation_score"].iloc[0]
        lpeg_val = ranked.loc[ranked["ticker"] == "LPEG", "valuation_score"].iloc[0]

        assert lpeg_val > npeg_val, (
            "Same P/E but having a low PEG gives a higher valuation_score — "
            "mature no-growth companies are systematically disadvantaged."
        )
