"""Sort by score and bucket into High / Medium / Low for a weekly call list."""

from __future__ import annotations

import pandas as pd

from agent.config import HIGH_QUANTILE, MEDIUM_QUANTILE


def rank_and_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Sort descending by score; add 1-based rank and High/Medium/Low tier.

    Thresholds are within-batch quantiles, not absolute P(convert):
      High   = score >= 90th percentile of this file (top 10%)
      Medium = score >= 70th percentile (next 20%)
      Low    = the rest

    Why quantiles: the brief's base rates are well under 1% cold / low-single-
    digits engaged. Model scores will likely cluster low. A 0.70 cutoff would
    not mean "70% will convert" and might create an empty High bucket.

    These are call-priority tiers for this batch, not calibrated probabilities.
    Absolute quality (score ≥ 0.10, p90 vs training) lives in monitoring, not
    here — so a worse batch still gets a full High list for SDR capacity, but
    the monitor can trip when those Highs no longer clear the historical bar.
    """
    out = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)

    high_cut = out["score"].quantile(HIGH_QUANTILE)
    med_cut = out["score"].quantile(MEDIUM_QUANTILE)

    def _tier(score: float) -> str:
        if score >= high_cut:
            return "High"
        if score >= med_cut:
            return "Medium"
        return "Low"

    out["tier"] = out["score"].map(_tier)
    return out
