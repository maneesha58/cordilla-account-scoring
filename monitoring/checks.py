"""Concrete checks: input/score/feature drift (now) and calibration (when labels exist)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from monitoring.config import (
    CALIBRATION_GAP_FLOOR,
    CALIBRATION_GAP_MULT,
    FEATURE_DRIFT_WATCH_COUNT,
    HIGH_SCORE_BAR,
    INTENT_MISSING_DELTA,
    MIN_BUCKET_N,
    N_SCORE_BUCKETS,
    P90_DROP,
    PSI_SHIFT,
    PSI_WATCH,
    SCORE_MEAN_STD_MULT,
    SHARE_ABOVE_BAR_DROP,
)
from monitoring.feature_drift import feature_psi_table


def _as_converted(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0) > 0).astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(["1", "true", "yes", "y"]).astype(int)


def intent_missing_rate(df: pd.DataFrame) -> float:
    return float(df["intent_score"].isna().mean())


def score_mean(df: pd.DataFrame) -> float:
    return float(df["score"].mean())


def score_std(df: pd.DataFrame) -> float:
    return float(df["score"].std(ddof=0) or 0.0)


def calibration_table(df: pd.DataFrame, n_buckets: int = N_SCORE_BUCKETS) -> list[dict[str, Any]]:
    """Bucket by score; compare mean predicted vs actual conversion.

    This is the Cordilla failure: scores still look 'healthy' (job ran,
    mean didn't explode) while high-score accounts stop converting.
    """
    if "converted_within_90d" not in df.columns:
        return []
    work = df[["score", "converted_within_90d"]].copy()
    work["y"] = _as_converted(work["converted_within_90d"])
    try:
        work["bucket"] = pd.qcut(
            work["score"],
            q=n_buckets,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return []

    rows: list[dict[str, Any]] = []
    for b, grp in work.groupby("bucket", sort=True):
        n = int(len(grp))
        mean_pred = float(grp["score"].mean())
        actual = float(grp["y"].mean())
        rows.append(
            {
                "bucket": int(b),
                "n": n,
                "mean_predicted": mean_pred,
                "actual_rate": actual,
                "gap": abs(mean_pred - actual),
                "enough_n": n >= MIN_BUCKET_N,
            }
        )
    return rows


def weighted_calibration_gap(table: list[dict[str, Any]]) -> float | None:
    usable = [r for r in table if r["enough_n"]]
    if not usable:
        return None
    total = sum(r["n"] for r in usable)
    return float(sum(r["gap"] * r["n"] for r in usable) / total)


def rank_inverted(table: list[dict[str, Any]]) -> bool:
    """High-score bucket converts worse than low-score bucket — ordering died."""
    if len(table) < 2:
        return False
    low = table[0]
    high = table[-1]
    if not (low["enough_n"] and high["enough_n"]):
        return False
    return high["actual_rate"] < low["actual_rate"]


def score_p90(df: pd.DataFrame) -> float:
    return float(df["score"].quantile(0.90))


def share_above_bar(df: pd.DataFrame, bar: float = HIGH_SCORE_BAR) -> float:
    return float((df["score"] >= bar).mean())


def check_intent_drift(batch_rate: float, baseline_rate: float) -> dict[str, Any]:
    delta = abs(batch_rate - baseline_rate)
    return {
        "name": "intent_drift",
        "tripped": delta > INTENT_MISSING_DELTA,
        "batch": batch_rate,
        "baseline": baseline_rate,
        "delta": delta,
        "threshold": INTENT_MISSING_DELTA,
        "what": (
            "Share of accounts with missing intent_score vs training. "
            "A coverage jump means scores are resting on a different mix "
            "(vendor gap, not 'low intent')."
        ),
    }


def check_score_drift(batch_mean: float, baseline_mean: float, baseline_std: float) -> dict[str, Any]:
    std = baseline_std if baseline_std > 1e-9 else 1.0
    delta = abs(batch_mean - baseline_mean)
    threshold = SCORE_MEAN_STD_MULT * std
    return {
        "name": "score_drift",
        "tripped": delta > threshold,
        "batch": batch_mean,
        "baseline": baseline_mean,
        "delta": delta,
        "threshold": threshold,
        "what": (
            "Mean model score vs training. Catches a silent mix shift "
            "(smaller companies, fewer trials) before labels exist."
        ),
    }


def check_p90_drop(batch_p90: float, baseline_p90: float) -> dict[str, Any]:
    """Quantile High still names 30 accounts even if this week's 'best' got worse.

    p90 falling while the call list still looks tidy is the Cordilla mask:
    relative tiers renormalize every batch; this check does not.
    """
    drop = baseline_p90 - batch_p90
    return {
        "name": "p90_drop",
        "tripped": drop > P90_DROP,
        "batch": batch_p90,
        "baseline": baseline_p90,
        "delta": drop,
        "threshold": P90_DROP,
        "what": (
            f"90th-percentile score vs training. Call-list High is always "
            f"top 10% of *this* file. If p90 drops >{P90_DROP:.2f}, those Highs "
            f"are relatively less-bad, not still above the historical bar."
        ),
    }


def check_share_above_bar(
    batch_share: float,
    baseline_share: float,
    bar: float = HIGH_SCORE_BAR,
) -> dict[str, Any]:
    drop = baseline_share - batch_share
    return {
        "name": "share_above_bar",
        "tripped": drop > SHARE_ABOVE_BAR_DROP,
        "batch": batch_share,
        "baseline": baseline_share,
        "delta": drop,
        "threshold": SHARE_ABOVE_BAR_DROP,
        "bar": bar,
        "what": (
            f"Share of accounts with score ≥ {bar:.2f} (the calibration bar "
            f"where actual conversion beat the 6.5% baseline). A 10pp drop "
            f"means fewer accounts clearing a real quality line, even if "
            f"quantile High still has 30 rows."
        ),
    }


def check_feature_drift(
    df: pd.DataFrame,
    feature_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Trip if mix of *inputs* moved vs training, even when mean score did not.

    Bins are whatever period 0 stored. Re-binning this file would hide the shift.
    """
    if not feature_baseline:
        return {
            "name": "feature_drift",
            "tripped": False,
            "skipped": True,
            "reason": "no frozen feature baseline",
            "what": "PSI of model features vs training, bins frozen at period 0.",
        }
    table = feature_psi_table(df, feature_baseline)
    if not table:
        return {
            "name": "feature_drift",
            "tripped": False,
            "skipped": True,
            "reason": "no overlapping feature columns",
            "what": "PSI of model features vs training, bins frozen at period 0.",
        }
    max_psi = max(r["psi"] for r in table)
    n_watch = sum(1 for r in table if r["psi"] > PSI_WATCH)
    n_shift = sum(1 for r in table if r["psi"] > PSI_SHIFT)
    tripped = n_shift >= 1 or n_watch >= FEATURE_DRIFT_WATCH_COUNT
    return {
        "name": "feature_drift",
        "tripped": tripped,
        "skipped": False,
        "max_psi": max_psi,
        "n_watch": n_watch,
        "n_shift": n_shift,
        "threshold_watch": PSI_WATCH,
        "threshold_shift": PSI_SHIFT,
        "features": table,
        "what": (
            "Population Stability Index on model features vs training "
            f"(bins frozen at period 0). Trips if any PSI > {PSI_SHIFT:.2f} "
            f"or {FEATURE_DRIFT_WATCH_COUNT}+ features > {PSI_WATCH:.2f}. "
            "Catches mix shifts that cancel in the mean score. "
            "intent_score uses observed values only; missingness is intent_drift."
        ),
    }


def check_calibration(
    table: list[dict[str, Any]],
    baseline_table: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Trip if a large-enough bucket's gap blows past 2× train gap or 3pp.

    baseline_table is period-0 (training). Comparing train to itself never
    trips the multiplier; we still surface rank inversion as a watch.
    """
    if not table:
        return {
            "name": "calibration",
            "tripped": False,
            "skipped": True,
            "reason": "no converted_within_90d labels on this file yet",
            "what": "Predicted vs actual conversion by score bucket.",
        }

    baseline_by_bucket = {}
    if baseline_table:
        baseline_by_bucket = {r["bucket"]: r for r in baseline_table}

    bucket_trips = []
    for row in table:
        if not row["enough_n"]:
            continue
        base = baseline_by_bucket.get(row["bucket"])
        base_gap = float(base["gap"]) if base else 0.0
        limit = max(CALIBRATION_GAP_FLOOR, CALIBRATION_GAP_MULT * base_gap)
        # Period 0: baseline is this same table, multiplier never trips.
        # Still record the limit so later labeled batches have a number.
        same_as_baseline = (
            base is not None
            and abs(base["gap"] - row["gap"]) < 1e-12
            and abs(base["mean_predicted"] - row["mean_predicted"]) < 1e-12
        )
        tripped = (not same_as_baseline) and row["gap"] > limit
        bucket_trips.append({**row, "limit": limit, "tripped": tripped})

    inverted = rank_inverted(table)
    overall = weighted_calibration_gap(table)
    tripped = any(b["tripped"] for b in bucket_trips)
    return {
        "name": "calibration",
        "tripped": tripped,
        "skipped": False,
        "watch_inversion": inverted,
        "overall_gap": overall,
        "buckets": bucket_trips,
        "what": (
            "Does a high score still mean a higher conversion rate? "
            "This is the quiet failure that burned the last Cordilla scorer."
        ),
    }
