"""Concrete checks: input/score drift (now) and calibration (when labels exist)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from monitoring.config import (
    CALIBRATION_GAP_FLOOR,
    CALIBRATION_GAP_MULT,
    INTENT_MISSING_DELTA,
    MIN_BUCKET_N,
    N_SCORE_BUCKETS,
    SCORE_MEAN_STD_MULT,
)


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
