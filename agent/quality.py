"""Flag data-quality issues. Do not impute or drop rows — scores stay as-is."""

from __future__ import annotations

import pandas as pd

from agent.config import TODAY

VALID_ACCOUNT_TYPES = {"Prospect", "Suspect", "Former Customer"}


def _truthy(series: pd.Series) -> pd.Series:
    """trial_started may arrive as bool, 0/1, or strings."""
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return series.isin([1, 1.0, True]) | lowered.isin(["1", "true", "yes", "y"])


def flag_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flag columns. Missingness is a signal, not a hole to fill.

    The frozen model may already impute intent_score internally. We still flag
    it so a rep does not treat every score as equally well-supported. That is
    the trust-calibration the agent exists for — ranking alone is not enough.
    """
    out = df.copy()

    # Required by the brief. Coverage is ~40% missing and not at random
    # (skews toward smaller companies). Missing ≠ low intent.
    out["intent_score_missing"] = out["intent_score"].isna()

    # Product usage should not exist without a trial. If it does, the row is
    # internally inconsistent — worth showing, not "fixing."
    trial_users = pd.to_numeric(out.get("trial_active_users"), errors="coerce").fillna(0)
    started = _truthy(out["trial_started"]) if "trial_started" in out.columns else False
    out["trial_users_without_trial"] = (trial_users > 0) & (~started)

    out["duplicate_account_id"] = out["account_id"].duplicated(keep=False)

    out["invalid_account_type"] = ~out["account_type"].isin(VALID_ACCOUNT_TYPES)

    # Placeholder bounds until training_data.csv is in hand. employee_count <= 0
    # is impossible; > 50k is rare for this B2B set and worth a glance.
    # Tighten after we see actual percentiles.
    emp = pd.to_numeric(out.get("employee_count"), errors="coerce")
    out["employee_count_outlier"] = emp.isna() | (emp <= 0) | (emp > 50_000)

    # Counts cannot be negative. Flag rather than clip.
    count_cols = [
        "mql_count_90d",
        "trial_active_users",
        "web_touchpoints_90d",
        "sales_contacts_90d",
    ]
    negative = pd.Series(False, index=out.index)
    for col in count_cols:
        if col in out.columns:
            negative = negative | (pd.to_numeric(out[col], errors="coerce") < 0)
    out["negative_activity"] = negative

    # Intent vendor scores are typically 0–100. If the file uses 0–1, values
    # stay inside this range and this flag stays quiet. Revisit after seeing data.
    intent = pd.to_numeric(out["intent_score"], errors="coerce")
    present = out["intent_score"].notna()
    out["intent_score_out_of_range"] = present & ((intent < 0) | (intent > 100))

    if "snapshot_date" in out.columns:
        snap = pd.to_datetime(out["snapshot_date"], errors="coerce")
        # Stale vs the exercise "today". Not a model feature; a batch-hygiene flag.
        out["snapshot_not_as_of_today"] = snap.dt.date != TODAY
    else:
        out["snapshot_not_as_of_today"] = False

    # Composite for the call list: score rests on thinner evidence.
    # Missing intent AND no trial → the two strongest "someone is in-market"
    # signals are both absent. Still show the score; label the uncertainty.
    no_trial = ~started if isinstance(started, pd.Series) else True
    out["weak_signal"] = out["intent_score_missing"] & no_trial

    return out
