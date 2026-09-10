"""Load the frozen model and attach conversion probabilities. Do not retrain."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from agent.config import DEFAULT_FEATURE_COLUMNS, NON_FEATURE_COLUMNS


def _positive_class_index(model) -> int:
    """Index of the conversion class in predict_proba columns.

    sklearn usually puts the positive class at index 1 (classes_ = [0, 1] or
    [False, True]). Pipelines store classes_ on the final estimator, not the
    pipeline itself. If we cannot tell, fall back to column 1.
    """
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        last = list(model.named_steps.values())[-1]
        classes = getattr(last, "classes_", None)
    if classes is None:
        return 1
    classes = list(classes)
    for marker in (1, True, "1"):
        if marker in classes:
            return classes.index(marker)
    return 1 if len(classes) > 1 else 0


def _feature_columns(model, df: pd.DataFrame) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        first = list(model.named_steps.values())[0]
        names = getattr(first, "feature_names_in_", None)
    if names is not None:
        return [str(c) for c in names]
    available = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c not in available]
    # Prefer the brief's list; ignore leftover id/target-like columns.
    _ = extras
    return available


def load_and_score(csv_path: str | Path, model_path: str | Path) -> pd.DataFrame:
    """Load accounts + frozen pipeline; add `score` = P(convert).

    Does not fit, does not write back to data/. Feature columns come from the
    fitted model when available so we don't pass account_id into predict.
    """
    csv_path = Path(csv_path)
    model_path = Path(model_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Drop the matching starter CSV in place."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Drop the starter file model/model.pkl in place. "
            "Do not retrain a substitute."
        )

    df = pd.read_csv(csv_path)
    model = joblib.load(model_path)

    feature_cols = _feature_columns(model, df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Model expects columns not in the CSV: {missing}")

    # Pass a DataFrame so a ColumnTransformer keyed on names still works.
    X = df[feature_cols]
    proba = model.predict_proba(X)
    pos = _positive_class_index(model)

    out = df.copy()
    out["score"] = proba[:, pos]
    return out
