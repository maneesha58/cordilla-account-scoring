"""Period-0 snapshot from labeled training data + the frozen model. No fit()."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.config import MODEL_PATH, TRAINING_PATH
from agent.score import load_and_score
from monitoring.checks import (
    calibration_table,
    intent_missing_rate,
    score_mean,
    score_p90,
    score_std,
    share_above_bar,
    weighted_calibration_gap,
)
from monitoring.feature_drift import build_feature_baseline
from monitoring.config import AS_OF, BASELINE_PATH, HIGH_SCORE_BAR


def build_baseline(
    training_path: Path = TRAINING_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    scored = load_and_score(training_path, model_path)
    cal = calibration_table(scored)
    return {
        "as_of": AS_OF,
        "source": str(training_path),
        "n": int(len(scored)),
        "intent_missing_rate": intent_missing_rate(scored),
        "mean_score": score_mean(scored),
        "std_score": score_std(scored),
        "p90_score": score_p90(scored),
        "share_above_bar": share_above_bar(scored),
        "high_score_bar": HIGH_SCORE_BAR,
        "calibration": cal,
        "overall_calibration_gap": weighted_calibration_gap(cal),
        "feature_baseline": build_feature_baseline(scored),
        "note": (
            "Built by scoring training_data.csv with the frozen model. "
            "Calibration here is optimistic (the model was fit on these labels). "
            "Production should append a new labeled snapshot when 90-day "
            "outcomes land — do not treat this as a live accuracy number."
        ),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(baseline: dict[str, Any], path: Path = BASELINE_PATH) -> None:
    save_json(path, baseline)


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    return load_json(path)
