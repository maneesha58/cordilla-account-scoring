"""Monitoring job: compare this batch to a frozen-model training baseline.

Not LangGraph. There is no tool choice, no wait-on-human node, no retry loop.
Scoring is one cron-shaped job. This is a second cron-shaped job that reads
scores (and, later, outcomes) and decides whether to keep trusting them.

  scoring loop:   new batch -> frozen model -> ranked list -> reps act
  monitoring:     (now) drift vs training; (later) predicted vs actual convert
                  -> if a check trips N periods in a row -> alert a human
                  -> pause auto-priority / investigate / maybe later retrain
                  retraining is never automatic
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.config import ACCOUNTS_PATH, MODEL_PATH, TRAINING_PATH
from agent.score import load_and_score
from monitoring.baseline import build_baseline, save_baseline
from monitoring.checks import (
    calibration_table,
    check_calibration,
    check_intent_drift,
    check_score_drift,
    intent_missing_rate,
    score_mean,
)
from monitoring.config import (
    AS_OF,
    BASELINE_PATH,
    CONSECUTIVE_PERIODS,
    FALLBACK_HEURISTIC,
    HISTORY_PATH,
    REPORT_PATH,
)
from monitoring.history import consecutive_trips, load_history, save_history, upsert_snapshot
from monitoring.report import render_report, write_report


def _snapshot(
    *,
    period_id: str,
    source: str,
    scored,
    baseline: dict,
    labeled: bool,
) -> dict:
    intent = check_intent_drift(intent_missing_rate(scored), baseline["intent_missing_rate"])
    score = check_score_drift(
        score_mean(scored),
        baseline["mean_score"],
        baseline["std_score"],
    )
    if labeled:
        table = calibration_table(scored)
        cal = check_calibration(table, baseline.get("calibration"))
    else:
        cal = check_calibration([], baseline.get("calibration"))
    return {
        "period_id": period_id,
        "as_of": AS_OF,
        "source": source,
        "n": int(len(scored)),
        "metrics": {
            "intent_missing_rate": intent_missing_rate(scored),
            "mean_score": score_mean(scored),
        },
        "checks": {
            "intent_drift": intent,
            "score_drift": score,
            "calibration": cal,
        },
    }


def _status(consecutive: dict[str, bool], snapshots: list[dict]) -> str:
    if any(consecutive.values()):
        return "ALERT — notify model owner; do not keep auto-prioritizing on trust"
    watches = []
    for s in snapshots:
        cal = (s.get("checks") or {}).get("calibration") or {}
        if cal.get("watch_inversion"):
            watches.append("calibration rank inversion")
        for name, chk in (s.get("checks") or {}).items():
            if chk.get("tripped") and not chk.get("skipped"):
                watches.append(name)
    if watches:
        return "WATCH — a check tripped but not yet " + str(CONSECUTIVE_PERIODS) + " periods in a row"
    return "OK"


def run(
    training_path: Path = TRAINING_PATH,
    accounts_path: Path = ACCOUNTS_PATH,
    model_path: Path = MODEL_PATH,
) -> str:
    baseline = build_baseline(training_path, model_path)
    save_baseline(baseline)

    train_scored = load_and_score(training_path, model_path)
    batch_scored = load_and_score(accounts_path, model_path)

    train_snap = _snapshot(
        period_id="period0_training",
        source=str(training_path),
        scored=train_scored,
        baseline=baseline,
        labeled=True,
    )
    batch_snap = _snapshot(
        period_id=f"batch_{AS_OF}",
        source=str(accounts_path),
        scored=batch_scored,
        baseline=baseline,
        labeled="converted_within_90d" in batch_scored.columns,
    )

    history = load_history()
    history = upsert_snapshot(history, train_snap)
    history = upsert_snapshot(history, batch_snap)
    save_history(history)

    consecutive = {
        name: consecutive_trips(history, name)
        for name in ("intent_drift", "score_drift", "calibration")
    }
    status = _status(consecutive, [train_snap, batch_snap])
    report = render_report(
        baseline=baseline,
        batch_snapshot=batch_snap,
        train_snapshot=train_snap,
        consecutive=consecutive,
        status=status,
    )
    write_report(report)

    print(status)
    print(f"Baseline: {BASELINE_PATH}")
    print(f"History:  {HISTORY_PATH}")
    print(f"Report:   {REPORT_PATH}")
    if status.startswith("ALERT"):
        print(FALLBACK_HEURISTIC)
    return status


def main() -> int:
    try:
        run()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Monitoring needs model/model.pkl, data/training_data.csv, and "
            "data/accounts_to_score.csv. Add the starter files, then re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
