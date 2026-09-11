"""Human-readable monitoring report. A VP will not open the JSON."""

from __future__ import annotations

from typing import Any

from monitoring.config import CONSECUTIVE_PERIODS, FALLBACK_HEURISTIC, REPORT_PATH


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}%"


def render_report(
    *,
    baseline: dict[str, Any],
    batch_snapshot: dict[str, Any],
    train_snapshot: dict[str, Any],
    consecutive: dict[str, bool],
    status: str,
) -> str:
    lines = [
        "# Cordilla scoring — monitoring report",
        "",
        f"**Status: {status}**",
        "",
        "Two loops, not one program. The agent scores a batch. This job asks "
        "whether those scores are still worth acting on. It does not retrain.",
        "",
        f"As-of date used everywhere: **{baseline.get('as_of')}** (not the system clock).",
        "",
        "## Period 0 — training snapshot (optimistic)",
        "",
        baseline.get("note", ""),
        "",
        f"- n = {baseline.get('n')}",
        f"- intent missing: {_pct(baseline.get('intent_missing_rate'))}",
        f"- mean score: {baseline.get('mean_score'):.4f} (std {baseline.get('std_score'):.4f})",
        f"- p90 score: {baseline.get('p90_score', 0):.4f}",
        f"- share with score ≥ {baseline.get('high_score_bar', 0.10):.2f}: {_pct(baseline.get('share_above_bar'))}",
        f"- weighted |predicted − actual| : {_pct(baseline.get('overall_calibration_gap'))}",
        "",
        _calibration_md((train_snapshot.get("checks") or {}).get("calibration")),
        "",
        "## This batch — label-free drift",
        "",
        f"Source: `{batch_snapshot.get('source')}` (n={batch_snapshot.get('n')})",
        "",
        _check_md((batch_snapshot.get("checks") or {}).get("intent_drift")),
        _check_md((batch_snapshot.get("checks") or {}).get("score_drift")),
        "",
        "PSI of model inputs vs training. Bins are frozen at period 0 "
        "(re-cutting this file would hide the shift). Complements mean-score "
        "drift: mix can move while the average score stays put.",
        "",
        _feature_drift_md((batch_snapshot.get("checks") or {}).get("feature_drift")),
        "",
        "## Absolute quality (not the call-list tiers)",
        "",
        "Reps still get quantile High/Medium/Low so workload stays ~top 10%. "
        "These two checks catch the case where that top 10% is quietly worse "
        "than the bar that beat baseline conversion on training data.",
        "",
        _check_md((batch_snapshot.get("checks") or {}).get("p90_drop")),
        _check_md((batch_snapshot.get("checks") or {}).get("share_above_bar")),
        "",
        "## Calibration on this batch",
        "",
        _check_md((batch_snapshot.get("checks") or {}).get("calibration")),
        "",
        "## Consecutive-period rule",
        "",
        f"Alert the model owner only if a check trips **{CONSECUTIVE_PERIODS} "
        "periods in a row**. One noisy week is not the failure mode. "
        "A quiet mismatch over a quarter is.",
        "",
    ]
    for name, tripped in consecutive.items():
        mark = "TRIP (would page)" if tripped else "not yet"
        lines.append(f"- `{name}`: {mark}")
    lines.extend(
        [
            "",
            "## If this pages",
            "",
            FALLBACK_HEURISTIC,
            "",
            "Retraining is a separate, deliberate decision. This job never calls `fit()`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(text: str, path=REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _check_md(check: dict[str, Any] | None) -> str:
    if not check:
        return "_No check._"
    if check.get("skipped"):
        return f"- **{check.get('name')}** — skipped. {check.get('reason', '')}"
    tripped = "TRIP" if check.get("tripped") else "ok"
    watch = " (WATCH: high bucket converts worse than low)" if check.get("watch_inversion") else ""
    extra = ""
    if "delta" in check and "threshold" in check:
        extra = f" delta={check['delta']:.4f} vs threshold={check['threshold']:.4f}."
    return f"- **{check.get('name')}** — {tripped}{watch}.{extra} {check.get('what', '')}"


def _feature_drift_md(check: dict[str, Any] | None) -> str:
    if not check or check.get("skipped"):
        return _check_md(check)
    max_psi = check.get("max_psi")
    extra = ""
    if max_psi is not None:
        extra = (
            f" max PSI={max_psi:.4f}; "
            f"{check.get('n_shift', 0)} feature(s) > {check.get('threshold_shift', 0.25):.2f}; "
            f"{check.get('n_watch', 0)} feature(s) > {check.get('threshold_watch', 0.10):.2f}."
        )
    head = (
        f"- **{check.get('name')}** — "
        f"{'TRIP' if check.get('tripped') else 'ok'}."
        f"{extra} {check.get('what', '')}"
    )
    rows = check.get("features") or []
    if not rows:
        return head
    lines = [
        head,
        "",
        "| feature | PSI | largest bin move (train -> batch) |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.get('feature')}` | {r.get('psi', 0):.4f} | {r.get('top_move', '—')} |"
        )
    return "\n".join(lines)


def _calibration_md(check: dict[str, Any] | None) -> str:
    if not check or check.get("skipped"):
        return _check_md(check)
    lines = [_check_md(check), "", "| bucket | n | mean predicted | actual convert | gap |", "|---|---|---|---|---|"]
    for b in check.get("buckets") or []:
        lines.append(
            f"| {b.get('bucket')} | {b.get('n')} | {b.get('mean_predicted', 0):.4f} "
            f"| {b.get('actual_rate', 0):.4f} | {b.get('gap', 0):.4f} |"
        )
    return "\n".join(lines)
