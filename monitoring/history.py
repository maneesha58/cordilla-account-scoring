"""Append one monitoring snapshot per batch. Trip after N consecutive failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monitoring.baseline import load_json, save_json
from monitoring.config import CONSECUTIVE_PERIODS, HISTORY_PATH


def load_history(path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    return data if isinstance(data, list) else []


def save_history(history: list[dict[str, Any]], path: Path = HISTORY_PATH) -> None:
    save_json(path, history)


def upsert_snapshot(history: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace a snapshot with the same period_id so re-runs don't fake extra weeks."""
    pid = snapshot.get("period_id")
    kept = [s for s in history if s.get("period_id") != pid]
    kept.append(snapshot)
    return kept


def consecutive_trips(
    history: list[dict[str, Any]],
    check_name: str,
    n: int = CONSECUTIVE_PERIODS,
) -> bool:
    """True only if the last n snapshots all tripped this check.

    Fewer than n periods → False. One bad week is noise; a quarter of
    mismatch is the Cordilla story.
    """
    flagged = []
    for snap in history:
        check = (snap.get("checks") or {}).get(check_name) or {}
        if check.get("skipped"):
            continue
        flagged.append(bool(check.get("tripped")))
    if len(flagged) < n:
        return False
    return all(flagged[-n:])
