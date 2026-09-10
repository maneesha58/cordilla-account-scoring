"""Write the ranked call list. CSV for the full batch; markdown for the SDR."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.config import OUTPUT_DIR

OUTPUT_COLUMNS = [
    "rank",
    "account_id",
    "account_type",
    "score",
    "tier",
    "intent_score_missing",
    "reasoning",
    "opener",
]


def build_output(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Persist full CSV + markdown for High-tier accounts only (rep queue)."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    table = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in table.columns:
            table[col] = ""
    table = table[OUTPUT_COLUMNS]

    high = table[table["tier"] == "High"] if "tier" in table.columns else table
    md_path = output_dir / "call_list.md"
    md_path.write_text(_to_markdown(high), encoding="utf-8")

    csv_path = output_dir / "prioritized_accounts.csv"
    try:
        table.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = output_dir / "prioritized_accounts_new.csv"
        table.to_csv(csv_path, index=False)
        print(
            f"Could not overwrite prioritized_accounts.csv (file is open). "
            f"Wrote {csv_path} instead — close the original and rerun, or rename this file."
        )
    return table


def _to_markdown(top: pd.DataFrame) -> str:
    lines = [
        "# Cordilla — prioritized call list",
        "",
        "High-tier queue this cycle. Ranking is relative priority, not a close probability.",
        "If intent is missing, that is a vendor coverage gap — not low intent.",
        "",
    ]
    for _, row in top.iterrows():
        missing = "yes" if bool(row["intent_score_missing"]) else "no"
        lines.extend(
            [
                f"## {int(row['rank'])}. `{row['account_id']}` — {row.get('account_type', '')} "
                f"— {row['tier']}",
                "",
                f"- Intent missing: {missing}",
                f"- Why: {row['reasoning'] or '—'}",
                f"- Opener: {row['opener'] or '—'}",
                "",
            ]
        )
    return "\n".join(lines)
