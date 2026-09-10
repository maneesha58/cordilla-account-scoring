"""Write the ranked call list. CSV for the full batch; markdown for the SDR."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.config import OUTPUT_DIR, REASONING_N

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
    top_n: int = REASONING_N,
) -> pd.DataFrame:
    """Persist full CSV + a short markdown the VP/SDR can actually open."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    table = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in table.columns:
            table[col] = ""
    table = table[OUTPUT_COLUMNS]

    md_path = output_dir / "call_list.md"
    md_path.write_text(_to_markdown(table.head(top_n)), encoding="utf-8")

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
        "Top accounts for this batch. Score is a model signal, not a close probability.",
        "If intent is missing, that is a vendor coverage gap — not low intent.",
        "",
    ]
    for _, row in top.iterrows():
        missing = "yes" if bool(row["intent_score_missing"]) else "no"
        lines.extend(
            [
                f"## {int(row['rank'])}. `{row['account_id']}` — {row.get('account_type', '')} "
                f"— {row['tier']} (score {float(row['score']):.3f})",
                "",
                f"- Intent missing: {missing}",
                f"- Why: {row['reasoning'] or '—'}",
                f"- Opener: {row['opener'] or '—'}",
                "",
            ]
        )
    return "\n".join(lines)
