"""Write the ranked call list. CSV for the full batch; markdown for the SDR."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent.config import OUTPUT_DIR, REASONING_N

OUTPUT_COLUMNS = [
    "rank",
    "account_id",
    "score",
    "tier",
    "intent_score_missing",
    "weak_signal",
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

    csv_path = output_dir / "prioritized_accounts.csv"
    table.to_csv(csv_path, index=False)

    md_path = output_dir / "call_list.md"
    md_path.write_text(_to_markdown(table.head(top_n)), encoding="utf-8")
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
        weak = "yes" if bool(row.get("weak_signal", False)) else "no"
        lines.extend(
            [
                f"## {int(row['rank'])}. `{row['account_id']}` — {row['tier']} "
                f"(score {float(row['score']):.3f})",
                "",
                f"- Intent missing: {missing} · Weak signal: {weak}",
                f"- Why: {row['reasoning'] or '—'}",
                f"- Opener: {row['opener'] or '—'}",
                "",
            ]
        )
    return "\n".join(lines)
