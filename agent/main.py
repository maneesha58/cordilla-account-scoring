"""Run the scoring agent end to end. Linear pipeline — no agent framework.

Why no LangGraph/LangChain: score → flag → rank → (mocked) explain → write
is a straight line with no branching, retries, or tool choice. A graph would
add dependency weight without changing what an SDR gets. If we later need a
HITL approval node or a real tool loop, that is when a framework earns its
place. Ranking by score alone is also not enough to justify this code;
flags + reasoning are the actual product.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.config import ACCOUNTS_PATH, MODEL_PATH, OUTPUT_DIR, REASONING_N
from agent.output import build_output
from agent.quality import flag_data_quality
from agent.rank import rank_and_tier
from agent.reasoning import attach_reasoning
from agent.score import load_and_score


def run(
    csv_path: Path = ACCOUNTS_PATH,
    model_path: Path = MODEL_PATH,
    output_dir: Path = OUTPUT_DIR,
    n: int = REASONING_N,
) -> None:
    scored = load_and_score(csv_path, model_path)
    flagged = flag_data_quality(scored)
    ranked = rank_and_tier(flagged)
    explained = attach_reasoning(ranked, n=n)
    table = build_output(explained, output_dir=output_dir, top_n=n)

    n_accounts = len(table)
    n_high = int((table["tier"] == "High").sum())
    n_med = int((table["tier"] == "Medium").sum())
    n_low = int((table["tier"] == "Low").sum())
    missing_pct = 100.0 * table["intent_score_missing"].mean()

    print(f"Scored {n_accounts} accounts. Model was not retrained.")
    print(f"Tiers — High: {n_high}, Medium: {n_med}, Low: {n_low}.")
    print(f"{missing_pct:.1f}% of accounts have missing intent_score.")
    print(f"Wrote reasoning for top {min(n, n_accounts)} accounts.")
    print(f"CSV: {output_dir / 'prioritized_accounts.csv'}")
    print(f"Call list: {output_dir / 'call_list.md'}")


def main() -> int:
    try:
        run()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Starter artifacts are not in the repo yet. "
            "Add model/model.pkl and data/accounts_to_score.csv, then re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
