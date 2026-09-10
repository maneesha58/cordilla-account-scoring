"""Attach mocked LLM reasoning to High-tier rows only."""

from __future__ import annotations

import pandas as pd

from agent.llm import SYSTEM_PROMPT, build_user_prompt, call_llm_mock, opener_word_count


def generate_reasoning(row: pd.Series) -> dict[str, str]:
    """One account → {reasoning, opener}. Builds the real prompt, then mocks."""
    account_data = row.to_dict()
    user_prompt = build_user_prompt(account_data)
    _full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    result = call_llm_mock(user_prompt, account_data)
    # Enforce the prompt's 25-word opener cap (soft "one sentence" is not enough).
    if opener_word_count(result["opener"]) > 25:
        raise ValueError(f"opener exceeds 25 words: {result['opener']!r}")
    return result


def attach_reasoning(df: pd.DataFrame) -> pd.DataFrame:
    """Fill reasoning/opener for every High-tier row. Medium/Low stay blank.

    High is the call-this-cycle set (top 10% of this batch). A separate
    top-20 cutoff left ranks 21–30 as High with empty copy — a live-demo
    hole. We do not LLM the other ~270: that copy would go stale before
    a rep worked it, and this is a batch job, not a per-call chat.
    """
    out = df.copy()
    out["reasoning"] = ""
    out["opener"] = ""
    high = out.index[out["tier"] == "High"]
    for idx in high:
        result = generate_reasoning(out.loc[idx])
        out.at[idx, "reasoning"] = result["reasoning"]
        out.at[idx, "opener"] = result["opener"]
    return out
