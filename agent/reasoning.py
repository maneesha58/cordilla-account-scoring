"""Attach mocked LLM reasoning to the top-N rows only."""

from __future__ import annotations

import pandas as pd

from agent.config import REASONING_N
from agent.llm import SYSTEM_PROMPT, build_user_prompt, call_llm_mock


def generate_reasoning(row: pd.Series) -> dict[str, str]:
    """One account → {reasoning, opener}. Builds the real prompt, then mocks."""
    account_data = row.to_dict()
    user_prompt = build_user_prompt(account_data)
    # Full prompt a real call would send (system + user). Mock uses the dict.
    _full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    return call_llm_mock(user_prompt, account_data)


def attach_reasoning(df: pd.DataFrame, n: int = REASONING_N) -> pd.DataFrame:
    """Fill reasoning/opener for the first n rows (already ranked). Rest stay blank.

    N defaults to 20: enough for a focused SDR block, cheap enough that we
    are not pretending to LLM-annotate the whole 300.
    """
    out = df.copy()
    out["reasoning"] = ""
    out["opener"] = ""
    top_n = min(n, len(out))
    for idx in range(top_n):
        result = generate_reasoning(out.iloc[idx])
        out.at[idx, "reasoning"] = result["reasoning"]
        out.at[idx, "opener"] = result["opener"]
    return out
