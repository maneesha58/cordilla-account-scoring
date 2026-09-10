"""Mocked LLM call. Same I/O a real provider call would use; no network."""

from __future__ import annotations

from typing import Any, Mapping

OPENER_MAX_WORDS = 25

# Fitted SimpleImputer median for intent_score (from model.pkl). Not a real reading.
INTENT_IMPUTE_MEDIAN = 25.3

# ---------------------------------------------------------------------------
# Prompt we WOULD send. This is the real instruction set, not a placeholder.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a sales copilot for Cordilla Systems, a B2B workflow company.

A frozen conversion model has scored one Salesforce account. Your job is to help
an SDR decide how to open the call — not to restate the score.

Rules:
- Use only the fields provided. Do not invent revenue, contacts, activity, or
  intent that isn't in the data.
- Cite 2-3 concrete signals from the account's actual values (trial activity,
  MQLs, web visits, sales contacts, intent score, company size). Prefer signals
  that are actually present over noting what's absent.
- intent_score_missing = true means the vendor had no coverage on this company,
  NOT that the company has low intent. The model fills a missing intent_score
  with the training-set median (25.3) internally — it is not a real reading.
  If intent_score_missing is true, say the ranking reflects incomplete signal,
  and lean on the account's other real activity (MQLs, trial, web visits,
  sales contacts) instead.
- Never describe the score as a percentage chance or calibrated probability
  (e.g. do not say "70% likely to convert" or "score 0.21"). This model's
  scores are a relative priority ranking within this batch, not a verified
  real-world probability — checked calibration data shows the model's stated
  numbers do not reliably match actual outcomes, particularly at the higher
  end. Refer to it only as "priority" or "ranking," never as a percentage
  or chance.
- account_type changes the opener:
    - "Former Customer": frame as re-engagement / what's changed since they left.
      Never use a cold-intro opener for a former customer.
    - "Prospect" or "Suspect": frame as discovery — first-touch, curious, no
      assumption of prior relationship.
- Tone: direct, specific, no fluff. The reader has 15 seconds before a call.
- Keep the opener to one sentence, under 25 words.

Return JSON only, no markdown:
{"reasoning": "<2-3 sentences for the SDR, grounded in this account's actual
data>", "opener": "<one sentence, under 25 words, the rep can say on the
call>"}
"""


def build_user_prompt(account_data: Mapping[str, Any]) -> str:
    # Column on the scored frame is `score`, not pred_score.
    fields = [
        "account_id",
        "account_type",
        "industry",
        "employee_count",
        "intent_score",
        "intent_score_missing",
        "mql_count_90d",
        "trial_started",
        "trial_active_users",
        "web_touchpoints_90d",
        "sales_contacts_90d",
        "score",
        "tier",
        "rank",
    ]
    lines = []
    for key in fields:
        if key in account_data:
            lines.append(f"- {key}: {account_data[key]}")
    return (
        "Explain why this account ranked where it did and draft a call opener.\n\n"
        "Account:\n" + "\n".join(lines)
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and value != value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def opener_word_count(text: str) -> int:
    return len(text.replace("—", " ").replace("-", " ").split())


def _cap_opener(text: str, max_words: int = OPENER_MAX_WORDS) -> str:
    words = text.replace("—", " ").split()
    if len(words) <= max_words:
        return text.strip()
    clipped = " ".join(words[:max_words]).rstrip(".,;:")
    if not clipped.endswith("?"):
        clipped += "?"
    return clipped


def call_llm_mock(prompt: str, account_data: Mapping[str, Any]) -> dict[str, str]:
    """Stand-in for a live LLM. Same return shape: {reasoning, opener}.

    Follows SYSTEM_PROMPT: no probability language, Former Customer =
    re-engagement first, missing intent ≠ low intent, opener ≤ 25 words.
    """
    _ = prompt

    # --- REAL API CALL GOES HERE ------------------------------------------
    #   from openai import OpenAI
    #   client = OpenAI()
    #   response = client.chat.completions.create(
    #       model="gpt-4o-mini",
    #       temperature=0.2,
    #       response_format={"type": "json_object"},
    #       messages=[
    #           {"role": "system", "content": SYSTEM_PROMPT},
    #           {"role": "user", "content": prompt},
    #       ],
    #   )
    #   return json.loads(response.choices[0].message.content)
    # ----------------------------------------------------------------------

    account_type = account_data.get("account_type", "account")
    industry = account_data.get("industry") or "their space"
    emp = _num(account_data.get("employee_count"))
    mqls = _num(account_data.get("mql_count_90d"))
    web = _num(account_data.get("web_touchpoints_90d"))
    contacts = _num(account_data.get("sales_contacts_90d"))
    trial_users = _num(account_data.get("trial_active_users"))
    trial = _as_bool(account_data.get("trial_started"))
    intent_missing = _as_bool(account_data.get("intent_score_missing"))
    rank = account_data.get("rank", "?")
    tier = account_data.get("tier", "High")
    intent = account_data.get("intent_score")

    signals: list[str] = []
    if trial and trial_users > 0:
        signals.append(f"an active trial ({int(trial_users)} users in product)")
    elif trial:
        signals.append("a trial on the books, though usage is thin")
    if mqls >= 1:
        signals.append(f"{int(mqls)} MQL(s) in the last 90 days")
    if web >= 3:
        signals.append(f"{int(web)} web touchpoints recently")
    if contacts >= 1:
        signals.append(f"{int(contacts)} sales contacts already logged")
    if not intent_missing and intent is not None and str(intent) not in {"", "nan"}:
        signals.append(f"vendor intent reading {intent}")
    if not signals:
        signals.append("mostly firmographic fit rather than recent engagement")

    cited = "; ".join(signals[:3])
    size = f"{int(emp)}-person" if emp else ""
    who = f"{size} {account_type} in {industry}".strip()

    if intent_missing:
        caveat = (
            f" Intent is missing — vendor coverage gap, not low intent; "
            f"the model fills {INTENT_IMPUTE_MEDIAN}, not a real reading. "
            f"Lean on the activity above."
        )
    else:
        caveat = ""

    reasoning = (
        f"Rank #{rank} ({tier} priority). {who} stands out because of {cited}."
        f"{caveat}"
    )

    former = account_type == "Former Customer"
    if former and trial:
        opener = (
            f"Since you left, I saw a new Cordilla trial — open to a short {industry} catch-up?"
        )
    elif former:
        opener = (
            f"Wanted to reconnect — what's changed for your {industry} team since Cordilla?"
        )
    elif trial:
        opener = (
            f"I noticed your team is in a Cordilla trial. Ten minutes to compare with other {industry} teams?"
        )
    elif mqls >= 1:
        opener = (
            f"Someone on your side showed interest recently. Still looking at workflow tooling?"
        )
    elif web >= 3:
        opener = (
            f"Your team has been on our site this quarter. Quick walkthrough of what {industry} teams ask first?"
        )
    else:
        opener = (
            f"We work with {industry} teams your size on workflow ops. Open to a brief call?"
        )

    opener = _cap_opener(opener)
    return {"reasoning": reasoning.strip(), "opener": opener}
