"""Mocked LLM call. Same I/O a real provider call would use; no network."""

from __future__ import annotations

from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Prompt we WOULD send. This is the real instruction set, not a placeholder.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a sales copilot for Cordilla Systems, a B2B workflow company.

A frozen conversion model has scored one Salesforce account. Your job is to help
an SDR decide whether and how to call — not to restate the percentage.

Rules:
- Use only the fields provided. Do not invent revenue, contacts, or activity.
- Cite 2–3 concrete signals (trial, MQLs, web visits, sales contacts, industry,
  account type, company size). Prefer present signals over missing ones.
- Missing intent_score is a coverage gap, NOT low intent. The vendor covers
  larger / better-known companies more often. Never say "low intent" because
  the field is empty. If intent_score_missing is true, say the score is less
  reliable because third-party intent was unavailable.
- If weak_signal is true, say so in one clause (missing intent and no trial).
- Do not treat the score as a calibrated probability ("87% chance they buy").
  Call it a model score / priority signal.
- Tone: direct, specific, no fluff. The reader has 15 seconds.

Return JSON only, no markdown:
{"reasoning": "<2-3 sentences for the SDR>", "opener": "<one sentence they can say on a call>"}
"""


def build_user_prompt(account_data: Mapping[str, Any]) -> str:
    fields = [
        "account_id",
        "account_type",
        "industry",
        "employee_count",
        "intent_score",
        "intent_score_missing",
        "weak_signal",
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
        if value is None or (isinstance(value, float) and value != value):  # NaN
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def call_llm_mock(prompt: str, account_data: Mapping[str, Any]) -> dict[str, str]:
    """Stand-in for a live LLM. Same return shape: {reasoning, opener}.

    `prompt` is the full user message we would send (logged / inspectable).
    The mock ignores the prose of the prompt and uses the structured fields
    so each account's text actually differs. Keep this logic cheap — it is
    not the product; the prompt and I/O contract are.
    """
    _ = prompt  # would be the user message in a real call

    # --- REAL API CALL GOES HERE ------------------------------------------
    # Example (do not uncomment without a key; judged the same as this mock):
    #
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
    #
    # Anthropic equivalent: messages.create(model="claude-sonnet-4-...", ...)
    # with SYSTEM_PROMPT as `system` and `prompt` as the user turn.
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
    weak = _as_bool(account_data.get("weak_signal"))
    score = _num(account_data.get("score"))
    rank = account_data.get("rank", "?")
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
    if not intent_missing and intent is not None:
        signals.append(f"vendor intent_score={intent}")
    if not signals:
        signals.append("mostly firmographic fit rather than recent engagement")

    cited = "; ".join(signals[:3])
    size = f"{int(emp)}-person" if emp else ""
    who = f"{size} {account_type} in {industry}".strip()

    caveats = []
    if intent_missing:
        caveats.append(
            "third-party intent is missing, so do not read that as low intent — "
            "the vendor just does not cover this account"
        )
    if weak:
        caveats.append("no trial either, so this score sits on thinner evidence")
    caveat = " " + ("Also: " + "; ".join(caveats) + ".") if caveats else ""

    reasoning = (
        f"Rank #{rank} (score {score:.3f}). {who} stands out because of {cited}."
        f"{caveat}"
    )

    if trial:
        opener = (
            f"I noticed your team is already in a Cordilla trial — got 10 minutes "
            f"to compare that to how other {industry} teams rolled out?"
        )
    elif mqls >= 1:
        opener = (
            f"You had recent interest from someone on your side — worth a quick "
            f"call to see if workflow tooling is still on the list?"
        )
    elif web >= 3:
        opener = (
            f"Your team has been on our site a few times this quarter — "
            f"happy to walk through the piece most {industry} teams ask about first."
        )
    elif account_type == "Former Customer":
        opener = (
            f"Wanted to reconnect now that some workflow teams in {industry} "
            f"are backfilling after churn — open to a short catch-up?"
        )
    else:
        opener = (
            f"We work with {industry} teams around your size on workflow ops — "
            f"open to a brief call this week?"
        )

    return {"reasoning": reasoning.strip(), "opener": opener}
