# RESEARCH-LOG

Live log for the Dialpad AI Engineer take-home. Entries are written as work happens, not reconstructed at the end.

---

## 2026-09-10 — Session 0: setup + domain framing (before data)

### Ground rules (from the brief)

- 24-hour hard deadline from receipt. Suggested time-box ~4 hours (soft).
- Evaluated on three things: (1) impact framing in business terms, (2) a real running agent, (3) monitoring designed for **silent** failure, not crashes.
- AI use is expected. Log actual prompts/sessions as they happen. Record at least one place I corrected or overrode the tool.
- Commit incrementally. Git history is part of the evaluation.
- Snapshot date to treat as "today": **2026-08-01**. Do not use the system clock for recency/age.

### Repo state at this point

- Empty scaffold committed: `README.md`, `requirements.txt`, `PROPOSAL.md`, `RESEARCH-LOG.md`, plus `model/`, `data/`, `agent/`, `monitoring/` (kept with `.gitkeep`).
- Starter artifacts not in the repo yet: `model/model.pkl`, `data/training_data.csv`, `data/accounts_to_score.csv`. Followed up by email (missing attachment, 24h clock start). Staying proactive rather than idle.

### Domain understanding (from the brief, not yet verified against CSVs)

**Account types**

- Prospect — shown some interest / fits target profile, hasn't bought.
- Suspect — earlier stage than Prospect, unconfirmed fit.
- Former Customer — churned.

**Columns**

- `intent_score` — third-party vendor signal for off-site shopping behavior. Missing on ~40% of rows, and **not missing at random**: coverage skews toward larger / better-known companies. Treating missing as "low intent" would systematically penalize smaller accounts.
- `mql_count_90d` — MQLs in last 90 days (form fills, webinars). Proxy for "someone there is paying attention."
- `trial_started` / `trial_active_users` — product engagement. Exists only for accounts that started a trial (coverage gap by construction).
- `web_touchpoints_90d` — website visit frequency from that account.
- `sales_contacts_90d` — how much a rep has already reached out. Possibly confounded: more contact can correlate with conversion for reasons other than account quality.
- `converted_within_90d` — target (training file only).

**Base rates given in the brief (unverified until `training_data.csv` is in hand)**

- Cold accounts: well under 1% historical conversion.
- Recently engaged accounts: low single digits.
- Do not assert these in the proposal. Compute them from the 1,200-row training file.

**Decision this is supposed to change**

Which accounts an SDR / AM should prioritize calling this week, and why — not just a bare score.

### Why monitoring exists (Cordilla backstory)

An earlier scoring effort looked strong in testing, shipped, then quietly drifted over a couple of quarters. Scores stopped matching what reps saw in the field. Nobody was watching the right signal. Trust collapsed.

Two drift types to keep separate:

- **Data drift** — production inputs no longer look like training (e.g. account-size mix shifts).
- **Concept drift** — the relationship between inputs and real outcomes changes (e.g. trial-starters used to convert; now they don't). The model keeps scoring as if nothing changed.

Silent wrongness is the failure mode. Crashes are the easy ones.

### Working interpretation of the three deliverables

**1. Impact framing**

Ground numbers in `training_data.csv`: overall conversion rate, rate by `account_type`, by intent present/absent, by `trial_started`. State the decision, the upside if right (redirected rep time → estimated lift), and the cost if wrong in each direction:

- False positive (call a dead account) → wasted rep time / opportunity cost.
- False negative (skip a real winner) → silently lost revenue. Worse, because it is undetected.

**2. Agent**

Real output an SDR could act on: prioritized call list + one-line reason per account + (mocked) outreach angle. Justify every tool. Open question: does this need a heavy framework (LangGraph etc.) or is a deterministic pipeline more honest given a mostly linear flow (score → rank → enrich → output)? Don't default to "agent framework because this is an AI role." Justify the choice.

**3. Monitoring**

Must be concrete code or a precisely specified check. Candidates:

- Score-distribution drift (batch mean/std vs rolling baseline).
- `intent_score` missingness rate drift batch-over-batch.
- Calibration: bucket by predicted score, compare to actual conversion once labels arrive — closest match to the Cordilla failure story.

Define normal noise vs a real trip (threshold + consecutive periods) and what action fires.

### Open questions / to verify once data is in hand

- Actual base rates by `account_type` and by `intent_score` bucket.
- Class balance of `converted_within_90d` — how imbalanced, and how that should change how we read `predict_proba`.
- Other data-quality issues worth flagging (outliers, impossible values, duplicate `account_id`s).

### AI session (this chat)

- Asked whether the take-home PDF was visible; it was not in the workspace at first. Later attached `Take-Home Exercise — Candidate Copy.pdf`; I read all 7 pages.
- Asked for the empty repo structure so they could commit first. I created the scaffold only — no fake `model.pkl` or CSVs.
- Provided this log draft and asked me to edit it as I saw fit and keep writing here as building starts.

Prompts (paraphrased, this session): "can u see the pdf?" → "there is a takeHome exercise pdf" → "@Take-Home Exercise — Candidate Copy.pdf" → create empty structure for first commit → write this log.

### Corrections / overrides of AI output

None in this session. First override is in Session 1 (ranking alone does not justify an agent).

---

## 2026-09-10 — Session 1: agent design & impact framing

Discussion, still before `training_data.csv` / `model.pkl`. No code written this session.

### Impact framing — working understanding

The decision: which accounts an SDR/AM should prioritize calling, given limited rep time against a much larger account list.

Value is not the model score itself. It is redirecting rep effort away from near-zero-odds cold accounts toward accounts more likely to convert.

Cost of being wrong is directional, not symmetric:

- False positive (called a dead account) → wasted rep time, opportunity cost.
- False negative (skipped a real winner) → silently lost revenue. Worse, because it is undetected.

Impact numbers must be grounded in computed base rates from `training_data.csv` (overall conversion, by `account_type`, by `intent_score` presence/absence) — not asserted from the brief's stated ranges.

### Agent design — working plan

Pipeline:

1. Load `data/accounts_to_score.csv`.
2. Score via `model.predict_proba()` (frozen, not retrained).
3. Flag data-quality issues (especially missing `intent_score`).
4. Rank / tier.
5. For top-N accounts, a mocked LLM call generates a plain-language "why this score" explanation plus a suggested outreach opener.
6. Write a structured ranked list.

Tools, each justified rather than added by default:

- `score_accounts()` — wraps `model.predict_proba()`.
- `flag_data_quality()` — flags missing `intent_score` etc., so downstream reasoning does not imply all scores rest on equally solid signal.
- `generate_reasoning()` (mocked LLM) — synthesizes raw feature values into a natural-language explanation + call opener. Document the prompt / inputs / tools / output a real call would use.
- (optional) `format_output()` — writes the final artifact.

**Framework decision: still open.** Either (a) use LangGraph and justify it via future extensibility (e.g. a HITL approval node later), or (b) skip a framework because the pipeline is currently linear with no real branching, and say so explicitly. Leaning toward honesty about actual complexity rather than defaulting to "agent framework because this is an AI role."

**Model stays frozen.** "Runs on a fresh batch daily/weekly" means re-scoring new incoming data through the same trained model — not retraining. Retraining is a separate, deliberate decision and is explicitly out of scope ("Don't retrain it").

### AI session

Working discussion of impact framing and agent shape. No implementation this session.

### Corrections / overrides of AI output

**Point 1 — "why do we need an agent at all if the model already gives a percentage?"**

Initial AI answer leaned on generic value-add. I pushed back: a rep could just sort by the model's raw percentage. Ranking alone is not a real justification for building anything beyond `df.sort_values()`.

AI response was corrected/sharpened to concede this directly: **ranking alone does not justify an agent.** The only genuinely defensible value-adds are:

1. **Trust calibration** — surfacing when a score rests on weak/missing signal (tied to the `intent_score` coverage gap) rather than presenting all scores as equally reliable.
2. **Actionability** — turning a number into something immediately usable (a call opener), which a raw percentage cannot do on its own.

Also the more mundane but real justification: **repeatability** — someone does not have to manually re-run this analysis every batch.

**Decision for PROPOSAL.md:** lead with those two justifications. Explicitly acknowledge that sorting alone is not sufficient reasoning. Do not oversell agent complexity that isn't earned.

---

## 2026-09-10 — Session 2: implement scoring agent (starter files still absent)

Coded the pipeline without `model.pkl` / CSVs. `run.py` fails with a clear `FileNotFoundError` until those files land. Did not invent a substitute model or fake accounts.

### Framework decision — closed

**Skip LangGraph / LangChain.** Flow is linear (score → flag → rank → mock explain → write). A graph would add weight without changing the SDR artifact. Comment lives in `agent/main.py` and README. Revisit only if we add HITL or a real tool loop.

This is the honest answer to Session 1's open question, not "no framework because we were lazy."

### What landed in the repo

- `agent/score.py` — `joblib.load` + `predict_proba`; feature list from `feature_names_in_` when the model is present, else the brief's columns.
- `agent/quality.py` — flags only, no imputation. Required `intent_score_missing`. Extra (pending data check): `trial_users_without_trial`, `duplicate_account_id`, `invalid_account_type`, `employee_count_outlier` (>50k or ≤0), `negative_activity`, `intent_score_out_of_range`, `snapshot_not_as_of_today` vs 2026-08-01, composite `weak_signal` (missing intent AND no trial).
- `agent/rank.py` — within-batch quantiles: High ≥ p90, Medium ≥ p70. Not absolute P(convert).
- `agent/llm.py` — full system prompt (missing intent ≠ low intent) + commented real API slot + rule-based mock that varies by trial / MQL / web / former customer.
- `agent/reasoning.py` — `generate_reasoning(row)` plus `attach_reasoning(df, n=20)`.
- `agent/output.py` — full CSV + top-20 markdown (the thing a non-technical reader opens).
- `run.py`, `requirements.txt` stand-in pins, README.

### Assumptions to verify the hour the starter files arrive

1. `scikit-learn==1.5.1` unpickles `model.pkl`. If not, take the starter pin.
2. `feature_names_in_` matches (or doesn't) `DEFAULT_FEATURE_COLUMNS`. Score path already prefers the model.
3. Score histogram: if everyone is 0.02–0.08, quantile tiers still work; if the model is well separated, we may switch to capacity (top 20 = High) instead.
4. `intent_score` scale (0–1 vs 0–100) — out-of-range flag assumes 0–100.
5. `employee_count` > 50k as outlier — tighten from training percentiles.
6. Compute real base rates from `training_data.csv` before writing impact numbers in PROPOSAL.md. Do not use the brief's "<1% / low single digits" as asserted facts.

### AI session

Asked to start coding despite missing starter files, and to append this log. Built the modules above rather than waiting idle.

Prompt (this turn): implement the scoring/agent pipeline as specified; keep it simple; mock the LLM; no monitoring yet.

### Corrections / overrides of AI output

None new this session. Carried forward Session 1: did **not** wrap this in an agent framework, and did **not** treat missing intent as something to impute. `weak_signal` exists so the call list can show unequal score reliability — that is the product, not `sort_values`.

### Still not done

- Run end-to-end (blocked on starter files).
- Monitoring / drift.
- PROPOSAL.md impact numbers.
- Last research-log entry that packs presentation raw material.

---

