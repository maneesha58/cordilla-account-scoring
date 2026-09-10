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
- Monitoring / drift — started in Session 3.
- PROPOSAL.md impact numbers.
- Last research-log entry that packs presentation raw material.

---

## 2026-09-10 — Session 3: monitoring loop + why LangGraph is still not required

Built the second job (`python monitor.py`) without starter files. Same pattern as the agent: clear `FileNotFoundError` until `model.pkl` + both CSVs land. Did not synthesize fake weekly batches to force an ALERT.

### Two loops

```
[SCORING]  new batch -> frozen model -> ranked list -> reps act
[MONITOR]  (now) input/score drift vs training
           (later, when 90-day labels exist) predicted vs actual, by score bucket
           -> if a check trips N consecutive periods -> ALERT a human
           -> pause auto-priority / investigate / maybe later retrain
           retraining is never automatic and is not in this job
```

These are two cron-shaped scripts, not one agent graph.

### Why LangGraph is not required (for now)

The take-home asks for an agent that *does something with scores* and for monitoring that could actually be built. It does not ask for a graph runtime.

LangGraph is useful when the next step is **chosen at runtime**: tool A vs tool B, retry, wait on a human, loop until a condition. Here:

- Scoring is always load → predict_proba → flag → rank → mock LLM on top N → write files.
- Monitoring is always score training (baseline) → score this batch → compare → append history → write a report.

There is no branch that needs an LLM to pick a tool. "If calibration trips 3 times, alert" is an `if`, not a graph node. Putting that in LangGraph would look more "AI" and would not change the SDR list or the alert. Session 1 already decided not to oversell unearned agent complexity; wrapping monitoring in a graph would be the same mistake.

Revisit a framework only if we add something that actually branches (HITL approval before a rep sees the list, or a real tool that writes back to Salesforce). Not now.

### What landed

- `monitoring/baseline.py` — score `training_data.csv` with the **frozen** model; store missingness, score mean/std, calibration table. Comment in the JSON: this calibration is optimistic (model was fit on these labels).
- `monitoring/checks.py` — `intent_drift` (10pp), `score_drift` (0.5 std), `calibration` (bucketed predicted vs `converted_within_90d`). Rank inversion (high bucket converts worse than low) is a WATCH even on train.
- `monitoring/history.py` — upsert by `period_id` so re-running locally does not look like three weeks passing. Consecutive trip = last 3 *non-skipped* snapshots all tripped.
- `monitoring/report.py` + `output/monitoring_report.md` — OK / WATCH / ALERT in prose. Alert action: pause score-based auto-priority; fall back to trial then MQLs then web; do not `fit()`.
- `monitor.py` entrypoint.

Unlabeled `accounts_to_score.csv` can only run **drift** today. Calibration on that file is skipped with an explicit reason until outcomes exist. The calibration *code* still runs for real against the training file so reviewers can see a table, not a comment that says "add monitoring."

### Thresholds are drafts

Same rule as quality flags: do not pretend 10pp / 0.5 std / 3pp came from these CSVs. When the files arrive, compute actual missingness and the train calibration table, then tighten. Until then the checks are the right *shape* (what we'd watch, what noise vs signal means, what fires).

### AI session

Asked to implement monitoring and to log why LangGraph is not required. Implemented the second job as plain functions + a history JSON, not a graph.

### Corrections / overrides of AI output

Carried forward: someone (including an AI default) might reach for LangGraph because "monitoring loop" sounds like a cyclic agent. Overrode that. A loop over weeks is a job scheduler + a function, not LangGraph.

### Still not done

- Run both jobs — done in Session 4.
- Revisit quality flags and monitoring thresholds after looking at the CSVs.
- PROPOSAL.md with impact numbers from training_data.csv.
- Final research-log entry packing presentation raw material.

---

## 2026-09-10 — Session 4: first successful run (Python / sklearn mismatch)

`pip install -r requirements.txt` failed on the machine Python **3.13**: pins were sklearn 1.5.1 / numpy 1.26, which have no 3.13 wheels, so pip tried to compile sklearn and died. `run.py` then failed with `No module named pandas` because that install never finished.

Tried sklearn 1.6.1 wheels for 3.13. Unpickle warned (estimators from **1.5.2**) then crashed: `Can't get attribute '__pyx_unpickle_CyHalfBinomialLoss'`. The frozen model is 1.5.2; bumping sklearn to get 3.13 wheels is not an option.

**Fix:** `uv python install 3.12`, new `.venv` on 3.12, pin `scikit-learn==1.5.2` (plus pandas 2.2.2 / numpy 1.26.4 / joblib 1.4.2). README updated. Do not use system `python` if it is 3.13 — activate `.venv` or call `.venv\Scripts\python.exe`.

### First live numbers (not proposal-ready yet)

Scoring: 300 accounts; High 30 / Medium 60 / Low 210 (10%/20%/70% quantiles); **38.7% missing intent_score** (brief said ~40%). Top 20 got reasoning. Wrote `output/prioritized_accounts.csv` and `output/call_list.md`.

Monitoring: **OK**. No consecutive trips. Calibration skipped on the unlabeled score file, as designed.

Next: look at the CSVs and tighten flags / thresholds / impact numbers. PROPOSAL.md still empty.

### Corrections / overrides

Overrode “install latest sklearn on 3.13 so pandas exists.” Matching **1.5.2** on Python 3.12 is the constraint. A newer sklearn that installs is not a model that loads.

---

## 2026-09-10 — Session 5: account_type on the call list

Added `account_type` (Prospect / Suspect / Former Customer) to `output/prioritized_accounts.csv` and the markdown headings. Rank is still by model score; type is context for the rep, not a new ranking formula.

---

## 2026-09-10 — Session 6: drop weak_signal from the published queue

Checked both CSVs: **only `intent_score` has NaNs** (~40% train, ~39% to-score). `trial_started` is always 0/1.

`weak_signal` (missing intent AND no trial) is not a second missing-data field — “no trial” is observed. On the call list it duplicated `intent_score_missing`. Removed it from `prioritized_accounts.csv` and `call_list.md`. Kept `intent_score_missing`.

Still computed internally in `quality.py` so the mock “Why” can mention thin evidence; not shown as a column.

Proposal later: in production, a composite coverage flag could roll up intent + other vendor holes. Not earned on this batch.

---

## 2026-09-10 — Session 7: data, calibration, quantile vs absolute

Live exploration of both CSVs and `model.pkl` (not taken on faith). Applied some of it in code earlier (flags, sklearn 1.5.2); this session records the numbers and the **tiering decision**.

### Data

- Train 1,200 × 12; score 300 × 11 (no target). No duplicate ids; no id overlap; account_type mix matches closely.
- **Conversion 6.5% (78/1200)** — use this in the proposal, not the brief’s “&lt;1% / low single digits.”
- By type, almost flat: Former Customer 7.2%, Prospect 6.6%, Suspect 6.0%.
- `intent_score` missing 40.2% train / 38.7% score; **only NaN column**. Mild size skew (mean emp ~126 present vs ~113 missing).
- Weak feature–target correlations (max ~0.09 `sales_contacts_90d`). Intent quartiles vs conversion are not a clean gradient.
- One 4,429-employee account vs median 67 — note, not a code “fix.” Our outlier flag is still &gt;50k so this row is not auto-flagged.

### Model

- Pickle is sklearn **1.5.2**. Pipeline: OHE (`account_type`, `industry`) + `SimpleImputer(median)` on seven numerics + `GradientBoostingClassifier`.
- Fitted medians (from the object): employee_count=67, **intent_score=25.3**, mql=1, trial_started=0, trial_users=0, web=2, sales_contacts=1.
- Live batch: max score **0.21**, mean **~0.065**. The “83% likelihood” sketch was never this model.

### In-sample calibration (train, optimistic)

| predicted | n | actual |
|---|---|---|
| 0–5% | 555 | 2.7% |
| 5–10% | 475 | 5.7% |
| 10–15% | 138 | 17.4% |
| 15–20% | 27 | 37% |
| 20–30% | 5 | 40% |

Model underpredicts in upper buckets. Small n at the top. **0.10** is where actual conversion clearly beats 6.5%. A low max score is **not** proof of Cordilla-style drift (that is change **over time**). It is also not a bad model in a 6.5% world — 83% would be overconfidence.

### Output already changed

- Collapsed published DQ to `intent_score_missing` only (Session 6).
- Agent flags; model still median-imputes. Flag exists because the model cannot say “this intent was fabricated.”

### Tiering — decided, not flipped in `rank.py`

This batch: **~37 accounts ≥ 0.10** vs **30 quantile High**; p90 ≈ **0.106**. Almost the same *today*.

**Quantiles stay on the call list** (top 10% / next 20% / rest). Guarantees ~30 High every week (SDR capacity). Hidden cost: a worse batch still looks like 30 Highs — relative “less bad,” not a real bar. That is the Cordilla *mask* if you only look at the list.

**Absolute 0.10 lives in monitoring** (`p90_drop`, `share_above_bar`). Empty-High would be honest but starves the floor some weeks (expensive FNs) and floods it others. Two jobs: predictable queue vs notice when p90 or share ≥ 0.10 falls vs training.

Precision if asked live: tiering **does not branch** on missing intent. Missing intent **does** change the score (imputer → 25.3) and can therefore change the tier.

### Scoring vs monitoring (override)

Running the frozen model on a fresh batch does **not** detect drift. The old Cordilla scorer kept running. Scoring without the second loop is that trap. Kept explicit: `run.py` vs `monitor.py`.

### AI session

Logged this exploration; kept quantile ranks; added p90 / share-above-0.10 checks; wrote the split into `PROPOSAL.md`.

---

## 2026-09-10 — Session 8: High-tier / reasoning boundary mismatch

Caught in review: High = top 10% (**30** accounts) but reasoning + opener only ran for **20**. Ranks 21–30 would show as High on the call list with blank why/opener — the kind of hole a panelist probes live. “20 was a round number” is not a defense.

**Fix:** write-ups follow the **tier**, not a separate N. Every High row gets reasoning + opener. Medium/Low stay blank on purpose: a rep is not working the other ~270 this cycle; generating that copy is wasted cost/latency for text that goes stale. Stated in `PROPOSAL.md`, not left implicit.

**Cadence:** batch job daily/weekly, not per-call. Each run: fresh file → frozen model → rank all 300 → reason High only → `prioritized_accounts.csv` (all 300, audit) + `call_list.md` (High, rep-facing). No cache/dedup for accounts that stay High across runs — regenerate every time so the list matches current data. Caching noted as a future optimization, not built (4-hour scope).

---

## 2026-09-10 — Session 9: system prompt + mock aligned to findings

Updated `SYSTEM_PROMPT` and the mock together. A prompt-only change would not have moved `call_list.md` — the mock writes those fields.

What went into the prompt (and the mock, so output matches):

- Dropped `weak_signal` from prompt fields. Only `intent_score_missing` (only NaN in these CSVs). Kept dataframe column `score` — a draft used `pred_score`, which is not a column here.
- Former Customer opener = re-engagement first, even if they have a trial. Prospect/Suspect = discovery. This was agreed earlier but the **prompt** never said it; the mock used trial before account type, so **ACC-00371** (Former Customer + trial) got a trial opener.
- Never describe the score as a % chance / calibrated probability, **with the calibration reason** (stated numbers don’t match outcomes, especially high end). Mock now says “High priority,” not `score 0.209`.
- Opener hard cap **25 words** (checked in `generate_reasoning`). Max on this run: 18.
- Intent missing: say coverage gap, mention the **25.3** impute is not a real reading, lean on other activity.

Typo `YSTEM_PROMPT` was in an **earlier draft outside this file**, not in the repo — constant was already `SYSTEM_PROMPT`.

### Four-account check (after re-run)

| ID | Expected | Actual |
|---|---|---|
| ACC-01491 Prospect, trial, High | Discovery + trial | Rank #1 High priority; trial/MQL/contacts; opener 17 words, trial discovery. OK |
| ACC-00371 Former Customer, trial, High | Re-engage, not cold/trial-first | Opener now “Since you left, I saw a new Cordilla trial…” — **this was the real gap**; old mock failed here |
| ACC-00646 Prospect, intent missing, High | Not “low intent”; cite 25.3 | Caveat + MQLs/web/contacts; MQL opener. OK |
| ACC-01212 Prospect, Low, rank 167 | No write-up (High-only) | Blank. Not a prompt-drift test — Low never hits this prompt |

Also stripped `(score 0.209)` from `call_list.md` headings so the SDR view matches “priority, not a percentage.” Raw score stays on the CSV for audit/monitoring.

---

## 2026-09-10 — Session 10: impact numbers into PROPOSAL.md

Replaced the weaker “14–17% / a few more conversions” framing with the historical **quantile-cut** lift: High 24.4% (32/131) vs 6.5% baseline (~3.8×); Medium 8.5%; Low 3.2%. Same 30 calls: ~7 vs ~2 expected. Cutoffs are **this scoring batch’s 90th/70th**, applied back to training — not “top 10% of train.” Caveats in the proposal: association not causation, in-sample, roughly +5 not a guarantee, Low still converts, Medium is the second pass.

---

