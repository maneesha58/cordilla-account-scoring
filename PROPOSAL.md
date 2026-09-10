# Cordilla Sales Prioritization Agent — proposal

The goal is not to predict conversion for its own sake. It is to help SDRs and account managers decide where to spend scarce selling time, and to know when that ranking has quietly stopped matching the field.

Cordilla already has a frozen conversion model (`model.pkl`). This work puts its output on a weekly call list, flags where a score was built on a filled-in intent value, and runs a second job that can page a human when the list is still printing “High” while the underlying quality has dropped.

## Impact

**Decision maker.** An SDR or AM chooses which account gets the next 30 minutes. The VP of Sales cares about the aggregate: are those minutes landing on accounts with higher expected conversion than a guess?

**What the data actually says.** On `training_data.csv` (1,200 labeled accounts, snapshot treated as 2026-08-01), **78 converted (6.5%)**. That is the measured baseline — not the brief’s “well under 1% cold / low single digits engaged” range.

Conversion is almost flat by `account_type`: Former Customer 7.2%, Prospect 6.6%, Suspect 6.0%. Type alone does not tell a rep who to call. `intent_score` is missing on 40.2% of training rows and 38.7% of the 300-row score file; it is the **only** column with NaNs. Missingness skews mildly toward smaller companies (mean employees ~113 missing vs ~126 present), not a dramatic split.

Scoring the training file with the frozen model (in-sample — optimistic):

| Predicted score | n | Actual conversion |
|---|---|---|
| 0–5% | 555 | 2.7% |
| 5–10% | 475 | 5.7% |
| 10–15% | 138 | 17.4% |
| 15–20% | 27 | 37% (small n) |
| 20–30% | 5 | 40% (very small n) |

Above about **0.10**, actual conversion clearly beats 6.5%. Below 0.05 it is worse. The model **underpredicts** in the upper buckets even on its own training data. The 300-account live batch has **max score 0.21, mean ~0.065**. A row that said “83% likelihood” would have been a lie. In a 6.5% world, frequent 80% scores would be overconfidence, not quality. Lift is **relative**: a 0.15–0.21 account is on the order of 2–3× the baseline, not a sure close.

**Capacity, not accuracy.** Suppose a rep can work ~30 accounts with care this cycle (the size of our High bucket). Random 30 at 6.5% is about 2 expected conversions. Thirty accounts from the region that actually converted ~14–17% is a few more — small counts, in-sample, but the mechanism is concentrating time, not contacting more people. False positives waste a call. False negatives (skipping a winner) lose revenue quietly and are worse. We still show High names; we do not auto-drop accounts just because intent is missing.

**Wrong in each direction.** Calling a dead High wastes SDR time and can annoy a prospect. Parking a real winner in Low because the score is 0.04 is a silent miss. The agent must not pretend 0.21 is 83%, and must not hide missing intent behind the imputer.

## Agent

No LangGraph. The flow is linear: load → `predict_proba` (frozen) → flag missing intent → quantile rank → mocked “why” + opener for **every High-tier account** → write files. There is no tool choice or retry loop. Ranking alone would be `sort_values`; the product is **trust** (`intent_score_missing`) plus **a line a rep can say**.

The pipeline inside `model.pkl` is OneHotEncoder (`account_type`, `industry`) + SimpleImputer median on seven numerics + GradientBoostingClassifier. Fitted medians include **intent_score = 25.3**. Missing intent is **not** 0 and is **not** dropped. The model always scores as if intent were typical. The agent flags that so a rep can see a High that sits on a fill-in. Tier **logic** does not special-case missingness; missingness **does** move the score, so it can move the tier. That distinction matters live.

**Call-list tiers stay quantiles** of this batch: High = top 10%, Medium = next 20%, Low = rest. On this file that is 30 / 60 / 210. This week’s 90th percentile is ~0.106, next to the 0.10 calibration bar (~37 accounts ≥ 0.10), so the two rules almost agree **today**. They will not agree if the world gets worse: quantiles still emit 30 Highs. That is a feature for **workload** (VP still gets a list) and a bug for **honesty**. Absolute 0.10 / 0.05 on the queue would make empty-High weeks visible, but would also starve the floor some weeks (bad when false negatives are expensive) and flood it others. So: **quantiles for the SDR, absolute bar for monitoring.**

**Why + opener only on High, not a separate top-20.** An earlier cut generated copy for 20 rows while High was 30, so ranks 21–30 showed as High with blank fields. That was a bug, not a product choice. Write-ups now follow the **tier boundary**. Medium and Low stay blank on purpose: a rep is not working 270 extra accounts this cycle, and LLM copy would go stale before it was read. Cost/latency for unused text is not worth it.

This is a **batch job** (daily/weekly): pull the file → frozen model → rank all 300 → reason High only → `prioritized_accounts.csv` (all 300, audit/monitoring) and `call_list.md` (High only, rep-facing). Accounts that stay High across weeks are **regenerated each run** so the list matches current features. Caching/dedup is a future optimization, not built here.

Output columns: rank, type, score (audit), tier, intent-missing, reasoning, opener. The LLM is mocked with a real system prompt (Former Customer = re-engagement, never a cold intro; never describe the score as a % chance — calibration does not match; opener ≤ 25 words). Commented API slot. Copy is grounded in actual fields. Production coverage gaps could become one composite flag later; this batch only has intent holes.

Runs: `python run.py`. As-of date is **2026-08-01**, not the clock. New batches mean the **same** pickle, not `fit()`.

## Monitoring

The last Cordilla scorer kept running. Scores stopped matching the field. Nobody was watching the right thing. **Re-scoring every week is not monitoring.** It is fuel for monitoring. Two jobs: `python run.py` then `python monitor.py`.

**Data quality.** Missing-intent rate vs training (10pp trip). Only intent is missing here.

**Distribution.** Mean score vs training (0.5 σ). Plus the absolute checks quantile tiers hide:

- **p90_drop** — this batch’s 90th-percentile score vs training. If p90 falls by more than 0.02, “High” is still top 10% but the best of a worse file.
- **share_above_bar** — share of accounts with score ≥ **0.10** (the calibration line that beat 6.5%). A 10pp drop means fewer accounts clearing a real bar while the call list still has 30 Highs.

**Outcomes (when 90-day labels exist).** Bucket predicted vs actual conversion. Trip if a large bucket’s gap exceeds max(3pp, 2× train gap) for **three consecutive** periods, or if high-score accounts convert worse than low-score ones. Training-set calibration is shown as period 0 and will look too good.

**If it pages.** Pause auto-priority by model score. Fall back to trial, then MQLs, then web. Investigate. **Do not auto-retrain.** Retrain is a separate decision.

Deploy as two scheduled jobs, not a graph. A well-documented mock LLM is in-scope; a live key is not required.

---
