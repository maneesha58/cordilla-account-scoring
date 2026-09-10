# Cordilla Sales Prioritization Agent — proposal

The goal is not to predict conversion for its own sake. It is to help SDRs and account managers decide where to spend scarce selling time, and to know when that ranking has quietly stopped matching the field.

Cordilla already has a frozen conversion model (`model.pkl`). This work puts its output on a weekly call list, flags where a score was built on a filled-in intent value, and runs a second job that can page a human when the list is still printing “High” while the underlying quality has dropped.

## Impact

**The decision this changes.** Cordilla has thousands of Prospects, Suspects, and Former Customers and a sales team with limited time. Reps currently guess who to call. The unused frozen model scores likelihood of converting within 90 days. The decision it informs: **which accounts does a rep call first this week.** Type alone does not answer that (Former Customer 7.2%, Prospect 6.6%, Suspect 6.0% on the 1,200-row training file). Measured baseline: **78/1,200 converted (6.5%)** — not the brief’s “well under 1%” range.

**What it’s worth if it works.** Tiers in code are quantiles of *this* scoring batch (High = top 10%, Medium = next 20%). I took **those same score cutoffs** (this week’s 90th/70th percentile, not “top 10% of training”) and applied them back to historical training outcomes:

| Tier | Historical conversion | vs 6.5% baseline |
|---|---|---|
| High | 24.4% (32/131) | ~3.8× |
| Medium | 8.5% (19/223) | ~1.3× |
| Low | 3.2% (27/846) | below baseline |

Accounts the system would label High converted at **nearly 4×** a random account. On this 300-row batch that is 30 High / 60 Medium / 210 Low. If the batch behaves like history: **30 High calls → ~7 expected conversions** vs **30 random calls → ~2**. Same effort, **roughly five more conversions** — pointing reps at the right 10%, not calling more people. The model’s top score is still only **0.21**; this is a rare-event business, not 83% confidence on any one row. Medium (8.5%) is the second pass, not “ignore everyone else.”

**If it’s wrong, in each direction.** A High account that does not convert costs **rep time** on the wrong 30 instead of a different 30 — real, bounded, recoverable next batch. A winner sitting in Low is worse because it is **silent**: Low still held 27 historical conversions (3.2%, not zero). Working only High is a **bet** that concentrating on the highest-lift segment beats spreading thin — not a claim that Low never converts.

**Caveats, said plainly.** This is **association, not causation**. High accounts already looked more engaged (trial, MQLs, intent); we are better at *finding* accounts trending toward conversion, not proven to *create* conversions that would not have happened. This split is **in-sample** (same file the model was trained on), so it is a best-case read; live batches may be softer. High n=131 is enough to trust the pattern, not to treat “+5 conversions” as a guarantee. Treat it as a real historical signal worth acting on and watching (see Monitoring).

## Agent

No LangGraph. The flow is linear: load → `predict_proba` (frozen) → flag missing intent → quantile rank → mocked “why” + opener for **every High-tier account** → write files. There is no tool choice or retry loop. Ranking alone would be `sort_values`; the product is **trust** (`intent_score_missing`) plus **a line a rep can say**.

The pipeline inside `model.pkl` is OneHotEncoder (`account_type`, `industry`) + SimpleImputer median on seven numerics + GradientBoostingClassifier. Fitted medians include **intent_score = 25.3**. Missing intent is **not** 0 and is **not** dropped. The model always scores as if intent were typical. The agent flags that so a rep can see a High that sits on a fill-in. Tier **logic** does not special-case missingness; missingness **does** move the score, so it can move the tier. That distinction matters live.

**Call-list tiers stay quantiles** of this batch: High = top 10%, Medium = next 20%, Low = rest. On this file that is 30 / 60 / 210. This week’s 90th percentile is ~0.106, next to the 0.10 calibration bar (~37 accounts ≥ 0.10), so the two rules almost agree **today**. They will not agree if the world gets worse: quantiles still emit 30 Highs. That is a feature for **workload** (VP still gets a list) and a bug for **honesty**. Absolute 0.10 / 0.05 on the queue would make empty-High weeks visible, but would also starve the floor some weeks (bad when false negatives are expensive) and flood it others. So: **quantiles for the SDR, absolute bar for monitoring.**

**Why + opener only on High.** An earlier cut generated copy for 20 rows while High was 30, so ranks 21–30 showed as High with blank fields. Write-ups now follow the **tier**. Medium and Low stay blank: a rep is not working 270 extra accounts this cycle.

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
