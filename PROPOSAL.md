# Cordilla Sales Prioritization Agent — proposal

The goal is not to predict conversion for its own sake. It is to decide where scarce selling time goes, and to know when that ranking has quietly stopped matching the field. Cordilla already has a frozen model (`model.pkl`). This work puts it on a weekly call list, flags scores built on filled-in intent, and runs a second job that can page a human when the list still prints “High” while quality has dropped.

## Impact

**The decision.** Which accounts a rep calls first this week. Type alone does not answer that (Former Customer 7.2%, Prospect 6.6%, Suspect 6.0% on training). Baseline: **78/1,200 converted (6.5%)** — not the brief’s “well under 1%.”

**If it works.** Tiers are quantiles of *this* scoring batch (High = top 10%, Medium = next 20%). Those same cutoffs (this week’s 90th/70th, not “top 10% of training”) applied back to historical outcomes:

| Tier | Historical conversion | vs 6.5% |
|---|---|---|
| High | 24.4% (32/131) | ~3.8× |
| Medium | 8.5% (19/223) | ~1.3× |
| Low | 3.2% (27/846) | below baseline |

This batch is 30 High / 60 Medium / 210 Low. If it behaves like history: **30 High calls → ~7 expected conversions** vs **30 random → ~2**. Same effort, **roughly five more** — pointing reps at the right 10%, not calling more people. Max score is still **0.21**; this is a rare-event business, not 83% confidence on a row. Medium is the second pass, not “ignore everyone else.”

**If it’s wrong.** A bad High costs bounded rep time, recoverable next batch. A winner in Low is worse because it is silent: Low still held 27 conversions. Working only High is a bet that the highest-lift segment beats spreading thin.

**Caveats.** Association, not causation. High accounts already looked more engaged; we are better at *finding* converters than creating them. The split is in-sample (best case). n=131 supports the pattern, not “+5” as a guarantee. These two CSVs sit on a wider stack — Salesforce, marketing automation, an intent vendor, enrichment, web/ad attribution, product telemetry — each with coverage holes (intent skews toward larger/known accounts; product usage only exists once a trial has started); live ranking inherits those gaps, not just the columns we scored.

## Agent

No LangGraph. The flow has no tool choice or retry loop. Ranking by `predict_proba` alone would be `sort_values`. The product is **trust** plus **a line a rep can say**. `python run.py`. As-of **2026-08-01**, not the clock. Same pickle, never `fit()`.

### Architecture

```
accounts_to_score.csv + frozen model.pkl
        │
        ▼
   1. score     predict_proba → score
        │
        ▼
   2. flag      intent_score_missing
        │
        ▼
   3. rank      High / Medium / Low (within-batch quantiles)
        │
        ▼
   4. reason    mocked LLM: why + opener for every High row
        │
        ▼
   5. write     prioritized_accounts.csv (all 300) + call_list.md (High only)
```

| Step | Module | Does | Does not |
|---|---|---|---|
| Score | `agent/score.py` | `predict_proba` | Retrain, impute, drop rows |
| Flag | `agent/quality.py` | Mark missing intent | Fill 25.3, hide a High |
| Rank | `agent/rank.py` | Top 10% / next 20% / rest | Use 0.10 as a call-list cut |
| Reason | `agent/llm.py` | Why + opener ≤ 25 words | Live API, copy for Medium/Low, “21% chance” |
| Write | `agent/output.py` | CSV audit; markdown queue | Extra DQ flags on the rep view |

Pipeline inside the pickle: OHE (`account_type`, `industry`) + median impute (intent **25.3**) + GBT. Missing intent is not 0 and is not dropped. The model scores as if intent were typical. The agent flags that so a High on a fill-in is visible.

Batch job, not a per-call chat. High-stayers are regenerated each run. CSV = all 300 (audit). Markdown = High only (SDR). Score stays on the CSV; the call list says priority, not `score 0.209`.

### Why these choices

**No framework.** Score → flag → rank → explain → write has no branch. A graph would not change the SDR list.

**Quantiles on the list, absolute bar in monitoring.** Scores cluster low (max ~0.21). A 0.70 cut would empty High; a 0.10 cut on the queue would starve or flood the floor. Quantiles keep ~30 Highs (capacity). They still print 30 names if the world gets worse (honesty bug). Absolute 0.10 / p90 live in monitoring. This week they almost agree (p90 ~0.106, ~37 accounts ≥ 0.10). They will not if the mix worsens.

**Flag missing intent; do not impute again.** Coverage is not missing at random; treating missing as 0 punishes smaller accounts. Missing ≠ low intent. Tier logic does not special-case it; missingness still moves the score, so it can move the tier. Only `intent_score_missing` is published — the only NaN on these files.

**Mock LLM, real prompt.** Documented mock is in-scope. Prompt contract: Former Customer = re-engagement, never a cold intro; never a % chance (calibration does not match); opener ≤ 25 words; cite real fields; missing intent = vendor gap, 25.3 is not a reading.

## Monitoring

Re-scoring is not monitoring. Deploys as two scheduled jobs — `run.py` weekly, `monitor.py` immediately after — not a live service, not an orchestration graph. Never `fit()`.

### What we watch

| Layer | Check | Labels? | Trip (then 3 periods in a row) |
|---|---|---|---|
| Coverage | `intent_drift` | No | Missing-intent rate vs train >10pp |
| Input mix | `feature_drift` | No | PSI vs train: any >0.25 or 2+ >0.10; bins frozen at period 0 |
| Output | `score_drift` | No | Mean score vs train >0.5 σ |
| Honesty | `p90_drop` | No | p90 falls >0.02 vs train |
| Honesty | `share_above_bar` | No | Share with score ≥ 0.10 falls >10pp vs train |
| Outcomes | `calibration` | Yes | Bucket \|pred − actual\| > max(3pp, 2× train gap), or high converts worse than low |

Architecture: **this week → frozen training baseline → report.** The first five checks run now on the 300-row file. Calibration is wired, **skipped** here (no outcomes). Period 0 on training shows the table and will look too good (the model saw those labels).

**Noise vs real.** Thresholds are sized for n=300; a check tripping once is WATCH. Alert only if the same check trips **3 weeks in a row** — one noisy week is not the failure; persistence is.

**Why p90 / 0.10 exist.** Quantile High always names 30 accounts. If p90 or the share ≥ 0.10 falls, those Highs are relatively less-bad, not still above the bar that beat 6.5%.

**Storage.** This prototype overwrites the call-list CSV. Production **appends** a dated snapshot (`account_id`, `scored_at`, score, features, model version). CRM conversions land in a second table. After 90 days a job joins; calibration runs on the join. Drift still runs the week you score. `converted_within_90d` on a batch file here is a stand-in for that join.

**If it pages.** Pause auto-priority. Fall back to trial → MQLs → web. Investigate. **Do not auto-retrain.**

---
