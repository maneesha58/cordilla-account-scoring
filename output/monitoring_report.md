# Cordilla scoring — monitoring report

**Status: OK**

Two loops, not one program. The agent scores a batch. This job asks whether those scores are still worth acting on. It does not retrain.

As-of date used everywhere: **2026-08-01** (not the system clock).

## Period 0 — training snapshot (optimistic)

Built by scoring training_data.csv with the frozen model. Calibration here is optimistic (the model was fit on these labels). Production should append a new labeled snapshot when 90-day outcomes land — do not treat this as a live accuracy number.

- n = 1200
- intent missing: 40.2%
- mean score: 0.0661 (std 0.0316)
- p90 score: 0.1088
- share with score ≥ 0.10: 14.2%
- weighted |predicted − actual| : 2.8%

- **calibration** — ok. Does a high score still mean a higher conversion rate? This is the quiet failure that burned the last Cordilla scorer.

| bucket | n | mean predicted | actual convert | gap |
|---|---|---|---|---|
| 0 | 400 | 0.0423 | 0.0225 | 0.0198 |
| 1 | 400 | 0.0540 | 0.0300 | 0.0240 |
| 2 | 400 | 0.1021 | 0.1425 | 0.0404 |

## This batch — label-free drift

Source: `D:\job\dialpad\cordillaAccountScoring\data\accounts_to_score.csv` (n=300)

- **intent_drift** — ok. delta=0.0150 vs threshold=0.1000. Share of accounts with missing intent_score vs training. A coverage jump means scores are resting on a different mix (vendor gap, not 'low intent').
- **score_drift** — ok. delta=0.0006 vs threshold=0.0158. Mean model score vs training. Catches a silent mix shift (smaller companies, fewer trials) before labels exist.

PSI of model inputs vs training. Bins are frozen at period 0 (re-cutting this file would hide the shift). Complements mean-score drift: mix can move while the average score stays put.

- **feature_drift** — ok. max PSI=0.0465; 0 feature(s) > 0.25; 0 feature(s) > 0.10. Population Stability Index on model features vs training (bins frozen at period 0). Trips if any PSI > 0.25 or 2+ features > 0.10. Catches mix shifts that cancel in the mean score. intent_score uses observed values only; missingness is intent_drift.

| feature | PSI | largest bin move (train -> batch) |
|---|---|---|
| `intent_score` | 0.0465 | [-inf, 13.6) 20.2% -> 14.7% |
| `employee_count` | 0.0305 | [165, inf) 19.9% -> 14.7% |
| `trial_active_users` | 0.0230 | 0 87.5% -> 89.7% |
| `industry` | 0.0211 | Healthcare 16.3% -> 13.0% |
| `mql_count_90d` | 0.0174 | [-inf, 1) 69.1% -> 63.0% |
| `web_touchpoints_90d` | 0.0131 | [-inf, 1) 40.2% -> 45.3% |
| `sales_contacts_90d` | 0.0109 | [2, 3) 13.2% -> 10.0% |
| `account_type` | 0.0031 | Former Customer 13.8% -> 12.0% |
| `trial_started` | 0.0025 | 0 81.4% -> 83.3% |

## Absolute quality (not the call-list tiers)

Reps still get quantile High/Medium/Low so workload stays ~top 10%. These two checks catch the case where that top 10% is quietly worse than the bar that beat baseline conversion on training data.

- **p90_drop** — ok. delta=0.0025 vs threshold=0.0200. 90th-percentile score vs training. Call-list High is always top 10% of *this* file. If p90 drops >0.02, those Highs are relatively less-bad, not still above the historical bar.
- **share_above_bar** — ok. delta=0.0183 vs threshold=0.1000. Share of accounts with score ≥ 0.10 (the calibration bar where actual conversion beat the 6.5% baseline). A 10pp drop means fewer accounts clearing a real quality line, even if quantile High still has 30 rows.

## Calibration on this batch

- **calibration** — skipped. no converted_within_90d labels on this file yet

## Consecutive-period rule

Alert the model owner only if a check trips **3 periods in a row**. One noisy week is not the failure mode. A quiet mismatch over a quarter is.

- `intent_drift`: not yet
- `feature_drift`: not yet
- `score_drift`: not yet
- `p90_drop`: not yet
- `share_above_bar`: not yet
- `calibration`: not yet

## If this pages

Pause auto-prioritization by model score. Until a human investigates, rank by observable engagement: trial_started, then mql_count_90d, then web_touchpoints_90d. Do not auto-retrain.

Retraining is a separate, deliberate decision. This job never calls `fit()`.
