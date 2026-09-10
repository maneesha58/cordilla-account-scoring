# Cordilla account scoring

Working prototype for the Dialpad AI Engineer take-home. A frozen conversion model scores a batch of accounts; this agent flags where that score is on weak evidence and turns the top of the list into something an SDR can call from.

Two **jobs**, not a graph:

1. **Scoring** — new batch → frozen model → ranked call list.
2. **Monitoring** — compare this batch (and later, real conversions) to a training baseline; alert a human if trust should stop.

No LangGraph/LangChain. Both flows are straight lines: no tool choice, no retries, no HITL node. Ranking by `predict_proba` alone would be `df.sort_values`; flags + call copy are why the agent exists. Monitoring exists because the last Cordilla scorer died quietly.

## Setup

Needs **Python 3.11 or 3.12**. `model.pkl` was trained with scikit-learn 1.5.2, which has no 3.13 wheels. This machine can do:

```bash
uv python install 3.12
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\activate          # Windows
```

Or, if 3.12 is already on PATH:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Starter artifacts (do not retrain or regenerate): `model/model.pkl`, `data/accounts_to_score.csv`, `data/training_data.csv`.

## Run

```bash
python run.py         # scoring agent
python monitor.py     # monitoring job
```

Until the starter files are in place, both exit with a clear error. They do not invent a model or CSV.

Treat **2026-08-01** as today. The code does not use the system clock.

## What it writes

Scoring:

- `output/prioritized_accounts.csv` — all accounts: rank, account_id, account_type, score, tier, intent-missing, reasoning, opener
- `output/call_list.md` — High tier only (this batch: 30), readable for a rep or VP

Monitoring:

- `output/monitoring_baseline.json` — period 0 from scoring the training file (optimistic — model saw those labels)
- `output/monitoring_history.json` — one snapshot per period; re-runs upsert, they do not fake extra weeks
- `output/monitoring_report.md` — status OK / WATCH / ALERT and what to do if it pages

## Judgment calls (change in `agent/config.py` / `monitoring/config.py`)

| Choice | Default | Why |
|---|---|---|
| Tiers | High = top 10% of *this batch*, Medium = next 20% | Predictable SDR workload. Absolute bar (score ≥ 0.10, p90) is monitoring, not the call list |
| p90 / share ≥ 0.10 | Drop vs training (0.02 / 10pp) | Quantile High can still print 30 names while the batch got worse |
| Reasoning | All High-tier rows; not Medium/Low | Matches the call-this-cycle set. Top-20 left 21–30 as High with blank copy |
| Extra flags | trial-without-trial, dupes, bad type, employee outliers, negative counts | Flag only — no imputation. Missing intent is not low intent. Call list shows `intent_score_missing` only (only NaN in these CSVs). |
| LLM | Mock with a real prompt and a commented API slot | Brief: a documented mock is judged the same as a live call |
| Intent drift | 10pp vs training missingness | Coverage mix change, not sampling noise on n≈300 |
| Score drift | 0.5 × training std of mean score | Mix shift before labels exist |
| Calibration | gap > max(3pp, 2× train gap), n≥30 in bucket | Cordilla failure: scores look fine, conversion doesn't |
| Consecutive | 3 periods | One noisy week is not an alert |

## Layout

```
agent/              scoring loop
monitoring/         second job: drift + calibration
run.py
monitor.py
```

Monitoring never calls `fit()`. If it alerts: pause auto-priority, fall back to trial / MQL / web, investigate. Retrain is a separate decision.
