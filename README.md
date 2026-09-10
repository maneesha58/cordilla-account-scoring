# Cordilla account scoring

Working prototype for the Dialpad AI Engineer take-home. A frozen conversion model scores a batch of accounts; this agent flags where that score is on weak evidence and turns the top of the list into something an SDR can call from.

Two **jobs**, not a graph:

1. **Scoring** — new batch → frozen model → ranked call list.
2. **Monitoring** — compare this batch (and later, real conversions) to a training baseline; alert a human if trust should stop.

No LangGraph/LangChain. Both flows are straight lines: no tool choice, no retries, no HITL node. Ranking by `predict_proba` alone would be `df.sort_values`; flags + call copy are why the agent exists. Monitoring exists because the last Cordilla scorer died quietly.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Put the starter artifacts here (do not retrain or regenerate):

- `model/model.pkl`
- `data/accounts_to_score.csv`
- `data/training_data.csv` (needed for monitoring baseline and later for impact numbers)

`requirements.txt` is a stand-in until the starter pins arrive. If `model.pkl` fails to unpickle, match the starter's `scikit-learn` version.

## Run

```bash
python run.py         # scoring agent
python monitor.py     # monitoring job
```

Until the starter files are in place, both exit with a clear error. They do not invent a model or CSV.

Treat **2026-08-01** as today. The code does not use the system clock.

## What it writes

Scoring:

- `output/prioritized_accounts.csv` — all accounts: rank, score, tier, intent-missing, weak-signal, reasoning, opener
- `output/call_list.md` — top 20 only, readable for a rep or VP

Monitoring:

- `output/monitoring_baseline.json` — period 0 from scoring the training file (optimistic — model saw those labels)
- `output/monitoring_history.json` — one snapshot per period; re-runs upsert, they do not fake extra weeks
- `output/monitoring_report.md` — status OK / WATCH / ALERT and what to do if it pages

## Judgment calls (change in `agent/config.py` / `monitoring/config.py`)

| Choice | Default | Why |
|---|---|---|
| Tiers | High = top 10% of *this batch*, Medium = next 20% | Conversion is rare; a 0.70 probability cutoff is the wrong shape |
| Reasoning N | 20 | One focused call block; not 300 mocked LLM calls |
| Extra flags | trial-without-trial, dupes, bad type, employee outliers, negative counts, weak_signal | Flag only — no imputation. Missing intent is not low intent |
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
