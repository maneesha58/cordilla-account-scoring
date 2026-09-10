# Cordilla account scoring

Working prototype for the Dialpad AI Engineer take-home. A frozen conversion model scores a batch of accounts; this agent flags where that score is on weak evidence and turns the top of the list into something an SDR can call from.

It is a linear pipeline, not a LangGraph/LangChain app. Score → flag → rank → mocked explanation → files. No branching, no tool choice. Ranking by `predict_proba` alone would be `df.sort_values`; the flags and call copy are the reason this exists.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Put the starter artifacts here (do not retrain or regenerate):

- `model/model.pkl`
- `data/accounts_to_score.csv`
- `data/training_data.csv` (not required to run the agent; needed later for impact numbers)

`requirements.txt` is a stand-in until the starter pins arrive. If `model.pkl` fails to unpickle, match the starter's `scikit-learn` version.

## Run

```bash
python run.py
```

or `python -m agent.main`.

Until the starter files are in place, this exits with a clear error. It does not invent a model or CSV.

Treat **2026-08-01** as today. The code does not use the system clock.

## What it writes

- `output/prioritized_accounts.csv` — all accounts: rank, score, tier, intent-missing, weak-signal, reasoning, opener
- `output/call_list.md` — top 20 only, readable for a rep or VP

Console prints how many scored, tier counts, and % missing `intent_score`.

## Judgment calls (change in `agent/config.py`)

| Choice | Default | Why |
|---|---|---|
| Tiers | High = top 10% of *this batch*, Medium = next 20% | Conversion is rare; a 0.70 probability cutoff is the wrong shape |
| Reasoning N | 20 | One focused call block; not 300 mocked LLM calls |
| Extra flags | trial-without-trial, dupes, bad type, employee outliers, negative counts, weak_signal | Flag only — no imputation. Missing intent is not low intent |
| LLM | Mock with a real prompt and a commented API slot | Brief: a documented mock is judged the same as a live call |

## Layout

```
agent/
  score.py        load CSV + model.pkl, predict_proba → score
  quality.py      data-quality flags (no fills)
  rank.py         sort + High/Medium/Low
  llm.py          SYSTEM_PROMPT + call_llm_mock
  reasoning.py    top-N explanations
  output.py       csv + markdown
  main.py         orchestrator
run.py
```

Monitoring / drift is a separate next step. Do not look for it here.
