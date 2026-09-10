"""Paths and judgment-call defaults. Change these; don't scatter magic numbers."""

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Brief: treat this as "today" for any recency/age calc — never datetime.now().
TODAY = date(2026, 8, 1)

MODEL_PATH = REPO_ROOT / "model" / "model.pkl"
ACCOUNTS_PATH = REPO_ROOT / "data" / "accounts_to_score.csv"
TRAINING_PATH = REPO_ROOT / "data" / "training_data.csv"
OUTPUT_DIR = REPO_ROOT / "output"

# Columns the brief lists as model inputs. Overridden at load time if the
# fitted pipeline exposes feature_names_in_ (preferred — don't guess).
DEFAULT_FEATURE_COLUMNS = [
    "account_type",
    "employee_count",
    "industry",
    "intent_score",
    "mql_count_90d",
    "trial_started",
    "trial_active_users",
    "web_touchpoints_90d",
    "sales_contacts_90d",
]

NON_FEATURE_COLUMNS = {"account_id", "snapshot_date", "converted_within_90d"}

# Top-N get LLM reasoning. ~20 is one focused SDR call block, not 300 API calls.
REASONING_N = 20

# Within-batch tiers (not calibrated P(convert)). Conversion is rare; fixed
# 0.5/0.7 cutoffs would likely dump everyone into Low. Revisit once we see
# the actual score histogram from model.pkl.
HIGH_QUANTILE = 0.90  # top 10% of this batch → High (call this week)
MEDIUM_QUANTILE = 0.70  # next 20% → Medium (second pass)
