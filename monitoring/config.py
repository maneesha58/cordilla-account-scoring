"""Monitoring thresholds. Draft until training_data.csv is in hand — then tighten."""

from agent.config import OUTPUT_DIR, TODAY

BASELINE_PATH = OUTPUT_DIR / "monitoring_baseline.json"
HISTORY_PATH = OUTPUT_DIR / "monitoring_history.json"
REPORT_PATH = OUTPUT_DIR / "monitoring_report.md"

# How many consecutive tripped periods before we page a human.
# One noisy week is not the Cordilla failure; a quiet drift over a quarter is.
CONSECUTIVE_PERIODS = 3

# Label-free drift (can run on accounts_to_score immediately).
# Intent coverage in the brief is ~40% missing. A 10pp swing is "the vendor
# mix changed," not sampling noise on n=300. Revisit after seeing real rates.
INTENT_MISSING_DELTA = 0.10

# Mean-score shift vs training, in units of training std. 0.5 std on a 300-row
# batch is a real mix change; smaller wiggles wait for consecutive periods.
SCORE_MEAN_STD_MULT = 0.5

# Calibration: |mean predicted − actual conversion| in a bucket.
# Floor 3pp so we don't trip on 0.4% vs 0.6% with tiny base rates.
# Also trip if 2× the training-set gap for that bucket (once labels exist).
CALIBRATION_GAP_FLOOR = 0.03
CALIBRATION_GAP_MULT = 2.0
MIN_BUCKET_N = 30
N_SCORE_BUCKETS = 3

# Period 0 on the training set will look optimistic (the model saw those
# labels). Still run it so the check is real. Rank inversion (high bucket
# converts worse than low) is worth a WATCH even on train.
TRAIN_INVERSION_IS_WATCH = True

AS_OF = TODAY.isoformat()
FALLBACK_HEURISTIC = (
    "Pause auto-prioritization by model score. Until a human investigates, "
    "rank by observable engagement: trial_started, then mql_count_90d, "
    "then web_touchpoints_90d. Do not auto-retrain."
)
