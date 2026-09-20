# Usage: ./run_duett_preprocessing_mimic.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

python3 "$SCRIPT_DIR/make_hourly_resampled_dataset.py" \
  --sequence-path "$REPO_ROOT/preprocessing/processed/mimic_preprocessed/mimic_preprocessed_window24.0.parquet" \
  --cohort-path "$REPO_ROOT/preprocessing/processed/mimic_preprocessed/mimic_cohort_preprocessed_window24.0.parquet" \
  --duett-feature-map "$SCRIPT_DIR/YAIB_label_map.xlsx" \
  --split-pkl-dir "$REPO_ROOT/datasets/new_data_preparation" \
  --output-dir "$SCRIPT_DIR/mimic_splits" \
  --window-hours 24 \
  --num-time-bins 32 \
  --feature-col label_idx \
  --source-aware \
  --chart-agg last \
  --medication-agg sum \
  --fill zero \
  --normalize \
  --fold 0 \
  --duett \
  "$@"
