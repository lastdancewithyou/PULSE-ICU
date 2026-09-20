# Usage: ./run_duett_preprocessing_eicu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

python3 "$SCRIPT_DIR/make_hourly_resampled_eICU.py" \
  --sequence-path "$REPO_ROOT/datasets/eicu_mapped.parquet" \
  --timeline-path "$REPO_ROOT/datasets/eicu_timeline.parquet" \
  --cohort-path "$REPO_ROOT/data_statistic/finetune_eicu_YAIB_cohort.parquet" \
  --raw-cohort-path "$REPO_ROOT/cohort/eicu_cohort_labeled.parquet" \
  --yaib-map "$SCRIPT_DIR/YAIB_label_map.xlsx" \
  --units-map "$SCRIPT_DIR/unit_map.csv" \
  --split-pkl-dir "$REPO_ROOT/datasets/new_data_preparation" \
  --output-dir "$SCRIPT_DIR/eicu_splits" \
  --window-hours 24 \
  --num-time-bins 32 \
  --feature-col label_idx \
  --chart-agg last \
  --medication-agg sum \
  --fill zero \
  --normalize \
  --fold 0 \
  --duett \
  "$@"
