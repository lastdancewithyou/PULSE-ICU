#!/bin/bash

set -e
cd "$(dirname "$0")"

SEEDS=(41 42 43)
DEVICE=${DEVICE:-1}
EXTRA=""
TAG=pretrain
if [ -n "$RANDOM_INIT" ]; then EXTRA="--random-init"; TAG=randominit; fi

for SEED in "${SEEDS[@]}"; do
  JSON="checkpoints/eicu_finetune_${TAG}_seed${SEED}/test_metrics.json"
  if [ -f "$JSON" ] && grep -q '"auprc_macro"' "$JSON"; then
    echo "[skip] seed ${SEED}: test_metrics.json exists"
  elif [ -f "$JSON" ]; then
    echo "[eval-only] seed ${SEED}: JSON predates SOFA AUPRC, re-evaluating the saved best checkpoint"
    python finetune_duett.py --seed "$SEED" --devices "$DEVICE" $EXTRA --eval-only
  else
    python finetune_duett.py --seed "$SEED" --devices "$DEVICE" $EXTRA
  fi
done

python aggregate_duett.py --seeds "${SEEDS[@]}" $EXTRA
