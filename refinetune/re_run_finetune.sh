#!/bin/bash
set -euo pipefail

TUNING_MODE="finetune"  # finetune | linearprobing

# Replace with the checkpoint created by the REVISED pretraining pipeline.
PRETRAIN_PATH="/home/DAHS2/DAHS_EHR/review_checkpoints/pretrain_model_best_pretrain_mlm_VP0.1_epoch30.pth"

if [[ "$PRETRAIN_PATH" == "REPLACE_WITH_REVISED_PRETRAIN_CHECKPOINT.pth" ]]; then
  echo "ERROR: Set PRETRAIN_PATH to the revised pretraining checkpoint."
  exit 1
fi

INDEX=0
SEEDS=(41 42 43)
DATASETS=(P12) # candidates: final, benchmark, hirid, eicu, P12
RATIO_LEVELS=(10 30 100)   # zeroshot == ratio 0
WINDOW=48

run_job () {
  local selected_data="$1"
  local ratio_level="$2"
  local seed="$3"
  local init_flags="$4"

  local ratio="$ratio_level"
  local extra_flags=""
  if [[ "$ratio_level" == "zeroshot" ]]; then
    ratio=100
    extra_flags="--inference_mode"
  fi

  local locate_flags=""
  if [[ "$selected_data" == "hirid" ]]; then
    locate_flags="--locate last"
  fi

  local exp_name="MIMIC_${TUNING_MODE}_${EXP_TAG}_w${WINDOW}_fold${INDEX}_${selected_data}_${ratio_level}_seed${seed}"

  echo "========================================"
  echo "INIT: ${EXP_TAG} | DATA: ${selected_data} | RATIO: ${ratio_level} | SEED: ${seed}"
  echo "EXP_NAME: ${exp_name}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=/home/DAHS2/DAHS_EHR:${PYTHONPATH:-} \
  /home/DAHS2/anaconda3/envs/sj/bin/torchrun --nproc_per_node=1 --master_port 29502 /home/DAHS2/DAHS_EHR/refinetune/re_finetune_normal.py \
    --num_hidden_layers 6 \
    --num_attention_heads 8 \
    --intermediate_size 1024 \
    --embedding_size 512 \
    --batch_size 16 \
    --clip_interval 20 \
    --seed "$seed" \
    --acc 8 \
    --epochs 10 \
    --mode finetune \
    --task multitask \
    --tuning_mode "$TUNING_MODE" \
    --exp_name "$exp_name" \
    --learning_rate 1e-4 \
    --dropout_prob 0.3 \
    --classifier_weight 10 \
    --loss binary_cross_entropy \
    --patience 3 \
    --classifier_dropout 0.3 \
    --num_binary_tasks 11 \
    --num_sofa_tasks 6 \
    --num_multiclass_labels 4 \
    --ratio "$ratio" \
    --index "$INDEX" \
    --window "$WINDOW" \
    --selected_data "$selected_data" \
    --data_path /home/DAHS2/DAHS_EHR/datasets/new_data_preparation \
    --num_workers 0 \
    --value_embedding_type continuous \
    --single_task mortality_inhospital \
    $init_flags \
    $locate_flags \
    $extra_flags

  echo "========================================"
  echo "FINISHED DATA=${selected_data} RATIO=${ratio_level} SEED=${seed}"
  echo "========================================"
}

# for INIT_MODE in pretrain randominit
for INIT_MODE in pretrain
do
  if [[ "$INIT_MODE" == "pretrain" ]]; then
    EXP_TAG="external_cohort_pretrain_single_mortality" # mimic_pretrain / external_cohort_pretrain
    INIT_FLAGS="--pretrain --pretrain_path $PRETRAIN_PATH"
  else
    EXP_TAG="external_cohort_randominit_single_mortality" # mimic_randominit / external_cohort_randominit
    INIT_FLAGS=""
  fi

  for DATA in "${DATASETS[@]}"
  do
    for RATIO_LEVEL in "${RATIO_LEVELS[@]}"
    do
      for SEED in "${SEEDS[@]}"
      do
        run_job "$DATA" "$RATIO_LEVEL" "$SEED" "$INIT_FLAGS"
      done
    done
  done
done

echo "========================================"
echo "ALL RUNS FINISHED"
echo "========================================"
