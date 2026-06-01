#!/bin/bash
set -uo pipefail

# ---- Ida's Nordia multitask training launcher ----
# This updates your previous script by adding a regression task (lat/lon) with MSE loss
# and normalizing coordinates to [-1, 1] during training.


REG_WEIGHT="${1:-0.1}"                            # default stays 0.4 if not given
# Ensure decimal dot and 1 decimal precision (e.g., 0.0, 0.1, ... 1.0)
REG_WEIGHT_FMT="$(LC_NUMERIC=C printf '%.1f' "$REG_WEIGHT")"
RUN_TAG="${2:-nordia_fgdid_t4_a${REG_WEIGHT_FMT}}"   # e.g., nordia_fgdid_t4_a0.5

BASE_OUT_ROOT="/home/idatro/dialect_project/repo/did_prosody_whisper-main/hid64-64_new_model_output"
unmod_model_output_dir="${BASE_OUT_ROOT}/${RUN_TAG}"  # <— replaces hard-coded dir
#unmod_model_output_dir="${BASE_OUT_ROOT}" # Just for checkpoint training


# (Optional) Weights & Biases tracking#
export WANDB_ENTITY="ida-roedsjoe-ntnu"
export WANDB_PROJECT="hidden-64-64_fr_a1.0_new"


# 1) Concatenate folds (now configurable via env vars)

FOLD_DIR="/home/idatro/dialect_project/ndc_folds_loc/fold_pkls"
EXCEPTIONS_FILE="${EXCEPTIONS_FILE:-${FOLD_DIR}/exceptions_lt4.pkl}"

# Driver can set these; defaults keep your current behavior
TEST_FOLD="${TEST_FOLD:-4}"              # not used in concat, but helpful for tags
VAL_FOLD="${VAL_FOLD:-1}"
TRAIN_FOLDS="${TRAIN_FOLDS:-2 3}"

# Build train file list
train_files=()
for f in ${TRAIN_FOLDS}; do
  train_files+=("${FOLD_DIR}/fold${f}.pkl")
done
train_files+=("${EXCEPTIONS_FILE}")

val_file="${FOLD_DIR}/fold${VAL_FOLD}.pkl"

python /home/idatro/dialect_project/repo/did_prosody_whisper-main/shell_scripts/concat_folds.py \
  --train_files "${train_files[@]}" \
  --val_file "${val_file}" \
  --out_train tmp_run/train_concat.pkl \
  --out_val tmp_run/val_single.pkl \
  --drop_dupes


train_file=tmp_run/train_concat.pkl
eval_file=tmp_run/val_single.pkl
label_column=fg_dialect_region   # change as necessary

# Model output dirs (if you want to iterate variants)
#unmod_model_output_dir=/home/idatro/dialect_project/repo/did_prosody_whisper-main/t3_model_output/nordia_fgdid_t3_a1.0
low_pass_model_output_dir=/path/to/model_output/low_pass
monotonize_model_output_dir=/path/to/model_output/monotonize
output_dirs=($unmod_model_output_dir $low_pass_model_output_dir $monotonize_model_output_dir)

# Audio column variants
unmod_audio_column=full_audio_file_path
low_pass_audio_column=low_pass_audio
monotonize_audio_column=monotonize_audio
audio_columns=($unmod_audio_column $low_pass_audio_column $monotonize_audio_column)

for ((i=0 ; i < 1 ; i++)); do
  model=${output_dirs[$i]}
  echo "Creating model(s) in directory: ${model}"
  if [ ! -d $model ]; then
    mkdir -p $model
  else
    echo "${model} already exists. Continuing..."
  fi

  CUDA_VISIBLE_DEVICES=0,1 deepspeed /home/idatro/dialect_project/repo/did_prosody_whisper-main/whisper_did_training_multitask.py \
    --deepspeed original_ds_config.json \
    --model_name_or_path NbAiLab/nb-whisper-medium \
    --overwrite_output_dir True \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --audio_column_name ${audio_columns[$i]} \
    --label_column_name ${label_column} \
    --output_dir ${model} \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --fp16 \
    --learning_rate 3e-5 \
    --max_length_seconds 30 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 2 \
    --logging_strategy steps \
    --logging_steps 25 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model obj_score \
    --seed 0 \
    --freeze_feature_encoder True \
    --latitude_column latitude \
    --longitude_column longitude \
    --regression_weight 1.0

done