#!/bin/bash
set -uo pipefail

OUT_ROOT="/home/idatro/dialect_project/repo/did_prosody_whisper-main/t4_model_output"

for i in $(seq 8 10); do
    REG_WEIGHT="$(awk -v i="$i" 'BEGIN { printf "%.1f", i/10 }')"
    RUN_TAG="nordia_fgdid_t4_a${REG_WEIGHT}"
    OUT_DIR="${OUT_ROOT}/${RUN_TAG}/eval_results"
    MODEL_DIR="${OUT_ROOT}/${RUN_TAG}/checkpoint-492"
    echo "Evaluating ${RUN_TAG} with model from ${MODEL_DIR}..."

    if [ ! -d "$OUT_DIR" ]; then
        mkdir -p "$OUT_DIR"
    else
        echo "${OUT_DIR} already exists. Continuing..."
    fi

    python /home/idatro/dialect_project/repo/did_prosody_whisper-main/whisper_did_eval_tmp_multitask.py --model_path "${MODEL_DIR}" --save_path "${OUT_DIR}"
    echo "Finished evaluation for ${RUN_TAG} at $(date)"
done