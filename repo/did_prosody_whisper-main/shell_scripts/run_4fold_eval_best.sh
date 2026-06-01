#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Run evaluation sequentially for test folds 1..4.
# For each run dir:
#   1) find latest checkpoint-* (highest number)
#   2) read <latest>/trainer_state.json -> best_model_checkpoint (absolute path)
#   3) run eval using that best checkpoint
#
# Usage:
#   ./run_4fold_eval_best.sh [REG_WEIGHT] [TAG_PREFIX] [BASE_OUT_ROOT] [FOLD_DIR] [EVAL_PY]
#
# Example:
#   ./run_4fold_eval_best.sh 0.1 nordia_fgdid \
#     /home/idatro/dialect_project/repo/did_prosody_whisper-main/hid-64_model_output \
#     /home/idatro/dialect_project/ndc_folds_loc/fold_pkls \
#     /home/idatro/dialect_project/repo/did_prosody_whisper-main/whisper_did_eval_tmp_multitask.py
# ------------------------------------------------------------

REG_WEIGHT="${1:-0.1}"
TAG_PREFIX="${2:-nordia_fgdid}"
BASE_OUT_ROOT="${3:-/home/idatro/dialect_project/repo/did_prosody_whisper-main/hid64-32_new_model_output}"
FOLD_DIR="${4:-/home/idatro/dialect_project/ndc_folds_loc/fold_pkls}"
EVAL_PY="${5:-/home/idatro/dialect_project/repo/did_prosody_whisper-main/whisper_did_eval_tmp_multitask.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

REG_WEIGHT_FMT="$(LC_NUMERIC=C printf '%.1f' "$REG_WEIGHT")"

CURRENT_CHILD_PID=""

cleanup() {
  echo "[eval-driver] Caught stop signal. Stopping current evaluation..."
  if [[ -n "${CURRENT_CHILD_PID}" ]]; then
    kill -TERM -- "-${CURRENT_CHILD_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup INT TERM HUP

# Find latest checkpoint-* by numeric suffix.
latest_checkpoint_dir() {
  local run_dir="$1"
  local latest=""
  local maxn=-1

  shopt -s nullglob
  for d in "${run_dir}"/checkpoint-*; do
    [[ -d "$d" ]] || continue
    local bn
    bn="$(basename "$d")"
    local n="${bn#checkpoint-}"
    if [[ "$n" =~ ^[0-9]+$ ]]; then
      if (( n > maxn )); then
        maxn=$n
        latest="$d"
      fi
    fi
  done
  shopt -u nullglob

  if [[ -z "$latest" ]]; then
    echo ""
  else
    echo "$latest"
  fi
}

# Read best_model_checkpoint from trainer_state.json, print empty if missing.
read_best_ckpt_from_trainer_state() {
  local trainer_state_json="$1"
  python - <<'PY' "$trainer_state_json"
import json, sys
p = sys.argv[1]
try:
    with open(p, "r") as f:
        j = json.load(f)
    v = j.get("best_model_checkpoint", "")
    print(v if isinstance(v, str) else "")
except Exception:
    print("")
PY
}

run_eval_one() {
  local test_fold="$1"
  #local run_tag="${TAG_PREFIX}_h64-64_a${REG_WEIGHT_FMT}_t${test_fold}_a${REG_WEIGHT_FMT}"
  local run_tag="nordia_fgdid_t${test_fold}_a1.0"
  local run_dir="${BASE_OUT_ROOT}/${run_tag}"
  local test_pkl="${FOLD_DIR}/fold${test_fold}.pkl"
  local eval_out_dir="${run_dir}/eval_best"
  local log_file="${LOG_DIR}/eval_${run_tag}.out"

  echo "============================================================"
  echo "[eval-driver] Run tag:     ${run_tag}"
  echo "[eval-driver] Run dir:     ${run_dir}"
  echo "[eval-driver] Test fold:   ${test_fold} (${test_pkl})"
  echo "[eval-driver] Eval script: ${EVAL_PY}"
  echo "[eval-driver] Log file:    ${log_file}"
  echo "============================================================"

  if [[ ! -d "${run_dir}" ]]; then
    echo "[eval-driver] ERROR: Run dir not found: ${run_dir}" | tee -a "${log_file}"
    return 1
  fi
  if [[ ! -f "${test_pkl}" ]]; then
    echo "[eval-driver] ERROR: Test fold file not found: ${test_pkl}" | tee -a "${log_file}"
    return 1
  fi

  local latest_ckpt
  latest_ckpt="$(latest_checkpoint_dir "${run_dir}")"
  if [[ -z "${latest_ckpt}" ]]; then
    echo "[eval-driver] ERROR: No checkpoint-* dirs found in ${run_dir}" | tee -a "${log_file}"
    return 1
  fi

  local trainer_state="${latest_ckpt}/trainer_state.json"
  if [[ ! -f "${trainer_state}" ]]; then
    echo "[eval-driver] ERROR: Missing trainer_state.json in latest checkpoint: ${trainer_state}" | tee -a "${log_file}"
    return 1
  fi

  local best_ckpt
  best_ckpt="$(read_best_ckpt_from_trainer_state "${trainer_state}")"
  if [[ -z "${best_ckpt}" ]]; then
    echo "[eval-driver] WARNING: best_model_checkpoint missing; falling back to latest checkpoint: ${latest_ckpt}" | tee -a "${log_file}"
    best_ckpt="${latest_ckpt}"
  fi

  echo "[eval-driver] Latest checkpoint: ${latest_ckpt}" | tee -a "${log_file}"
  echo "[eval-driver] Best checkpoint:   ${best_ckpt}"   | tee -a "${log_file}"

  mkdir -p "${eval_out_dir}"

  # Run evaluation in its own process group for clean kill.
  (
    set -euo pipefail
    exec > >(tee -a "${log_file}") 2>&1

    python "${EVAL_PY}" \
      --model_path "${best_ckpt}" \
      --test_dataset "${test_pkl}" \
      --save_path "${eval_out_dir}"
  ) &

  CURRENT_CHILD_PID=$!
  wait "${CURRENT_CHILD_PID}"
  CURRENT_CHILD_PID=""

  echo "[eval-driver] Done: ${run_tag}" | tee -a "${log_file}"
  echo
}

# Sequential evaluations
#run_eval_one 1
#run_eval_one 2
#run_eval_one 3
run_eval_one 4

echo "[eval-driver] All evaluations completed."