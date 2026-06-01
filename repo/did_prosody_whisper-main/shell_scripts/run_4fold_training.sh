#!/usr/bin/env bash
set -uo pipefail

# Usage:
#   ./run_4fold_trainings.sh [REG_WEIGHT] [TAG_PREFIX]
#
# Examples:
#   ./run_4fold_trainings.sh 0.1 nordia_fgdid
#   ./run_4fold_trainings.sh 0.4 nordia_fgdid

REG_WEIGHT="${1:-0.1}"
TAG_PREFIX="${2:-nordia_fgdid}"

# Match formatting used inside your training script (1 decimal, decimal dot)
REG_WEIGHT_FMT="$(LC_NUMERIC=C printf '%.1f' "$REG_WEIGHT")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_model_multitask_nordia.sh"

LOG_DIR="/home/idatro/dialect_project/repo/did_prosody_whisper-main/logs/hid-64-64_fr"
mkdir -p "${LOG_DIR}"

# Fold rotation: val = next fold (wrap 4->1), train = remaining two folds, test = t
# exceptions always added inside the patched training script via EXCEPTIONS_FILE default

run_one () {
  local test_fold="$1"
  local val_fold="$2"
  local train_folds="$3"

  local run_tag="${TAG_PREFIX}_t${test_fold}_a${REG_WEIGHT_FMT}"
  local log_file="${LOG_DIR}/${run_tag}.log"

  echo "============================================================"
  echo "Starting run: ${run_tag}"
  echo "  TEST_FOLD=${test_fold}"
  echo "  VAL_FOLD=${val_fold}"
  echo "  TRAIN_FOLDS=${train_folds} (+ exceptions)"
  echo "  Log: ${log_file}"
  echo "============================================================"

  # Export fold config so train_model_multitask_nordia.sh can use it
  TEST_FOLD="${test_fold}" \
  VAL_FOLD="${val_fold}" \
  TRAIN_FOLDS="${train_folds}" \
  bash "${TRAIN_SCRIPT}" "${REG_WEIGHT}" "${run_tag}" |& tee "${log_file}"

  echo "Finished run: ${run_tag}"
  echo
}

# Define the 4 runs explicitly (clear + matches your description exactly)
run_one 1 2 "3 4"
run_one 2 3 "1 4"
run_one 3 4 "1 2"
run_one 4 1 "2 3"

echo "All 4 trainings completed successfully."