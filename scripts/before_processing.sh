#!/usr/bin/env bash
# Check WAV files for clipping using SoX
# Requirements: sox
# Usage: set DIR to the directory you want to scan.

set -u  # treat unset variables as an error
IFS=$'\n\t'

# === Set this before running ===
DIR="/home/idatro/talebase/data/speech_raw/ScanDia/NorDia/norske-lydfiler-fra-ndc-1"

# Optional: treat values >= this amplitude as "near clipping" warnings (float)
NEAR_CLIP_THRESHOLD=0.9990

# Internal counters
total=0
clipped=0
nearclip=0
ok=0

# Make sure directory exists
if [[ ! -d "$DIR" ]]; then
  echo "Error: Directory does not exist: $DIR" >&2
  exit 1
fi

# Handle case of no wav files
shopt -s nullglob
files=("$DIR"/*.wav "$DIR"/*.WAV)
if (( ${#files[@]} == 0 )); then
  echo "No .wav files found in: $DIR"
  exit 0
fi

printf "Scanning %d WAV files in: %s\n\n" "${#files[@]}" "$DIR"

for f in "${files[@]}"; do
  ((total++))

  # sox 'stat' writes to stderr; capture it
  stats="$(sox "$f" -n stat 2>&1 || true)"

  # Extract values
  # Example lines:
  #   Maximum amplitude:     1.000000
  #   Minimum amplitude:    -1.000000
  #   Clipping:                   42
  max_amp="$(printf '%s\n' "$stats" | awk -F: '/Maximum amplitude/ {gsub(/^ +| +$/,"",$2); print $2}')"
  min_amp="$(printf '%s\n' "$stats" | awk -F: '/Minimum amplitude/ {gsub(/^ +| +$/,"",$2); print $2}')"
  clip_cnt="$(printf '%s\n' "$stats" | awk '/^Clipping:/ {print $2}')"

  # Fallbacks if parsing fails
  max_amp="${max_amp:-NA}"
  min_amp="${min_amp:-NA}"
  clip_cnt="${clip_cnt:-0}"

  # Decide status
  status=""
  if [[ "$clip_cnt" =~ ^[0-9]+$ ]] && (( clip_cnt > 0 )); then
    status="CLIPPED"
    ((clipped++))
  else
    # If not flagged as clipped, still warn if max amplitude is very close to 1.0
    if [[ "$max_amp" != "NA" ]]; then
      # Compare as floats using awk to avoid bash float issues
      is_near="$(awk -v x="$max_amp" -v t="$NEAR_CLIP_THRESHOLD" 'BEGIN{ if (x >= t) print "1"; else print "0"; }')"
      if [[ "$is_near" == "1" ]]; then
        status="NEAR-CLIP"
        ((nearclip++))
      fi
    fi
  fi

  if [[ -z "$status" ]]; then
    status="OK"
    ((ok++))
  fi

  printf "%-9s | %s\n" "$status" "$f"
  printf "           max_amp=%s  min_amp=%s  clipped_samples=%s\n" "$max_amp" "$min_amp" "$clip_cnt"
done

echo
echo "Summary:"
printf "  Total files   : %d\n" "$total"
printf "  CLIPPED       : %d\n" "$clipped"
printf "  NEAR-CLIP (>= %.4f): %d\n" "$NEAR_CLIP_THRESHOLD" "$nearclip"
printf "  OK            : %d\n" "$ok"
exit 0