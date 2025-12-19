
#!/usr/bin/env bash
# wav_stats_summary.sh — Summary-only stats for .wav audio in a directory.
# Prints: sampling rates x count, channels x count,
# total/mean/median/longest/shortest durations.
#
# Dependencies: ffprobe (FFmpeg) OR soxi (SoX)
# Usage:
#   ./wav_stats_summary.sh [DIRECTORY]
# Examples:
#   ./wav_stats_summary.sh
#   ./wav_stats_summary.sh /path/to/wavs

set -euo pipefail

DIR="."
if [ $# -ge 1 ]; then
  DIR="$1"
fi

if [ ! -d "$DIR" ]; then
  echo "ERROR: Directory not found: $DIR" >&2
  exit 1
fi

# Dependency detection
HAVE_FFPROBE=0
HAVE_SOXI=0
if command -v ffprobe >/dev/null 2>&1; then
  HAVE_FFPROBE=1
elif command -v soxi >/dev/null 2>&1; then
  HAVE_SOXI=1
else
  echo "ERROR: Need either ffprobe (FFmpeg) or soxi (SoX) installed." >&2
  echo "Install tips:" >&2
  echo "  - Ubuntu/Debian: sudo apt-get install ffmpeg    (or: sudo apt-get install sox)" >&2
  echo "  - macOS (Homebrew): brew install ffmpeg         (or: brew install sox)" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

DURATIONS_TXT="$tmpdir/durations.txt"
RATES_TXT="$tmpdir/rates.txt"
CH_TXT="$tmpdir/channels.txt"
: > "$DURATIONS_TXT"
: > "$RATES_TXT"
: > "$CH_TXT"

fmt_hms() {
  local sec_float="$1"
  awk -v t="$sec_float" 'BEGIN{
    sec=int(t+0.5);
    h=int(sec/3600);
    m=int((sec%3600)/60);
    s=sec%60;
    printf "%02d:%02d:%02d", h,m,s
  }'
}

probe_with_ffprobe() {
  local f="$1"
  local sr ch dur
  sr="$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=nk=1:nw=1 "$f" 2>/dev/null || true)"
  ch="$(ffprobe -v error -select_streams a:0 -show_entries stream=channels     -of default=nk=1:nw=1 "$f" 2>/dev/null || true)"
  dur="$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$f" 2>/dev/null || true)"
  echo "$sr|$ch|$dur"
}

probe_with_soxi() {
  local f="$1"
  local sr ch dur
  sr="$(soxi -r "$f" 2>/dev/null || true)"
  ch="$(soxi -c "$f" 2>/dev/null || true)"
  dur="$(soxi -D "$f" 2>/dev/null || true)"
  echo "$sr|$ch|$dur"
}

count=0
while IFS= read -r -d '' f; do
  line=""
  if [ "$HAVE_FFPROBE" -eq 1 ]; then
    line="$(probe_with_ffprobe "$f")"
  else
    line="$(probe_with_soxi "$f")"
  fi

  sr="${line%%|*}"; rest="${line#*|}"
  ch="${rest%%|*}"; dur="${rest#*|}"

  if [[ -z "$sr" || -z "$ch" || -z "$dur" ]]; then
    echo "WARN: Skipping (could not parse): $f" >&2
    continue
  fi
  # numeric duration?
  if ! awk -v x="$dur" 'BEGIN{exit(!(x+0==x))}'; then
    echo "WARN: Skipping (non-numeric duration): $f" >&2
    continue
  fi

  echo "$dur" >> "$DURATIONS_TXT"
  echo "$sr"  >> "$RATES_TXT"
  echo "$ch"  >> "$CH_TXT"
  count=$((count+1))
done < <(find "$DIR" -type f \( -iname "*.wav" -o -iname "*.wave" \) -print0)

if [ "$count" -eq 0 ]; then
  echo "No .wav files found under: $DIR"
  exit 0
fi

# Sort durations and compute stats
readarray -t sorted < <(LC_ALL=C sort -n "$DURATIONS_TXT")
N=${#sorted[@]}

sum=$(awk '{s+=$1} END{printf "%.6f", s+0}' "$DURATIONS_TXT")
mean=$(awk -v s="$sum" -v n="$N" 'BEGIN{if(n>0) printf "%.6f", s/n; else printf "0.000000"}')

if [ $((N % 2)) -eq 1 ]; then
  mid=$((N/2))
  median="${sorted[$mid]}"
else
  m1=$((N/2 - 1))
  m2=$((N/2))
  median=$(awk -v a="${sorted[$m1]}" -v b="${sorted[$m2]}" 'BEGIN{printf "%.6f", (a+b)/2}')
fi

min="${sorted[0]}"
max="${sorted[$((N-1))]}"

# Frequency tables
RATES_SUMMARY="$(LC_ALL=C sort "$RATES_TXT" | uniq -c | LC_ALL=C sort -nr)"
CH_SUMMARY="$(LC_ALL=C sort "$CH_TXT" | uniq -c | LC_ALL=C sort -nr)"

# Print summary
echo ""
echo "Analyzed $N WAV file(s) under: $DIR"
echo "--------------------------------------------"
echo "Total duration:    $(fmt_hms "$sum")  (${sum}s)"
echo "Mean duration:     $(fmt_hms "$mean") (${mean}s)"
echo "Median duration:   $(fmt_hms "$median") (${median}s)"
echo "Longest duration:  $(fmt_hms "$max")  (${max}s)"
echo "Shortest duration: $(fmt_hms "$min")  (${min}s)"
echo "--------------------------------------------"
echo "Sampling rates (Hz) × count:"
# shellcheck disable=SC2086
awk '{printf "  %10s Hz  × %s\n", $2, $1}' <<< "$RATES_SUMMARY"
echo "Channels × count:"
# shellcheck disable=SC2086
awk '{printf "  %10s ch  × %s\n", $2, $1}' <<< "$CH_SUMMARY"
echo "--------------------------------------------"
