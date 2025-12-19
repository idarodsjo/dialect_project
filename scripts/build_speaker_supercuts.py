#!/usr/bin/env python3
"""
build_speaker_supercuts.py  (WAV-only, supercut-only, no-ffmpeg)

Creates ONE concatenated WAV per speaker using Audacity label files.

Label filename convention (from parse_trs.py output):
    <location>_<speakerToken>_spkX.txt
Example:
    alvdal_01um_spk2.txt  -> location='alvdal', speakerToken='01um', speakerId='spk2'

Audio assumptions:
- Source WAV files live under --audio (default: data/audio)
- Filenames contain the location token (e.g., alvdal_01um-02uk.wav)
- You can override per-label audio path via --audio-map CSV:
      label_base,audio_path
  where label_base == label stem without extension (e.g., "alvdal_01um_spk2")

Outputs:
- One WAV per label (speaker) to --audio-out:
      out/audio/<label_stem>.wav

Usage example:
    python build_speaker_supercuts.py \\
      --labels out/labels \\
      --audio data/audio \\
      --include "^alvdal$" \\
      --mono --sr 16000 \\
      --audio-out out/audio
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from pydub import AudioSegment

# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_label_filename(p: Path):
    """
    For 'alvdal_01um_spk2.txt' -> ('alvdal','01um','spk2','alvdal_01um_spk2')
    """
    stem = p.stem
    m = re.match(r"^([^_]+)_([^_]+)_(spk\d+)$", stem)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), stem

def read_label_spans(label_path: Path) -> List[Tuple[float, float]]:
    """
    Reads Audacity label lines: start<TAB>end<TAB>label
    Returns list of (start_sec, end_sec).
    """
    spans = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            s, e = float(parts[0]), float(parts[1])
            if e > s:
                spans.append((s, e))
    return spans

def load_audio_map(csv_path: Path) -> dict:
    """
    CSV with columns: label_base,audio_path
    Returns dict: base -> absolute/relative audio path string
    """
    m = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row or row.startswith("#"):
                continue
            parts = [x.strip() for x in row.split(",")]
            if len(parts) < 2:
                continue
            base, apath = parts[0], parts[1]
            m[base] = apath
    return m

def find_wav_for_location(audio_dir: Path, location: str, preferred_tokens=None) -> Optional[Path]:
    """
    Heuristic locator: return a WAV file in audio_dir whose name contains the location token.
    If preferred_tokens is given, pick the candidate that contains the most of those tokens.
    """
    candidates = []
    for p in audio_dir.rglob("*.wav"):
        name = p.name.lower()
        if location.lower() in name:
            candidates.append(p)
    if not candidates:
        return None
    if preferred_tokens:
        best = None
        best_score = -1
        for p in candidates:
            lname = p.name.lower()
            score = sum(1 for t in preferred_tokens if t.lower() in lname)
            if score > best_score:
                best, best_score = p, score
        return best
    return candidates[0]

def load_wav_via_pydub(wav_path: Path, sr: Optional[int], mono: bool) -> AudioSegment:
    """
    Load WAV without ffmpeg and apply optional resampling/mono.
    """
    audio = AudioSegment.from_wav(wav_path)  # pydub uses Python's wave module for WAV
    if mono:
        audio = audio.set_channels(1)
    if sr:
        audio = audio.set_frame_rate(sr)
    return audio

def export_supercut_pydub(src_wav: Path, spans: List[Tuple[float,float]],
                          out_wav: Path, sr: Optional[int], mono: bool):
    """
    Concatenate spans into one AudioSegment and export WAV.
    """
    audio = load_wav_via_pydub(src_wav, sr=sr, mono=mono)

    # Build result incrementally to avoid holding two huge copies:
    result = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    if mono and result.channels != 1:
        result = result.set_channels(1)

    for s, e in spans:
        # pydub slicing is in milliseconds
        clip = audio[int(s*1000):int(e*1000)]
        result += clip

    ensure_dir(out_wav.parent)
    result.export(out_wav, format="wav")

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="out/labels", help="Directory with label .txt files")
    ap.add_argument("--audio", default="data/audio", help="Directory with WAV sources")
    ap.add_argument("--include", required=True, help="Regex for location filter (e.g., '^alvdal$')")
    ap.add_argument("--audio-map", help="CSV mapping: label_base,audio_path (optional)")

    # Output
    ap.add_argument("--audio-out", default="out/audio", help="Output dir for per-speaker supercuts")

    # Audio formatting
    ap.add_argument("--sr", type=int, default=None, help="Resample to this sample rate (e.g., 16000)")
    ap.add_argument("--mono", action="store_true", help="Downmix to mono")

    # Misc
    ap.add_argument("--dry-run", action="store_true", help="List actions but do not write outputs")
    args = ap.parse_args()

    labels_dir = Path(args.labels)
    audio_dir = Path(args.audio)
    include_re = re.compile(args.include, re.IGNORECASE)

    audio_map = {}
    if args.audio_map:
        audio_map = load_audio_map(Path(args.audio_map))

    # Find labels that match the location filter and conform to the expected naming
    label_files = []
    for p in labels_dir.iterdir():
        if not p.name.lower().endswith(".txt"):
            continue
        parsed = parse_label_filename(p)
        if not parsed:
            print(f"[SKIP] Unrecognized label filename format: {p.name}", file=sys.stderr)
            continue
        location, speaker_token, speaker_id, base = parsed
        if include_re.search(location):
            label_files.append(p)

    if not label_files:
        print("[INFO] No label files matched --include filter.")
        return

    print(f"[INFO] Matched {len(label_files)} label file(s) for pattern: {args.include}")

    # Group by location to select the best WAV per location
    by_location = {}
    for p in label_files:
        location, speaker_token, speaker_id, base = parse_label_filename(p)
        by_location.setdefault(location, []).append(p)

    for location, files in by_location.items():
        tokens = set(parse_label_filename(p)[1] for p in files)  # speakerToken set
        default_wav = find_wav_for_location(audio_dir, location, preferred_tokens=tokens)

        if not default_wav:
            print(f"[WARN] No WAV found for location '{location}' under {audio_dir}", file=sys.stderr)

        for label_path in files:
            location, speaker_token, speaker_id, base = parse_label_filename(label_path)
            spans = read_label_spans(label_path)
            if not spans:
                print(f"[WARN] {label_path.name}: no spans -> skipping.")
                continue

            # Choose WAV: mapping overrides default heuristic
            mapped = audio_map.get(base)
            wav_path = Path(mapped) if mapped else default_wav
            if not wav_path or not wav_path.exists():
                print(f"[WARN] {label_path.name}: WAV not found (tried {wav_path}). Skipping.", file=sys.stderr)
                continue

            out_wav = Path(args.audio_out) / f"{base}.wav"
            print(f"[DO] Supercut <- {label_path.name}  using {wav_path.name}  -> {out_wav}")
            if not args.dry_run:
                export_supercut_pydub(wav_path, spans, out_wav, sr=args.sr, mono=args.mono)

    print("[DONE]")

if __name__ == "__main__":
    main()