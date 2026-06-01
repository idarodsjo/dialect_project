#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path
import wave
import pandas as pd

# ---------- Reused utilities from your previous build_fold_splits.py ----------
# (1) Read WAV duration in seconds
def read_duration_seconds(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        return frames / float(sr)

# (2) Map speaker_id -> list of WAV files (search speaker_id/ and subdirs)
def list_speaker_files(base_dir: str):
    result = {}
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Audio base directory not found: {base_dir}")
    for speaker_dir in sorted(base.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        wavs = []
        for p in sorted(speaker_dir.glob("*.wav")):
            wavs.append(str(p))
        # include one level of subfolders (as in your prior script)
        for sub in sorted(speaker_dir.iterdir()):
            if sub.is_dir():
                for p in sorted(sub.glob("*.wav")):
                    wavs.append(str(p))
        if wavs:
            result[speaker_id] = sorted(set(wavs))
    return result

# (3) Create (start_ms, end_ms) windows along the duration
def chunk_times(duration_sec: float, chunk_sec=30.0, overlap_sec=0.0, min_last=3.0):
    if duration_sec <= 0:
        return
    hop = max(0.001, chunk_sec - overlap_sec)
    start = 0.0
    while start < duration_sec:
        end = min(start + chunk_sec, duration_sec)
        seg_len = end - start
        if seg_len >= min_last:  # skip too-short tail segments
            yield int(round(start * 1000)), int(round(end * 1000))
        start = start + hop

# (4) Segment ID helper (file stem)
def filename_to_conversation_id(wav_path: str):
    return Path(wav_path).stem
# ---------------------------------------------------------------------------

def build_segments_for_fold(fold_csv: str,
                            speaker_files: dict,
                            chunk_sec: float,
                            overlap_sec: float,
                            min_last_sec: float,
                            allowed_exts=(".wav", ".WAV")) -> pd.DataFrame:
    dff = pd.read_csv(fold_csv)
    if "speaker_id" not in dff.columns:
        raise ValueError(f"{fold_csv} missing 'speaker_id' column.")

    # Keep only speakers for whom we have audio
    dff = dff[dff["speaker_id"].isin(speaker_files.keys())].copy()
    if dff.empty:
        print(f"[WARN] No speakers with audio for {fold_csv}.")
        return pd.DataFrame()

    all_rows = []
    for spk in dff["speaker_id"].unique():
        wav_list = speaker_files.get(spk, [])
        for wav_path in wav_list:
            if Path(wav_path).suffix not in allowed_exts:
                continue
            try:
                dur = read_duration_seconds(wav_path)
            except Exception as e:
                print(f"[WARN] Failed reading {wav_path}: {e}")
                continue

            conv_id = filename_to_conversation_id(wav_path)
            for idx, (st_ms, et_ms) in enumerate(
                chunk_times(dur, chunk_sec=chunk_sec, overlap_sec=overlap_sec, min_last=min_last_sec)
            ):
                seg_id = f"{spk}__{conv_id}__seg{idx:04d}"
                seg_dur_sec = (et_ms - st_ms) / 1000.0
                all_rows.append(
                    {
                        "speaker_id": spk,
                        "segment_id": seg_id,
                        "full_audio_file_path": wav_path,
                        "duration_sec": seg_dur_sec,
                        "offset_start_ms": st_ms,
                        "offset_end_ms": et_ms,
                    }
                )

    seg_df = pd.DataFrame(all_rows)
    if seg_df.empty:
        return seg_df

    # Bring over fold metadata so each segment row carries class labels & coords.
    # We include fg_dialect_region (from your updated fold CSVs).
    keep_cols = [
        "speaker_id", "place", "sex", "age_group",
        "dialect_region", "fg_dialect_region", "latitude", "longitude"
    ]
    meta_cols = [c for c in keep_cols if c in dff.columns]
    seg_df = seg_df.merge(
        dff[meta_cols].drop_duplicates(),
        on="speaker_id",
        how="left"
    )

    # Final column order
    ordered = [
        "speaker_id", "segment_id", "full_audio_file_path", "duration_sec",
        "offset_start_ms", "offset_end_ms",
    ] + meta_cols
    seg_df = seg_df[ordered]
    return seg_df

def main():
    ap = argparse.ArgumentParser(description="Convert fold CSVs to per-fold pickle files with 30s segments + offsets.")
    ap.add_argument("--folds_dir", required=True, help="Directory containing fold*.csv files (e.g., folds_out).")
    ap.add_argument("--audio_dir", required=True, help="Root directory containing per-speaker WAV folders.")
    ap.add_argument("--out_dir", required=True, help="Where to write the per-fold .pkl files.")
    ap.add_argument("--chunk_sec", type=float, default=30.0, help="Segment length in seconds.")
    ap.add_argument("--overlap_sec", type=float, default=0.0, help="Segment overlap in seconds.")
    ap.add_argument("--min_last_sec", type=float, default=3.0, help="Skip tail segments shorter than this.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Find the fold CSVs emitted by tmp_create_folds.py (e.g., fold1.csv ... fold4.csv)
    fold_csvs = sorted(glob.glob(os.path.join(args.folds_dir, "exceptions_lt4.csv")))
    if not fold_csvs:
        raise FileNotFoundError(f"No fold*.csv found in {args.folds_dir}")

    # Build speaker -> files map once
    speaker_files = list_speaker_files(args.audio_dir)

    for fold_csv in fold_csvs:
        seg_df = build_segments_for_fold(
            fold_csv, speaker_files,
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
            min_last_sec=args.min_last_sec
        )
        if seg_df.empty:
            print(f"[WARN] No segments created for {fold_csv}; skipping.")
            continue

        out_name = Path(fold_csv).stem + ".pkl"  # e.g., fold1.pkl
        out_path = os.path.join(args.out_dir, out_name)
        seg_df.to_pickle(out_path)
        print(f"Wrote {out_path} with {len(seg_df):,} segments.")

if __name__ == "__main__":
    main()