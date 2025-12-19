
#!/usr/bin/env python3
"""
detect_clipping.py — Detect likely clipping segments in WAV files without ffmpeg.

Outputs lines:
<file_path>\t<start_seconds>\t<end_seconds>\tClipCount=<N>

Clipping definition:
- Integer PCM: samples exactly at full-scale (min or max).
- IEEE float PCM: |sample| >= 1.0 - eps OR > 1.0 (overs).
"""

import argparse
import os
import sys
import wave
import numpy as np

# ----------------------------- Helpers -----------------------------

def count_clipped_int(block: np.ndarray, bits_per_sample: int) -> int:
    if block.size == 0:
        return 0
    if bits_per_sample == 8:
        return int(np.sum((block <= 0) | (block >= 255)))
    else:
        min_val = -(1 << (bits_per_sample - 1))
        max_val = (1 << (bits_per_sample - 1)) - 1
        return int(np.sum((block <= min_val) | (block >= max_val)))

def count_clipped_float(block: np.ndarray, eps: float = 1e-6) -> int:
    if block.size == 0:
        return 0
    ab = np.abs(block)
    return int(np.sum((ab >= (1.0 - eps)) | (ab > 1.0)))

def find_wavs(root: str, recursive: bool):
    exts = ('.wav', '.WAV', '.wave', '.WAVE')
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                if fn.endswith(exts):
                    yield os.path.join(dirpath, fn)
    else:
        for fn in sorted(os.listdir(root)):
            if fn.endswith(exts):
                yield os.path.join(root, fn)

# ----------------------------- Core logic -----------------------------

def process_file(path: str, window: float, threshold: int, quiet: bool, eps: float):
    try:
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames_per_window = int(sr * window)
            bits_per_sample = sw * 8
            audio_format = 'float' if bits_per_sample in (32, 64) and sw >= 4 else 'int'

            if not quiet:
                print(f"→ {path}")

            start_frame = 0
            while True:
                raw = wf.readframes(frames_per_window)
                if not raw:
                    break
                if audio_format == 'int':
                    block = np.frombuffer(raw, dtype=np.int16 if sw == 2 else np.int8)
                else:
                    block = np.frombuffer(raw, dtype=np.float32)
                if block.size == 0:
                    break
                clip_count = count_clipped_float(block, eps) if audio_format == 'float' else count_clipped_int(block, bits_per_sample)
                if clip_count >= threshold:
                    start_sec = start_frame / sr
                    end_sec = start_sec + window
                    print(f"{path}\t{start_sec:.3f}\t{end_sec:.3f}\tClipCount={clip_count}")
                start_frame += frames_per_window
    except Exception as e:
        print(f"[ERROR] Cannot process {path}: {e}", file=sys.stderr)

# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect clipping in WAV files.")
    parser.add_argument("-d", "--dir", default=".", help="Directory to scan (default: current dir)")
    parser.add_argument("-n", "--limit", type=int, default=0, help="Max number of files to process (0 = all)")
    parser.add_argument("-w", "--window", type=float, default=0.05, help="Analysis window in seconds (default: 0.05)")
    parser.add_argument("-t", "--threshold", type=int, default=1, help="Minimum clipped samples per window to report (default: 1)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode (no progress lines)")
    parser.add_argument("--eps", type=float, default=1e-6, help="Tolerance for float clipping (default: 1e-6)")
    args = parser.parse_args()

    files = list(find_wavs(args.dir, args.recursive))
    if not files:
        print(f"No WAV files found in {args.dir}", file=sys.stderr)
        sys.exit(0)

    if args.limit > 0:
        files = files[:args.limit]

    if not args.quiet:
        print(f"Scanning {len(files)} file(s) in '{args.dir}' (window={args.window}s, threshold={args.threshold})...")

    for f in files:
        process_file(f, args.window, args.threshold, args.quiet, args.eps)

if __name__ == "__main__":
    main()
