#!/usr/bin/env python3
"""
Remove duplicate-named columns from pandas pickles—but only if later duplicates
are value-identical to the first occurrence. Keeps the first; drops only later copies.

Usage examples:
  # Dry run: report duplicates and planned removals (no writes)
  python dedupe_pickle_columns.py --inputs /path/to/*.pkl --dry-run --debug

  # In place, with backups (*.bak)
  python dedupe_pickle_columns.py --inputs /path/to/dir --recursive

  # Write cleaned files to a separate folder (no touching originals)
  python dedupe_pickle_columns.py --inputs /path/to/dir --recursive --out-dir /path/to/cleaned
"""
import argparse
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


def expand_inputs(inputs: List[str], recursive: bool) -> List[Path]:
    paths: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if any(ch in inp for ch in "*?[]"):
            for g in glob.glob(inp, recursive=recursive):
                gp = Path(g)
                if gp.is_file() and gp.suffix.lower() == ".pkl":
                    paths.append(gp)
        elif p.is_dir():
            pattern = "**/*.pkl" if recursive else "*.pkl"
            for gp in p.glob(pattern):
                if gp.is_file():
                    paths.append(gp)
        elif p.is_file() and p.suffix.lower() == ".pkl":
            paths.append(p)
        else:
            print(f"[WARN] Skipping non-existing path or non-pickle: {inp}", file=sys.stderr)

    # de-duplicate and keep order
    seen = set()
    out: List[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def values_match(s1: pd.Series, s2: pd.Series) -> bool:
    """
    Return True if two Series have the same values at every position.
    NaNs at the same locations count as equal. Dtype differences are ignored.
    """
    try:
        if s1.equals(s2):
            return True
        a = s1.astype("object")
        b = s2.astype("object")
        a_null = a.isna()
        b_null = b.isna()
        if not a_null.equals(b_null):
            return False
        mask = ~a_null
        return np.array_equal(a[mask].values, b[mask].values)
    except Exception:
        return False


def dedupe_columns(df: pd.DataFrame, *, verbose: bool = True, debug: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    For each duplicated column name:
      - Compare each later occurrence to the first; if all identical -> mark for drop.
      - If any differs -> keep all and report mismatch.

    Returns (clean_df, summary)
    """
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("This script does not support MultiIndex columns.")

    col_index = pd.Index(df.columns)
    dupe_mask = col_index.duplicated(keep=False)

    summary = {
        "total_dupe_names": 0,
        "dropped_columns": 0,
        "mismatched_names": 0,
    }

    if not dupe_mask.any():
        return df, summary

    # Build name -> list of positions with that name
    name_to_pos: Dict[str, List[int]] = {}
    for pos, name in enumerate(col_index):
        if dupe_mask[pos]:
            name_to_pos.setdefault(name, []).append(pos)

    summary["total_dupe_names"] = len(name_to_pos)

    to_drop_positions: List[int] = []
    for name, positions in name_to_pos.items():
        first_pos = positions[0]
        s_first = df.iloc[:, first_pos]
        all_equal = True
        for pos in positions[1:]:
            s_other = df.iloc[:, pos]
            if values_match(s_first, s_other):
                to_drop_positions.append(pos)  # drop later identical copies
            else:
                all_equal = False
        if not all_equal and verbose:
            print(f"[MISMATCH] Column '{name}' has duplicates with differing values; none removed for this name.", file=sys.stderr)

    if to_drop_positions:
        if debug:
            to_drop_names = [df.columns[pos] for pos in to_drop_positions]
            print(f"[DEBUG] Will drop {len(to_drop_positions)} duplicate column instances (keeping FIRST occurrence): {to_drop_names}")

        # *** CRITICAL FIX: drop by POSITION, not by label ***
        keep_mask = np.ones(df.shape[1], dtype=bool)
        keep_mask[to_drop_positions] = False
        before = df.shape[1]
        clean_df = df.iloc[:, keep_mask].copy()  # positional keep
        after = clean_df.shape[1]
        summary["dropped_columns"] = before - after
        return clean_df, summary

    # nothing to drop
    return df, summary


def main():
    ap = argparse.ArgumentParser(description="Remove duplicate columns from pickle files if (and only if) later duplicates are identical to the first.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Pickle files, directories, or glob patterns (e.g., /path/*.pkl).")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories if inputs contain directories or ** globs.")
    ap.add_argument("--out-dir", type=str, default=None, help="Write cleaned pickles to this directory (mirrors filenames). If omitted, edits are in-place.")
    ap.add_argument("--backup-suffix", type=str, default=".bak", help="When editing in place, save a backup as <file>.bak (set to '' to disable).")
    ap.add_argument("--dry-run", action="store_true", help="Analyze and report, but do not write any files.")
    ap.add_argument("--quiet", action="store_true", help="Reduce verbosity.")
    ap.add_argument("--debug", action="store_true", help="Print names of columns that are removed in each file.")
    args = ap.parse_args()

    targets = expand_inputs(args.inputs, recursive=args.recursive)
    if not targets:
        print("[ERROR] No pickle files found.", file=sys.stderr)
        sys.exit(2)

    total_files = 0
    changed_files = 0
    total_dropped = 0
    total_mismatched_names = 0

    for p in targets:
        total_files += 1
        try:
            df = pd.read_pickle(p)
        except Exception as e:
            print(f"[ERROR] Failed to read {p}: {e}", file=sys.stderr)
            continue

        if not isinstance(df, pd.DataFrame):
            print(f"[WARN] {p} does not contain a pandas DataFrame; skipping.", file=sys.stderr)
            continue

        before_cols = df.shape[1]
        clean_df, summary = dedupe_columns(df, verbose=not args.quiet, debug=args.debug)
        after_cols = clean_df.shape[1]

        total_mismatched_names += summary.get("mismatched_names", 0)

        if before_cols == after_cols:
            if not args.quiet:
                print(f"[OK] {p}: no removable duplicates (or none found).")
        else:
            changed_files += 1
            total_dropped += (before_cols - after_cols)
            if args.dry_run:
                print(f"[DRY-RUN] {p}: would drop {before_cols - after_cols} duplicate columns.")
            else:
                if args.out_dir:
                    out_dir = Path(args.out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / p.name
                    clean_df.to_pickle(out_path)
                    print(f"[WRITE] {p} → {out_path} (dropped {before_cols - after_cols})")
                else:
                    if args.backup_suffix:
                        backup_path = Path(str(p) + args.backup_suffix)
                        try:
                            pd.read_pickle(p).to_pickle(backup_path)
                        except Exception as e:
                            print(f"[WARN] Could not write backup for {p}: {e}", file=sys.stderr)
                    clean_df.to_pickle(p)
                    print(f"[WRITE] {p} (in-place, dropped {before_cols - after_cols})")

    if not args.quiet:
        print("\n==== Summary ====")
        print(f"Files scanned:   {total_files}")
        print(f"Files changed:   {changed_files}")
        print(f"Columns dropped: {total_dropped}")
        print(f"Names mismatched (kept as-is): {total_mismatched_names}")


if __name__ == "__main__":
    main()