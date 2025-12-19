# -*- coding: utf-8 -*-
"""
create_ssc_data_splits.py (extended)

This script retains the original SSC merge logic and adds support for
creating NorDia (NDC) train/validation/test splits with enriched metadata.

New for NorDia:
- Recursively collect all WAV files under norske-lydfiler-fra-ndc-*/
- Join with NDC_norge_metadata.tsv on base id (e.g., "alvdal_01um" from
  a filename like "alvdal_01um-jbj.wav").
- Add 'place', 'area', 'country' columns from metadata.
- Create a 'county' column and update it to 2024 county names by merging with
  muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv
  using municipality/place mappings (old_muni/new_muni).
- Add a 'cardinal four' column from the CSV ('cardinal_four').
- RANDOM split **by audio file** (rows), default ratios 80/10/10.
- Write both {split}_key.pkl (minimal) and {split}_data.pkl (enriched),
  with 'audio_path' as the left-most column.

Usage examples:

# Keep SSC behavior (exactly as before)
python create_ssc_data_splits.py --dataset ssc \
  --ssc-json ssc/ssc_v1_0.jsonl \
  --ssc-split-dir repo/did_prosody_whisper-main/datasets/ssc_data

# Create NorDia splits (by audio files, random 80/10/10)
python create_ssc_data_splits.py --dataset nordia \
  --nordia-root /path/to/preprocessed_NorDia \
  --ndc-metadata NDC_norge_metadata.tsv \
  --muni-map muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv \
  --out-dir datasets/nordia_data \
  --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import sys
import os

# Get the absolute path of the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
new_dir = os.path.dirname(parent_dir)

sys.path.insert(0, new_dir)
print(sys.path)


from CoordMapper.coord_mapper import CoordMapper
import time




# ------------------------------
# Helpers
# ------------------------------

def _ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _normalize_name(s: str) -> str:
    """Normalize municipality/place names for robust matching.
    - lowercases
    - strips leading/trailing whitespace
    - normalizes whitespace
    (Keeps Norwegian letters; no diacritic stripping to avoid collisions.)
    """
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _collect_nordia_wavs(root: Path) -> pd.DataFrame:
    """Collect all .wav files from norske-lydfiler-fra-ndc-*/ directories.
    Returns a DataFrame with columns: audio_path, file_stem, tid, filename, relative_audio_path.
    """
    wavs: List[Path] = []
    for sub in root.glob("norske-lydfiler-fra-ndc-*/*.wav"):
        if sub.is_file():
            wavs.append(sub)
    if not wavs:
        raise FileNotFoundError(
            f"No WAV files found under {root}/norske-lydfiler-fra-ndc-*/"
        )

    rows = []
    for p in wavs:
        stem = p.stem  # e.g., "alvdal_01um-jbj_prcs"
        tid = stem.split("-")[0]  # "alvdal_01um"
        rows.append({
            "audio_path": str(p),
            "file_stem": stem,
            "tid": tid,
            "filename": p.name,
            "relative_audio_path": str(p.relative_to(root)) if str(p).startswith(str(root)) else str(p),
        })
    return pd.DataFrame(rows)


def _maybe_add_trs_paths(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Add a best-effort 'trs_path' column pointing to a matching .trs file.
    We try multiple candidates:
      - file_stem (as-is)
      - file_stem without known processing suffixes (e.g., _prcs, _processed, _proc, _prc)
      - tid
      - first two dash parts from the cleaned stem (e.g., 'alvdal_03gm-04gk')
    """
    trs_dir = root / "fonetisk-ortografisk-transkripsjoner-til-nedlasting"
    if not trs_dir.exists():
        df["trs_path"] = np.nan
        return df

    # Index all trs once for faster lookup
    all_trs = list(trs_dir.glob("**/*.trs"))
    trs_by_stem = {t.stem: t for t in all_trs}

    def strip_known_suffixes(stem: str) -> str:
        suffixes = ["_prcs", "_processed", "_proc", "_prc"]
        for sfx in suffixes:
            if stem.endswith(sfx):
                return stem[: -len(sfx)]
        return stem

    def candidate_keys(file_stem: str, tid: str):
        keys = []
        # as-is
        keys.append(file_stem)
        # without known processing suffix
        clean = strip_known_suffixes(file_stem)
        if clean != file_stem:
            keys.append(clean)
        # base tid
        if tid:
            keys.append(tid)
        # first two dash parts (e.g., alvdal_03gm-04gk_prcs -> alvdal_03gm-04gk)
        parts = clean.split("-")
        if len(parts) >= 2:
            keys.append("-".join(parts[:2]))
        # unique order
        seen = set()
        uniq = []
        for k in keys:
            if k and k not in seen:
                uniq.append(k)
                seen.add(k)
        return uniq

    trs_paths = []
    for _, row in df.iterrows():
        stem = row["file_stem"]
        tid = row["tid"]
        found = None
        for k in candidate_keys(stem, tid):
            hit = trs_by_stem.get(k)
            if hit is not None:
                found = str(hit)
                break
        trs_paths.append(found if found is not None else np.nan)

    df["trs_path"] = trs_paths
    return df


def _merge_ndc_metadata(df_files: pd.DataFrame, ndc_tsv: Path) -> pd.DataFrame:
    """Join WAV list with NDC metadata TSV on tid.
    Brings in: birth, sex, age, agegroup, place, area, region, country.
    """
    meta = pd.read_csv(ndc_tsv, sep="\t")
    for col in ["tid", "place", "area", "region", "country", "agegroup", "sex"]:
        if col in meta.columns:
            meta[col] = meta[col].astype(str)
    merged = df_files.merge(meta, how="left", left_on="tid", right_on="tid")
    return merged


def _apply_muni_mapping(df: pd.DataFrame, muni_csv: Path) -> pd.DataFrame:
    """Update county to 2024 names and attach 'cardinal four' using mapping CSV.
    Matching is done on place against both old_muni and new_muni.
    """
    mapdf = pd.read_csv(muni_csv)

    long_parts = []
    for col in ["old_muni", "new_muni"]:
        if col in mapdf.columns:
            part = mapdf[[col, "new_county_2024", "cardinal_four"]].copy()
            part = part.rename(columns={col: "muni"})
            long_parts.append(part)

    if not long_parts:
        # Fallback: keep old county from TSV's 'area'
        df["county"] = df.get("area", np.nan)
        df["cardinal four"] = np.nan
        return df

    map_long = pd.concat(long_parts, axis=0, ignore_index=True)
    map_long = map_long.dropna(subset=["muni"]).drop_duplicates(subset=["muni"], keep="first")

    # Normalize keys for matching
    map_long["muni_norm"] = map_long["muni"].map(_normalize_name)
    df["place_norm"] = df["place"].map(_normalize_name)

    df = df.merge(
        map_long[["muni_norm", "new_county_2024", "cardinal_four"]],
        how="left",
        left_on="place_norm",
        right_on="muni_norm",
    )

    # Create/Update county and 'cardinal four'
    if "county" not in df.columns:
        df["county"] = df.get("area")
    df["county"] = np.where(df["new_county_2024"].notna(), df["new_county_2024"], df["county"]).astype(str)

    # expose 'cardinal four' (with space, as requested)
    df["cardinal four"] = df.get("cardinal four", pd.Series([np.nan] * len(df)))
    df["cardinal four"] = np.where(df["cardinal_four"].notna(), df["cardinal_four"], df["cardinal four"])

    # Clean up helper columns
    for c in ["muni_norm", "place_norm", "new_county_2024", "cardinal_four"]:
        if c in df.columns:
            del df[c]

    return df


# ------------------------------
# NEW: Random split by audio files (rows), not by tid
# ------------------------------
def _random_file_split(df: pd.DataFrame, train: float, val: float, test: float, seed: int = 42) -> pd.DataFrame:
    """Assign split labels randomly per audio file (row) with global targets.

    Guarantees approximately the requested 80/10/10 (subject to rounding):
      - Shuffle rows
      - First N_train rows -> train
      - Next N_val rows   -> validation
      - Remainder         -> test
    """
    assert abs((train + val + test) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    rng = np.random.default_rng(seed)

    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_total = len(df)
    n_train = int(round(n_total * train))
    n_val   = int(round(n_total * val))
    n_test  = max(0, n_total - n_train - n_val)

    # Adjust if rounding off by 1
    if n_train + n_val + n_test != n_total:
        n_test = n_total - n_train - n_val

    split = np.array([""] * n_total, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "validation"
    split[idx[n_train + n_val:]] = "test"

    df = df.copy()
    df["split"] = split
    return df


def build_nordia_splits(nordia_root: Path, ndc_metadata: Path, muni_map: Path, out_dir: Path,
                        train: float, val: float, test: float, seed: int):
    print("[NorDia] Collecting WAV files...")
    df_files = _collect_nordia_wavs(nordia_root)

    print("[NorDia] Adding .trs paths (best effort)...")
    df_files = _maybe_add_trs_paths(df_files, nordia_root)

    print("[NorDia] Merging NDC metadata...")
    df = _merge_ndc_metadata(df_files, ndc_metadata)

    print("[NorDia] Applying municipality → county(2024) and cardinal_four mapping...")
    df = _apply_muni_mapping(df, muni_map)

    # --- Add latitude/longitude using CoordMapper (place, county, country) ---
    print("[NorDia] Geocoding to add latitude/longitude (this can take a while)...")
    mapper = CoordMapper()  # you can set user_agent/timeout if you modified the class

    # Helper that prefers updated county, falls back to 'area' if county missing
    def _row_to_coords(row):
        place = str(row.get("place", "")).strip()
        country = str(row.get("country", "Norway")).strip() or "Norway"
        county = str(row.get("county", "")).strip()
        # Try with county first (updated 2024), else fall back to original area
        area_for_lookup = county if county else str(row.get("area", "")).strip()
        if not place:
            return (None, None)
        coords = mapper.get_coordinates(place, area_for_lookup, country)
        # Light pacing to avoid hammering the service (if RateLimiter not used)
        time.sleep(0.2)
        if coords is None:
            return (None, None)
        return coords

    # Compute coordinates
    lat_lon = df.apply(_row_to_coords, axis=1, result_type="reduce")
    df["latitude"], df["longitude"] = zip(*lat_lon)

    print("[NorDia] Creating RANDOM split by audio files (rows)...")
    df = _random_file_split(df, train=train, val=val, test=test, seed=seed)

    # Standardize audio_path as string and also keep relative path
    df["audio_path"] = df["audio_path"].astype(str)

    # Columns to include in enriched data (audio_path first, as requested)
    
    cols_pref = [
        "audio_path", "relative_audio_path", "trs_path", "filename", "file_stem", "tid",
        # Metadata
        "birth", "sex", "age", "agegroup", "place", "area", "region", "country",
        # Updated county + cardinal direction
        "county", "cardinal four",
        # NEW geo columns
        "latitude", "longitude",  # <-- NEW
        # Split label
        "split",
    ]

    cols_present = [c for c in cols_pref if c in df.columns]
    df_out = df[cols_present].copy()

    # Write outputs
    _ensure_outdir(out_dir)
    for split in ["train", "validation", "test"]:
        part = df_out[df_out["split"] == split].copy()
        key = part[["audio_path"]].copy()
        key.to_pickle(out_dir / f"{split}_key.pkl")
        part.to_pickle(out_dir / f"{split}_data.pkl")
        print(f"[NorDia] Wrote {split}: {len(part)} rows")


# ------------------------------
# Original SSC behavior (kept for convenience)
# ------------------------------

def build_ssc_merges(ssc_json: Path, ssc_split_dir: Path):
    ssc_json_official = pd.read_json(ssc_json, lines=True)
    ssc_json_official["audio_path"] = ssc_json_official["audio_path"].astype(str)

    for split in ["train", "validation", "test"]:
        split_key_path = ssc_split_dir / f"{split}_key.pkl"
        if not split_key_path.exists():
            raise FileNotFoundError(f"Missing {split_key_path}")
        split_key = pd.read_pickle(split_key_path)
        split_key["audio_path"] = split_key["audio_path"].astype(str)
        split_data = split_key.merge(ssc_json_official, on="audio_path", how="inner")
        split_data.to_pickle(ssc_split_dir / f"{split}_data.pkl")
        print(f"[SSC] Merged {split}: {len(split_data)} rows → {ssc_split_dir / f'{split}_data.pkl'}")


# ------------------------------
# CLI
# ------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create SSC and/or NorDia data splits with metadata.")
    p.add_argument("--dataset", choices=["ssc", "nordia"], required=True, help="Which dataset operation to run.")

    # SSC args
    p.add_argument("--ssc-json", type=Path, default=Path("ssc/ssc_v1_0.jsonl"))
    p.add_argument("--ssc-split-dir", type=Path, default=Path("repo/did_prosody_whisper-main/datasets/ssc_data"))

    # NorDia args
    p.add_argument("--nordia-root", type=Path, default=Path("/home/idatro/preprocessed_NorDia"))
    p.add_argument("--ndc-metadata", type=Path, default=Path("/home/idatro/preprocessed_NorDia/NDC_norge_metadata.tsv"))
    p.add_argument("--muni-map", type=Path, default=Path("/home/idatro/repo/DialectMapper/dialect_mapper/mapping_data/muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("datasets/nordia_data"))
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args(argv)


def main(argv: List[str] = None):
    ns = parse_args(argv if argv is not None else sys.argv[1:])

    if ns.dataset == "ssc":
        build_ssc_merges(ns.ssc_json, ns.ssc_split_dir)
    elif ns.dataset == "nordia":
        build_nordia_splits(
            nordia_root=ns.nordia_root,
            ndc_metadata=ns.ndc_metadata,
            muni_map=ns.muni_map,
            out_dir=ns.out_dir,
            train=ns.train,
            val=ns.val,
            test=ns.test,
            seed=ns.seed,
        )
    else:
        raise ValueError(ns.dataset)


if __name__ == "__main__":
    main()