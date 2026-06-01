#!/usr/bin/env python3
"""
Create 4 location-balanced folds + exceptions for NDC.

- Each place with >=4 speakers contributes exactly 4 speakers: one to each fold.
- Sex (M/F) and age group (A/B) are balanced across folds using a rotation and a greedy balancing step.
- Places with <4 speakers go to exceptions_lt4.csv (to be concatenated into training only).
- For places with >4 speakers, only 4 are used for the folds; the remainder go to exceptions_extra.csv.
- dialect_region is merged from ndc_cardinal4_by_muni.csv.

Usage:
    python create_ndc_folds.py \
        --metadata NDC_norge_metadata.tsv \
        --mapping ndc_cardinal4_by_muni.csv \
        --outdir ./folds_out \
        --rotation-offset 0

Outputs:
    folds_out/fold1.csv, fold2.csv, fold3.csv, fold4.csv
    folds_out/exceptions_lt4.csv
    folds_out/exceptions_extra.csv
    folds_out/folds_summary.csv  (quick QC)
"""

import argparse
import os
import sys
from typing import Dict, List
import pandas as pd
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from repo.CoordMapper.coord_mapper import CoordMapper

BASE_CATS = ["MA", "FA", "MB", "FB"]  # sex+age_group

def load_and_merge(metadata_path: str, mapping_path: str) -> pd.DataFrame:
    # Read metadata (tab-separated)
    meta = pd.read_csv(metadata_path, sep="\t")
    # Expect columns: tid, place, sex, agegroup
    # Rename for consistent merging
    meta = meta.rename(columns={"tid": "speaker_id", "agegroup": "age_group"})
    meta = meta[["speaker_id", "place", "area", "sex", "age_group"]]

    # Read mapping (dialect region)
    mapping = pd.read_csv(mapping_path)[["speaker_id", "dialect_region"]]

    # Merge and validate
    df = meta.merge(mapping, on="speaker_id", how="left", validate="one_to_one")
    missing = df["dialect_region"].isna()
    if missing.any():
        missing_ids = df.loc[missing, "speaker_id"].tolist()
        raise ValueError(
            "dialect_region missing for these speaker_ids from mapping file:\n"
            + "\n".join(missing_ids)
        )

    # Normalize types
    df["sex"] = df["sex"].astype(str).str.upper().str.strip()
    df["age_group"] = df["age_group"].astype(str).str.upper().str.strip()
    df["cat"] = df["sex"] + df["age_group"]
    return df

def make_output_dir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def balance_score(counts, fold: int, sex: str, age_group: str) -> float:
    """
    Compute how 'unbalanced' the global fold counts would be after assigning
    a (sex, age_group) speaker to `fold`. Lower is better.
    """
    score = 0.0
    # Sex balance
    for s in ["M", "F"]:
        total = sum(counts[f][s] for f in range(4)) + (1 if sex == s else 0)
        avg = total / 4.0
        for f in range(4):
            val = counts[f][s] + (1 if (f == fold and sex == s) else 0)
            score += abs(val - avg)
    # Age-group balance
    for a in ["A", "B"]:
        total = sum(counts[f][a] for f in range(4)) + (1 if age_group == a else 0)
        avg = total / 4.0
        for f in range(4):
            val = counts[f][a] + (1 if (f == fold and age_group == a) else 0)
            score += abs(val - avg)
    return score

def assign_place_to_folds(
    g: pd.DataFrame,
    counts: Dict[int, Dict[str, int]],
    rotation_offset: int,
    place_index: int,
    rng: random.Random,
):
    """
    Given all speakers from one place (g), return:
      assigned_rows_by_fold: dict[fold] -> row (Series)
      extras_df: DataFrame of remaining speakers (not used in folds)
      lt4_df: DataFrame of all speakers when len(g) < 4 (else empty)
    """
    if len(g) < 4:
        return {}, pd.DataFrame(), g  # all are exceptions_lt4

    # Sort to keep determinism; compute category
    g = g.sort_values("speaker_id").reset_index(drop=True)
    # For speed: precompute indices per category
    cat_to_idxs = {c: g.index[g["cat"] == c].tolist() for c in BASE_CATS}

    # Choose at most one representative per category (if present)
    chosen = {c: rng.choice(idxs) for c, idxs in cat_to_idxs.items() if len(idxs) > 0}

    # Rotation assigns which category each fold "expects" for this place
    r = (rotation_offset + place_index) % 4
    fold_to_expected_cat = {f: BASE_CATS[(r + f) % 4] for f in range(4)}

    used = set()
    assigned = {}

    # First pass: if the expected category exists, use it
    for f in range(4):
        cat = fold_to_expected_cat[f]
        if cat in chosen:
            idx = chosen[cat]
            if idx not in used:
                assigned[f] = idx
                used.add(idx)
                row = g.loc[idx]
                counts[f][row["sex"]] += 1
                counts[f][row["age_group"]] += 1

    # Second pass: fill remaining folds by minimizing global imbalance
    for f in range(4):
        if f in assigned:
            continue
        remaining_idxs = [i for i in g.index if i not in used]
        if not remaining_idxs:
            break  # shouldn't happen since len(g) >= 4

        # Prefer expected category if available
        expected_cat = fold_to_expected_cat[f]
        preferred = [i for i in remaining_idxs if g.loc[i, "cat"] == expected_cat]
        pool = preferred if preferred else remaining_idxs

        
        scores = []
        for idx in pool:
            sex = g.loc[idx, "sex"]
            ag  = g.loc[idx, "age_group"]
            s   = balance_score(counts, f, sex, ag)
            scores.append((idx, s))
        # find minimal score
        min_s = min(s for _, s in scores)
        # all candidates that achieve the min score
        best_idxs = [idx for idx, s in scores if s == min_s]
        # choose reproducibly at random among ties
        best_idx = rng.choice(best_idxs)


        assigned[f] = best_idx
        used.add(best_idx)
        row = g.loc[best_idx]
        counts[f][row["sex"]] += 1
        counts[f][row["age_group"]] += 1

    # Build fold rows
    assigned_rows_by_fold = {
        f: g.loc[idx, ["speaker_id", "place", "area", "sex", "age_group", "dialect_region"]]
        for f, idx in assigned.items()
    }
    # Extras are anything from this place not used for folds
    extras_df = g.drop(index=list(used))[
        ["speaker_id", "place", "area", "sex", "age_group", "dialect_region"]
    ].copy()

    return assigned_rows_by_fold, extras_df, pd.DataFrame()


def load_county_candidates(county_map_path: Optional[str]) -> dict:
    """
    Build old_county -> [new_county_2024 ...] mapping from your county CSV.
    """
    county_candidates = {}
    if county_map_path and os.path.exists(county_map_path):
        county_df = pd.read_csv(county_map_path)
        need = {"old_county", "new_county_2024"}
        if need.issubset(set(county_df.columns)):
            tmp = (
                county_df[["old_county", "new_county_2024"]]
                .dropna()
                .drop_duplicates()
            )
            county_candidates = (
                tmp.groupby("old_county")["new_county_2024"]
                   .apply(lambda s: list(dict.fromkeys(s.tolist())))
                   .to_dict()
            )
        else:
            print("WARNING: county map missing {old_county,new_county_2024}; skipping modernization.")
    else:
        print("NOTE: county map not found; geocoding will use old `area` values.")
    return county_candidates


def main(args):
    df = load_and_merge(args.metadata, args.mapping)
    make_output_dir(args.outdir)
    rng = random.Random(args.seed)
    county_candidates = load_county_candidates(args.county_map)

    # --- Load county modernization map (old_county -> list of new_county_2024) ---
    county_candidates = {}
    if args.county_map and os.path.exists(args.county_map):
        county_df = pd.read_csv(args.county_map)
        # Keep only the columns we need and drop missing
        need_cols = [c for c in ["old_county", "new_county_2024"] if c in county_df.columns]
        if set(["old_county", "new_county_2024"]).issubset(need_cols):
            tmp = (
                county_df[["old_county", "new_county_2024"]]
                .dropna()
                .drop_duplicates()
            )
            # Build: old_county -> [candidate new counties...]
            county_candidates = (
                tmp.groupby("old_county")["new_county_2024"]
                .apply(lambda s: list(dict.fromkeys(s.tolist())))
                .to_dict()
            )
        else:
            print("WARNING: county map file does not contain expected columns "
                "`old_county`, `new_county_2024`; skipping county modernization.")
    else:
        print("NOTE: county map not found; geocoding will use the old `area` values.")



    unique_places = df[["place", "area"]].drop_duplicates().reset_index(drop=True)

    mapper = CoordMapper(user_agent="ndc_fold_builder", timeout=10)  # rate-limited internally
    place_lat = []
    place_lon = []

    for r in unique_places.itertuples(index=False):
        # Old county (metadata)
        old_county = r.area if pd.notna(r.area) else None
        # Candidate 2024 counties for this old county (may be multiple)
        candidates = county_candidates.get(old_county, []) if old_county else []
        coords = None

        # 1) Try each 2024 county candidate with the given place
        for new24 in candidates:
            coords = mapper.get_coordinates(r.place, new24, country="Norway")
            if coords is not None:
                break

        # 2) If still no hit, try with the old county (as some names still resolve)
        if coords is None and old_county:
            coords = mapper.get_coordinates(r.place, old_county, country="Norway")

        # 3) Final fallback: just (place, Norway)
        if coords is None:
            coords = mapper.get_coordinates(r.place, None, country="Norway")

        if coords is None:
            place_lat.append(None)
            place_lon.append(None)
        else:
            lat, lon = coords
            place_lat.append(lat)
            place_lon.append(lon)


    unique_places["latitude"] = place_lat
    unique_places["longitude"] = place_lon

    def attach_coords(d: pd.DataFrame) -> pd.DataFrame:
        # join by (place, area), then drop area from outputs
        if d.empty:
            d["latitude"] = []
            d["longitude"] = []
            return d
        out = d.merge(unique_places, on=["place", "area"], how="left")
        # drop area from final outputs, reorder columns
        return out[["speaker_id", "place", "sex", "age_group", "dialect_region", "latitude", "longitude"]]


    # Group by place
    groups = list(df.groupby("place"))
    groups.sort(key=lambda kv: kv[0])  # deterministic order

    if args.shuffle_places:
        rng.shuffle(groups)  # reproducible shuffle based on --seed

    # Prepare fold containers and running counts
    folds = {i: [] for i in range(4)}
    counts = {i: {"M": 0, "F": 0, "A": 0, "B": 0} for i in range(4)}

    exceptions_lt4 = []
    exceptions_extra = []

    for place_idx, (place, g) in enumerate(groups):
        assigned, extras_df, lt4_df = assign_place_to_folds(
            g,
            counts=counts,
            rotation_offset=args.rotation_offset,
            place_index=place_idx,
            rng=rng
        )
        if len(lt4_df) > 0:
            exceptions_lt4.append(lt4_df)
            continue

        # add to folds
        for f, row in assigned.items():
            folds[f].append(row)

        if len(extras_df) > 0:
            exceptions_extra.append(extras_df)

    # Build DataFrames
    fold_dfs = {i: (pd.DataFrame(folds[i]).reset_index(drop=True) if len(folds[i]) else
                    pd.DataFrame(columns=["speaker_id", "place", "area", "sex", "age_group", "dialect_region"]))
                for i in range(4)}
    exc_lt4_df = pd.concat(exceptions_lt4, ignore_index=True) if exceptions_lt4 else pd.DataFrame(
        columns=["speaker_id", "place", "area", "sex", "age_group", "dialect_region"]
    )
    exc_extra_df = pd.concat(exceptions_extra, ignore_index=True) if exceptions_extra else pd.DataFrame(
        columns=["speaker_id", "place", "area", "sex", "age_group", "dialect_region"]
    )

    fold_dfs = {i: attach_coords(d) for i, d in fold_dfs.items()}
    exc_lt4_df = attach_coords(exc_lt4_df)
    exc_extra_df = attach_coords(exc_extra_df)


    # Save folds
    fold_paths = []
    for i in range(4):
        path = os.path.join(args.outdir, f"fold{i+1}.csv")
        fold_dfs[i].to_csv(path, index=False)
        fold_paths.append(path)

    # Save exceptions (with reason column to be explicit)
    if len(exc_lt4_df):
        exc_lt4_df = exc_lt4_df.copy()
        exc_lt4_df["exception_reason"] = "place_has_<4_speakers"
    if len(exc_extra_df):
        exc_extra_df = exc_extra_df.copy()
        exc_extra_df["exception_reason"] = "extra_from_place_>4"

    exc_lt4_df.to_csv(os.path.join(args.outdir, "exceptions_lt4.csv"), index=False)
    exc_extra_df.to_csv(os.path.join(args.outdir, "exceptions_extra.csv"), index=False)

    # Quick QC summary (counts by fold)
    summary_rows = []
    for i, d in fold_dfs.items():
        row = {
            "fold": i + 1,
            "n": len(d),
            "M": int((d["sex"] == "M").sum()),
            "F": int((d["sex"] == "F").sum()),
            "A": int((d["age_group"] == "A").sum()),
            "B": int((d["age_group"] == "B").sum()),
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.outdir, "folds_summary.csv"), index=False)

    print("Saved folds to:")
    for p in fold_paths:
        print(" -", p)
    print("Exceptions:")
    print(" -", os.path.join(args.outdir, "exceptions_lt4.csv"))
    print(" -", os.path.join(args.outdir, "exceptions_extra.csv"))
    print("\nQC summary:\n", summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="/home/idatro/preprocessed_NorDia/NDC_norge_metadata.tsv",
                        help="Path to NDC metadata TSV (tab-separated).")
    parser.add_argument("--mapping", type=str, default="/home/idatro/dialect_project/ndc_speaker_outputs/ndc_cardinal4_by_muni.csv",
                        help="Path to speaker_id -> dialect_region CSV.")
    parser.add_argument("--outdir", type=str, default="./folds_out",
                        help="Output directory for CSVs.")
    parser.add_argument("--rotation-offset", type=int, default=0,
                        help="Rotation offset for distributing sex/age categories across folds.")
    parser.add_argument("--county-map", type=str, default="/home/idatro/dialect_project/repo/DialectMapper/dialect_mapper/mapping_data/muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv", help="CSV with mappings from old county to new mapping")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for reproducabile randomization.")
    parser.add_argument("--shuffle-places", action="store_true", help="Shuffle the order of places before assignment.")
    parser.add_argument("--spread-extras-penalty", type=float, default=0.25, help="Soft penalty per existing speaker from the same place in a fold when placing extras. 0.0 disables.")

    args = parser.parse_args()
    main(args)