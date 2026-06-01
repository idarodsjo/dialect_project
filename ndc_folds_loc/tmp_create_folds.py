#!/usr/bin/env python3
"""
Create 4 location-balanced folds + exceptions for NDC, with coordinates and extras assigned.

Key properties:
- For every place with >= 4 speakers: at least one speaker goes to each fold (location coverage).
- Sex (M/F) and age group (A/B) are globally balanced using rotation + greedy balancing + tie RNG.
- All EXTRA speakers from >4-speaker places are ALSO assigned to folds (not to exceptions),
  minimizing the same global balance objective. Optionally spread extras across folds for that place.
- Places with <4 speakers -> exceptions_lt4.csv (for training-only concatenation).
- dialect_region merged from ndc_cardinal4_by_muni.csv.
- fg_dialect_region (fg=fine-grained) filled by looking up place in the county map's named_dialect, with fallback to (place, county) disambiguation.
- latitude, longitude added via CoordMapper, trying 2024 counties first based on a county map.

Inputs:
  --metadata  NDC_norge_metadata.tsv          (tid, place, area, sex, agegroup, ...)
  --mapping   ndc_cardinal4_by_muni.csv       (speaker_id, dialect_region)
  --county-map muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv
  --transform-places ndc_transform.csv        (optional: legacy place -> modern municipality)

Outputs:
  folds_out/fold1.csv ... fold4.csv
  folds_out/exceptions_lt4.csv
  folds_out/folds_summary.csv
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
import pandas as pd
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from repo.CoordMapper.coord_mapper import CoordMapper 

BASE_CATS = ["MA", "FA", "MB", "FB"]  # sex+age_group

# -------------------------- LOADING & MERGE --------------------------

def load_and_merge(metadata_path: str, mapping_path: str, transform_path: Optional[str]) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path, sep="\t")
    # Expect columns: tid, place, area, sex, agegroup                      
    meta = meta.rename(columns={"tid": "speaker_id", "agegroup": "age_group"})
    meta = meta[["speaker_id", "place", "area", "sex", "age_group"]]

    # Optional: normalize legacy place names -> modern municipalities       
    if transform_path and os.path.exists(transform_path):
        tr_df = pd.read_csv(transform_path, header=None, names=["old_place", "new_place"])
        tr_map = dict(zip(tr_df["old_place"], tr_df["new_place"]))
        meta["place"] = meta["place"].apply(lambda p: tr_map.get(p, p))

    mapping = pd.read_csv(mapping_path)[["speaker_id", "dialect_region"]]     
    df = meta.merge(mapping, on="speaker_id", how="left", validate="one_to_one")
    missing = df["dialect_region"].isna()
    if missing.any():
        missing_ids = df.loc[missing, "speaker_id"].tolist()
        raise ValueError(
            "dialect_region missing for these speaker_ids in mapping file:\n"
            + "\n".join(missing_ids)
        )

    df["sex"] = df["sex"].astype("string").str.upper().str.strip()
    df["age_group"] = df["age_group"].astype("string").str.upper().str.strip()
    df["cat"] = df["sex"] + df["age_group"]
    return df

def load_county_candidates(county_map_path: Optional[str]) -> Dict[str, List[str]]:
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
            print("WARNING: county map missing expected columns {old_county,new_county_2024}; skipping modernization.")
    else:
        print("NOTE: county map not found; geocoding will use the old `area` values.")
    return county_candidates

def add_fg_dialect_region(df: pd.DataFrame, county_map_path: str) -> pd.DataFrame:
    """
    Fill df['fg_dialect_region'] by looking up named_dialect from the municipality mapping CSV.
    Primary key: place name (after ndc_transform is applied).
    Fallback: (place, county) disambiguation via old/new county columns.
    """
    import pandas as pd

    if not county_map_path or not os.path.exists(county_map_path):
        print("WARNING: county map not provided; cannot compute fg_dialect_region.")
        df["fg_dialect_region"] = pd.NA
        return df

    cm = pd.read_csv(county_map_path)

    # --- First pass: join on municipality name (old_muni and new_muni)
    join_frames = []
    if "old_muni" in cm.columns:
        join_frames.append(cm[["old_muni", "named_dialect"]].rename(columns={"old_muni": "muni_name"}))
    if "new_muni" in cm.columns:
        join_frames.append(cm[["new_muni", "named_dialect"]].rename(columns={"new_muni": "muni_name"}))

    if not join_frames:
        print("WARNING: Mapping CSV missing old_muni/new_muni; fg_dialect_region will be empty.")
        df["fg_dialect_region"] = pd.NA
        return df

    muni_named = pd.concat(join_frames, ignore_index=True).dropna().drop_duplicates()
    muni_named["muni_name"] = muni_named["muni_name"].astype(str).str.strip()
    muni_named["named_dialect"] = muni_named["named_dialect"].astype(str).str.strip()

    out = df.copy()
    out["place_norm"] = out["place"].astype(str).str.strip()

    out = out.merge(muni_named, left_on="place_norm", right_on="muni_name", how="left")
    out = out.rename(columns={"named_dialect": "fg_dialect_region"}).drop(columns=["muni_name"])

    # --- Second pass (fallback): disambiguate by county if still missing
    missing_mask = out["fg_dialect_region"].isna()
    if missing_mask.any():
        county_frames = []
        if {"old_muni", "old_county"}.issubset(cm.columns):
            county_frames.append(
                cm[["old_muni", "old_county", "named_dialect"]]
                  .rename(columns={"old_muni": "muni_name", "old_county": "county"})
            )
        if {"new_muni", "new_county_2024"}.issubset(cm.columns):
            county_frames.append(
                cm[["new_muni", "new_county_2024", "named_dialect"]]
                  .rename(columns={"new_muni": "muni_name", "new_county_2024": "county"})
            )

        if county_frames:
            muni_county_named = pd.concat(county_frames, ignore_index=True).dropna().drop_duplicates()
            muni_county_named["muni_name"] = muni_county_named["muni_name"].astype(str).str.strip()
            muni_county_named["county"] = muni_county_named["county"].astype(str).str.strip()
            muni_county_named["named_dialect"] = muni_county_named["named_dialect"].astype(str).str.strip()

            tmp = (
                out.loc[missing_mask, ["speaker_id", "place_norm", "area"]]
                   .merge(muni_county_named, left_on=["place_norm", "area"],
                          right_on=["muni_name", "county"], how="left")
            )
            tmp = tmp[["speaker_id", "named_dialect"]].rename(columns={"named_dialect": "fg_dialect_region"})
            out = out.merge(tmp, on="speaker_id", how="left", suffixes=("", "_from_county"))
            out["fg_dialect_region"] = out["fg_dialect_region"].fillna(out["fg_dialect_region_from_county"])
            out = out.drop(columns=[c for c in out.columns if c.endswith("_from_county")])

    # Report unresolved cases (if any)
    unresolved = out["fg_dialect_region"].isna()
    if unresolved.any():
        examples = out.loc[unresolved, ["speaker_id", "place", "area"]].drop_duplicates()
        print(f"WARNING: fg_dialect_region not found for {len(examples)} speakers. Examples:\n"
              + examples.head(10).to_string(index=False))

    return out.drop(columns=["place_norm"])

# -------------------------- COORDINATES --------------------------

def build_place_coords(df: pd.DataFrame, county_candidates: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Geocode each unique (place, area) to (latitude, longitude).
    Try mapped 2024 counties first, then old county, then place-only.          
    """
    unique_places = df[["place", "area"]].drop_duplicates().reset_index(drop=True)
    mapper = CoordMapper(user_agent="ndc_fold_builder", timeout=10)           
    place_lat, place_lon = [], []

    for r in unique_places.itertuples(index=False):
        old_county = r.area if pd.notna(r.area) else None
        candidates = county_candidates.get(old_county, []) if old_county else []
        coords = None

        # 1) Any mapped 2024 counties
        for new24 in candidates:
            coords = mapper.get_coordinates(r.place, new24, country="Norway")
            if coords is not None:
                break
        # 2) Fall back to old county
        if coords is None and old_county:
            coords = mapper.get_coordinates(r.place, old_county, country="Norway")
        # 3) Final fallback: place only
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
    return unique_places

def attach_coords(d: pd.DataFrame, unique_places: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        d["latitude"] = []
        d["longitude"] = []
        return d
    out = d.merge(unique_places, on=["place", "area"], how="left")
    return out[
        ["speaker_id", "place", "sex", "age_group", "dialect_region", "fg_dialect_region", "latitude", "longitude"]
    ]

# -------------------------- BALANCING --------------------------

def balance_score(counts, fold: int, sex: str, age_group: str) -> float:
    """
    Sum of absolute deviations from current averages across folds for sex and age.
    Lower is better.
    """
    score = 0.0
    for s in ["M", "F"]:
        total = sum(counts[f][s] for f in range(4)) + (1 if sex == s else 0)
        avg = total / 4.0
        for f in range(4):
            val = counts[f][s] + (1 if (f == fold and sex == s) else 0)
            score += abs(val - avg)
    for a in ["A", "B"]:
        total = sum(counts[f][a] for f in range(4)) + (1 if age_group == a else 0)
        avg = total / 4.0
        for f in range(4):
            val = counts[f][a] + (1 if (f == fold and age_group == a) else 0)
            score += abs(val - avg)
    return score

# -------------------------- ASSIGNMENT --------------------------

def assign_place_to_folds(
    g: pd.DataFrame,
    counts: Dict[int, Dict[str, int]],
    place_load: Dict[int, Dict[str, int]],
    rotation_offset: int,
    place_index: int,
    rng: random.Random,
    spread_penalty: float = 0.25,  # small penalty to spread extras across folds per place
):
    """
    Assign one speaker per fold (location coverage), then place extras.
    Returns: assigned_rows_by_fold (list incl. extras),  lt4_df (if len<4 else empty)
    """
    if len(g) < 4:
        return {i: [] for i in range(4)}, g  # nothing assigned; send all to exceptions_lt4

    g = g.sort_values("speaker_id").reset_index(drop=True)

    # --- 1) Location coverage: try to hit MA, FA, MB, FB across folds with rotation
    cat_to_idxs = {c: g.index[g["cat"] == c].tolist() for c in BASE_CATS}
    chosen = {c: rng.choice(idxs) for c, idxs in cat_to_idxs.items() if len(idxs) > 0}

    r = (rotation_offset + place_index) % 4
    fold_to_expected_cat = {f: BASE_CATS[(r + f) % 4] for f in range(4)}

    used = set()
    assigned = {}  # fold -> idx

    # First pass: expected category available?
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
                place_load[f][row["place"]] = place_load[f].get(row["place"], 0) + 1

    # Second pass: fill remaining coverage folds by minimizing global imbalance
    for f in range(4):
        if f in assigned:
            continue
        remaining_idxs = [i for i in g.index if i not in used]
        if not remaining_idxs:
            break
        expected_cat = fold_to_expected_cat[f]
        preferred = [i for i in remaining_idxs if g.loc[i, "cat"] == expected_cat]
        pool = preferred if preferred else remaining_idxs

        scores = []
        for idx in pool:
            sex = g.loc[idx, "sex"]
            ag = g.loc[idx, "age_group"]
            s = balance_score(counts, f, sex, ag)
            scores.append((idx, s))
        min_s = min(s for _, s in scores)
        best_idxs = [idx for idx, s in scores if s == min_s]
        best_idx = rng.choice(best_idxs)

        assigned[f] = best_idx
        used.add(best_idx)
        row = g.loc[best_idx]
        counts[f][row["sex"]] += 1
        counts[f][row["age_group"]] += 1
        place_load[f][row["place"]] = place_load[f].get(row["place"], 0) + 1

    # --- 2) Assign EXTRAS (any rows not yet used) into folds
    assigned_rows_by_fold = {i: [] for i in range(4)}
    # First, push the 4 coverage picks
    for f, idx in assigned.items():
        assigned_rows_by_fold[f].append(
            g.loc[idx, ["speaker_id", "place", "area", "sex", "age_group", "dialect_region", "fg_dialect_region"]]
        )

    # Then place extras
    extras = [i for i in g.index if i not in used]
    # Optional: randomize the order in which extras are placed, for variability
    rng.shuffle(extras)

    for idx in extras:
        # Choose the fold that minimizes the global balance objective, with a soft spread penalty
        fold_scores: List[Tuple[int, float]] = []
        for f in range(4):
            sex = g.loc[idx, "sex"]
            ag = g.loc[idx, "age_group"]
            base = balance_score(counts, f, sex, ag)
            # soft penalty: more speakers from this place already in this fold -> slightly worse
            penalty = spread_penalty * place_load[f].get(g.loc[idx, "place"], 0)
            fold_scores.append((f, base + penalty))

        min_s = min(s for _, s in fold_scores)
        best_folds = [f for f, s in fold_scores if s == min_s]
        target_fold = rng.choice(best_folds)

        # Assign
        assigned_rows_by_fold[target_fold].append(
            g.loc[idx, ["speaker_id", "place", "area", "sex", "age_group", "dialect_region", "fg_dialect_region"]]
        )
        # Update counts & load
        counts[target_fold][g.loc[idx, "sex"]] += 1
        counts[target_fold][g.loc[idx, "age_group"]] += 1
        place_load[target_fold][g.loc[idx, "place"]] = place_load[target_fold].get(g.loc[idx, "place"], 0) + 1

    return assigned_rows_by_fold, pd.DataFrame()

# -------------------------- MAIN --------------------------

def main(args):
    rng = random.Random(args.seed)

    df = load_and_merge(args.metadata, args.mapping, args.transform_places)
    os.makedirs(args.outdir, exist_ok=True)
    df = add_fg_dialect_region(df, args.county_map)
    # County modernization map for coordinates
    county_candidates = load_county_candidates(args.county_map)

    # Build coordinates once per (place, area)
    unique_places = build_place_coords(df, county_candidates)

    # Group by place
    groups = list(df.groupby("place"))
    # Baseline deterministic sort
    groups.sort(key=lambda kv: kv[0])
    if args.shuffle_places:
        rng.shuffle(groups)

    # Prepare fold containers and running counts
    folds = {i: [] for i in range(4)}
    counts = {i: {"M": 0, "F": 0, "A": 0, "B": 0} for i in range(4)}
    place_load = {i: {} for i in range(4)}  # count of speakers per place in each fold

    exceptions_lt4 = []

    for place_idx, (place, g) in enumerate(groups):
        assigned_dict, lt4_df = assign_place_to_folds(
            g=g,
            counts=counts,
            place_load=place_load,
            rotation_offset=args.rotation_offset,
            place_index=place_idx,
            rng=rng,
            spread_penalty=args.spread_extras_penalty,
        )

        if len(lt4_df) > 0:
            exceptions_lt4.append(lt4_df)
            continue

        # Accumulate all rows (coverage + extras) into folds
        for f in range(4):
            if assigned_dict[f]:
                folds[f].extend(assigned_dict[f])

    # Build DataFrames
    fold_dfs = {
        i: (pd.DataFrame(folds[i]).reset_index(drop=True) if len(folds[i]) else
            pd.DataFrame(columns=["speaker_id", "place", "area", "sex", "age_group", "dialect_region", "fg_dialect_region"]))
        for i in range(4)
    }
    exc_lt4_df = pd.concat(exceptions_lt4, ignore_index=True) if exceptions_lt4 else pd.DataFrame(
        columns=["speaker_id", "place", "area", "sex", "age_group", "dialect_region", "fg_dialect_region"]
    )

    # Attach coordinates and drop `area` from the final schema
    fold_dfs = {i: attach_coords(d, unique_places) for i, d in fold_dfs.items()}
    exc_lt4_df = attach_coords(exc_lt4_df, unique_places)
    if len(exc_lt4_df):
        exc_lt4_df["exception_reason"] = "place_has_<4_speakers"

    # Save folds
    fold_paths = []
    for i in range(4):
        path = os.path.join(args.outdir, f"fold{i+1}.csv")
        fold_dfs[i].to_csv(path, index=False)
        fold_paths.append(path)

    # Save exceptions
    exc_lt4_df.to_csv(os.path.join(args.outdir, "exceptions_lt4.csv"), index=False)

    # QC summary
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
    print("\nQC summary:\n", summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="/home/idatro/preprocessed_NorDia/NDC_norge_metadata.tsv",
                        help="Path to NDC metadata TSV (tab-separated).")
    parser.add_argument("--mapping", type=str, default="/home/idatro/dialect_project/ndc_speaker_outputs/ndc_cardinal4_by_muni.csv",
                        help="Path to speaker_id -> dialect_region CSV.")
    parser.add_argument("--county-map", type=str,
                        default="/home/idatro/dialect_project/repo/DialectMapper/dialect_mapper/mapping_data/muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv",
                        help="CSV with old_county -> new_county_2024 mapping.")
    parser.add_argument("--transform-places", type=str, default="/home/idatro/dialect_project/repo/DialectMapper/dialect_mapper/mapping_data/ndc_transform.csv",
                        help="Optional CSV mapping legacy place names to modern municipalities (ndc_transform.csv).")
    parser.add_argument("--outdir", type=str, default="./folds_out",
                        help="Output directory for CSVs.")
    parser.add_argument("--rotation-offset", type=int, default=0,
                        help="Rotation offset for distributing sex/age categories across folds.")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Seed for reproducible randomness.")
    parser.add_argument("--shuffle-places", action="store_true",
                        help="Shuffle the order of places before assignment.")
    parser.add_argument("--spread-extras-penalty", type=float, default=0.25,
                        help="Soft penalty added per existing speaker from the same place in a fold when placing extras. "
                             "Use 0.0 to disable spreading preference.")
    args = parser.parse_args()
    main(args)