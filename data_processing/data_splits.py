#!/usr/bin/env python3
"""
Attach `dialect_region` (cardinal_four) to metadata by mapping county -> dialect region.

- Uses only county names from metadata (`area`) and the county fields in the mapping CSV
  (`old_county`, `new_county`, `new_county_2024`).
- Robust to historic and 2024 county renames.
"""

import pandas as pd
from typing import List

# ---------- Config ----------
META_PATH = "/home/idatro/preprocessed_NorDia/NDC_norge_metadata.tsv"
MAP_PATH  = "/home/idatro/dialect_project/repo/DialectMapper/dialect_mapper/mapping_data/muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv"  # <-- update if needed
OUT_PATH  = "/home/idatro/preprocessed_NorDia/NDC_norge_metadata_with_dialect_region.tsv"  # optional output


# ---------- Helpers ----------
def norm_series(s: pd.Series) -> pd.Series:
    """Normalize text for matching: strip, collapse whitespace, normalize hyphen, case-fold."""
    return (
        s.fillna("")
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace("\\-", "-", regex=False)
         .str.casefold()
    )

def build_county_to_region(mapdf: pd.DataFrame,
                           county_cols: List[str],
                           value_col: str = "cardinal_four") -> pd.DataFrame:
    """
    Create a tidy county -> dialect_region table by melting county columns.
    Ensures one row per normalized county.
    """
    # Keep only columns we need and drop duplicates to reduce fan-out
    cols = [c for c in county_cols + [value_col] if c in mapdf.columns]
    m = mapdf[cols].copy().drop_duplicates()

    # Melt county columns into a single "county" column
    tidy = m.melt(value_vars=[c for c in county_cols if c in m.columns],
                  value_name="county", var_name="source_col").dropna(subset=["county"])
    tidy["county_norm"] = norm_series(tidy["county"])
    tidy[value_col] = tidy[value_col].astype(str)

    # Deduplicate per county_norm
    # If multiple rows exist per county_norm, we keep the first non-null value.
    tidy = tidy.sort_values(["county_norm", "source_col"]).drop_duplicates(subset=["county_norm"])

    # Final mapping: one row per county_norm
    county_map = tidy[["county_norm", value_col]].drop_duplicates()
    return county_map


# ---------- Main ----------
def main():
    # Load metadata (TSV) and mapping CSV
    meta = pd.read_csv(META_PATH, sep="\t", dtype=str)

    mapdf = pd.read_csv(MAP_PATH, dtype=str)

    print("\n[DEBUG] Mapping file path:", MAP_PATH)
    print("[DEBUG] Mapping shape:", mapdf.shape)

    # Basic view
    print("[DEBUG] MAP columns (raw):", list(mapdf.columns))

    # Reveal hidden characters (helps explain KeyError on 'cardinal_four')
    print("[DEBUG] MAP columns (repr):", [repr(c) for c in mapdf.columns])

    # Quick filter to see any columns containing 'cardinal'
    print("[DEBUG] Columns containing 'cardinal':",
        [c for c in mapdf.columns if "cardinal" in c.casefold()])

    # Optional: first rows
    print("[DEBUG] MAP head():")
    print(mapdf.head(3))


    # Normalize county name in metadata ("area" = county)
    if "area" not in meta.columns:
        raise KeyError("Metadata missing 'area' column (county).")
    meta["area_norm"] = norm_series(meta["area"])  # area ~ county  [1](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/data_splits.py)

    # Build county -> dialect_region lookup from mapping CSV
    county_cols = ["old_county", "new_county", "new_county_2024"]   # county fields in mapper  [2](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/NDC_norge_metadata.tsv)
    county_map = build_county_to_region(mapdf, county_cols, value_col="cardinal_four")  # target dialect region  [2](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/NDC_norge_metadata.tsv)

    # Join (left) by normalized county
    out = meta.merge(county_map, how="left", left_on="area_norm", right_on="county_norm")
    out = out.rename(columns={"cardinal_four": "dialect_region"}).drop(columns=["county_norm"])

    # Diagnostics: unmatched counties
    unmatched_mask = out["dialect_region"].isna()
    unmatched_cnt = unmatched_mask.sum()
    print(f"[County mapping] matched: {len(out) - unmatched_cnt}, unmatched: {unmatched_cnt}")

    if unmatched_cnt:
        # Show DISTINCT county names that failed to map, to help patch mapping or metadata
        bad_counties = (out.loc[unmatched_mask, ["area"]]
                          .assign(area_norm=out.loc[unmatched_mask, "area_norm"])
                          .drop_duplicates()
                          .sort_values("area"))
        print("\nUnmatched counties in metadata (distinct):")
        print(bad_counties.to_string(index=False))

    # Optional: save a new TSV with `dialect_region` attached
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"\nWrote enriched metadata: {OUT_PATH}")

if __name__ == "__main__":
    main()