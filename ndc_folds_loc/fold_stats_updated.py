from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# USER SETTINGS
# =========================================================

# Define your labeled pickle files here:
PICKLE_FILES = [
    ("fold_1", "/home/idatro/dialect_project/ndc_folds_loc/fold_pkls/fold1.pkl"),
    ("fold_2", "/home/idatro/dialect_project/ndc_folds_loc/fold_pkls/fold2.pkl"),
    ("fold_3", "/home/idatro/dialect_project/ndc_folds_loc/fold_pkls/fold3.pkl"),
    ("fold_4", "/home/idatro/dialect_project/ndc_folds_loc/fold_pkls/fold4.pkl"),
    ("exceptions", "/home/idatro/dialect_project/ndc_folds_loc/fold_pkls/exceptions_lt4.pkl"),
]

# Output folder for CSV summaries
OUTPUT_DIR = Path("fold_stats_csv")

# CSV separator:
# Use "," for standard CSV.
# If Excel in your locale prefers semicolon-separated files, change to ";"
CSV_SEP = ","

# Text to use for missing values in categorical variables
MISSING_LABEL = "<MISSING>"


# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "speaker_id",
    "segment_id",
    "full_audio_file_path",
    "duration_sec",
    "offset_start_ms",
    "offset_end_ms",
    "place",
    "sex",
    "age_group",
    "dialect_region",
    "fg_dialect_region",
    "latitude",
    "longitude",
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def validate_columns(df: pd.DataFrame, label: str):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{label}] Missing required columns: {missing}"
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure duration is numeric
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")

    # Standardize missing values in categorical columns
    for col in ["sex", "age_group", "dialect_region", "fg_dialect_region", "speaker_id"]:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), MISSING_LABEL)
            df[col] = df[col].astype(str)

    return df


def duration_summary_row(df: pd.DataFrame, label: str, source_path: str) -> dict:
    dur = df["duration_sec"].dropna()

    if len(dur) == 0:
        return {
            "label": label,
            "source_file": source_path,
            "n_segments": len(df),
            "n_speakers": df["speaker_id"].nunique(dropna=True),
            "missing_duration_segments": int(df["duration_sec"].isna().sum()),
            "total_hours": 0.0,
            "mean_duration_sec": np.nan,
            "median_duration_sec": np.nan,
            "std_duration_sec": np.nan,
            "min_duration_sec": np.nan,
            "p10_duration_sec": np.nan,
            "p25_duration_sec": np.nan,
            "p75_duration_sec": np.nan,
            "p90_duration_sec": np.nan,
            "max_duration_sec": np.nan,
            "iqr_duration_sec": np.nan,
        }

    q10 = dur.quantile(0.10)
    q25 = dur.quantile(0.25)
    q75 = dur.quantile(0.75)
    q90 = dur.quantile(0.90)

    return {
        "label": label,
        "source_file": source_path,
        "n_segments": len(df),
        "n_speakers": df["speaker_id"].nunique(dropna=True),
        "missing_duration_segments": int(df["duration_sec"].isna().sum()),
        "total_hours": dur.sum() / 3600.0,
        "mean_duration_sec": dur.mean(),
        "median_duration_sec": dur.median(),
        "std_duration_sec": dur.std(),
        "min_duration_sec": dur.min(),
        "p10_duration_sec": q10,
        "p25_duration_sec": q25,
        "p75_duration_sec": q75,
        "p90_duration_sec": q90,
        "max_duration_sec": dur.max(),
        "iqr_duration_sec": q75 - q25,
    }


def grouped_balance_stats(df: pd.DataFrame, label: str, group_col: str) -> pd.DataFrame:
    """
    Returns stats for one grouping column, e.g. sex or age_group:
    - segment counts and percentages
    - unique speaker counts and percentages
    - hours and percentage of hours
    - mean / median segment duration
    """
    work = df.copy()
    work[group_col] = work[group_col].fillna(MISSING_LABEL).astype(str)

    total_segments = len(work)
    total_speakers = work["speaker_id"].nunique(dropna=True)
    total_hours = work["duration_sec"].dropna().sum() / 3600.0

    grouped = (
        work.groupby(group_col, dropna=False)
        .agg(
            n_segments=("segment_id", "count"),
            n_speakers=("speaker_id", pd.Series.nunique),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            median_duration_sec=("duration_sec", "median"),
            std_duration_sec=("duration_sec", "std"),
            min_duration_sec=("duration_sec", "min"),
            max_duration_sec=("duration_sec", "max"),
        )
        .reset_index()
        .rename(columns={group_col: "group_value"})
    )

    grouped["label"] = label
    grouped["group_column"] = group_col
    grouped["total_hours"] = grouped["total_duration_sec"] / 3600.0
    grouped["segment_pct"] = grouped["n_segments"] / total_segments * 100 if total_segments else np.nan
    grouped["speaker_pct"] = grouped["n_speakers"] / total_speakers * 100 if total_speakers else np.nan
    grouped["hours_pct"] = grouped["total_hours"] / total_hours * 100 if total_hours else np.nan

    grouped = grouped[
        [
            "label",
            "group_column",
            "group_value",
            "n_segments",
            "segment_pct",
            "n_speakers",
            "speaker_pct",
            "total_hours",
            "hours_pct",
            "mean_duration_sec",
            "median_duration_sec",
            "std_duration_sec",
            "min_duration_sec",
            "max_duration_sec",
        ]
    ]

    return grouped.sort_values(["label", "group_value"]).reset_index(drop=True)


def grouped_pair_stats(df: pd.DataFrame, label: str, col1: str, col2: str) -> pd.DataFrame:
    """
    Cross-tab style summary, e.g. sex x age_group.
    Useful for spotting interaction imbalances.
    """
    work = df.copy()
    work[col1] = work[col1].fillna(MISSING_LABEL).astype(str)
    work[col2] = work[col2].fillna(MISSING_LABEL).astype(str)

    total_segments = len(work)
    total_hours = work["duration_sec"].dropna().sum() / 3600.0

    grouped = (
        work.groupby([col1, col2], dropna=False)
        .agg(
            n_segments=("segment_id", "count"),
            n_speakers=("speaker_id", pd.Series.nunique),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            median_duration_sec=("duration_sec", "median"),
        )
        .reset_index()
        .rename(columns={col1: "group_1", col2: "group_2"})
    )

    grouped["label"] = label
    grouped["group_1_column"] = col1
    grouped["group_2_column"] = col2
    grouped["total_hours"] = grouped["total_duration_sec"] / 3600.0
    grouped["segment_pct"] = grouped["n_segments"] / total_segments * 100 if total_segments else np.nan
    grouped["hours_pct"] = grouped["total_hours"] / total_hours * 100 if total_hours else np.nan

    grouped = grouped[
        [
            "label",
            "group_1_column",
            "group_1",
            "group_2_column",
            "group_2",
            "n_segments",
            "segment_pct",
            "n_speakers",
            "total_hours",
            "hours_pct",
            "mean_duration_sec",
            "median_duration_sec",
        ]
    ]

    return grouped.sort_values(["label", "group_1", "group_2"]).reset_index(drop=True)


def dialect_stats(df: pd.DataFrame, label: str, region_col: str) -> pd.DataFrame:
    """
    Summary for dialect region columns:
    - counts
    - % of segments
    - unique speakers
    - hours
    - % of hours
    """
    work = df.copy()
    work[region_col] = work[region_col].fillna(MISSING_LABEL).astype(str)

    total_segments = len(work)
    total_speakers = work["speaker_id"].nunique(dropna=True)
    total_hours = work["duration_sec"].dropna().sum() / 3600.0

    grouped = (
        work.groupby(region_col, dropna=False)
        .agg(
            n_segments=("segment_id", "count"),
            n_speakers=("speaker_id", pd.Series.nunique),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            median_duration_sec=("duration_sec", "median"),
        )
        .reset_index()
        .rename(columns={region_col: "region_value"})
    )

    grouped["label"] = label
    grouped["region_column"] = region_col
    grouped["total_hours"] = grouped["total_duration_sec"] / 3600.0
    grouped["segment_pct"] = grouped["n_segments"] / total_segments * 100 if total_segments else np.nan
    grouped["speaker_pct"] = grouped["n_speakers"] / total_speakers * 100 if total_speakers else np.nan
    grouped["hours_pct"] = grouped["total_hours"] / total_hours * 100 if total_hours else np.nan

    grouped = grouped[
        [
            "label",
            "region_column",
            "region_value",
            "n_segments",
            "segment_pct",
            "n_speakers",
            "speaker_pct",
            "total_hours",
            "hours_pct",
            "mean_duration_sec",
            "median_duration_sec",
        ]
    ]

    return grouped.sort_values(["label", "region_value"]).reset_index(drop=True)


def save_csv(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep=CSV_SEP, encoding="utf-8-sig")


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    sex_rows = []
    age_rows = []
    sex_age_rows = []
    dialect_rows = []
    four_dialect_rows = []
    fg_dialect_rows = []

    for label, pickle_path in PICKLE_FILES:
        pickle_path = str(pickle_path)
        print(f"Reading [{label}] from {pickle_path}")

        df = pd.read_pickle(pickle_path)
        validate_columns(df, label)
        df = prepare_dataframe(df)

        # Overall duration / dataset summary
        overall_rows.append(duration_summary_row(df, label, pickle_path))

        # Sex summary
        sex_rows.append(grouped_balance_stats(df, label, "sex"))

        # Age summary
        age_rows.append(grouped_balance_stats(df, label, "age_group"))

        # Sex x age-group cross summary
        sex_age_rows.append(grouped_pair_stats(df, label, "sex", "age_group"))

        # Dialect-region summaries
        dialect_region_stats = dialect_stats(df, label, "dialect_region")
        dialect_rows.append(dialect_region_stats)
        four_dialect_rows.append(dialect_region_stats.copy())
        fg_dialect_rows.append(dialect_stats(df, label, "fg_dialect_region"))

    # Combine all outputs
    overall_df = pd.DataFrame(overall_rows).sort_values("label").reset_index(drop=True)
    sex_df = pd.concat(sex_rows, ignore_index=True) if sex_rows else pd.DataFrame()
    age_df = pd.concat(age_rows, ignore_index=True) if age_rows else pd.DataFrame()
    sex_age_df = pd.concat(sex_age_rows, ignore_index=True) if sex_age_rows else pd.DataFrame()
    dialect_df = pd.concat(dialect_rows, ignore_index=True) if dialect_rows else pd.DataFrame()
    four_dialect_df = pd.concat(four_dialect_rows, ignore_index=True) if four_dialect_rows else pd.DataFrame()
    fg_dialect_df = pd.concat(fg_dialect_rows, ignore_index=True) if fg_dialect_rows else pd.DataFrame()

    # Dedicated four-region summary (same schema as fg_dialect_region_summary.csv,
    # but region_value comes from dialect_region)

    # Optional combined region summary
    combined_region_df = pd.concat([dialect_df, fg_dialect_df], ignore_index=True)

    # Save files
    save_csv(overall_df, OUTPUT_DIR / "overall_fold_summary.csv")
    save_csv(sex_df, OUTPUT_DIR / "sex_summary.csv")
    save_csv(age_df, OUTPUT_DIR / "age_group_summary.csv")
    save_csv(sex_age_df, OUTPUT_DIR / "sex_by_age_group_summary.csv")
    save_csv(dialect_df, OUTPUT_DIR / "dialect_region_summary.csv")
    save_csv(four_dialect_df, OUTPUT_DIR / "four_dialect_region_summary.csv")
    save_csv(fg_dialect_df, OUTPUT_DIR / "fg_dialect_region_summary.csv")
    save_csv(combined_region_df, OUTPUT_DIR / "combined_region_summary.csv")

    print("\nDone. Wrote the following files:")
    print(f"  - {OUTPUT_DIR / 'overall_fold_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'sex_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'age_group_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'sex_by_age_group_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'dialect_region_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'four_dialect_region_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'fg_dialect_region_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'combined_region_summary.csv'}")


if __name__ == "__main__":
    main()