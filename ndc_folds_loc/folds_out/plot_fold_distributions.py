from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: statistical test
try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = Path(".")  # folder containing csv files
FOLD_PATTERN = "fold*.csv"  # e.g., fold1.csv ... fold4.csv
EXCEPTIONS_FILE = "exceptions_lt4.csv"

FEATURES = ["sex", "age_group"]  # characteristics to analyze
ID_COL = "speaker_id"

# Plot style
sns.set_context("talk")
sns.set_style("whitegrid")


# ---------------------------
# LOADING
# ---------------------------
def load_split_csv(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["split"] = split_name
    return df

def load_all_splits(data_dir: Path) -> pd.DataFrame:
    fold_paths = sorted(data_dir.glob(FOLD_PATTERN))
    if not fold_paths:
        raise FileNotFoundError(f"No files matching pattern {FOLD_PATTERN} in {data_dir.resolve()}")

    dfs = []
    for p in fold_paths:
        # split name from filename, e.g. fold1.csv -> fold1
        split_name = p.stem
        dfs.append(load_split_csv(p, split_name))

    exc_path = data_dir / EXCEPTIONS_FILE
#    if exc_path.exists():
#        dfs.append(load_split_csv(exc_path, "exceptions"))
#    else:
#        print(f"[WARN] Exceptions file not found: {exc_path}. Proceeding without exceptions.")

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


# ---------------------------
# SANITY CHECKS
# ---------------------------
def check_speaker_uniqueness(df: pd.DataFrame):
    # Each speaker_id should appear in only one split.
    # If a speaker appears multiple times within the same split (e.g. duplicate rows), that's also flagged.
    counts = df.groupby(ID_COL)["split"].nunique()
    bad = counts[counts > 1]
    if len(bad) > 0:
        example = bad.index[:10].tolist()
#        raise ValueError(
#            f"Found speakers appearing in multiple splits (should be unique). "
#            f"Count={len(bad)}. Examples: {example}"
#        )

    # Optional: duplicates inside split
    dup = df.duplicated(subset=[ID_COL, "split"]).sum()
    if dup > 0:
        print(f"[WARN] Found {dup} duplicated (speaker_id, split) rows.")


# ---------------------------
# DISTRIBUTIONS
# ---------------------------
def compute_distribution(df: pd.DataFrame, feature: str):
    """
    Returns:
      counts: pivot table [split x category]
      props:  pivot table [split x category] normalized row-wise
      total_prop: Series [category] for full dataset
    """
    # Clean missing as explicit category if you want:
    # df[feature] = df[feature].fillna("MISSING")

    # Counts per split
    counts = (
        df.pivot_table(index="split", columns=feature, values=ID_COL, aggfunc="count", fill_value=0)
        .astype(int)
    )

    # Total counts across all splits
    total_counts = df[feature].value_counts(dropna=False).sort_index()
    total_prop = total_counts / total_counts.sum()

    # Ensure columns are aligned (same categories in each split)
    counts = counts.reindex(columns=total_prop.index, fill_value=0)

    # Proportions per split
    props = counts.div(counts.sum(axis=1), axis=0)

    return counts, props, total_prop


# ---------------------------
# PLOTTING
# ---------------------------
def plot_feature(props: pd.DataFrame, total_prop: pd.Series, feature: str, out_dir: Path):
    """
    Plot bars = per-split proportions, markers = total proportions.
    """
    splits = props.index.tolist()
    categories = props.columns.tolist()

    # Long-form for seaborn bars
    plot_df = (
        props.reset_index()
        .melt(id_vars="split", var_name="category", value_name="proportion")
    )
    plot_df["feature"] = feature

    # Total points
    total_df = pd.DataFrame({
        "category": categories,
        "total_proportion": [total_prop.get(c, 0.0) for c in categories]
    })

    # Figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars per split
    sns.barplot(
        data=plot_df,
        x="category",
        y="proportion",
        hue="split",
        ax=ax
    )

    # Overlay total distribution as black diamonds
    # We place them at the center of each category group.
    x_positions = np.arange(len(categories))
    ax.scatter(
        x_positions,
        total_df["total_proportion"].values,
        color="black",
        marker="D",
        s=60,
        label="TOTAL"
    )

    ax.set_title(f"Distribution of {feature} by split vs TOTAL")
    ax.set_xlabel(feature)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 0.6)

    # Improve legend: include total marker
    handles, labels = ax.get_legend_handles_labels()
    # Add total handle manually if not present
    if "TOTAL" not in labels:
        handles.append(plt.Line2D([0], [0], marker='D', color='black', linestyle='None', markersize=8))
        labels.append("TOTAL")
    ax.legend(handles, labels, title="Split", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"distribution_{feature}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved plot: {out_path}")


def print_deviation_table(props: pd.DataFrame, total_prop: pd.Series, feature: str):
    """
    Print absolute deviation from total per split/category.
    """
    # Align
    total = total_prop.reindex(props.columns).fillna(0)
    deviation = (props - total).abs()
    deviation.index.name = "split"
    print(f"\nAbsolute deviation from TOTAL for feature='{feature}':")
    print(deviation.round(4))
    print("\nMean abs deviation per split:")
    print(deviation.mean(axis=1).round(4))


def chi_square_against_total(counts: pd.DataFrame, feature: str):
    """
    Chi-square test of independence across splits for this feature.
    This tests whether distributions differ across splits (global test).
    """
    if not SCIPY_AVAILABLE:
        print("[INFO] scipy not installed; skipping chi-square test.")
        return

    # counts is split x category
    table = counts.values
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"\nChi-square test for feature='{feature}': chi2={chi2:.3f}, dof={dof}, p={p:.4g}")
    if p < 0.05:
        print("  -> Distributions differ significantly across splits (p < 0.05).")
    else:
        print("  -> No significant evidence of differing distributions across splits (p >= 0.05).")


# ---------------------------
# MAIN
# ---------------------------
def main():
    df = load_all_splits(DATA_DIR)

    # Keep only relevant columns (optional)
    needed = [ID_COL, "split"] + FEATURES
    df = df[[c for c in needed if c in df.columns]].copy()

    # Sanity check speaker uniqueness across splits
    check_speaker_uniqueness(df)

    out_dir = Path("plots")

    for feature in FEATURES:
        if feature not in df.columns:
            print(f"[WARN] Missing column '{feature}', skipping.")
            continue

        counts, props, total_prop = compute_distribution(df, feature)

        # Plot
        plot_feature(props, total_prop, feature, out_dir)

        # Diagnostics (optional but useful)
        #print_deviation_table(props, total_prop, feature)
        #chi_square_against_total(counts, feature)

    print("\nDone.")


if __name__ == "__main__":
    main()