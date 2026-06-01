import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ========================
# Configurable paths
# ========================
# Update these paths to match your local setup.
FINE_GRAINED_CSV = "/home/idatro/dialect_project/ndc_folds_loc/fold_stats_csv/fg_dialect_region_summary.csv"
FOUR_REGION_CSV = "/home/idatro/dialect_project/ndc_folds_loc/fold_stats_csv/four_dialect_region_summary.csv"
OUTPUT_PDF = "dialect_region_heatmaps_4_and_11.pdf"

# ========================
# Configurable font sizes
# ========================
ANNOT_FONT_SIZE = 22
TICK_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 20
TITLE_FONT_SIZE = 22
SUPTITLE_FONT_SIZE = 24
COLORBAR_FONT_SIZE = 18

# ========================
# Figure/layout settings
# ========================
FIGSIZE = (22, 10)
CMAP = "Blues"
DPI = 300
ANNOTATE = True
VALUE_COLUMN = "total_hours"  # Change to e.g. "hours_pct" if needed

FOUR_REGION_ORDER = ["north", "mid", "east", "west"]
ELEVEN_REGION_ORDER = [
    "Troms-Finnmarks-mål",
    "Nordlandsk",
    "Helgelandsk",
    "Namdalsk",
    "Østtrøndsk",
    "Uttrøndersk",
    "Nordvestlandsk",
    "Østlandsk",
    "Midlandsk",
    "Sørvestlandsk",
    "Sørlandsk",
]


def fold_sort_key(label: str):
    """Sort fold labels numerically (fold_1, fold_2, ...)."""
    match = re.match(r"fold_(\d+)$", str(label))
    if match:
        return (0, int(match.group(1)))
    return (1, str(label))



def build_pivot(csv_path: str, region_order: list[str], value_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["label"].astype(str).str.startswith("fold_")].copy()

    fold_order = sorted(df["label"].unique(), key=fold_sort_key)
    pivot = df.pivot(index="region_value", columns="label", values=value_column)
    pivot = pivot.reindex(index=region_order, columns=fold_order)
    return pivot



def heatmap_on_axis(ax, pivot: pd.DataFrame, title: str, vmin: float, vmax: float, show_colorbar: bool):
    hm = sns.heatmap(
        pivot,
        ax=ax,
        annot=ANNOTATE,
        fmt=".1f",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        cbar=show_colorbar,
        annot_kws={"size": ANNOT_FONT_SIZE},
    )

    # Center x-ticks in the heatmap cells and label folds as 1..N.
    n_cols = pivot.shape[1]
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels([str(i + 1) for i in range(n_cols)], fontsize=TICK_FONT_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=TICK_FONT_SIZE)

    ax.set_xlabel("Fold", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Dialect region", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    if show_colorbar:
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)
        cbar.set_label("Hours", fontsize=AXIS_LABEL_FONT_SIZE)



def main():
    four_pivot = build_pivot(FOUR_REGION_CSV, FOUR_REGION_ORDER, VALUE_COLUMN)
    eleven_pivot = build_pivot(FINE_GRAINED_CSV, ELEVEN_REGION_ORDER, VALUE_COLUMN)

    common_vmin = 0
    common_vmax = max(four_pivot.max().max(), eleven_pivot.max().max())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

    heatmap_on_axis(
        ax=axes[0],
        pivot=four_pivot,
        title="Four dialect regions",
        vmin=common_vmin,
        vmax=common_vmax,
        show_colorbar=False,
    )

    heatmap_on_axis(
        ax=axes[1],
        pivot=eleven_pivot,
        title="Eleven dialect regions",
        vmin=common_vmin,
        vmax=common_vmax,
        show_colorbar=True,
    )

    fig.suptitle("Dialect region duration across folds", fontsize=SUPTITLE_FONT_SIZE)
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()