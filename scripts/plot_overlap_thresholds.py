#!/usr/bin/env python3
"""
plot_overlap_thresholds.py

Visualize and/or summarize how much data would be REMOVED if you drop all
overlapping speech, for multiple minor-words thresholds. It loads the
`summary<threshold>.csv` files produced by trs_overlap_stats.py.

Key features:
  • Robust absolute/relative glob matching for summary files.
  • Computes per-threshold aggregates (kept/removed seconds & %).
  • Plots removal (overlap after threshold) in % and hours, plus distributions.
  • CLI flags to produce only CSV tables or only plots from the summaries.

Example usage:
  # Both CSV and plots
  python plot_overlap_thresholds.py \
      --summary-glob "/abs/path/to/reports/summary*.csv" \
      --out-dir "/abs/path/to/reports/plots"

  # CSV only
  python plot_overlap_thresholds.py \
      --summary-glob "/abs/path/to/reports/summary*.csv" \
      --out-dir "/abs/path/to/reports/plots" \
      --csv-only

  # Plots only (no CSV tables written)
  python plot_overlap_thresholds.py \
      --summary-glob "/abs/path/to/reports/summary*.csv" \
      --out-dir "/abs/path/to/reports/plots" \
      --plots-only
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

plt.rcParams.update({
    'figure.dpi': 140,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

THRESHOLD_RE = re.compile(r"summary(\d+)\.csv$")


def load_summaries(glob_pattern: str, verbose: bool=False) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pattern))  # supports absolute & relative
    paths = [Path(p) for p in paths]
    if verbose:
        print("Matched files:", [str(p) for p in paths])
    if not paths:
        raise SystemExit(f"No files matched pattern: {glob_pattern}")
    frames = []
    for p in paths:
        m = THRESHOLD_RE.search(p.name)
        if not m:
            # Skip files that don't match expected naming
            if verbose:
                print(f"Skipping {p.name} (doesn't match summary<N>.csv)")
            continue
        thr = int(m.group(1))
        df = pd.read_csv(p)
        df['threshold'] = thr
        frames.append(df)
    if not frames:
        raise SystemExit("Found files but none matched 'summary<N>.csv' pattern.")
    df_all = pd.concat(frames, ignore_index=True)
    # Ensure numeric types for key metrics
    for col in [
        'total_duration_sec',
        'overlap_sec_original', 'overlap_pct_original',
        'overlap_sec_after_threshold', 'overlap_pct_after_threshold',
        'unresolved_multi_no_who_sec'
    ]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    return df_all


def compute_aggregates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # per-file derived fields
    df = df.copy()
    df['removed_sec'] = df['overlap_sec_after_threshold']
    df['removed_pct'] = df['overlap_pct_after_threshold']
    df['kept_sec'] = df['total_duration_sec'] - df['removed_sec']
    df['kept_pct'] = np.where(df['total_duration_sec']>0,
                              100.0 - df['removed_pct'], np.nan)
    df['unresolved_pct'] = np.where(df['total_duration_sec']>0,
                                    df['unresolved_multi_no_who_sec'] / df['total_duration_sec'] * 100,
                                    np.nan)

    # Aggregates by threshold
    agg = df.groupby('threshold').agg(
        n_files=('file', 'nunique'),
        total_duration_sec=('total_duration_sec', 'sum'),
        total_removed_sec=('removed_sec', 'sum'),
        total_kept_sec=('kept_sec', 'sum'),
        mean_removed_pct=('removed_pct', 'mean'),
        std_removed_pct=('removed_pct', 'std'),
        median_removed_pct=('removed_pct', 'median'),
        mean_kept_pct=('kept_pct', 'mean'),
        std_kept_pct=('kept_pct', 'std'),
        mean_unresolved_pct=('unresolved_pct', 'mean'),
        baseline_overlap_pct_original=('overlap_pct_original', 'mean'),
    ).reset_index().sort_values('threshold')

    # Also compute hours for readability
    for c in ['total_duration_sec', 'total_removed_sec', 'total_kept_sec']:
        agg[c.replace('_sec','_hours')] = agg[c] / 3600.0

    return df, agg


def save_csv_tables(df_all: pd.DataFrame, agg: pd.DataFrame, out_dir: Path, verbose: bool=False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / 'aggregates_by_threshold.csv'
    agg.to_csv(agg_path, index=False)
    if verbose:
        print(f"Wrote {agg_path}")

    removal_summary = agg[['threshold', 'n_files',
                           'total_duration_hours',
                           'total_removed_sec', 'total_removed_hours',
                           'total_kept_sec', 'total_kept_hours',
                           'mean_removed_pct', 'median_removed_pct']]
    removal_csv = out_dir / 'removal_summary_by_threshold.csv'
    removal_summary.to_csv(removal_csv, index=False)
    if verbose:
        print(f"Wrote {removal_csv}")


def plot_lines_with_points(thresholds, y_mean, y_std, per_file_df, y_col, title, ylabel, out_path, baseline=None):
    # Defensive check
    if y_col not in per_file_df.columns:
        raise KeyError(f"Column '{y_col}' not found in per_file_df.columns: {list(per_file_df.columns)}")

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(thresholds, y_mean, marker='o', label='Mean across files')
    # error band (std)
    if y_std is not None:
        ax.fill_between(thresholds, y_mean - y_std, y_mean + y_std,
                        alpha=0.15, color=ax.lines[0].get_color(), label='±1 SD')
    # overlay per-file jittered points
    rng = np.random.default_rng(42)
    x_jitter = 0.07
    for thr in thresholds:
        sub = per_file_df[per_file_df['threshold']==thr]
        xs = thr + (rng.random(len(sub)) - 0.5) * x_jitter
        ax.scatter(xs, sub[y_col], s=12, color='tab:gray', alpha=0.6, label='_nolegend_')
    if baseline is not None:
        ax.plot(thresholds, baseline, linestyle='--', color='tab:red', alpha=0.7, label='Baseline (original)')
    ax.set_title(title)
    ax.set_xlabel('Minor-words threshold (allowed words by other speakers)')
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_boxplot(df: pd.DataFrame, out_path: Path):
    thresholds = sorted(df['threshold'].unique())
    data = [df.loc[df['threshold']==thr, 'removed_pct'].dropna().values for thr in thresholds]
    fig, ax = plt.subplots(figsize=(7,4))
    bp = ax.boxplot(data, labels=[str(t) for t in thresholds], showfliers=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set(facecolor='#88ccee', alpha=0.6)
    ax.set_title('Removed % (overlap after threshold) — distribution across files')
    ax.set_xlabel('Minor-words threshold')
    ax.set_ylabel('Removed % of duration')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_total_removed_hours(agg: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(agg['threshold'].astype(str), agg['total_removed_hours'], color='#dd8452')
    ax.set_title('Total hours removed if dropping overlap (by threshold)')
    ax.set_xlabel('Minor-words threshold')
    ax.set_ylabel('Removed hours (sum across files)')
    for i, v in enumerate(agg['total_removed_hours']):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_stacked_kept_removed(agg: pd.DataFrame, out_path: Path):
    # Use mean percentages so bars are comparable across thresholds despite varying durations
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(agg))
    kept = agg['mean_kept_pct'].values
    removed = agg['mean_removed_pct'].values
    ax.bar(x, kept, label='Kept %', color='#55a868')
    ax.bar(x, removed, bottom=kept, label='Removed %', color='#c44e52')
    ax.set_xticks(x)
    ax.set_xticklabels(agg['threshold'].astype(str))
    ax.set_title('Kept vs Removed (mean % across files)')
    ax.set_xlabel('Minor-words threshold')
    ax.set_ylabel('% of duration')
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_file_spaghetti(df: pd.DataFrame, out_path: Path):
    # One line per file: removed % vs threshold
    pivot = df.pivot_table(index='file', columns='threshold', values='removed_pct')
    thresholds = sorted(df['threshold'].unique())
    fig, ax = plt.subplots(figsize=(8,5))
    for _, row in pivot.iterrows():
        ax.plot(thresholds, row[thresholds].values, color='tab:gray', alpha=0.5)
    ax.plot(thresholds, df.groupby('threshold')['removed_pct'].mean().values,
            color='tab:blue', marker='o', linewidth=2.5, label='Mean across files')
    ax.set_title('Per-file removed % (overlap after threshold) across thresholds')
    ax.set_xlabel('Minor-words threshold')
    ax.set_ylabel('Removed % of duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Plot overlap/removal vs. minor-words threshold from summary*.csv files.')
    ap.add_argument('--summary-glob', type=str, default='./reports/summary*.csv', help='Glob pattern for summary CSVs (e.g., ./reports/summary*.csv)')
    ap.add_argument('--out-dir', type=str, default='./reports/plots', help='Directory to write plots and aggregates')
    ap.add_argument('--csv-only', action='store_true', help='Only produce CSV tables (no plots)')
    ap.add_argument('--plots-only', action='store_true', help='Only produce plots (no CSV tables)')
    ap.add_argument('--verbose', action='store_true', help='Print matched files and outputs')
    args = ap.parse_args()

    if args.csv_only and args.plots_only:
        raise SystemExit('Choose either --csv-only or --plots-only, or neither (to do both).')

    out_dir = Path(args.out_dir)
    df = load_summaries(args.summary_glob, verbose=args.verbose)
    df_all, agg = compute_aggregates(df)

    # CSVs
    if not args.plots_only:
        save_csv_tables(df_all, agg, out_dir, verbose=args.verbose)

    # Plots
    if not args.csv_only:
        thresholds = agg['threshold'].values
        # Removed % vs threshold (mean ± SD with per-file points)
        plot_lines_with_points(
            thresholds=thresholds,
            y_mean=agg['mean_removed_pct'].values,
            y_std=agg['std_removed_pct'].values,
            per_file_df=df_all,
            y_col='removed_pct',
            title='Removed % (overlap after threshold): mean ± SD with per-file points',
            ylabel='Removed % of duration',
            out_path=out_dir / 'removable_pct_vs_threshold.png',
            baseline=agg['baseline_overlap_pct_original'].values
        )

        # Removed seconds (mean across files)
        y_mean_sec = df_all.groupby('threshold')['overlap_sec_after_threshold'].mean().reindex(thresholds).values
        plot_lines_with_points(
            thresholds=thresholds,
            y_mean=y_mean_sec,
            y_std=None,
            per_file_df=df_all,
            y_col='overlap_sec_after_threshold',
            title='Removed seconds (overlap after threshold): mean across files',
            ylabel='Removed seconds (mean)',
            out_path=out_dir / 'overlap_sec_vs_threshold.png',
            baseline=None
        )

        # Unresolved percentage (no <Who>)
        plot_lines_with_points(
            thresholds=thresholds,
            y_mean=agg['mean_unresolved_pct'].values,
            y_std=None,
            per_file_df=df_all,
            y_col='unresolved_pct',
            title='Unresolved (no <Who>) as % of duration',
            ylabel='Unresolved % of duration',
            out_path=out_dir / 'unresolved_pct_vs_threshold.png',
            baseline=None
        )

        # Boxplot of removed % by threshold
        plot_boxplot(df_all, out_dir / 'overlap_pct_boxplot.png')

        # Total removed hours (sum across files)
        plot_total_removed_hours(agg, out_dir / 'removable_total_hours_vs_threshold.png')

        # Stacked kept vs removed (mean %)
        plot_stacked_kept_removed(agg, out_dir / 'kept_vs_removed_stacked_pct.png')

        # Per-file spaghetti
        plot_per_file_spaghetti(df_all, out_dir / 'per_file_removed_pct_spaghetti.png')

    if args.verbose:
        print('Done.')

if __name__ == '__main__':
    main()