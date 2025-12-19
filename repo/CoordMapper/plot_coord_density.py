#!/usr/bin/env python3
"""
Simple KDE-based heatmap of coordinates using scipy.stats.gaussian_kde.

Features:
- Reads a .pkl file with 'latitude' and 'longitude' columns.
- Applies Gaussian KDE in lat/lon space.
- Plots heatmap with colorbar using matplotlib.
- Optionally masks to Norway bounding box.

Usage:
  python plot_coord_density_simple.py /path/to/data.pkl --out heatmap.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def parse_args():
    ap = argparse.ArgumentParser(description="Simple KDE heatmap of coordinates")
    ap.add_argument("pickle_path", type=str, help="Path to .pkl with latitude/longitude")
    ap.add_argument("--out", type=str, default="heatmap.png", help="Output image file (.png/.pdf)")
    ap.add_argument("--bins", type=int, default=300, help="Grid resolution (number of pixels per axis)")
    ap.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap")
    ap.add_argument("--bw", type=float, default=None, help="Bandwidth for KDE (None = auto)")
    ap.add_argument("--clip-norway", action="store_true", help="Clip to Norway bounding box")
    return ap.parse_args()

def main():
    args = parse_args()

    # Load data
    df = pd.read_pickle(args.pickle_path)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Expected 'latitude' and 'longitude' columns.")
    coords = df.dropna(subset=["latitude", "longitude"])
    if coords.empty:
        raise ValueError("No valid coordinates to plot.")

    lat = coords["latitude"].values
    lon = coords["longitude"].values

    # Norway bounding box (approx): lat 57–71, lon 4–32
    if args.clip_norway:
        mask = (lat >= 57) & (lat <= 71) & (lon >= 4) & (lon <= 32)
        lat = lat[mask]
        lon = lon[mask]

    # KDE
    xy = np.vstack([lon, lat])  # Note: KDE expects shape (2, N)
    kde = gaussian_kde(xy, bw_method=args.bw)

    # Grid
    lon_min, lon_max = (4, 32) if args.clip_norway else (lon.min(), lon.max())
    lat_min, lat_max = (57, 71) if args.clip_norway else (lat.min(), lat.max())

    lon_grid = np.linspace(lon_min, lon_max, args.bins)
    lat_grid = np.linspace(lat_min, lat_max, args.bins)
    X, Y = np.meshgrid(lon_grid, lat_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(Z, origin="lower", cmap=args.cmap,
                   extent=[lon_min, lon_max, lat_min, lat_max],
                   aspect="auto")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Coordinate Density (Gaussian KDE)")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label("Density")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[INFO] Saved heatmap to {args.out}")

if __name__ == "__main__":
    main()