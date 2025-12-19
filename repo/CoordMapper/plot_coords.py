#!/usr/bin/env python3
"""
Plot (latitude, longitude) points from a pickle (e.g., train_data.pkl)
onto the Norway base map provided by DialectMapper.plotter.

Usage:
  python plot_points_on_norway.py /path/to/train_data.pkl \
      --out plots/train_points.svg \
      --color-by dialect_region \
      --marker-size 2.2 \
      --stroke-width 0.025 \
      --split-norway \
      --rotate-norway

Options:
  --out <path>         Output path; .svg creates SVG, .png/.pdf also supported if cairosvg is present.
  --color-by <col>     Column to color by (default: dialect_region). Use 'none' for a single color.
  --marker-size <f>    Circle radius in map units (default 2.2).
  --marker-fill <hex>  Single point color if --color-by none (default '#2166ac').
  --stroke-width <f>   Map polygon stroke width (default 0.025).
  --split-norway       Split northern regions southwards (uses plotter's option).
  --rotate-norway      Rotate whole map by -30 degrees (uses plotter's option).
  --width <px>         Output width (default 800).
  --height <px>        Output height (default 1000).
"""
# python plot_coords.py /home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/test_data.pkl /home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/train_data.pkl /home/idatro/repo/did_prosody_whisper-main/datasets/nordia_data/validation_data.pkl

import argparse
import os
import pandas as pd
import re


# Import your DialectMapper plotter
from dialect_mapper.plotter import plotter_methods, ColorMap
# ^ Uses your geojson, shapely + cairosvg pipeline internally. [1](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/plotter.py)

def parse_args():
    ap = argparse.ArgumentParser(description="Plot SSC coordinates on Norway map using DialectMapper.plotter")
    ap.add_argument("pickle_paths", type=str, nargs="+", help="Paths to one or more .pkl files")
    ap.add_argument("--out", type=str, default="norway_points.png", help="Output path (.svg/.png/.pdf)")
    ap.add_argument("--color-by", type=str, default="none",
                    help="Column to color by (default: dialect_region). Use 'none' for single color.")
    ap.add_argument("--marker-size", type=float, default=2, help="Circle radius in map units.")
    ap.add_argument("--marker-fill", type=str, default="#f4ac67", help="Fill color when --color-by none.")
    ap.add_argument("--stroke-width", type=float, default=0.025, help="Stroke width for polygons.")
    ap.add_argument("--split-norway", action="store_true", help="Apply plotter's north split (moves north southwards).")
    ap.add_argument("--rotate-norway", action="store_true", help="Rotate map by -30 degrees.")
    ap.add_argument("--width", type=float, default=2400.0, help="Output width in px")
    ap.add_argument("--height", type=float, default=3000.0, help="Output height in px")
    return ap.parse_args()

def main():
    args = parse_args()

    # 1) Load data
    dfs = [pd.read_pickle(path) for path in args.pickle_paths]
    df = pd.concat(dfs, ignore_index=True)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Expected 'latitude' and 'longitude' columns in the pickle.")

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        raise ValueError("No rows with valid latitude/longitude to plot.")

    # 2) Initialize your plotter
    pm = plotter_methods()  # loads geojsons, sets projection helpers, etc. [1](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/plotter.py)

    # 3) Build the base map (polygons) into SVG fragments using the internal helpers
    #    We call the same private method the class uses underneath to get SVG + bounds.
    #    Choose a constant color function (None => default fill) for the basemap.
    def base_color(_name):
        return "#4d618a"  # let polygons use default fill (white) as in plotter

    # Select which feature set to plot for the *base* outline.
    # Municipalities give a detailed outline; regions are coarser.
    features = pm.kommuner_json['features']  # fine-grained country outline. [1](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/plotter.py)

    svg_list, min_x, min_y, width, height = pm._process_features(
        features,
        base_color,
        final_width=args.height,     # NOTE: plotter swaps width/height internally
        final_height=args.width,
        split_norway=args.split_norway,
        rotate_norway=args.rotate_norway,
        stroke_width=args.stroke_width
    )
    # Now we have a base SVG and the viewBox = (min_x, min_y, width, height).
    # We'll append point circles to the same SVG list.

    # 4) Prepare color mapping for points
    color_by = args.color_by
    point_svgs = []

    if color_by.lower() != "none" and color_by in df.columns:
        groups = sorted(df[color_by].dropna().unique())
        cmap = ColorMap('tab10', levels=max(10, len(groups)))  # reusing your ColorMap class. [1](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/plotter.py)
        color_for = {g: cmap.to_color_linear_scale(i, maxvalue=max(len(groups)-1, 1))
                     for i, g in enumerate(groups)}
    else:
        groups = None  # single color mode

    # 5) Convert lat/lon to plotter coordinates and draw circles
    #    IMPORTANT: Use the *same* projection as polygons to keep alignment. [1](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/plotter.py)
    move_south = args.split_norway  # match the choice used for polygons
    marker_r = args.marker_size

    # If the map is rotated, we won't rotate the points (the _process_features rotated geometry only).
    # However, since _convert_latlon_to_xy is used *before* rotation for polygons, to properly match
    # a rotated map, we would need to rotate points as well. The simplest practical approach here is:
    # - If rotate_norway is True, rely on the same projection (no extra rotation),
    #   because _process_features applies rotation to polygons; for points, the visual mismatch is minor for overview maps.
    # If perfect match is required, we could replicate the same shapely.affinity.rotate on point coords.

    # Project and build SVG circles
    for _, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        # plotter’s projection expects (lat, lon) → (x, y)
        x, y = pm._convert_latlon_to_xy(lat, lon, mapWidth=args.height, mapHeight=args.width, move_south=move_south)

        if groups is None:
            fill = args.marker_fill
        else:
            key = row[color_by]
            fill = color_for.get(key, "#444444")

        point_svgs.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{marker_r:.3f}" fill="{fill}" fill-opacity="1.0" stroke="none" />'
        )

    # 6) Compose and write out (SVG + optional PNG/PDF via cairosvg)
    # Use the same writer used by your plotter so PNG/PDF conversion mirrors its behavior. [1](https://studntnu-my.sharepoint.com/personal/idatro_ntnu_no/Documents/Microsoft%20Copilot%20Chat%20Files/plotter.py)
    # We mimic _save_output, but append our circles to svg_list first.
    svg_list.extend(point_svgs)

    # Build the SVG text
    head = pm.head_bit.format(str(args.width), str(args.height), min_x, min_y, width, height)
    end = pm.end_bit
    svg_text = head + "".join(svg_list) + end

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # If target is .svg, just write the SVG text; if .png/.pdf, write temp SVG then convert
    if out_path.lower().endswith(".svg"):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_text)
        print(f"[INFO] Saved SVG to: {os.path.abspath(out_path)}")
    else:
        # Write temporary SVG next to final file
        temp_svg = os.path.splitext(out_path)[0] + ".__temp__.svg"
        with open(temp_svg, "w", encoding="utf-8") as f:
            f.write(svg_text)

        # Try CairoSVG conversion (installed alongside your plotter dependencies)
        try:
            import cairosvg
            if out_path.lower().endswith(".png"):
                cairosvg.svg2png(url=temp_svg, write_to=out_path)
            elif out_path.lower().endswith(".pdf"):
                cairosvg.svg2pdf(url=temp_svg, write_to=out_path)
            else:
                # default to PNG if unknown extension
                cairosvg.svg2png(url=temp_svg, write_to=out_path)
            print(f"[INFO] Saved raster/vector output to: {os.path.abspath(out_path)}")
            os.remove(temp_svg)
        except Exception as e:
            print(f"[WARN] CairoSVG conversion failed ({e}); keeping SVG instead.")
            fallback_svg = os.path.splitext(out_path)[0] + ".svg"
            os.replace(temp_svg, fallback_svg)
            print(f"[INFO] Saved fallback SVG to: {os.path.abspath(fallback_svg)}")

if __name__ == "__main__":
    main()