#!/usr/bin/env python3
"""
Plot a geographic centre point together with its error radius on top of the Norway
base map produced by DialectMapper.plotter.

Features
--------
- Reads centre latitude / longitude and radius (km) from a text summary file.
- Uses the same DialectMapper projection helper as plot_coords.py so overlays line up
  with the map in the same way as your existing point plots.
- Optionally plots latitude/longitude points from one or more pickle files.
- Expands the SVG viewBox to include the full projected radius circle, so the error
  radius is never clipped in the saved plot.
- Intentionally does NOT add legends, titles, labels, or any other text to the plot.

Examples
--------
# Only centre + radius from the summary text file
python plot_center_radius.py geographic_center_summary.txt \
    --out plots/geographic_center_radius.png

# Centre + radius + points from one or more pickles
python plot_center_radius.py geographic_center_summary.txt \
    datasets/nordia_data/test_data.pkl \
    datasets/nordia_data/train_data.pkl \
    datasets/nordia_data/validation_data.pkl \
    --out plots/geographic_center_radius_with_points.png \
    --split-norway
"""

import argparse
import math
import os
import re
from pathlib import Path

import pandas as pd
from dialect_mapper.plotter import plotter_methods


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Plot the geographic centre and its distance radius on the Norway map, "
            "optionally together with points from one or more pickle files."
        )
    )
    ap.add_argument(
        "summary_path",
        type=str,
        default="/home/idatro/dialect_project/ndc_folds_loc/geographic_center_summary.txt",
        help="Path to the text file containing centre latitude/longitude and radius in km.",
    )
    ap.add_argument(
        "pickle_paths",
        nargs="*",
        help="Optional paths to one or more .pkl files with latitude/longitude columns.",
    )
    ap.add_argument("--out", type=str, default="center_radius_plot.pdf", help="Output path (.svg/.png/.pdf)")
    ap.add_argument("--width", type=float, default=2400.0, help="Output width in px")
    ap.add_argument("--height", type=float, default=3000.0, help="Output height in px")
    ap.add_argument("--stroke-width", type=float, default=0.00, help="Stroke width for map polygons")
    ap.add_argument("--split-norway", action="store_true", help="Apply plotter's north split (moves north southwards)")
    ap.add_argument("--rotate-norway", action="store_true", help="Rotate whole map by -30 degrees (same option as plot_coords.py)")

    ap.add_argument("--point-size", type=float, default=1.5, help="Radius of optional pickle points in map units")
    ap.add_argument("--point-fill", type=str, default="#417a38", help="Fill colour for optional pickle points")

    ap.add_argument("--center-size", type=float, default=2.5, help="Radius of the centre marker in map units")
    ap.add_argument("--center-fill", type=str, default="#206e80", help="Fill colour of the centre marker")

    ap.add_argument("--circle-stroke", type=str, default="#206e80", help="Stroke colour of the radius circle")
    ap.add_argument("--circle-stroke-width", type=float, default=0.8, help="Stroke width of the radius circle in map units")
    ap.add_argument("--circle-fill", type=str, default="#206e80", help="Fill colour for the radius polygon (default: none)")
    ap.add_argument("--circle-fill-opacity", type=float, default=0.08, help="Fill opacity for the radius polygon")
    ap.add_argument("--circle-points", type=int, default=360, help="Number of vertices used to approximate the circle")

    ap.add_argument("--radius-km", type=float, default=None, help="Override the radius from the summary file (in km)")
    ap.add_argument("--center-lat", type=float, default=None, help="Override the centre latitude from the summary file")
    ap.add_argument("--center-lon", type=float, default=None, help="Override the centre longitude from the summary file")

    return ap.parse_args()


def read_summary(summary_path):
    text = Path(summary_path).read_text(encoding="utf-8")

    lat_match = re.search(r"Latitude:\s*([-+]?\d+(?:\.\d+)?)", text)
    lon_match = re.search(r"Longitude:\s*([-+]?\d+(?:\.\d+)?)", text)
    radius_match = re.search(
        r"Mean distance from unique locations to center:\s*([-+]?\d+(?:\.\d+)?)\s*km",
        text,
        flags=re.IGNORECASE,
    )

    if not lat_match or not lon_match:
        raise ValueError(
            "Could not find 'Latitude:' and 'Longitude:' in the summary file."
        )
    if not radius_match:
        raise ValueError(
            "Could not find 'Mean distance from unique locations to center: ... km' in the summary file. "
            "You can also supply --radius-km explicitly."
        )

    return float(lat_match.group(1)), float(lon_match.group(1)), float(radius_match.group(1))


def destination_point(lat_deg, lon_deg, bearing_deg, distance_km):
    """Return lat/lon reached by travelling distance_km from (lat, lon) along a great-circle bearing."""
    radius_earth_km = 6371.0088
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    ang_dist = distance_km / radius_earth_km

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist)
        + math.cos(lat1) * math.sin(ang_dist) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )

    lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(lat2), math.degrees(lon2)


def project_latlon(pm, lat, lon, width_px, height_px, split_norway):
    # Keep the same width/height swap as in plot_coords.py
    return pm._convert_latlon_to_xy(
        lat,
        lon,
        mapWidth=height_px,
        mapHeight=width_px,
        move_south=split_norway,
    )


def load_points(pickle_paths):
    if not pickle_paths:
        return pd.DataFrame(columns=["latitude", "longitude"])

    dfs = [pd.read_pickle(path) for path in pickle_paths]
    df = pd.concat(dfs, ignore_index=True)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Expected 'latitude' and 'longitude' columns in the pickle file(s).")

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    return df


def compute_combined_bounds(base_min_x, base_min_y, base_width, base_height, overlay_bounds, margin):
    base_max_x = base_min_x + base_width
    base_max_y = base_min_y + base_height

    min_x = min(base_min_x, overlay_bounds[0]) - margin
    min_y = min(base_min_y, overlay_bounds[1]) - margin
    max_x = max(base_max_x, overlay_bounds[2]) + margin
    max_y = max(base_max_y, overlay_bounds[3]) + margin

    return min_x, min_y, max_x - min_x, max_y - min_y


def write_output(pm, svg_text, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if out_path.lower().endswith(".svg"):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_text)
        print(f"[INFO] Saved SVG to: {os.path.abspath(out_path)}")
        return

    temp_svg = os.path.splitext(out_path)[0] + ".__temp__.svg"
    with open(temp_svg, "w", encoding="utf-8") as f:
        f.write(svg_text)

    try:
        import cairosvg
        if out_path.lower().endswith(".png"):
            cairosvg.svg2png(url=temp_svg, write_to=out_path)
        elif out_path.lower().endswith(".pdf"):
            cairosvg.svg2pdf(url=temp_svg, write_to=out_path)
        else:
            cairosvg.svg2png(url=temp_svg, write_to=out_path)
        os.remove(temp_svg)
        print(f"[INFO] Saved output to: {os.path.abspath(out_path)}")
    except Exception as e:
        fallback_svg = os.path.splitext(out_path)[0] + ".svg"
        os.replace(temp_svg, fallback_svg)
        print(f"[WARN] CairoSVG conversion failed ({e}). Saved fallback SVG to: {os.path.abspath(fallback_svg)}")


def main():
    args = parse_args()

    summary_lat, summary_lon, summary_radius_km = read_summary(args.summary_path)
    center_lat = args.center_lat if args.center_lat is not None else summary_lat
    center_lon = args.center_lon if args.center_lon is not None else summary_lon
    radius_km = args.radius_km if args.radius_km is not None else summary_radius_km

    if radius_km <= 0:
        raise ValueError("Radius must be positive.")
    if args.circle_points < 12:
        raise ValueError("--circle-points must be at least 12.")

    points_df = load_points(args.pickle_paths)

    pm = plotter_methods()

    def base_color(_name):
        return "#adadad"

    features = pm.kommuner_json["features"]
    svg_list, base_min_x, base_min_y, base_width, base_height = pm._process_features(
        features,
        base_color,
        final_width=args.height,
        final_height=args.width,
        split_norway=args.split_norway,
        rotate_norway=args.rotate_norway,
        stroke_width=args.stroke_width,
    )

    # Project centre and geodesic circle vertices with the SAME helper used for point plotting.
    center_x, center_y = project_latlon(pm, center_lat, center_lon, args.width, args.height, args.split_norway)

    circle_xy = []
    for i in range(args.circle_points):
        bearing = 360.0 * i / args.circle_points
        lat_i, lon_i = destination_point(center_lat, center_lon, bearing, radius_km)
        x_i, y_i = project_latlon(pm, lat_i, lon_i, args.width, args.height, args.split_norway)
        circle_xy.append((x_i, y_i))

    circle_points_attr = " ".join(f"{x:.3f},{y:.3f}" for x, y in circle_xy)

    overlay_min_x = min([center_x] + [x for x, _ in circle_xy])
    overlay_max_x = max([center_x] + [x for x, _ in circle_xy])
    overlay_min_y = min([center_y] + [y for _, y in circle_xy])
    overlay_max_y = max([center_y] + [y for _, y in circle_xy])

    point_svgs = []
    if not points_df.empty:
        for _, row in points_df.iterrows():
            x, y = project_latlon(
                pm,
                float(row["latitude"]),
                float(row["longitude"]),
                args.width,
                args.height,
                args.split_norway,
            )
            point_svgs.append(
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{args.point_size:.3f}" '
                f'fill="{args.point_fill}" fill-opacity="1.0" stroke="none" />'
            )
            overlay_min_x = min(overlay_min_x, x - args.point_size)
            overlay_max_x = max(overlay_max_x, x + args.point_size)
            overlay_min_y = min(overlay_min_y, y - args.point_size)
            overlay_max_y = max(overlay_max_y, y + args.point_size)
    circle_margin = max(args.circle_stroke_width, args.center_size, args.point_size)
    view_min_x, view_min_y, view_width, view_height = compute_combined_bounds(
        base_min_x,
        base_min_y,
        base_width,
        base_height,
        (overlay_min_x, overlay_min_y, overlay_max_x, overlay_max_y),
        margin=circle_margin,
    )

    overlay_svgs = [
        (
            f'<polygon points="{circle_points_attr}" fill="{args.circle_fill}" '
            f'fill-opacity="{args.circle_fill_opacity:.3f}" '
            f'stroke="{args.circle_stroke}" '
            f'stroke-width="{args.circle_stroke_width:.3f}" />'
        ),
        (
            f'<circle cx="{center_x:.3f}" cy="{center_y:.3f}" r="{args.center_size:.3f}" '
            f'fill="{args.center_fill}" fill-opacity="1.0" stroke="none" />'
        ),
    ]


    overlay_svgs.extend(point_svgs)


    text_x = center_x + 50    # horizontal offset
    text_y = center_y - 60    # small upward shift



    label = f"R = {radius_km:.2f} km"

    text_svg = (
        f'<text x="{text_x:.3f}" y="{text_y:.3f}" '
        f'fill="{args.circle_stroke}" '
        f'font-size="12" '
        f'font-family="Helvetica" '
        f'text-anchor="start" '
        f'dominant-baseline="middle">'
        f'{label}'
        f'</text>'
    )
    overlay_svgs.append(text_svg)


    # Build the final SVG with an expanded viewBox so the entire radius remains visible.
    head = pm.head_bit.format(str(args.width), str(args.height), view_min_x, view_min_y, view_width, view_height)
    svg_text = head + "".join(svg_list) + "".join(overlay_svgs) + pm.end_bit

    write_output(pm, svg_text, args.out)


if __name__ == "__main__":
    main()
