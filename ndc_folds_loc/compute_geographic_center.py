#!/usr/bin/env python3
"""
Plot a geographical centre, a mean-error-radius ring, and the three shortest /
three longest distances on top of the Norway base map using the SAME DialectMapper
projection pipeline as plot_coords.py.

Why this version fixes the skew:
- The base map is generated with plotter_methods()._process_features(...), exactly
  like plot_coords.py.
- Every coordinate is projected with plotter_methods()._convert_latlon_to_xy(...),
  exactly like plot_coords.py.
- The mean-error radius is not drawn in raw lon/lat space. Instead, we build a
  geodesic ring in lat/lon around the centre, then project each ring point with the
  same plotter projection, so the ring follows the same map geometry.
- If the map is rotated, the points and the ring are rotated too, so they stay
  aligned with the base map.

Edit the HARD-CODED INPUTS section below to swap in new coordinates, labels, or
radius values.

Examples:
  python plot_geographical_center_projected.py --out geographical_center_map.svg
  python plot_geographical_center_projected.py --out geographical_center_map.png --split-norway --rotate-norway
"""

import argparse
import html
import math
import os
from typing import Iterable, List, Sequence, Tuple

from dialect_mapper.plotter import plotter_methods

try:
    import cairosvg  # optional, only needed for PNG/PDF output
except Exception:
    cairosvg = None

try:
    from pyproj import Geod
    _GEOD = Geod(ellps="WGS84")
except Exception:
    _GEOD = None


# =============================================================================
# HARD-CODED INPUTS
# =============================================================================
GEOGRAPHIC_CENTER = {
    "label": "Geographical centre",
    "latitude": 62.741861,
    "longitude": 11.447117,
}

MEAN_ERROR_DISTANCE_KM = 416.720

LONGEST_DISTANCES = [
    {
        "speaker_id": "vardoe_01um",
        "distance_km": 1197.560,
        "latitude": 70.330816,
        "longitude": 30.961821,
    },
    {
        "speaker_id": "kirkenes_03gm",
        "distance_km": 1108.741,
        "latitude": 69.537566,
        "longitude": 29.726137,
    },
    {
        "speaker_id": "tana_04gk",
        "distance_km": 1103.701,
        "latitude": 70.282334,
        "longitude": 27.891327,
    },
]

SHORTEST_DISTANCES = [
    {
        "speaker_id": "roeros_02uk",
        "distance_km": 22.045,
        "latitude": 62.569515,
        "longitude": 11.660441,
    },
    {
        "speaker_id": "dalsbygda_04gk",
        "distance_km": 36.375,
        "latitude": 62.415186,
        "longitude": 11.409663,
    },
    {
        "speaker_id": "selbu_03gm",
        "distance_km": 50.353,
        "latitude": 63.175425,
        "longitude": 11.159626,
    },
]

# SVG styling / rendering defaults
DEFAULT_OUT = "geographical_center_projected.svg"
DEFAULT_WIDTH = 2400.0
DEFAULT_HEIGHT = 3000.0
DEFAULT_MARKER_SIZE = 8.0
DEFAULT_CENTER_SIZE = 12.0
DEFAULT_STROKE_WIDTH = 0.025
DEFAULT_FONT_SIZE = 28
DEFAULT_LABEL_DX = 18.0
DEFAULT_LABEL_DY = -18.0
DEFAULT_RING_STROKE = 7.0
DEFAULT_RADIUS_LINE_STROKE = 5.0
DEFAULT_SPLIT_NORWAY = False
DEFAULT_ROTATE_NORWAY = False

BASE_FILL = "#ffffff"
BASE_STROKE = "#4d618a"
CENTER_FILL = "#111111"
RING_COLOR = "#1f77b4"
SHORTEST_FILL = "#2ca02c"
LONGEST_FILL = "#d62728"
LABEL_HALO = "#ffffff"
LEGEND_BG = "rgba(255,255,255,0.92)"


# =============================================================================
# ARGUMENTS
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot a projected geographical centre + error radius + distance extremes on the DialectMapper Norway map."
    )
    ap.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output path (.svg, .png, or .pdf)")
    ap.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="Output width in px")
    ap.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="Output height in px")
    ap.add_argument("--stroke-width", type=float, default=DEFAULT_STROKE_WIDTH, help="Map polygon stroke width")
    ap.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE, help="Radius of shortest/longest distance markers")
    ap.add_argument("--center-size", type=float, default=DEFAULT_CENTER_SIZE, help="Radius of centre marker")
    ap.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE, help="Base label font size")
    ap.add_argument("--split-norway", action="store_true", default=DEFAULT_SPLIT_NORWAY,
                    help="Apply plotter's northern split to the base map and all projected geometry")
    ap.add_argument("--rotate-norway", action="store_true", default=DEFAULT_ROTATE_NORWAY,
                    help="Rotate the map and all projected geometry by -30 degrees")
    return ap.parse_args()


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def normalize_longitude_delta(delta_deg: float) -> float:
    """Normalize longitude delta to [-180, 180]."""
    while delta_deg <= -180:
        delta_deg += 360
    while delta_deg > 180:
        delta_deg -= 360
    return delta_deg


def destination_point_approx(lat: float, lon: float, distance_km: float, azimuth_deg: float) -> Tuple[float, float]:
    """Fallback destination point if pyproj is unavailable."""
    earth_radius_km = 6371.0088
    angular_distance = distance_km / earth_radius_km

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(azimuth_deg)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )

    lon2_deg = ((math.degrees(lon2) + 540) % 360) - 180
    lat2_deg = math.degrees(lat2)
    return lat2_deg, lon2_deg


def geodesic_ring_latlon(center_lat: float, center_lon: float, radius_km: float, n_points: int = 360) -> List[Tuple[float, float]]:
    """Create a geodesic ring as a list of (lat, lon) points."""
    coords = []
    if _GEOD is not None:
        for az in range(n_points):
            lon2, lat2, _ = _GEOD.fwd(center_lon, center_lat, az, radius_km * 1000.0)
            coords.append((lat2, lon2))
    else:
        for i in range(n_points):
            az = 360.0 * i / n_points
            coords.append(destination_point_approx(center_lat, center_lon, radius_km, az))
    if coords:
        coords.append(coords[0])
    return coords


def project_latlon(pm, lat: float, lon: float, map_width: float, map_height: float, move_south: bool) -> Tuple[float, float]:
    """Project a single lat/lon point with the exact same DialectMapper helper as plot_coords.py."""
    x, y = pm._convert_latlon_to_xy(lat, lon, mapWidth=map_height, mapHeight=map_width, move_south=move_south)
    return float(x), float(y)


def rotate_point(x: float, y: float, angle_deg: float, origin_x: float, origin_y: float) -> Tuple[float, float]:
    """Rotate a point around an origin in SVG/map coordinates."""
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = x - origin_x
    dy = y - origin_y
    xr = origin_x + cos_t * dx - sin_t * dy
    yr = origin_y + sin_t * dx + cos_t * dy
    return xr, yr


def apply_optional_rotation(points_xy: Sequence[Tuple[float, float]], rotate_norway: bool,
                            min_x: float, min_y: float, width: float, height: float,
                            angle_deg: float = -30.0) -> List[Tuple[float, float]]:
    """Rotate projected points if the map was rotated in _process_features."""
    if not rotate_norway:
        return list(points_xy)

    origin_x = min_x + width / 2.0
    origin_y = min_y + height / 2.0
    return [rotate_point(x, y, angle_deg, origin_x, origin_y) for x, y in points_xy]


def project_many(pm, latlon_points: Sequence[Tuple[float, float]], map_width: float, map_height: float,
                 move_south: bool, rotate_norway: bool, min_x: float, min_y: float,
                 width: float, height: float) -> List[Tuple[float, float]]:
    """Project many lat/lon points and apply the same optional map rotation."""
    projected = [project_latlon(pm, lat, lon, map_width, map_height, move_south) for lat, lon in latlon_points]
    return apply_optional_rotation(projected, rotate_norway, min_x, min_y, width, height)


# =============================================================================
# SVG HELPERS
# =============================================================================
def svg_circle(cx: float, cy: float, r: float, fill: str, stroke: str = "none", stroke_width: float = 0.0,
               fill_opacity: float = 1.0) -> str:
    return (
        f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
        f'fill="{fill}" fill-opacity="{fill_opacity:.3f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.3f}" />'
    )


def svg_polyline(points_xy: Sequence[Tuple[float, float]], stroke: str, stroke_width: float,
                 fill: str = "none", fill_opacity: float = 1.0, linecap: str = "round", linejoin: str = "round") -> str:
    pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in points_xy)
    return (
        f'<polyline points="{pts}" fill="{fill}" fill-opacity="{fill_opacity:.3f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.3f}" '
        f'stroke-linecap="{linecap}" stroke-linejoin="{linejoin}" />'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: float,
             dasharray: str = None) -> str:
    dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return (
        f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.3f}"{dash_attr} />'
    )


def svg_text(x: float, y: float, text: str, font_size: int, fill: str,
             anchor: str = "start", weight: str = "normal") -> str:
    # White halo keeps text readable against the map.
    escaped = html.escape(text)
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{font_size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}" stroke="{LABEL_HALO}" stroke-width="8" paint-order="stroke">{escaped}</text>'
    )


def svg_multiline_text(x: float, y: float, lines: Sequence[str], font_size: int, fill: str,
                       anchor: str = "start", weight: str = "normal", line_spacing: float = 1.2) -> str:
    escaped_lines = [html.escape(line) for line in lines]
    text_parts = [
        f'<text x="{x:.3f}" y="{y:.3f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{font_size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}" stroke="{LABEL_HALO}" stroke-width="8" paint-order="stroke">'
    ]
    first = True
    for line in escaped_lines:
        if first:
            text_parts.append(f'<tspan x="{x:.3f}" dy="0">{line}</tspan>')
            first = False
        else:
            text_parts.append(f'<tspan x="{x:.3f}" dy="{font_size * line_spacing:.1f}">{line}</tspan>')
    text_parts.append('</text>')
    return ''.join(text_parts)


def svg_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str,
             stroke_width: float, rx: float = 10.0, ry: float = 10.0) -> str:
    return (
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" '
        f'rx="{rx:.3f}" ry="{ry:.3f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.3f}" />'
    )


def estimate_text_box(lines: Sequence[str], font_size: int) -> Tuple[float, float]:
    """Rough text box estimate for legend/background layout in SVG units."""
    max_len = max((len(line) for line in lines), default=0)
    width = max_len * font_size * 0.62 + 28
    height = len(lines) * font_size * 1.25 + 20
    return width, height


# =============================================================================
# MAIN PLOT CREATION
# =============================================================================
def main():
    args = parse_args()

    pm = plotter_methods()

    # Use the same detailed municipality features and the same backend helper as plot_coords.py.
    features = pm.kommuner_json["features"]

    def base_color(_name):
        return BASE_STROKE

    svg_list, min_x, min_y, width, height = pm._process_features(
        features,
        base_color,
        final_width=args.height,   # Note: plot_coords.py intentionally swaps width/height here
        final_height=args.width,
        split_norway=args.split_norway,
        rotate_norway=args.rotate_norway,
        stroke_width=args.stroke_width,
    )

    center_lat = float(GEOGRAPHIC_CENTER["latitude"])
    center_lon = float(GEOGRAPHIC_CENTER["longitude"])

    # Project the centre.
    center_xy = project_many(
        pm,
        [(center_lat, center_lon)],
        args.width,
        args.height,
        args.split_norway,
        args.rotate_norway,
        min_x,
        min_y,
        width,
        height,
    )[0]

    # Build and project the mean-error ring.
    ring_latlon = geodesic_ring_latlon(center_lat, center_lon, MEAN_ERROR_DISTANCE_KM, n_points=360)
    ring_xy = project_many(
        pm,
        ring_latlon,
        args.width,
        args.height,
        args.split_norway,
        args.rotate_norway,
        min_x,
        min_y,
        width,
        height,
    )

    # Get a projected point along the ring in the due-east direction for the radius line/label.
    if _GEOD is not None:
        east_lon, east_lat, _ = _GEOD.fwd(center_lon, center_lat, 90.0, MEAN_ERROR_DISTANCE_KM * 1000.0)
        east_latlon = [(east_lat, east_lon)]
    else:
        east_latlon = [destination_point_approx(center_lat, center_lon, MEAN_ERROR_DISTANCE_KM, 90.0)]
    east_xy = project_many(
        pm,
        east_latlon,
        args.width,
        args.height,
        args.split_norway,
        args.rotate_norway,
        min_x,
        min_y,
        width,
        height,
    )[0]

    # Project shortest and longest points.
    shortest_latlon = [(float(item["latitude"]), float(item["longitude"])) for item in SHORTEST_DISTANCES]
    longest_latlon = [(float(item["latitude"]), float(item["longitude"])) for item in LONGEST_DISTANCES]

    shortest_xy = project_many(
        pm, shortest_latlon, args.width, args.height,
        args.split_norway, args.rotate_norway, min_x, min_y, width, height
    )
    longest_xy = project_many(
        pm, longest_latlon, args.width, args.height,
        args.split_norway, args.rotate_norway, min_x, min_y, width, height
    )

    overlay = []

    # Mean-error ring and radius line.
    overlay.append(svg_polyline(ring_xy, stroke=RING_COLOR, stroke_width=DEFAULT_RING_STROKE, fill="none"))
    overlay.append(svg_line(center_xy[0], center_xy[1], east_xy[0], east_xy[1], stroke=RING_COLOR,
                            stroke_width=DEFAULT_RADIUS_LINE_STROKE, dasharray="24,18"))

    # Centre marker and label.
    overlay.append(svg_circle(center_xy[0], center_xy[1], args.center_size, CENTER_FILL))
    center_label_x = center_xy[0] + DEFAULT_LABEL_DX
    center_label_y = center_xy[1] + DEFAULT_LABEL_DY
    overlay.append(
        svg_multiline_text(
            center_label_x,
            center_label_y,
            [GEOGRAPHIC_CENTER["label"], f"({center_lat:.6f}, {center_lon:.6f})"],
            font_size=args.font_size,
            fill=CENTER_FILL,
            weight="bold",
        )
    )

    # Radius label near the centre of the radius line.
    radius_mid_x = (center_xy[0] + east_xy[0]) / 2.0
    radius_mid_y = (center_xy[1] + east_xy[1]) / 2.0
    overlay.append(
        svg_text(
            radius_mid_x + DEFAULT_LABEL_DX,
            radius_mid_y + DEFAULT_LABEL_DY,
            f"Radius = {MEAN_ERROR_DISTANCE_KM:.3f} km",
            font_size=max(args.font_size, 30),
            fill=RING_COLOR,
            weight="bold",
        )
    )

    # Shortest-distance points + labels.
    for (x, y), item in zip(shortest_xy, SHORTEST_DISTANCES):
        overlay.append(svg_circle(x, y, args.marker_size, SHORTEST_FILL, stroke="white", stroke_width=2.5))
        overlay.append(
            svg_multiline_text(
                x + DEFAULT_LABEL_DX,
                y + DEFAULT_LABEL_DY,
                [str(item["speaker_id"]), f'{float(item["distance_km"]):.3f} km'],
                font_size=args.font_size,
                fill=SHORTEST_FILL,
                weight="bold",
            )
        )

    # Longest-distance points + labels.
    for (x, y), item in zip(longest_xy, LONGEST_DISTANCES):
        overlay.append(svg_circle(x, y, args.marker_size, LONGEST_FILL, stroke="white", stroke_width=2.5))
        overlay.append(
            svg_multiline_text(
                x + DEFAULT_LABEL_DX,
                y + DEFAULT_LABEL_DY,
                [str(item["speaker_id"]), f'{float(item["distance_km"]):.3f} km'],
                font_size=args.font_size,
                fill=LONGEST_FILL,
                weight="bold",
            )
        )

    # Simple legend in the upper-left corner of the rendered viewBox.
    legend_x = min_x + width * 0.03
    legend_y = min_y + height * 0.04
    legend_lines = [
        "Geographical centre",
        "Mean error ring",
        "3 shortest distances",
        "3 longest distances",
    ]
    leg_w, leg_h = estimate_text_box(legend_lines, args.font_size)
    overlay.append(svg_rect(legend_x, legend_y, leg_w + 70, leg_h + 58, LEGEND_BG, "#333333", 2.0))

    icon_x = legend_x + 24
    text_x = legend_x + 56
    row_y = legend_y + 36
    row_gap = args.font_size * 1.55

    # Legend row 1: centre
    overlay.append(svg_circle(icon_x, row_y - 7, args.center_size * 0.7, CENTER_FILL))
    overlay.append(svg_text(text_x, row_y, "Geographical centre", args.font_size, CENTER_FILL, weight="bold"))

    # Legend row 2: ring
    row_y += row_gap
    overlay.append(svg_line(icon_x - 12, row_y - 7, icon_x + 12, row_y - 7, RING_COLOR, 5.0, dasharray="14,10"))
    overlay.append(svg_text(text_x, row_y, "Mean error ring", args.font_size, RING_COLOR, weight="bold"))

    # Legend row 3: shortest
    row_y += row_gap
    overlay.append(svg_circle(icon_x, row_y - 7, args.marker_size * 0.75, SHORTEST_FILL, stroke="white", stroke_width=1.8))
    overlay.append(svg_text(text_x, row_y, "3 shortest distances", args.font_size, SHORTEST_FILL, weight="bold"))

    # Legend row 4: longest
    row_y += row_gap
    overlay.append(svg_circle(icon_x, row_y - 7, args.marker_size * 0.75, LONGEST_FILL, stroke="white", stroke_width=1.8))
    overlay.append(svg_text(text_x, row_y, "3 longest distances", args.font_size, LONGEST_FILL, weight="bold"))

    # Compose final SVG.
    svg_list.extend(overlay)
    head = pm.head_bit.format(str(args.width), str(args.height), min_x, min_y, width, height)
    end = pm.end_bit
    svg_text_output = head + "".join(svg_list) + end

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if out_path.lower().endswith(".svg"):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_text_output)
        print(f"[INFO] Saved SVG to: {os.path.abspath(out_path)}")
        return

    temp_svg = os.path.splitext(out_path)[0] + ".__temp__.svg"
    with open(temp_svg, "w", encoding="utf-8") as f:
        f.write(svg_text_output)

    if cairosvg is None:
        fallback_svg = os.path.splitext(out_path)[0] + ".svg"
        os.replace(temp_svg, fallback_svg)
        print("[WARN] CairoSVG is not available; saved SVG instead.")
        print(f"[INFO] Saved fallback SVG to: {os.path.abspath(fallback_svg)}")
        return

    try:
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
        print(f"[WARN] CairoSVG conversion failed ({e}); saved SVG instead.")
        print(f"[INFO] Saved fallback SVG to: {os.path.abspath(fallback_svg)}")


if __name__ == "__main__":
    main()