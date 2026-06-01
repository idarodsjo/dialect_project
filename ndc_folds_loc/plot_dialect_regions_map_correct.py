# scripts/plot_dialects_by_kommune.py

import os
import re
import json
import pandas as pd
from pathlib import Path
from collections import OrderedDict, Counter, defaultdict

# Geometry support used to detect neighboring municipalities
from shapely.geometry import shape

# 1) Import your plotter class
# Adjust this import if your package path differs
from dialect_mapper.plotter import plotter_methods


# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = "/home/idatro/dialect_project/repo/DialectMapper/dialect_mapper/development/final_pass_dialects_supplemental.csv"

# IMPORTANT:
# Use SVG first so we can strip municipality border lines afterward.
OUTPUT_SVG_PATH = os.path.join(
    PROJECT_ROOT,
    "dialect_map_by_municipality.svg"
)
OUTPUT_SVG_PATH = "/home/idatro/dialect_project/ndc_folds_loc/map_regions/dialect_map_by_municipality.svg"

# Optional PDF export after border removal
WRITE_PDF_COPY = True

# Choose a categorical palette for discrete dialects (tab20 works well)
COLOR_MAP_NAME = "tab20b"

# Final canvas size (in pixels) – tweak as you like
FINAL_WIDTH = 1200
FINAL_HEIGHT = 1800

# If a municipality is not in the mapping, this fill color is used
DEFAULT_COLOR = "#eeeeee"

# ----------------------------
# Optional automatic cleanup
# ----------------------------
# Set to True to automatically fix isolated outlier municipalities
AUTO_CORRECT_OUTLIERS = True

# Require at least this many neighbors with known values before correction
MIN_NEIGHBORS_FOR_CORRECTION = 3

# Fraction of neighbors that must agree on one value for correction to happen
# 0.75 means at least 75% of known neighbors must agree
MAJORITY_THRESHOLD = 0.75

# Keep this at 1 to avoid over-smoothing real dialect boundaries
SMOOTHING_ROUNDS = 1


# ----------------------------
# Helpers
# ----------------------------
def normalize(s: str) -> str:
    """Trim whitespace and normalize small inconsistencies."""
    return (s or "").strip()


# Known historical-to-current municipality redirects (extend as needed).
# These map CSV municipality names to the current name used in kommuner_komprimert.json.
HISTORICAL_TO_CURRENT = {
    # Møre og Romsdal / Vestland etc.
    "Eide": "Hustadvika",
    "Fræna": "Hustadvika",
    "Flora": "Kinn",
    "Eid": "Stad",
    "Selje": "Stad",
    "Hornindal": "Volda",
    "Jølster": "Sunnfjord",
    "Førde": "Sunnfjord",

    # Øygarden/Alver/Bjørnafjorden mergers
    "Fjell": "Øygarden",
    "Sund": "Øygarden",
    "Lindås": "Alver",
    "Meland": "Alver",
    "Radøy": "Alver",
    "Fusa": "Bjørnafjorden",

    # Ålesund (2019/2020 mergers)
    "Sandøy": "Ålesund",
    "Skodje": "Ålesund",

    # Trøndelag mergers
    "Rissa": "Indre Fosen",
    "Agdenes": "Orkland",
    "Meldal": "Orkland",
    "Hemne": "Heim",

    # Viken/Oslo region
    "Skedsmo": "Lillestrøm",
    "Sørum": "Lillestrøm",
    "Fet": "Lillestrøm",

    # Agder mergers
    "Søgne": "Kristiansand",
    "Songdalen": "Kristiansand",

    # Senja region
    "Torsken": "Senja",

    # Legacy / typos occasionally seen in CSV
    "Nord-reisa": "Nordreisa",
    "Kværangen": "Kvænangen",
    "Hyllestand": "Hyllestad",
    "Vulda": "Volda",
    "Hisøy": "Arendal",
    "Fana": "Bergen",  # very historic (borough today), included just in case
}


def apply_name_corrections(muni: str) -> str:
    muni = normalize(muni)
    return HISTORICAL_TO_CURRENT.get(muni, muni)


def make_kommune_key(row) -> str:
    """
    Make the key that plotter._process_features will request for get_color(...).

    Special-case 'Herøy' because plotter renames it internally to avoid collisions:
      - Nordland -> 'Herøy_Helgelandsk'
      - Møre og Romsdal -> 'Herøy_Nordvestlandsk'

    Everything else: just use the (possibly corrected) municipality name.
    """
    muni = apply_name_corrections(row["municipality"])
    county = normalize(row.get("county", ""))

    if muni == "Herøy":
        if county == "Nordland":
            return "Herøy_Helgelandsk"
        elif county == "Møre og Romsdal":
            return "Herøy_Nordvestlandsk"

    return muni


def make_feature_key(feature) -> str:
    """
    Build the same kind of municipality key for GeoJSON features as for CSV rows.
    """
    props = feature["properties"]
    muni = normalize(props.get("navn", ""))

    # Try a few possible county property names in the GeoJSON
    county = normalize(
        props.get("fylke")
        or props.get("county")
        or props.get("fylkesnavn")
        or props.get("fylke_navn")
        or ""
    )

    if muni == "Herøy":
        if county == "Nordland":
            return "Herøy_Helgelandsk"
        elif county == "Møre og Romsdal":
            return "Herøy_Nordvestlandsk"

    return muni


def build_neighbor_graph(features):
    """
    Build a neighbor graph from municipality polygons.

    Two municipalities count as neighbors if they share a boundary segment
    with positive length (not just a corner point).
    """
    geoms = []
    keys = []

    for feat in features:
        keys.append(make_feature_key(feat))
        geoms.append(shape(feat["geometry"]))

    neighbors = defaultdict(set)

    for i in range(len(geoms)):
        g1 = geoms[i]
        b1 = g1.bounds  # (minx, miny, maxx, maxy)

        for j in range(i + 1, len(geoms)):
            g2 = geoms[j]
            b2 = g2.bounds

            # Fast bounding-box rejection
            separated = (
                b1[2] < b2[0] or  # g1.maxx < g2.minx
                b2[2] < b1[0] or  # g2.maxx < g1.minx
                b1[3] < b2[1] or  # g1.maxy < g2.miny
                b2[3] < b1[1]     # g2.maxy < g1.miny
            )
            if separated:
                continue

            # Shared boundary with positive length => true neighbors
            inter = g1.boundary.intersection(g2.boundary)
            if not inter.is_empty and inter.length > 0:
                neighbors[keys[i]].add(keys[j])
                neighbors[keys[j]].add(keys[i])

    return neighbors


def smooth_outlier_municipalities(
    kommune_to_value,
    neighbor_graph,
    min_neighbors=3,
    majority_threshold=0.75,
    rounds=1
):
    """
    Replace isolated outlier municipality values with the dominant value among
    neighbors, provided the agreement is strong enough.

    Returns:
        new_mapping, list_of_changes
    """
    current = dict(kommune_to_value)
    all_changes = []

    for _ in range(rounds):
        updated = dict(current)
        round_changes = []

        for muni, val in current.items():
            neighs = neighbor_graph.get(muni, set())
            neigh_vals = [current[n] for n in neighs if n in current]

            if len(neigh_vals) < min_neighbors:
                continue

            counts = Counter(neigh_vals)
            majority_val, majority_count = counts.most_common(1)[0]
            majority_fraction = majority_count / len(neigh_vals)

            # Only flip if it is clearly an outlier among its neighbors
            if val != majority_val and majority_fraction >= majority_threshold:
                updated[muni] = majority_val
                round_changes.append(
                    (muni, val, majority_val, majority_fraction, len(neigh_vals))
                )

        current = updated
        all_changes.extend(round_changes)

        if not round_changes:
            break

    return current, all_changes


def remove_svg_borders(svg_path: str):
    """
    Remove municipality border lines by stripping stroke / stroke-width from
    the generated SVG.
    """
    svg_file = Path(svg_path)
    text = svg_file.read_text(encoding="utf-8")

    # Replace inline style strokes
    text = re.sub(r"stroke\\s*:\\s*[^;\"']+;?", "stroke:none;", text)
    text = re.sub(r"stroke-width\\s*:\\s*[^;\"']+;?", "stroke-width:0;", text)

    # Replace attribute strokes
    text = re.sub(r'stroke="[^"]+"', 'stroke="none"', text)
    text = re.sub(r'stroke-width="[^"]+"', 'stroke-width="0"', text)

    svg_file.write_text(text, encoding="utf-8")


def write_pdf_from_svg(svg_path: str, pdf_path: str):
    """
    Convert cleaned SVG to PDF if cairosvg is available.
    """
    try:
        import cairosvg
        cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
        print(f"[INFO] Wrote borderless PDF to {pdf_path}")
    except ImportError:
        print("[WARN] cairosvg is not installed, so PDF copy was not written.")
    except Exception as e:
        print(f"[WARN] Failed to convert SVG to PDF: {e}")


# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load CSV with municipality → dialect
    df = pd.read_csv(CSV_PATH)
    required_cols = {"municipality", "county", "dialect"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    # Normalize text
    for col in ["municipality", "county", "dialect"]:
        df[col] = df[col].astype(str).map(normalize)

    # 2) Build unique dialect list and encode categories into ints
    dialects = list(OrderedDict.fromkeys(df["dialect"].tolist()))  # preserve first appearance
    dialect_to_id = {d: i for i, d in enumerate(dialects)}

    # 3) Build kommune → value mapping for the plotter
    kommune_to_value = {}
    for _, row in df.iterrows():
        key = make_kommune_key(row)
        val = dialect_to_id[row["dialect"]]
        kommune_to_value[key] = val  # if duplicates appear, later rows win

    # Instantiate plotter now so we can access the municipalities GeoJSON
    pm = plotter_methods()

    # 4) Optional automatic cleanup of isolated outlier municipalities
    if AUTO_CORRECT_OUTLIERS:
        print("[INFO] Building municipality neighbor graph...")
        neighbor_graph = build_neighbor_graph(pm.kommuner_json["features"])

        kommune_to_value, changes = smooth_outlier_municipalities(
            kommune_to_value,
            neighbor_graph,
            min_neighbors=MIN_NEIGHBORS_FOR_CORRECTION,
            majority_threshold=MAJORITY_THRESHOLD,
            rounds=SMOOTHING_ROUNDS,
        )

        if changes:
            print("[INFO] Auto-corrected municipality outliers:")
            for muni, old_val, new_val, frac, n_neigh in changes:
                old_dialect = dialects[old_val]
                new_dialect = dialects[new_val]
                print(
                    f"   - {muni}: {old_dialect} -> {new_dialect} "
                    f"(neighbor agreement={frac:.2f}, n_neighbors={n_neigh})"
                )
        else:
            print("[INFO] No municipality outliers were auto-corrected.")

    # 5) Sanity check against municipalities present in the GeoJSON
    geo_munis = {feat["properties"]["navn"] for feat in pm.kommuner_json["features"]}

    # The plotter internally renames Herøy to suffixed keys, so emulate those here
    geo_munis_with_heroy = set(geo_munis)
    geo_munis_with_heroy.add("Herøy_Helgelandsk")
    geo_munis_with_heroy.add("Herøy_Nordvestlandsk")

    missing_in_geo = sorted(set(kommune_to_value.keys()) - geo_munis_with_heroy)
    if missing_in_geo:
        print("[WARN] The following municipalities from CSV did not match the GeoJSON names "
              "(consider adding to HISTORICAL_TO_CURRENT or fixing typos):")
        for m in missing_in_geo:
            print("   -", m)

    # 6) Render to SVG first
    pm.plot_kommune_regions(
        output_svg_filepath=OUTPUT_SVG_PATH,
        kommune_region_to_value=kommune_to_value,
        color_map_name=COLOR_MAP_NAME,
        color_map_levels=max(10, len(dialects)),  # ensure enough discrete bins
        max_region_value=max(0, len(dialects) - 1),
        default_color=DEFAULT_COLOR,
        final_width=str(FINAL_WIDTH),
        final_height=str(FINAL_HEIGHT),
    )

    print(f"[INFO] Wrote raw SVG map to {OUTPUT_SVG_PATH}")

    # 7) Remove municipality border lines from the SVG
    remove_svg_borders(OUTPUT_SVG_PATH)
    print(f"[INFO] Removed municipality border lines in {OUTPUT_SVG_PATH}")

    # 8) Optionally also create a borderless PDF copy
    if WRITE_PDF_COPY:
        pdf_path = os.path.splitext(OUTPUT_SVG_PATH)[0] + ".pdf"
        write_pdf_from_svg(OUTPUT_SVG_PATH, pdf_path)

    # 9) Also write a small legend (dialect → hex color) next to the map for reference
    from matplotlib import colormaps
    import matplotlib.colors as mcolors

    cmap = colormaps[COLOR_MAP_NAME]
    levels = max(10, len(dialects))

    legend = []
    for d, idx in dialect_to_id.items():
        # mimic the discrete step your plotter uses
        value = (idx / max(1, len(dialects) - 1)) if len(dialects) > 1 else 0

        # snap to discrete bins
        value = int(levels * value) * 1.0 / (levels - 1) if levels > 1 else 0

        hexcolor = mcolors.to_hex(cmap(value))
        legend.append({"dialect": d, "color": hexcolor})

    legend_df = pd.DataFrame(legend)
    legend_path = os.path.splitext(OUTPUT_SVG_PATH)[0] + "_legend.csv"
    legend_df.to_csv(legend_path, index=False, encoding="utf-8")

    print(f"[INFO] Wrote legend to {legend_path}")
    print(f"[INFO] Done.")


if __name__ == "__main__":
    main()