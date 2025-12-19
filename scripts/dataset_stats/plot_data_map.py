
#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from PIL import Image

# --- Resolve paths similarly to your existing script ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
new_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)

from dialect_mapper.plotter import plotter_methods  # assumes your local package layout

# --- CONFIG ---
CSV_PATH = os.path.join(current_dir, "dialect_counts.csv")  # change if needed
OUTPUT_MAP_PATH = os.path.join(current_dir, "card4_map.png")
FINAL_OUTPUT_WITH_COLORBAR = os.path.join(current_dir, "card4_map_with_colorbar.png")

# Use the same colormap you like
COLOR_MAP_NAME = "PuBuGn"  # e.g., "PuBuGn", "viridis", "plasma", etc.
COLOR_MAP_LEVELS = 256      # higher = smoother gradient
DEFAULT_COLOR = "#66cc99"   # used if a region is missing in the dict
FINAL_WIDTH = "2400"        # as strings to match your API
FINAL_HEIGHT = "3000"

# --- Helper to generate the region->value dict from CSV ---
def load_region_values_from_csv(csv_path):
    """
    Reads the CSV and returns:
      - region_to_value: dict {east|mid|north|west: mean_segment_duration_s}
      - vmin, vmax: min and max of the mean_segment_duration_s column
    """
    df = pd.read_csv(csv_path)
    # Normalize keys to lowercase to match your API usage ("north", "mid", "west", "east")
    df["dialect_cardinal4"] = df["dialect_cardinal4"].str.strip().str.lower()

    # Safety: ensure all four regions are present; if not, we’ll still proceed with what's there.
    required = {"east", "mid", "north", "west"}
    present = set(df["dialect_cardinal4"].unique())
    missing = required - present
    if missing:
        print(f"[WARN] Missing regions in CSV: {missing}. Proceeding with available ones.")

    # Build mapping to mean_segment_duration_s
    region_to_value = {
        row["dialect_cardinal4"]: float(row["total_duration_s"])
        for _, row in df.iterrows()
    }

    # Compute vmin/vmax for color scaling
    vmin = float(df["total_duration_s"].min())
    vmax = float(df["total_duration_s"].max())

    # covnert values to hours
    region_to_value = {k: v / 3600.0 for k, v in region_to_value.items()}
    vmin /= 3600.0
    vmax /= 3600.0

    return region_to_value, vmin, vmax


def try_plot_with_colorbar_direct(pm, region_to_value, vmin, vmax):
    """
    If pm.plot_card4_dialect_regions returns a Matplotlib figure/axes, attach a colorbar directly.
    If not, return False and we will do the fallback (compose images).
    """
    try:
        # Some libraries return fig/ax; others only write files.
        # We'll call the method and check if it returns something usable.
        result = pm.plot_card4_dialect_regions(
            output_svg_filepath=OUTPUT_MAP_PATH,
            dia_region_to_value=region_to_value,
            color_map_name=COLOR_MAP_NAME,
            color_map_levels=COLOR_MAP_LEVELS,
            max_region_value=vmax,   # IMPORTANT: align map scaling with the data
            default_color=DEFAULT_COLOR,
            final_width=FINAL_WIDTH,
            final_height=FINAL_HEIGHT
        )

        # If result is a Matplotlib Figure or (fig, ax), add a colorbar:
        # We try a few patterns to stay robust.
        fig = None
        ax = None

        if hasattr(result, "add_axes") or hasattr(result, "colorbar"):
            # Likely a Figure
            fig = result
        elif isinstance(result, tuple) and len(result) == 2:
            fig, ax = result
        elif hasattr(result, "figure"):
            ax = result
            fig = ax.figure

        if fig is not None:
            # Create a scalar mappable to anchor the colorbar
            cmap = plt.get_cmap(COLOR_MAP_NAME)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            # Add a colorbar at the right
            cbar = fig.colorbar(sm, ax=fig.axes, orientation="vertical", fraction=0.025, pad=0.02)
            cbar.set_label("Total segment duration (hrs)", rotation=90)

            # Save the final output
            fig.savefig(FINAL_OUTPUT_WITH_COLORBAR, dpi=300, bbox_inches="tight")
            print(f"[INFO] Saved map with colorbar: {FINAL_OUTPUT_WITH_COLORBAR}")
            return True

        # If we reached here, the method didn’t return a Matplotlib object we can use.
        return False

    except Exception as e:
        print(f"[WARN] Direct colorbar attachment failed: {e}")
        return False



def create_colorbar_image(
    vmin,
    vmax,
    height_px=3000,
    width_px=200,
    label="Total segment duration (hrs)",
    label_fontsize=32,      # <-- control axis label font size
    tick_fontsize=28,       # <-- control tick labels font size
    tick_values=None,       # <-- optional: set explicit tick positions
    tick_format=None        # <-- optional: callable or fmt string for labels
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FixedLocator, FuncFormatter

    dpi = 300
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)  # keep transparent
    ax = fig.add_axes([0.25, 0.05, 0.2, 0.9])

    cmap = plt.get_cmap(COLOR_MAP_NAME)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=ax, orientation="vertical")

    # --- Label and tick font sizes ---
    cbar.set_label(label, rotation=90, size=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # --- Optional: exact tick positions and formatting ---
    if tick_values is not None:
        # tick_values must be in the same scale as vmin/vmax
        cbar.locator = FixedLocator(tick_values)
        cbar.update_ticks()

    if tick_format is not None:
        # tick_format can be a format string like "{:.3f}s" or a callable
        if isinstance(tick_format, str):
            fmt = lambda x, pos: tick_format.format(x)
        else:
            fmt = tick_format
        cbar.formatter = FuncFormatter(fmt)
        cbar.update_ticks()

    colorbar_path = os.path.join(current_dir, "colorbar.png")
    fig.savefig(colorbar_path, dpi=dpi, bbox_inches="tight", transparent=True, facecolor='none')
    plt.close(fig)
    return colorbar_path




def compose_map_and_colorbar(map_path, colorbar_path, output_path, gap_px=40):
    # Open as RGBA to preserve transparency
    map_img = Image.open(map_path).convert("RGBA")
    cbar_img = Image.open(colorbar_path).convert("RGBA")

    # Resize colorbar to match map height while preserving aspect ratio
    map_w, map_h = map_img.size
    cbar_w, cbar_h = cbar_img.size
    new_cbar_h = map_h
    new_cbar_w = int(cbar_w * (new_cbar_h / cbar_h))
    cbar_img = cbar_img.resize((new_cbar_w, new_cbar_h), Image.LANCZOS)

    total_w = map_w + gap_px + new_cbar_w
    total_h = map_h

    # Transparent RGBA canvas
    out = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))

    # Paste preserving alpha (use the image itself as the mask)
    out.paste(map_img, (0, 0), map_img)
    out.paste(cbar_img, (map_w + gap_px, 0), cbar_img)

    # Save with transparency
    out.save(output_path)
    print(f"[INFO] Saved composed image (transparent): {output_path}")



def main():
    # Load data
    region_to_value, vmin, vmax = load_region_values_from_csv(CSV_PATH)

    # Create plotter and produce the base map
    pm = plotter_methods()
    # First attempt: if we can attach colorbar directly (Scenario A)
    attached = try_plot_with_colorbar_direct(pm, region_to_value, vmin, vmax)
    if attached:
        return

    # Fallback (Scenario B): just generate the map, then stitch a separate colorbar
    pm.plot_card4_dialect_regions(
        output_svg_filepath=OUTPUT_MAP_PATH,
        dia_region_to_value=region_to_value,
        color_map_name=COLOR_MAP_NAME,
        color_map_levels=COLOR_MAP_LEVELS,
        max_region_value=vmax,   # match color scaling to data max
        default_color=DEFAULT_COLOR,
        final_width=FINAL_WIDTH,
        final_height=FINAL_HEIGHT
    )
    # Create a standalone colorbar image at matching height
    # Note: FINAL_HEIGHT is a string; cast to int for pixel math
    colorbar_path = create_colorbar_image(vmin=vmin, vmax=vmax, height_px=int(FINAL_HEIGHT), width_px=220)
    # Compose map + colorbar side-by-side
    compose_map_and_colorbar(OUTPUT_MAP_PATH, colorbar_path, FINAL_OUTPUT_WITH_COLORBAR)


if __name__ == "__main__":
    main()
