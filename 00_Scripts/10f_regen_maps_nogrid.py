"""
10f — Regenerate the national panel maps without coordinate frames
==================================================================
The panel maps produced by `10a_figures_maps.py` inherit matplotlib's default
axis decoration: every panel carries a full easting/northing tick frame. At the
figure widths used in the manuscript (12 cm across five panels) those labels are
larger than the map features they surround, and they eat roughly a fifth of each
panel's area. This script re-renders the same two figures with the frames off
and moves all cartographic furniture — colour key, scale bar, north arrow — into
the empty sixth cell of the 2x3 grid, so no element is ever drawn over
territory.

Regenerates, into Manuscript/figures/main/:
  Fig11_probability_maps.png   — 5-panel continuous probability (per-size)
  Fig12_binary_flood_maps.png  — 5-panel CSI-thresholded binary extent

Deliberately NOT regenerated: Fig12 (ensemble) and Fig13 (agreement) draw
municipality boundaries and the national flood-zone reference layers from
`Vectors/`, which is not present on this machine. Re-rendering them here would
silently drop those layers, so the originals are left in place.

Paths are resolved relative to this file, not hardcoded.
"""

import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from rasterio.enums import Resampling
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZE = os.environ.get("MAP_SIZE", "10x")
PROB_DIR = ROOT / "06_Probability_Maps" / SIZE
CLSF_DIR = ROOT / "07_Classified_Maps"
OUT_DIR = ROOT / "Manuscript" / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["RF", "XGBoost", "CatBoost", "DNN", "Ensemble_mean"]
LABELS = {
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "DNN": "Deep Neural Network",
    "Ensemble_mean": "Ensemble (mean)",
}
PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)"]

# Palettes carried over unchanged from 10a so the figures stay comparable.
CMAP_PROB = LinearSegmentedColormap.from_list(
    "fsm_prob",
    [(0.00, "#f7f7f7"), (0.25, "#fee08b"), (0.50, "#fc8d59"),
     (0.75, "#d73027"), (1.00, "#4d0000")],
)
CMAP_BIN = ListedColormap(["#d1e5f0", "#b2182b"])
NORM_BIN = BoundaryNorm([-0.5, 0.5, 1.5], CMAP_BIN.N)

NODATA = -9999.0
DSAMP = 8


def read_raster(path, factor=DSAMP, resampling=Resampling.average):
    """Read at reduced resolution, nodata -> NaN.

    Note the out_shape form: `out_shape=(h, w)` with a scalar band index.
    Writing `src.read(1, out_shape=(1, h, w))[0]` looks equivalent but is not —
    a scalar band index already returns a 2-D array, so the trailing [0] slices
    off the first *row*. That bug silently reduced an analysis to six pixels
    elsewhere in this pipeline.
    """
    with rasterio.open(path) as src:
        h = max(1, src.height // factor)
        w = max(1, src.width // factor)
        data = src.read(1, out_shape=(h, w), resampling=resampling).astype("float32")
        nd = src.nodata if src.nodata is not None else NODATA
        ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    data[data == nd] = np.nan
    return data, ext


def panel_grid():
    """2x3 grid: five map panels, sixth cell reserved for the key."""
    fig = plt.figure(figsize=(15.0, 9.0))
    gs = fig.add_gridspec(2, 3, hspace=0.10, wspace=0.03,
                          left=0.012, right=0.988, top=0.955, bottom=0.02)
    axes = [fig.add_subplot(gs[r, c]) for r, c in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]]
    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")
    key_ax = fig.add_subplot(gs[1, 2])
    key_ax.set_axis_off()
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)
    return fig, axes, key_ax


def panel_label(ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.85, ec="none"),
            zorder=20)


def north_arrow(ax, x, y, h=0.085):
    """Simple filled triangle with an N above it, in axes coordinates."""
    ax.add_patch(mpatches.Polygon(
        [[x, y + h], [x - h * 0.34, y - h * 0.28], [x, y - h * 0.08],
         [x + h * 0.34, y - h * 0.28]],
        closed=True, fc="#1a1a1a", ec="#1a1a1a", lw=0.6,
        transform=ax.transAxes, zorder=25, clip_on=False))
    ax.text(x, y + h + 0.018, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", zorder=25)


def scale_bar(ax, x0, y0, width, ext, map_ax, fig, km_target=50):
    """Horizontal bar in the key axes, labelled in km.

    The bar has to represent distance on the *map* panels, but it is drawn in a
    different axes. `aspect='equal'` shrinks a map panel's box to the data
    aspect, so its rendered width is not necessarily the column width. The
    conversion is therefore taken from the drawn geometry rather than assumed:
    render once, measure both axes in device pixels, and scale accordingly.
    """
    fig.canvas.draw()
    map_px = map_ax.get_window_extent().width
    key_px = ax.get_window_extent().width
    span_km = (ext[1] - ext[0]) / 1000.0
    frac = (km_target / span_km) * (map_px / key_px)
    if frac > width:                      # too long for the cell — halve it
        km_target /= 2.0
        frac = (km_target / span_km) * (map_px / key_px)
    km = km_target
    seg = frac / 2.0
    for i, fc in enumerate(["#1a1a1a", "#ffffff"]):
        ax.add_patch(mpatches.Rectangle(
            (x0 + i * seg, y0), seg, 0.028, fc=fc, ec="#1a1a1a", lw=0.7,
            transform=ax.transAxes, zorder=25, clip_on=False))
    for v, lab in [(0.0, "0"), (0.5, f"{km/2:.0f}"), (1.0, f"{km:.0f} km")]:
        ax.text(x0 + v * frac, y0 - 0.012, lab, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, zorder=25)


def finish_key(key_ax, ext, map_ax, handles=None, title=None, mappable=None,
               cbar_label=None, fig=None):
    """Fill the sixth cell: colour key on top, then scale bar and north arrow."""
    if handles is not None:
        key_ax.legend(handles=handles, title=title, title_fontsize=10.5,
                      fontsize=10, loc="upper center",
                      bbox_to_anchor=(0.5, 0.97), framealpha=1.0,
                      edgecolor="#999999", handlelength=1.6, handleheight=1.4)
    if mappable is not None:
        cax = key_ax.inset_axes([0.10, 0.62, 0.80, 0.055])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label, fontsize=10, labelpad=5)
        cb.ax.tick_params(labelsize=9)
    scale_bar(key_ax, 0.10, 0.34, 0.80, ext, map_ax, fig)
    north_arrow(key_ax, 0.78, 0.13)
    key_ax.text(0.10, 0.16, "EPSG:3794\nSlovenia 1996 / Slovene National Grid",
                transform=key_ax.transAxes, fontsize=8.5, va="bottom",
                ha="left", color="#333333", linespacing=1.5)


# ── Fig 10 — probability maps ────────────────────────────────────────────────
print(f"Fig10 probability maps ({SIZE}) ...")
fig, axes, key_ax = panel_grid()
im = None
for i, (model, ax) in enumerate(zip(MODELS, axes)):
    data, ext = read_raster(PROB_DIR / f"FSM_{model}_probability.tif")
    im = ax.imshow(data, extent=ext, cmap=CMAP_PROB, vmin=0, vmax=1,
                   origin="upper", interpolation="bilinear")
    ax.set_title(LABELS[model], fontsize=11.5, fontweight="bold", pad=4)
    panel_label(ax, PANEL_LETTERS[i])
finish_key(key_ax, ext, axes[0], mappable=im, fig=fig,
           cbar_label="Flood susceptibility probability")
out = OUT_DIR / "Fig11_probability_maps.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved {out.name}")

# ── Fig 11 — binary maps ─────────────────────────────────────────────────────
print("Fig11 binary flood maps ...")
fig, axes, key_ax = panel_grid()
for i, (model, ax) in enumerate(zip(MODELS, axes)):
    data, ext = read_raster(CLSF_DIR / f"FSM_{model}_binary.tif",
                            resampling=Resampling.nearest)
    ax.imshow(data, extent=ext, cmap=CMAP_BIN, norm=NORM_BIN,
              origin="upper", interpolation="nearest")
    ax.set_title(LABELS[model], fontsize=11.5, fontweight="bold", pad=4)
    panel_label(ax, PANEL_LETTERS[i])
patches_bin = [
    mpatches.Patch(fc="#d1e5f0", ec="#777777", lw=0.4, label="Not flood-prone"),
    mpatches.Patch(fc="#b2182b", ec="#777777", lw=0.4, label="Flood-prone"),
]
finish_key(key_ax, ext, axes[0], handles=patches_bin,
           title="CSI-thresholded classification", fig=fig)
out = OUT_DIR / "Fig12_binary_flood_maps.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  saved {out.name}")
print("done.")
