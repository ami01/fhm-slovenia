#!/usr/bin/env python3
"""
10e_study_area_map.py — Figure 1 of the manuscript: study area.

Hillshaded national DEM, the rasterised flood-extent inventory used as the
dependent variable, and the major drainage network derived from flow
accumulation. Cartographic furniture (legend, scale bar, north arrow, graticule
labels) is placed in the emptiest corner of the frame, chosen programmatically
from the land mask, so that no part of the territory is obscured.

Inputs  (01_Input_Features/, 12.5 m, EPSG:3794)
    elevation.tif
    Flooded_NoFlood.tif
    flow_accumulation.tif

Output
    Manuscript/figures/main/Fig01_study_area.png

Paths are resolved relative to this file, so the script is machine-independent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (LightSource, LinearSegmentedColormap,
                               ListedColormap, Normalize)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "01_Input_Features"
OUT = ROOT / "Manuscript" / "figures" / "main" / "Fig01_study_area.png"

# Display grid. The native raster is 19909 x 12996 at 12.5 m; a 10x decimation
# gives 125 m display cells, ample for a 2-column figure and cheap to read.
DECIMATE = 8
DPI = 400
RIVER_PCTL = 99.70   # flow-accumulation percentile defining the drawn network
FLOOD_DILATE = 1     # display-cell dilation so the 5.6 % flood class stays legible


def log(msg: str) -> None:
    print(f"[10e] {msg}", flush=True)


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------
def read_decimated(path: Path, dec: int) -> tuple[np.ndarray, float, tuple]:
    """Read a raster at 1/dec resolution.

    NOTE: pass out_shape as (h, w) with a scalar band index. Writing
    ``src.read(1, out_shape=(1, h, w))[0]`` silently slices off the first *row*
    instead of the first band — a bug that has bitten this codebase before.
    """
    with rasterio.open(path) as src:
        h, w = src.height // dec, src.width // dec
        arr = src.read(1, out_shape=(h, w), resampling=Resampling.average)
        nodata = src.nodata
        bounds = src.bounds
    return arr, nodata, bounds


def read_decimated_max(path: Path, dec: int, rows_per_chunk: int = 40) -> np.ndarray:
    """Block-maximum decimation by a factor of dec.

    rasterio refuses Resampling.max on plain reads (warp only), and nearest
    would drop most of the 5.6 % flood class, so the reduction is done here:
    the raster is streamed in horizontal strips of dec*rows_per_chunk native
    rows and each dec x dec block is reduced to its maximum.
    """
    with rasterio.open(path) as src:
        h, w = src.height // dec, src.width // dec
        out = np.zeros((h, w), dtype=np.float32)
        step = rows_per_chunk
        for r0 in range(0, h, step):
            r1 = min(r0 + step, h)
            win = Window(0, r0 * dec, w * dec, (r1 - r0) * dec)
            blk = src.read(1, window=win).astype(np.float32)
            blk = blk.reshape(r1 - r0, dec, w, dec)
            out[r0:r1] = blk.max(axis=(1, 3))
        return out


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Binary dilation by n cells in 4-connectivity, without scipy."""
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        )
    return out


def outline(mask: np.ndarray) -> np.ndarray:
    """One-cell border of a binary mask."""
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    )
    return mask & ~eroded


# --------------------------------------------------------------------------
# Furniture placement
# --------------------------------------------------------------------------
def emptiest_corner(land: np.ndarray, frac: float = 0.30) -> str:
    """Return the corner ('nw','ne','sw','se') with the least land coverage."""
    h, w = land.shape
    ch, cw = int(h * frac), int(w * frac)
    cover = {
        "nw": land[:ch, :cw].mean(),
        "ne": land[:ch, -cw:].mean(),
        "sw": land[-ch:, :cw].mean(),
        "se": land[-ch:, -cw:].mean(),
    }
    for k, v in sorted(cover.items(), key=lambda kv: kv[1]):
        log(f"  corner {k}: {v:.1%} land")
    return min(cover, key=cover.get)


def main() -> int:
    if not FEATURES.exists():
        log(f"ERROR: {FEATURES} not found — run 00_Scripts/_repath.py first.")
        return 1

    log(f"reading elevation at 1/{DECIMATE} …")
    dem, dem_nd, bounds = read_decimated(FEATURES / "elevation.tif", DECIMATE)
    land = np.isfinite(dem) & (dem != dem_nd) & (dem > -1000)
    log(f"  display grid {dem.shape[1]} x {dem.shape[0]}, land {land.mean():.1%}")

    log("reading flood inventory …")
    flood_raw = read_decimated_max(FEATURES / "Flooded_NoFlood.tif", DECIMATE)
    flood = (flood_raw == 1) & land
    if FLOOD_DILATE:
        flood = dilate(flood, FLOOD_DILATE) & land
    log(f"  flood cells shown: {flood.sum():,}")

    log("reading flow accumulation …")
    facc = read_decimated_max(FEATURES / "flow_accumulation.tif", DECIMATE)
    facc = np.where(land, facc, np.nan)
    thr = np.nanpercentile(facc, RIVER_PCTL)
    rivers = np.nan_to_num(facc, nan=0.0) > thr
    rivers = dilate(rivers, 1) & land
    log(f"  river threshold {thr:,.0f}; cells {rivers.sum():,}")

    # ---------------------------------------------------------------- render
    dem_m = np.where(land, dem, np.nan)
    vmin, vmax = 0.0, float(np.nanpercentile(dem_m, 99.9))

    ls = LightSource(azdeg=315, altdeg=45)
    filled = np.where(land, dem, np.nanmin(dem_m))
    shade = ls.hillshade(filled, vert_exag=6, dx=12.5 * DECIMATE, dy=12.5 * DECIMATE)

    x0, y0, x1, y1 = bounds.left, bounds.bottom, bounds.right, bounds.top
    extent = (x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000)  # km, national grid

    fig, ax = plt.subplots(figsize=(9.0, 6.1))
    ax.set_facecolor("white")

    # Hypsometric ramp. matplotlib's built-in "terrain" starts at dark blue,
    # which renders the Pannonian lowlands as if they were water and directly
    # competes with the drainage network — hence a custom green-to-white ramp.
    terrain = LinearSegmentedColormap.from_list(
        "hypsometric",
        [
            (0.00, "#3f7f3f"),
            (0.12, "#88b96a"),
            (0.30, "#d6cf90"),
            (0.50, "#c39a68"),
            (0.72, "#8d6b4e"),
            (0.88, "#b9a99c"),
            (1.00, "#ffffff"),
        ],
    )
    rgb = terrain(Normalize(vmin, vmax)(dem_m))
    rgb[..., :3] *= (0.42 + 0.58 * shade)[..., None]
    rgb[..., 3] = np.where(land, 1.0, 0.0)
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="bilinear")

    # rivers, then flood on top
    riv = np.ma.masked_where(~rivers, np.ones_like(dem))
    ax.imshow(riv, extent=extent, origin="upper",
              cmap=ListedColormap(["#1f6fb4"]), interpolation="nearest", alpha=0.9)

    fl = np.ma.masked_where(~flood, np.ones_like(dem))
    ax.imshow(fl, extent=extent, origin="upper",
              cmap=ListedColormap(["#d7191c"]), interpolation="nearest")

    border = outline(land)
    bd = np.ma.masked_where(~border, np.ones_like(dem))
    ax.imshow(bd, extent=extent, origin="upper",
              cmap=ListedColormap(["#222222"]), interpolation="nearest", alpha=0.85)

    ax.set_xlabel("Easting (km, D96/TM — EPSG:3794)", fontsize=9)
    ax.set_ylabel("Northing (km)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, color="0.72", lw=0.35, ls=":", alpha=0.55)
    ax.set_axisbelow(False)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    # ------------------------------------------------------- furniture
    # Everything cartographic lives inside one panel, sized and positioned in
    # the emptiest corner of the land mask so no territory is covered. The
    # elements are laid out inside that panel rather than anchored
    # independently, which is what previously made them collide.
    corner = emptiest_corner(land)
    log(f"  placing furniture in {corner.upper()}")

    pw, ph = 0.335, 0.265               # panel size, axes fraction
    px = 0.012 if corner in ("nw", "sw") else 1.0 - pw - 0.012
    py = 1.0 - ph - 0.012 if corner in ("nw", "ne") else 0.012

    panel = ax.inset_axes([px, py, pw, ph], zorder=30)
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.set_xticks([])
    panel.set_yticks([])
    panel.set_facecolor("white")
    panel.patch.set_alpha(0.95)
    for sp in panel.spines.values():
        sp.set_linewidth(0.7)
        sp.set_color("0.35")

    # -- legend keys (top third of the panel)
    keys = [
        ("patch", "#d7191c", "Flood inventory (dependent variable)"),
        ("line", "#1f6fb4", "Major drainage network"),
        ("line", "#222222", "National boundary"),
    ]
    for i, (kind, colour, label) in enumerate(keys):
        yk = 0.93 - i * 0.115
        if kind == "patch":
            panel.add_patch(Rectangle((0.045, yk - 0.032), 0.075, 0.062,
                                      facecolor=colour, edgecolor="none",
                                      transform=panel.transAxes))
        else:
            panel.plot([0.045, 0.12], [yk, yk], color=colour, lw=1.8,
                       transform=panel.transAxes, solid_capstyle="butt")
        panel.text(0.145, yk, label, fontsize=7.4, va="center", ha="left",
                   transform=panel.transAxes)

    # -- elevation colour bar (middle band). The bar stops at x = 0.78 so the
    # north-arrow column on the right stays clear of the tick labels.
    cb_x0, cb_x1, cb_y, cb_h = 0.045, 0.780, 0.42, 0.070
    grad = np.linspace(0, 1, 256)[None, :]
    panel.imshow(grad, cmap=terrain, aspect="auto", origin="lower",
                 extent=(cb_x0, cb_x1, cb_y, cb_y + cb_h),
                 transform=panel.transAxes, zorder=31)
    panel.add_patch(Rectangle((cb_x0, cb_y), cb_x1 - cb_x0, cb_h,
                              facecolor="none", edgecolor="0.35", lw=0.5,
                              transform=panel.transAxes, zorder=32))
    # round tick values rather than exact quantile fractions
    step = 500.0
    ticks = np.arange(0, vmax + step, step)
    ticks = ticks[ticks <= vmax]
    for t in ticks:
        frac = (t - vmin) / (vmax - vmin)
        panel.text(cb_x0 + frac * (cb_x1 - cb_x0), cb_y - 0.030,
                   f"{t:.0f}", fontsize=6.3, ha="center", va="top",
                   transform=panel.transAxes)
    panel.text((cb_x0 + cb_x1) / 2, cb_y + cb_h + 0.020,
               "Elevation (m a.s.l.)", fontsize=7.2, ha="center", va="bottom",
               transform=panel.transAxes)

    # -- scale bar (bottom left of the panel)
    sb_x0, sb_w, sb_y, sb_h = 0.045, 0.50, 0.135, 0.052
    for j in range(4):
        panel.add_patch(Rectangle(
            (sb_x0 + j * sb_w / 4, sb_y), sb_w / 4, sb_h,
            facecolor="black" if j % 2 == 0 else "white",
            edgecolor="black", lw=0.5,
            transform=panel.transAxes, zorder=32))
    for frac, lab in ((0.0, "0"), (0.5, "25"), (1.0, "50 km")):
        panel.text(sb_x0 + frac * sb_w, sb_y - 0.030, lab, fontsize=6.3,
                   ha="center", va="top", transform=panel.transAxes)

    # -- north arrow, in its own clear column on the right
    na_x = 0.895
    panel.annotate("", xy=(na_x, 0.40), xytext=(na_x, 0.145),
                   xycoords=panel.transAxes, textcoords=panel.transAxes,
                   arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
                   zorder=33)
    panel.text(na_x, 0.42, "N", fontsize=8.5, fontweight="bold",
               ha="center", va="bottom", transform=panel.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
