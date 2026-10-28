#!/usr/bin/env python3
"""
raba_fractions.py
=================

Convert Slovenian RABA (Evidenca dejanske rabe kmetijskih in gozdnih zemljisc)
polygons into per-class *areal fraction* bands, snapped exactly to an existing
reference grid such as DMV0125 (12.5 m, EPSG:3794).

Why fractions instead of a categorical rasterisation
----------------------------------------------------
RABA's minimum mapping unit is 1000 m2 for agricultural land (~6.4 cells at
12.5 m), but its polygon *boundaries* are delineated from orthophotos at
sub-metre precision. A straight categorical burn at 12.5 m throws that away and
introduces aliasing: any feature smaller than one cell (156 m2 -- i.e. most
buildings) either flips or vanishes on the pixel-centre rule, and a learner will
happily fit that noise.

This script instead burns at 12.5/k metres (k=10 by default -> 1.25 m), then
block-aggregates to the target grid as per-class fractions in [0, 1]. You get
sub-cell mixing preserved, no arbitrary tie-breaking, and a continuous input
that suits both tree ensembles and CNNs.

Processing is windowed, so full-Slovenia runs do not need the whole fine grid
in memory.

Usage
-----
    # Whole reference-grid extent, default 7-group RABA mapping
    python raba_fractions.py \
        --vector RABA_2026_06_30.zip \
        --reference DMV0125.tif \
        --output raba_frac_202606.tif

    # Catchment subset, custom grouping, keep a modal band for comparison
    python raba_fractions.py \
        --vector RABA_2026_06_30.zip \
        --reference savinja_dem_12_5m.tif \
        --output savinja_raba_frac.tif \
        --groups my_groups.json \
        --also-modal --mask-from-reference

Requires: rasterio, geopandas, pyogrio, shapely, pyproj, numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default RABA_ID -> group mapping
# ---------------------------------------------------------------------------
# Codes are the RABA_ID sifrant published in RABA_Beri_me.pdf. Grouping below is
# oriented at runoff / infiltration behaviour rather than agronomy, so a few
# assignments differ from how MKGP groups them thematically:
#
#   * 1190 (rastlinjak / greenhouse) sits in `built_up`, not `arable` -- a glass
#     or plastic roof is hydrologically impervious. Move it if you care about
#     the agronomic reading instead.
#   * 1410 (zaraščanje / scrubbing over) and 1500 (drevesa in grmičevje) sit in
#     `woody` -- their canopy interception behaves far more like forest than
#     like the grassland they used to be.
#   * 1600 (neobdelano kmetijsko zemljisce) sits in `grassland`; it is usually
#     rank grass rather than bare soil.
#
# Order here defines band order in the output.
DEFAULT_GROUPS: dict[str, list[int]] = {
    "arable":         [1100, 1160, 1180],
    "permanent_crop": [1211, 1212, 1221, 1222, 1230, 1240],
    "grassland":      [1300, 1321, 1600],
    "woody":          [1410, 1420, 1500, 1800, 2000],
    "built_up":       [3000, 1190],
    "water_wetland":  [4100, 4210, 4220, 7000],
    "bare_open":      [5000, 6000],
}

# Historic CRS of the RABA archive. Layers up to and including 2018 are D48/GK,
# from January 2019 onward D96/TM.
EPSG_D48_GK = 3912
EPSG_D96_TM = 3794


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_groups(path: str | None) -> dict[str, list[int]]:
    """Load a {group_name: [codes...]} mapping from JSON, or return the default."""
    if path is None:
        return DEFAULT_GROUPS
    with open(path) as fh:
        groups = json.load(fh)
    if not isinstance(groups, dict) or not groups:
        raise ValueError(f"{path}: expected a non-empty object of group -> [codes]")
    out = {}
    for name, codes in groups.items():
        if not isinstance(codes, (list, tuple)) or not codes:
            raise ValueError(f"{path}: group '{name}' must map to a non-empty list")
        out[str(name)] = [int(c) for c in codes]
    return out


def build_code_lookup(groups: dict[str, list[int]]) -> dict[int, int]:
    """Map each source code to a 1-based group id. 0 is reserved for 'uncovered'."""
    lookup: dict[int, int] = {}
    for gid, (name, codes) in enumerate(groups.items(), start=1):
        for code in codes:
            if code in lookup:
                raise ValueError(
                    f"code {code} assigned to more than one group "
                    f"('{list(groups)[lookup[code] - 1]}' and '{name}')"
                )
            lookup[code] = gid
    return lookup


def resolve_vector_path(path: str) -> str:
    """Accept a .zip of the RABA shapefile directly and turn it into a /vsizip/ path."""
    p = Path(path)
    if p.suffix.lower() != ".zip":
        return str(p)
    with zipfile.ZipFile(p) as zf:
        shps = [n for n in zf.namelist() if n.lower().endswith((".shp", ".gpkg"))]
    if not shps:
        raise ValueError(f"{path}: no .shp or .gpkg found inside the archive")
    if len(shps) > 1:
        warnings.warn(f"{path}: multiple layers found, using {shps[0]!r}")
    return f"/vsizip/{p.resolve()}/{shps[0]}"


def describe_transformation(src_crs, dst_crs) -> None:
    """Report which datum transformation pyproj will use, and warn on ballpark fallbacks.

    D48/GK -> D96/TM is *not* a simple Helmert shift: the official Slovenian
    transformation is grid-based, and a generic 7-parameter fallback leaves
    metre-level distortion that varies regionally. That is enough to smear a
    12.5 m grid, so this is worth checking rather than assuming.
    """
    from pyproj import CRS
    from pyproj.transformer import TransformerGroup

    src, dst = CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs)
    if src.equals(dst):
        return

    tg = TransformerGroup(src, dst)
    if not tg.transformers:
        raise RuntimeError(f"no transformation available from {src.name} to {dst.name}")

    chosen = tg.transformers[0]
    print(f"  datum transformation: {chosen.description}", file=sys.stderr)
    if not tg.best_available:
        print(
            "  WARNING: the best-available transformation grid is missing.\n"
            "           pyproj has fallen back to a lower-accuracy operation.\n"
            "           Install the Slovenian grid via `pyproj sync --source-id si`\n"
            "           or set PROJ_NETWORK=ON before rerunning.",
            file=sys.stderr,
        )
    if "ballpark" in chosen.description.lower():
        print(
            "  WARNING: this is a BALLPARK transformation (accuracy unknown, often\n"
            "           metres). Do not use the result on a 12.5 m grid.",
            file=sys.stderr,
        )


def check_grid(ref) -> None:
    """Reject reference grids this script cannot honour exactly."""
    t = ref.transform
    if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
        raise ValueError("reference raster is rotated/sheared; not supported")
    if abs(abs(t.a) - abs(t.e)) > 1e-9:
        raise ValueError(f"non-square cells ({t.a} x {t.e}); not supported")
    if ref.crs is None:
        raise ValueError("reference raster has no CRS")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    import geopandas as gpd
    import rasterio
    from rasterio import features, windows
    from rasterio.transform import Affine
    from shapely.geometry import box

    ap = argparse.ArgumentParser(
        description="Rasterise RABA/SDR polygons to per-class areal fractions "
                    "on a reference grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--vector", required=True,
                    help="RABA shapefile/GeoPackage, or the .zip straight from rkg.gov.si")
    ap.add_argument("--reference", required=True,
                    help="reference raster defining the output grid (e.g. DMV0125.tif)")
    ap.add_argument("--output", required=True, help="output multiband GeoTIFF")
    ap.add_argument("--field", default="RABA_ID",
                    help="attribute holding the land-use code")
    ap.add_argument("--groups", default=None,
                    help="JSON file with {group: [codes]}; omit for the built-in RABA mapping")
    ap.add_argument("--oversample", type=int, default=10,
                    help="fine-grid subdivision per target cell (10 -> 1.25 m on a 12.5 m grid)")
    ap.add_argument("--tile", type=int, default=256,
                    help="tile size in target cells")
    ap.add_argument("--also-modal", action="store_true",
                    help="append a band with the modal (majority) group id")
    ap.add_argument("--mask-from-reference", action="store_true",
                    help="write nodata where the reference raster is nodata")
    ap.add_argument("--no-bbox-filter", action="store_true",
                    help="read the whole vector layer instead of clipping to the grid extent")
    args = ap.parse_args(argv)

    if args.oversample < 1:
        ap.error("--oversample must be >= 1")

    groups = load_groups(args.groups)
    lookup = build_code_lookup(groups)
    names = list(groups)
    n_groups = len(names)
    if n_groups > 254:
        ap.error("more than 254 groups is not supported (uint8 fine grid)")

    k = args.oversample
    k2 = k * k

    # -- reference grid -----------------------------------------------------
    with rasterio.open(args.reference) as ref:
        check_grid(ref)
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width, ref_height = ref.width, ref.height
        ref_bounds = ref.bounds
        ref_nodata = ref.nodata
        ref_mask = None
        if args.mask_from_reference:
            ref_mask = ref.read_masks(1)

    res = abs(ref_transform.a)
    print(f"reference grid: {ref_width} x {ref_height} @ {res} m, {ref_crs.to_string()}",
          file=sys.stderr)
    print(f"fine grid:      {res / k} m ({k}x oversample)", file=sys.stderr)

    # -- vector -------------------------------------------------------------
    src_path = resolve_vector_path(args.vector)
    read_kwargs = {"columns": [args.field], "engine": "pyogrio"}
    if not args.no_bbox_filter:
        # pyogrio wants the bbox in the *source* CRS, so peek at it first.
        src_crs = gpd.read_file(src_path, engine="pyogrio", max_features=1).crs
        bbox_geom = box(*ref_bounds)
        if src_crs is not None and not src_crs.equals(ref_crs):
            bbox_geom = (
                gpd.GeoSeries([bbox_geom], crs=ref_crs)
                .to_crs(src_crs)
                .iloc[0]
                .buffer(2 * res)  # pad so edge polygons are not clipped away
            )
        read_kwargs["bbox"] = bbox_geom.bounds

    print(f"reading {src_path} ...", file=sys.stderr)
    gdf = gpd.read_file(src_path, **read_kwargs)
    if gdf.empty:
        print("ERROR: no features in the reference-grid extent", file=sys.stderr)
        return 1
    print(f"  {len(gdf):,} features", file=sys.stderr)

    if gdf.crs is None:
        raise ValueError("vector layer has no CRS; cannot reproject safely")
    if not gdf.crs.equals(ref_crs):
        print(f"reprojecting {gdf.crs.to_string()} -> {ref_crs.to_string()}", file=sys.stderr)
        describe_transformation(gdf.crs, ref_crs)
        gdf = gdf.to_crs(ref_crs)

    # Codes not in the mapping are dropped rather than silently lumped anywhere.
    gdf["_gid"] = gdf[args.field].map(lookup)
    unmapped = gdf["_gid"].isna()
    if unmapped.any():
        codes = sorted(gdf.loc[unmapped, args.field].dropna().unique().tolist())
        print(f"  WARNING: dropping {int(unmapped.sum()):,} features with unmapped "
              f"codes: {codes}", file=sys.stderr)
        gdf = gdf.loc[~unmapped]
    gdf["_gid"] = gdf["_gid"].astype("uint8")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    sindex = gdf.sindex

    # -- output -------------------------------------------------------------
    n_bands = n_groups + 1 + (1 if args.also_modal else 0)  # +1 for coverage
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.float32(np.nan),
        "width": ref_width, "height": ref_height, "count": n_bands,
        "crs": ref_crs, "transform": ref_transform,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "deflate", "predictor": 3, "zlevel": 6, "BIGTIFF": "IF_SAFER",
    }

    n_tiles_x = -(-ref_width // args.tile)
    n_tiles_y = -(-ref_height // args.tile)
    total = n_tiles_x * n_tiles_y
    print(f"processing {total} tiles ...", file=sys.stderr)

    with rasterio.open(args.output, "w", **profile) as dst:
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, f"frac_{name}")
        dst.set_band_description(n_groups + 1, "coverage")
        if args.also_modal:
            dst.set_band_description(n_bands, "modal_gid")
        dst.update_tags(
            source=str(args.vector), field=args.field, oversample=str(k),
            group_order=",".join(names),
            group_ids=",".join(f"{n}={i}" for i, n in enumerate(names, 1)),
        )

        done = 0
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                col_off, row_off = tx * args.tile, ty * args.tile
                w = min(args.tile, ref_width - col_off)
                h = min(args.tile, ref_height - row_off)
                win = windows.Window(col_off, row_off, w, h)

                win_transform = windows.transform(win, ref_transform)
                fine_transform = win_transform * Affine.scale(1.0 / k, 1.0 / k)
                bounds = windows.bounds(win, ref_transform)

                idx = list(sindex.query(box(*bounds), predicate="intersects"))
                out = np.zeros((n_bands, h, w), dtype=np.float32)

                if idx:
                    sub = gdf.iloc[idx]
                    fine = features.rasterize(
                        ((geom, gid) for geom, gid in zip(sub.geometry, sub["_gid"])),
                        out_shape=(h * k, w * k),
                        transform=fine_transform,
                        fill=0,
                        dtype="uint8",
                        all_touched=False,  # centre rule; all_touched would overcount
                    )
                    # Block-reduce: (h*k, w*k) -> (h, k, w, k) -> count per target cell
                    blocks = fine.reshape(h, k, w, k)
                    for gid in range(1, n_groups + 1):
                        counts = (blocks == gid).sum(axis=(1, 3), dtype=np.int32)
                        out[gid - 1] = counts.astype(np.float32) / k2
                    out[n_groups] = 1.0 - (blocks == 0).sum(axis=(1, 3),
                                                            dtype=np.int32) / np.float32(k2)
                    if args.also_modal:
                        stack = out[:n_groups]
                        modal = stack.argmax(axis=0).astype(np.float32) + 1.0
                        modal[stack.max(axis=0) == 0] = 0.0
                        out[-1] = modal

                if ref_mask is not None:
                    invalid = ref_mask[row_off:row_off + h, col_off:col_off + w] == 0
                    out[:, invalid] = np.nan

                dst.write(out, window=win)

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total} tiles", file=sys.stderr)

    print(f"\nwrote {args.output}", file=sys.stderr)
    print("bands: " + ", ".join(
        [f"{i}=frac_{n}" for i, n in enumerate(names, 1)]
        + [f"{n_groups + 1}=coverage"]
        + ([f"{n_bands}=modal_gid"] if args.also_modal else [])
    ), file=sys.stderr)
    print("\nSanity check: the group fractions should sum to the coverage band.\n"
          "Coverage well below 1.0 means gaps in the source polygons over that cell.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
