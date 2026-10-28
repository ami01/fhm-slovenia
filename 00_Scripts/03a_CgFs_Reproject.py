"""
Script 2 — DEM Conditioning Factors: Reproject + Resample → EPSG:3794 @ 12.5 m
=================================================================================

Scans all .tif files in DEM_FACTORS_DIR, reprojects each to EPSG:3794 at
TARGET_RES_M, clips to the Slovenia boundary, and writes a companion output
file with the same stem + '_3794_12m5' suffix.

Inputs:
  - DEM_FACTORS_DIR/*.tif     (any CRS, any resolution)
  - Slovenia_Boundary.shp     (EPSG:3794)

Outputs (EPSG:3794, 12.5 m, LZW, tiled, nodata preserved):
  - OUTPUT_DIR/<original_name>_3794_12m5.tif

Resampling strategy (auto-selected per band):
  - Float / continuous data  → Resampling.cubic
  - Integer / categorical     → Resampling.nearest
  Override per-file via RESAMPLING_OVERRIDES dict.

PROJ fix (same as Script 1):
  CRS_DST is read from the shapefile (.crs attribute), not from CRS.from_epsg(),
  so no EPSG database lookup is ever made. Input raster CRS is read from each
  file via src.crs.

Dependencies:
  pip install rasterio geopandas numpy shapely
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEM_FACTORS_DIR = '/home/arora/Documents/FSM_Slovenia/Processed_Data/Rasters/DEM_Factors'
BOUNDARY_SHP    = '/home/arora/Documents/FSM_Slovenia/Processed_Data/Vectors/Slovenia_Boundary.shp'
OUTPUT_DIR      = '/home/arora/Documents/FSM_Slovenia/Processed_Data/Rasters/DEM_Factors_3794_12m5'

TARGET_RES_M    = 12.5    # output pixel size in metres
TILE_SIZE       = 512     # tile size for windowed reprojection

# Per-file resampling overrides — stem (no extension) → Resampling method.
# Any file NOT listed here is auto-detected (cubic for float, nearest for int).
# Example:
#   'TWI'       : Resampling.bilinear,
#   'LULC'      : Resampling.nearest,
#   'curvature' : Resampling.cubic,
RESAMPLING_OVERRIDES: dict = {
    # 'filename_stem': Resampling.method,
}

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import glob
import warnings
import numpy as np

import rasterio
import rasterio.windows
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform

import geopandas as gpd
from shapely.geometry import mapping

warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD BOUNDARY  (CRS_DST taken from shapefile — no EPSG lookup)
# ─────────────────────────────────────────────────────────────────────────────
slov_gdf    = gpd.read_file(BOUNDARY_SHP)
CRS_DST     = slov_gdf.crs                         # rasterio CRS from file
geom_dst    = [mapping(slov_gdf.geometry.values[0])]
slov_bounds = slov_gdf.total_bounds                 # xmin, ymin, xmax, ymax

print(f"Boundary CRS : {CRS_DST}")
print(f"Boundary bbox: xmin={slov_bounds[0]:.0f}  ymin={slov_bounds[1]:.0f}  "
      f"xmax={slov_bounds[2]:.0f}  ymax={slov_bounds[3]:.0f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVER INPUT FILES
# ─────────────────────────────────────────────────────────────────────────────
tif_files = sorted(
    glob.glob(os.path.join(DEM_FACTORS_DIR, '*.tif')) +
    glob.glob(os.path.join(DEM_FACTORS_DIR, '*.tiff'))
)

if not tif_files:
    print(f"ERROR: no .tif files found in {DEM_FACTORS_DIR}")
    sys.exit(1)

print(f"Found {len(tif_files)} raster(s) in {DEM_FACTORS_DIR}:")
for f in tif_files:
    with rasterio.open(f) as src:
        print(f"  {os.path.basename(f):<40}  "
              f"{src.height}×{src.width} px  "
              f"res={src.res[0]:.4g} m  "
              f"CRS={src.crs.to_epsg() or src.crs.to_string()[:30]}  "
              f"dtype={src.dtypes[0]}  bands={src.count}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — auto-select resampling method
# ─────────────────────────────────────────────────────────────────────────────
_INTEGER_DTYPES = {'uint8', 'uint16', 'int16', 'uint32', 'int32', 'int8'}

def pick_resampling(stem: str, dtype: str) -> Resampling:
    """Return Resampling method for this file: override > auto-detect."""
    if stem in RESAMPLING_OVERRIDES:
        return RESAMPLING_OVERRIDES[stem]
    if dtype in _INTEGER_DTYPES:
        return Resampling.nearest      # categorical / classified raster
    return Resampling.cubic            # continuous / float raster


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION — tiled reproject + per-tile mask + crop
# ─────────────────────────────────────────────────────────────────────────────
def reproject_mask_crop(src_path, out_path, crs_dst,
                        res_m, clip_geom, clip_bounds,
                        resampling, tile_size=512):
    """
    Reproject all bands of src_path to crs_dst at res_m, mask to clip_geom,
    crop to clip_bounds, and write to out_path as a LZW-tiled GeoTIFF.

    Processes one tile_size×tile_size block at a time so peak RAM stays
    ≈ n_bands × tile_size² × bytes_per_pixel — safe at 12.5 m over Slovenia.

    CRS is never looked up from the EPSG database; src.crs and the supplied
    crs_dst (read from the shapefile) are used directly.
    """
    with rasterio.open(src_path) as src:
        n_bands  = src.count
        src_crs  = src.crs
        nodata   = src.nodata
        src_dtype = src.dtypes[0]

        # Fill value for masked pixels
        fill_val = nodata if nodata is not None else (
            0 if src_dtype in _INTEGER_DTYPES else np.nan)

        full_tf, full_w, full_h = calculate_default_transform(
            src_crs, crs_dst, src.width, src.height, *src.bounds,
            resolution=res_m,
        )

        # Compute crop window from bounding box
        xmin, ymin, xmax, ymax = clip_bounds
        pad     = res_m * 10
        col_off = max(0, int((xmin - pad - full_tf.c) / full_tf.a))
        row_off = max(0, int((ymax + pad - full_tf.f) / full_tf.e))
        col_end = min(full_w, int((xmax + pad - full_tf.c) / full_tf.a) + 1)
        row_end = min(full_h, int((ymin - pad - full_tf.f) / full_tf.e) + 1)
        crop_w  = col_end - col_off
        crop_h  = row_end - row_off
        crop_tf = full_tf * rasterio.Affine.translation(col_off, row_off)

        out_meta = dict(
            driver='GTiff', dtype=src_dtype,
            nodata=fill_val,
            width=crop_w, height=crop_h, count=n_bands,
            crs=crs_dst, transform=crop_tf,
            compress='lzw', tiled=True,
            blockxsize=tile_size, blockysize=tile_size,
        )
        # Preserve band descriptions if present
        band_tags = [src.tags(i) for i in range(1, n_bands + 1)]

        with rasterio.open(out_path, 'w', **out_meta) as dst:
            # Copy band descriptions
            for b, tags in enumerate(band_tags, start=1):
                if tags:
                    dst.update_tags(b, **tags)

            total = sum(1 for _ in dst.block_windows(1))

            for idx, (_, win) in enumerate(dst.block_windows(1)):
                win_tf = rasterio.windows.transform(win, crop_tf)
                wh, ww = win.height, win.width

                # Build inside-Slovenia mask for this tile
                inside = geometry_mask(
                    clip_geom, transform=win_tf,
                    invert=True, out_shape=(wh, ww),
                )

                # Reproject + mask each band
                for band_idx in range(1, n_bands + 1):
                    tile = np.full((wh, ww), fill_val, dtype=src_dtype)
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=tile,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=win_tf,
                        dst_crs=crs_dst,
                        resampling=resampling,
                        src_nodata=nodata,
                        dst_nodata=fill_val,
                    )
                    tile[~inside] = fill_val
                    dst.write(tile, window=win, indexes=band_idx)

                if (idx + 1) % 200 == 0 or (idx + 1) == total:
                    print(f"    Tiles: {idx+1}/{total}", end='\r')
        print()

    return crop_h, crop_w, fill_val


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ALL DEM FACTOR FILES
# ─────────────────────────────────────────────────────────────────────────────
summary = []

for i, src_path in enumerate(tif_files, start=1):
    stem     = os.path.splitext(os.path.basename(src_path))[0]
    out_name = f'{stem}_3794_12m5.tif'
    out_path = os.path.join(OUTPUT_DIR, out_name)

    with rasterio.open(src_path) as src:
        dtype    = src.dtypes[0]
        n_bands  = src.count
        src_res  = src.res
        src_crs  = src.crs

    rsmp = pick_resampling(stem, dtype)

    print(f"[{i}/{len(tif_files)}] {stem}")
    print(f"    Source : {src_crs.to_epsg() or src_crs.to_string()[:40]}  "
          f"res={src_res[0]:.4g} m  dtype={dtype}  bands={n_bands}")
    print(f"    Method : {rsmp.name}")

    try:
        h, w, nodata_out = reproject_mask_crop(
            src_path=src_path, out_path=out_path,
            crs_dst=CRS_DST, res_m=TARGET_RES_M,
            clip_geom=geom_dst, clip_bounds=slov_bounds,
            resampling=rsmp, tile_size=TILE_SIZE,
        )

        # Quick validation — downsampled read
        with rasterio.open(out_path) as dst:
            ov = dst.read(
                1,
                out_shape=(max(1, h // 50), max(1, w // 50)),
                resampling=Resampling.average,
            )
        nodata_val = nodata_out
        valid = ov[ov != nodata_val] if not (
            isinstance(nodata_val, float) and np.isnan(nodata_val)
        ) else ov[~np.isnan(ov)]

        range_str = (f"{valid.min():.3g}–{valid.max():.3g}"
                     if valid.size > 0 else "all nodata")
        print(f"    Saved  : {out_path}")
        print(f"    Grid   : {h}×{w} px | range (band 1, downsampled): {range_str}\n")
        summary.append((stem, 'OK', range_str))

    except Exception as e:
        print(f"    ERROR  : {e}\n")
        summary.append((stem, 'FAILED', str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("── Summary ──────────────────────────────────────────────────────────")
print(f"  {'Factor':<40}  {'Status':<8}  Range / Note")
print(f"  {'-'*40}  {'-'*8}  {'-'*20}")
for stem, status, note in summary:
    print(f"  {stem:<40}  {status:<8}  {note}")
ok  = sum(1 for _, s, _ in summary if s == 'OK')
err = sum(1 for _, s, _ in summary if s == 'FAILED')
print(f"\n  {ok} succeeded, {err} failed.")
print(f"  Outputs → {OUTPUT_DIR}")
print("\n✓ Script 2 done.")