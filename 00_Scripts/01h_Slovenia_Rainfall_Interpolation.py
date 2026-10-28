"""
Script 1 — Slovenia Rainfall: Interpolate + Reproject + Clip → EPSG:3794 @ 12.5 m
====================================================================================

Inputs:
  - slovenia_mean_annual_rainfall.tif   (1 band,  mm/year,  E-OBS 0.1°)
  - slovenia_annual_max_daily.tif       (75 bands, mm/day,  E-OBS 0.1°)
  - Slovenia_Boundary.shp               (EPSG:3794)

Outputs (EPSG:3794, 12.5 m, LZW, tiled, nodata=NaN):
  - slovenia_mean_annual_rainfall_3794_12m5.tif
  - slovenia_annual_max_daily_mean_3794_12m5.tif
  - slovenia_annual_max_daily_p90_3794_12m5.tif

Pipeline:
  1. Read source rasters → build cell-centre lon/lat
  2. NaN-fill borders (nearest-neighbour) → cubic spline → 0.005° fine grid
  3. Gaussian smoothing (sigma=1.5)
  4. Write temp EPSG:4326 GeoTIFF (tiny: ~720×320 px)
  5. Reproject → EPSG:3794 @ 12.5 m in 512×512 tiles  (memory-safe)
  6. Per-tile Slovenia mask + crop to boundary bbox
  7. Save final LZW-compressed tiled GeoTIFF

PROJ fix:
  CRS objects are read directly from the input files (src.crs, gdf.crs) so
  CRS.from_epsg() is never called — no EPSG database lookup, no proj.db
  version conflict regardless of conda environment.

Dependencies:
  pip install rasterio geopandas scipy numpy shapely pyproj
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INPUT_MEAN_RAIN = '/home/arora/Documents/FSM_Slovenia/Scripts/slovenia_mean_annual_rainfall.tif'
INPUT_MAX_DAILY = '/home/arora/Documents/FSM_Slovenia/Scripts/slovenia_annual_max_daily.tif'
BOUNDARY_SHP    = '/home/arora/Documents/FSM_Slovenia/Processed_Data/Vectors/Slovenia_Boundary.shp'
OUTPUT_DIR      = '/home/arora/Documents/FSM_Slovenia/Processed_Data/Rasters/Rainfall'

TARGET_RES_M    = 12.5    # output pixel size in metres
INTERP_DEG      = 0.005   # intermediate spline grid (°) — ~555 m
GAUSS_SIGMA     = 1.5     # Gaussian smoothing kernel (cells on fine grid)
TILE_SIZE       = 512     # tile size for windowed reprojection

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import tempfile
import numpy as np

import rasterio
import rasterio.windows
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform

import geopandas as gpd
from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping

warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RASTERS  (CRS read from file — no EPSG lookup)
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 1: Loading rasters ──────────────────────────────────────────")

with rasterio.open(INPUT_MEAN_RAIN) as src:
    mean_rain = src.read(1)
    T         = src.transform
    CRS_SRC   = src.crs          # ← from file, avoids CRS.from_epsg()
    rows, cols = mean_rain.shape
    print(f"  Mean annual rain : {rows}×{cols}, "
          f"range {np.nanmin(mean_rain):.0f}–{np.nanmax(mean_rain):.0f} mm/year")
    print(f"  Source CRS       : {CRS_SRC}")

with rasterio.open(INPUT_MAX_DAILY) as src:
    all_years      = src.read()                          # (75, 15, 34)
    max_daily_mean = np.nanmean(all_years, axis=0)
    max_daily_p90  = np.nanpercentile(all_years, 90, axis=0)
    print(f"  Annual max daily : {all_years.shape[0]} bands, "
          f"mean {np.nanmin(max_daily_mean):.0f}–{np.nanmax(max_daily_mean):.0f} mm/day")
    print(f"  Annual max P90   : "
          f"{np.nanmin(max_daily_p90):.0f}–{np.nanmax(max_daily_p90):.0f} mm/day")

# Load Slovenia boundary — CRS_DST taken from shapefile
slov_gdf    = gpd.read_file(BOUNDARY_SHP)
CRS_DST     = slov_gdf.crs                # ← from file, EPSG:3794
geom_dst    = [mapping(slov_gdf.geometry.values[0])]
slov_bounds = slov_gdf.total_bounds       # xmin, ymin, xmax, ymax (metres)
print(f"  Boundary CRS     : {CRS_DST}")

# Cell-centre coordinates (positive y-step = SW origin, rows S→N)
lons = T.c + T.a * (np.arange(cols) + 0.5)
lats = T.f + T.e * (np.arange(rows) + 0.5)
print(f"  Lon: {lons[0]:.2f}–{lons[-1]:.2f}  Lat: {lats[0]:.2f}–{lats[-1]:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CUBIC SPLINE INTERPOLATION → 0.005° FINE GRID
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Step 2: Cubic spline → {INTERP_DEG}° (~{INTERP_DEG*111:.0f} km) fine grid ──────")

margin   = 0.10
lon_fine = np.arange(lons[0] - margin, lons[-1] + margin + INTERP_DEG, INTERP_DEG)
lat_fine = np.arange(lats[0] - margin, lats[-1] + margin + INTERP_DEG, INTERP_DEG)
print(f"  Fine grid: {len(lat_fine)}×{len(lon_fine)} cells")


def interp_cubic(data, lons, lats, lon_fine, lat_fine, sigma=GAUSS_SIGMA):
    """
    Bicubic spline on a regular grid.
    Fills NaN borders via nearest-neighbour before fitting.
    lats must be ascending (S→N) — matches our E-OBS positive y-step layout.
    """
    LON2D, LAT2D = np.meshgrid(lons, lats)
    valid = ~np.isnan(data)
    if not valid.all():
        nn   = NearestNDInterpolator(
            np.column_stack([LON2D[valid], LAT2D[valid]]), data[valid])
        data = data.copy()
        data[~valid] = nn(LON2D[~valid], LAT2D[~valid])
    fine = RectBivariateSpline(lats, lons, data, kx=3, ky=3)(lat_fine, lon_fine)
    if sigma and sigma > 0:
        fine = gaussian_filter(fine, sigma=sigma)
    return fine.astype(np.float32)


datasets = {
    'slovenia_mean_annual_rainfall': {
        'data': mean_rain,      'unit': 'mm/year',
        'desc': 'Mean annual precipitation',
    },
    'slovenia_annual_max_daily_mean': {
        'data': max_daily_mean, 'unit': 'mm/day',
        'desc': 'Mean annual max daily precipitation',
    },
    'slovenia_annual_max_daily_p90': {
        'data': max_daily_p90,  'unit': 'mm/day',
        'desc': '90th-pct annual max daily precipitation',
    },
}

for key, meta in datasets.items():
    meta['fine'] = interp_cubic(meta['data'], lons, lats, lon_fine, lat_fine)
    f = meta['fine']
    print(f"  {key}: {np.nanmin(f):.1f}–{np.nanmax(f):.1f} {meta['unit']}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — REPROJECT → EPSG:3794 @ 12.5 m (WINDOWED) + MASK + CROP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Step 3: Reproject → EPSG:3794 @ {TARGET_RES_M} m  "
      f"(tiles {TILE_SIZE}×{TILE_SIZE}) + mask + crop ──")

# Fine-grid transform: rasterio needs N→S convention (negative y-step)
fine_tf_src = from_origin(
    west  = lon_fine[0]  - INTERP_DEG / 2,
    north = lat_fine[-1] + INTERP_DEG / 2,
    xsize = INTERP_DEG,
    ysize = INTERP_DEG,
)


def reproject_mask_crop(src_path, out_path, crs_src, crs_dst,
                        res_m, clip_geom, clip_bounds, tile_size=512):
    """
    Tiled reproject → per-tile mask → crop.
    Peak RAM ≈ 2 × tile_size² × 4 B  — safe for a 12.5 m Slovenia grid.

    CRS objects come from open files, not from CRS.from_epsg(), so no
    EPSG database lookup is required.
    """
    with rasterio.open(src_path) as src:
        full_tf, full_w, full_h = calculate_default_transform(
            crs_src, crs_dst, src.width, src.height, *src.bounds,
            resolution=res_m,
        )

        # Crop to Slovenia bounding box + small pad
        xmin, ymin, xmax, ymax = clip_bounds
        pad     = res_m * 10
        col_off = max(0, int((xmin - pad - full_tf.c) / full_tf.a))
        row_off = max(0, int((ymax + pad - full_tf.f) / full_tf.e))   # e < 0
        col_end = min(full_w, int((xmax + pad - full_tf.c) / full_tf.a) + 1)
        row_end = min(full_h, int((ymin - pad - full_tf.f) / full_tf.e) + 1)
        crop_w  = col_end - col_off
        crop_h  = row_end - row_off
        crop_tf = full_tf * rasterio.Affine.translation(col_off, row_off)

        out_meta = dict(
            driver='GTiff', dtype='float32', nodata=np.nan,
            width=crop_w, height=crop_h, count=1,
            crs=crs_dst, transform=crop_tf,
            compress='lzw', tiled=True,
            blockxsize=tile_size, blockysize=tile_size,
        )
        print(f"    Output grid : {crop_h}×{crop_w} px  "
              f"({crop_h*res_m/1000:.1f} km × {crop_w*res_m/1000:.1f} km)")

        with rasterio.open(out_path, 'w', **out_meta) as dst:
            total = sum(1 for _ in dst.block_windows(1))
            for idx, (_, win) in enumerate(dst.block_windows(1)):
                win_tf = rasterio.windows.transform(win, crop_tf)
                wh, ww = win.height, win.width

                # Reproject source tile
                tile = np.full((1, wh, ww), np.nan, dtype='float32')
                reproject(
                    source=rasterio.band(src, 1), destination=tile,
                    src_transform=src.transform, src_crs=crs_src,
                    dst_transform=win_tf,         dst_crs=crs_dst,
                    resampling=Resampling.cubic,
                    src_nodata=np.nan, dst_nodata=np.nan,
                )

                # Mask: keep only pixels inside Slovenia
                inside = geometry_mask(
                    clip_geom, transform=win_tf,
                    invert=True, out_shape=(wh, ww),
                )
                tile[0][~inside] = np.nan
                dst.write(tile, window=win)

                if (idx + 1) % 200 == 0 or (idx + 1) == total:
                    print(f"    Tiles: {idx+1}/{total}", end='\r')
            print()


for key, meta in datasets.items():
    print(f"\n  ► {key}")

    # Write tiny 4326 intermediate (≈1 MB)
    tmp = tempfile.mktemp(suffix='.tif')
    with rasterio.open(
        tmp, 'w', driver='GTiff',
        height=len(lat_fine), width=len(lon_fine),
        count=1, dtype='float32',
        crs=CRS_SRC, transform=fine_tf_src, nodata=np.nan,
    ) as dst:
        dst.write(meta['fine'][::-1, :][np.newaxis])   # flip to N→S

    out_path = os.path.join(OUTPUT_DIR, f'{key}_3794_12m5.tif')
    reproject_mask_crop(
        src_path=tmp, out_path=out_path,
        crs_src=CRS_SRC, crs_dst=CRS_DST,
        res_m=TARGET_RES_M,
        clip_geom=geom_dst, clip_bounds=slov_bounds,
        tile_size=TILE_SIZE,
    )
    os.remove(tmp)

    # Validate with a downsampled read (avoids loading full raster)
    with rasterio.open(out_path) as src:
        ov = src.read(1,
                      out_shape=(src.height // 50, src.width // 50),
                      resampling=Resampling.average)
    v = ov[~np.isnan(ov)]
    print(f"    Saved : {out_path}")
    print(f"    Range : {v.min():.1f}–{v.max():.1f} {meta['unit']}")

print("\n✓ Script 1 done.")