"""
Mask all CgFs_Aligned rasters + Flooded_NoFlood.tif to Slovenia boundary
=========================================================================
- Builds the Slovenia polygon mask once at the reference grid resolution.
- Applies it in-place to every .tif in CgFs_Aligned/.
- Copies Flooded_NoFlood.tif (the reference/target) into CgFs_Aligned/
  with the same mask applied.
- Processes in row-strips to keep RAM usage low (~80 MB per strip).
"""

import os, warnings
warnings.filterwarnings('ignore')

ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs_Aligned'
TARGET_RASTER = '/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs/Flooded_NoFlood.tif'
BOUNDARY_SHP  = '/home/arora/Documents/FSM_Slovenia/For_Analysis/Vectors/Slovenia_Boundary.shp'

STRIP_ROWS = 500   # process this many rows at a time (~80 MB @ float32, 19k cols)

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import mapping

# ── Load boundary ─────────────────────────────────────────────────────────────
gdf   = gpd.read_file(BOUNDARY_SHP)
geoms = [mapping(g) for g in gdf.geometry]
print(f"Boundary CRS : {gdf.crs}")
print(f"Boundary bbox: {np.round(gdf.total_bounds, 1)}\n")

# ── Reference grid from first aligned file ────────────────────────────────────
ref_files = sorted(f for f in os.listdir(ALIGNED_DIR) if f.endswith('.tif'))
with rasterio.open(os.path.join(ALIGNED_DIR, ref_files[0])) as ref:
    ref_transform = ref.transform
    ref_width     = ref.width
    ref_height    = ref.height
    ref_crs       = ref.crs

print(f"Reference grid : {ref_width} × {ref_height} px | {ref_transform.a:.4f} m | {ref_crs.to_epsg()}\n")

# ── Build Slovenia mask in strips (True = inside Slovenia) ────────────────────
# We store it as a uint8 boolean array (0/1) to save RAM vs bool
print("Building Slovenia boundary mask ...")
slov_mask = np.zeros((ref_height, ref_width), dtype=np.uint8)

for row_start in range(0, ref_height, STRIP_ROWS):
    row_end  = min(row_start + STRIP_ROWS, ref_height)
    n_rows   = row_end - row_start
    win_tf   = ref_transform * rasterio.Affine.translation(0, row_start)
    strip    = geometry_mask(
        geoms, transform=win_tf, invert=True,
        out_shape=(n_rows, ref_width)
    ).astype(np.uint8)
    slov_mask[row_start:row_end, :] = strip

n_inside  = slov_mask.sum()
n_outside = ref_height * ref_width - n_inside
print(f"  Inside  : {n_inside:,} px  ({100*n_inside/(ref_height*ref_width):.1f}%)")
print(f"  Outside : {n_outside:,} px  ({100*n_outside/(ref_height*ref_width):.1f}%)\n")

# ── Helper: apply mask to one file (in-place) ─────────────────────────────────
def mask_raster_inplace(path, slov_mask, label=''):
    with rasterio.open(path, 'r+') as ds:
        nd = ds.nodata
        dtype = ds.dtypes[0]

        _INT = {'uint8','uint16','int16','uint32','int32','int8','uint64','int64'}
        if nd is None:
            fill = -9999 if dtype in _INT else np.nan
            ds.nodata = fill
        else:
            fill = nd

        changed = 0
        for row_start in range(0, ds.height, STRIP_ROWS):
            row_end = min(row_start + STRIP_ROWS, ds.height)
            win     = Window(0, row_start, ds.width, row_end - row_start)
            arr     = ds.read(1, window=win)
            outside = slov_mask[row_start:row_end, :] == 0

            # set outside pixels to fill value
            n_changed = outside.sum()
            if n_changed > 0:
                arr[outside] = fill
                ds.write(arr, 1, window=win)
                changed += n_changed

        print(f"  {label:<55}  {changed:>12,} px set to nodata  ({100*changed/(ds.height*ds.width):.1f}%)")


# ── Helper: copy + mask (for Flooded_NoFlood which isn't in ALIGNED_DIR yet) ──
def copy_and_mask(src_path, dst_path, slov_mask):
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        nd   = src.nodata
        dtype = src.dtypes[0]

        _INT = {'uint8','uint16','int16','uint32','int32','int8','uint64','int64'}
        if nd is None:
            fill = -9999 if dtype in _INT else np.nan
        else:
            fill = nd
        meta.update(nodata=fill, compress='lzw', tiled=True,
                    blockxsize=512, blockysize=512)

        with rasterio.open(dst_path, 'w', **meta) as dst:
            changed = 0
            for row_start in range(0, src.height, STRIP_ROWS):
                row_end = min(row_start + STRIP_ROWS, src.height)
                win     = Window(0, row_start, src.width, row_end - row_start)
                arr     = src.read(1, window=win).copy()
                outside = slov_mask[row_start:row_end, :] == 0
                arr[outside] = fill
                dst.write(arr, 1, window=win)
                changed += outside.sum()

    print(f"  {'Flooded_NoFlood.tif (copied+masked)':<55}  {changed:>12,} px set to nodata  ({100*changed/(src.height*src.width):.1f}%)")


# ── Process all aligned rasters ───────────────────────────────────────────────
print("Applying boundary mask ...")

# 1. All existing files in CgFs_Aligned
for fname in sorted(ref_files):
    path = os.path.join(ALIGNED_DIR, fname)
    mask_raster_inplace(path, slov_mask, label=fname)

# 2. Copy + mask Flooded_NoFlood into CgFs_Aligned
fno_dst = os.path.join(ALIGNED_DIR, 'Flooded_NoFlood.tif')
if not os.path.exists(fno_dst):
    copy_and_mask(TARGET_RASTER, fno_dst, slov_mask)
else:
    mask_raster_inplace(fno_dst, slov_mask, label='Flooded_NoFlood.tif')

# ── Verification ──────────────────────────────────────────────────────────────
print("\nVerification (sample: elevation.tif, Flooded_NoFlood.tif) ...")
for fname in ['elevation.tif', 'Flooded_NoFlood.tif']:
    path = os.path.join(ALIGNED_DIR, fname)
    with rasterio.open(path) as ds:
        nd   = ds.nodata
        arr  = ds.read(1, out_shape=(ds.height//20, ds.width//20),
                       resampling=rasterio.enums.Resampling.nearest)
        h2, w2 = arr.shape
        t2 = ds.transform * ds.transform.scale(20, 20)

    # build low-res Slovenia mask for verification
    slv_lr = geometry_mask(geoms, transform=t2, invert=True, out_shape=(h2, w2))
    if nd is None or (isinstance(nd, float) and np.isnan(nd)):
        data_mask = ~np.isnan(arr)
    else:
        data_mask = arr != nd

    outside_leak = (~slv_lr & data_mask).sum()
    inside_data  = ( slv_lr & data_mask).sum()
    print(f"  {fname:<40}  outside leak: {outside_leak:,}  inside data: {inside_data:,}  "
          f"{'PASS' if outside_leak == 0 else 'FAIL'}")

print("\nDone. All rasters masked to Slovenia boundary.")
print(f"Output: {ALIGNED_DIR}")
