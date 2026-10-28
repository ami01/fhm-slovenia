"""
Generates three derived rasters at 12.5 m resolution, EPSG:3794:
  1. River density      — Gaussian KDE smooth density surface (km/km²)
  2. Distance to river  — Euclidean distance (metres) to nearest river pixel
  3. Distance to road   — Euclidean distance (metres) to nearest road pixel

Inputs (already in EPSG:3794, 12.5 m):
  Slovenia_Rivers.tif   burn value = CODE (1-8; 9999 = unclassified)
  Slovenia_Roads.tif    burn value = CODE (0-4)
"""

import subprocess
import tempfile
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from pathlib import Path

RAS_DIR = Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/Rasters")
VEC_DIR = Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/Vector_CgFs/In Raster")

RIVERS_TIF = RAS_DIR / "Slovenia_Rivers_N.tif"
#ROADS_TIF  = RAS_DIR / "Slovenia_Roads.tif"

OUT_RIVER_DENSITY  = RAS_DIR / "River_Density_N.tif"
OUT_RIVER_DIST     = RAS_DIR / "Distance_to_River_N.tif"
#OUT_ROAD_DIST      = RAS_DIR / "Distance_to_Road.tif"

RES              = 12.5    # metres per pixel
KDE_BANDWIDTH_M  = 2000    # Gaussian KDE bandwidth (sigma) in metres — controls smoothness
NODATA           = -9999.0


# ── helpers ───────────────────────────────────────────────────────────────

def read_raster(path):
    with rasterio.open(path) as src:
        data     = src.read(1).astype(np.float32)
        profile  = src.profile.copy()
        nodata   = src.nodata
    return data, profile, nodata


def write_raster(path, array, profile, nodata=NODATA, dtype="float32"):
    p = profile.copy()
    p.update(dtype=dtype, nodata=nodata, count=1, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(dtype), 1)
    print(f"  Saved: {path.name}")


def binary_from_raster(data, nodata_val, exclude_vals=None):
    """Return binary array: 1 where feature present, 0 elsewhere."""
    exclude_vals = set(exclude_vals or [])
    mask = (data != nodata_val) & (~np.isin(data, list(exclude_vals)))
    binary = np.where(mask, 1, 0).astype(np.uint8)
    return binary


def gdal_proximity(src_tif: Path, out_tif: Path):
    """Run gdal_proximity.py to compute Euclidean distance in metres."""
    cmd = [
        "gdal_proximity.py",
        str(src_tif),
        str(out_tif),
        "-values", "1",
        "-distunits", "GEO",
        "-nodata", str(NODATA),
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr}")
    else:
        print(f"  Saved: {out_tif.name}")


# ── 1. River density (Gaussian KDE) ──────────────────────────────────────

print("\n=== River Density (Gaussian KDE) ===")

river_data, river_profile, river_nodata = read_raster(RIVERS_TIF)

# Binary presence: 1 for valid river pixels (CODE 1-8), 0 everywhere else
# IMPORTANT: treat nodata as 0 (not masked) so KDE spreads across full extent
river_bin = np.where(
    (river_data != river_nodata) & (~np.isin(river_data, [9999])),
    1.0, 0.0
).astype(np.float32)

# Sigma in pixels
sigma_px = KDE_BANDWIDTH_M / RES                   # 2000 m / 12.5 m = 160 px

# Gaussian KDE: convolves each river pixel with a Gaussian kernel.
# The result is a smooth continuous surface across the FULL extent —
# high values near dense river networks, fading smoothly to near-zero far away.
smoothed = gaussian_filter(river_bin, sigma=sigma_px, mode="constant", cval=0)

# Scale to km/km²: each pixel = RES metres of river length
density = smoothed * (1000.0 / RES)

# Do NOT re-apply the rivers nodata mask — that would blank most of the map.
# All pixels in the bounding box are valid density values (≥ 0).

write_raster(OUT_RIVER_DENSITY, density, river_profile)
print(f"  Gaussian sigma: {KDE_BANDWIDTH_M} m  ({sigma_px:.0f} px)")
print(f"  Density range: {density.min():.4f} – {density.max():.4f} km/km²")


# ── 2. Distance to river ──────────────────────────────────────────────────

print("\n=== Distance to River (Euclidean) ===")

# Write binary river raster to a temp file for gdal_proximity
with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
    tmp_river = Path(tmp.name)

write_raster(tmp_river, river_bin, river_profile, nodata=255, dtype="uint8")
gdal_proximity(tmp_river, OUT_RIVER_DIST)
tmp_river.unlink()


# ── 3. Distance to road ───────────────────────────────────────────────────

print("\n=== Distance to Road (Euclidean) ===")

road_data, road_profile, road_nodata = read_raster(ROADS_TIF)

# Binary: 1 for all road pixels (CODE 0-4), 0 for nodata
road_bin = binary_from_raster(road_data, road_nodata)

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
    tmp_road = Path(tmp.name)

write_raster(tmp_road, road_bin, road_profile, nodata=255, dtype="uint8")
gdal_proximity(tmp_road, OUT_ROAD_DIST)
tmp_road.unlink()

print("\nDone.")
