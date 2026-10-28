"""
extract_river_network_pysheds.py
─────────────────────────────────
Stream network extraction with Strahler order using PySheds.

Pipeline:
  1. Condition DEM (fill pits → resolve flats)
  2. D8 flow direction
  3. Flow accumulation (cells)
  4. Threshold → binary stream mask
  5. Strahler order per stream cell
  6. Vectorise → GeoDataFrame → .shp

Dependencies: pysheds, geopandas, rasterio, numpy, shapely
  pip install pysheds
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from pathlib import Path

from pysheds.grid import Grid

# ── Paths ──────────────────────────────────────────────────────────────────

CGFS_DIR = Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs")
OUT_DIR  = Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/Vector_CgFs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEM_TIF  = CGFS_DIR / "elevation.tif"
OUT_SHP  = OUT_DIR / "River_Network_Strahler_pysheds.shp"

# ── Parameters ─────────────────────────────────────────────────────────────

ACCUM_THRESHOLD = 5000   # cells (~0.78 km² at 12.5 m)

# ── Step 1: Load DEM and condition ─────────────────────────────────────────

print("[1/5] Loading and conditioning DEM ...")
grid = Grid.from_raster(str(DEM_TIF))
dem  = grid.read_raster(str(DEM_TIF))

# Fill pits, then resolve flats (epsilon gradient to ensure drainage)
pit_filled  = grid.fill_pits(dem)
flooded_dem = grid.fill_depressions(pit_filled)
conditioned  = grid.resolve_flats(flooded_dem)

# ── Step 2: D8 flow direction ───────────────────────────────────────────────

print("[2/5] Computing D8 flow direction ...")
# PySheds uses ESRI D8 encoding by default (matches WhiteboxTools)
fdir = grid.flowdir(conditioned)

# ── Step 3: Flow accumulation ───────────────────────────────────────────────

print("[3/5] Computing flow accumulation ...")
acc = grid.accumulation(fdir)

valid = acc[acc > 0]
print(f"      Accumulation range : {valid.min():.0f} – {valid.max():.0f} cells")
stream_px = (valid >= ACCUM_THRESHOLD).sum()
print(f"      Stream pixels      : {stream_px:,}")

# ── Step 4 + 5: Extract streams and compute Strahler order ─────────────────

print(f"[4/5] Extracting streams (threshold = {ACCUM_THRESHOLD} cells) ...")
# PySheds returns a labelled stream order raster directly
# extract_river_network masks at threshold and returns Strahler order
branches = grid.extract_river_network(
    fdir,
    acc > ACCUM_THRESHOLD,   # binary mask: True where stream
    dirmap=(64, 128, 1, 2, 4, 8, 16, 32),  # ESRI D8 encoding
)

# ── Step 6: Build GeoDataFrame ──────────────────────────────────────────────

print("[5/5] Building GeoDataFrame ...")

# branches is a dict with 'features' (GeoJSON-like) and 'properties'
features = branches['features']

if not features:
    raise RuntimeError("No stream features extracted — lower ACCUM_THRESHOLD.")

geoms      = [shape(f['geometry'])      for f in features]
strahler   = [f['properties']['order']  for f in features]

gdf = gpd.GeoDataFrame(
    {"STRAHLER": strahler},
    geometry=geoms,
    crs="EPSG:3794",
)

# Drop empty/null geometries
gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
gdf["STRAHLER"] = gdf["STRAHLER"].astype(int)

# ── Save ────────────────────────────────────────────────────────────────────

gdf.to_file(OUT_SHP, driver="ESRI Shapefile")

print(f"\n  Saved          : {OUT_SHP}")
print(f"  Total segments : {len(gdf)}")
print(f"  Strahler range : {gdf['STRAHLER'].min()} – {gdf['STRAHLER'].max()}")
print("\n  Segments by Strahler order:")
print(gdf["STRAHLER"].value_counts().sort_index().to_string())