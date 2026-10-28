"""
Flood Zone Merge + Difference Script
======================================
Workflow:
  1. Load IKPN_Q500.shp and dissolve all its features → code = 1
  2. Load Slovenia_Boundary.shp
  3. Difference: boundary − flood zones → code = 0
  4. Merge both → save as single output shapefile
"""

import os
import geopandas as gpd
from shapely.ops import unary_union

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FLOOD_SHP    = "/home/arora/Downloads/DRSV_Files/Flood Zones/Flood hazard – IKPN/DRSV_IKPN_Q500/IKPN_Q500.shp"
BOUNDARY_SHP = "/home/arora/Documents/FSM_Slovenia/For_Analysis/Vectors/Slovenia_Boundary.shp"
OUTPUT_SHP   = "/home/arora/Documents/FSM_Slovenia/For_Analysis/Vectors/Flood_Hazard_Binary.shp"
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Load flood zone shapefile ─────────────────────────────────────────
    print(f"Loading flood zones: {FLOOD_SHP}")
    flood_gdf = gpd.read_file(FLOOD_SHP)
    print(f"  → {len(flood_gdf)} feature(s) found in IKPN_Q500.shp")
    print(f"  → CRS: {flood_gdf.crs}")

    # ── 2. Load boundary shapefile ────────────────────────────────────────────
    print(f"\nLoading boundary: {BOUNDARY_SHP}")
    boundary_gdf = gpd.read_file(BOUNDARY_SHP)
    print(f"  → CRS: {boundary_gdf.crs}")

    # ── 3. Reproject flood zones to boundary CRS if needed ───────────────────
    if flood_gdf.crs is None:
        print("\nWARNING: Flood zone CRS is missing — assigning boundary CRS.")
        flood_gdf = flood_gdf.set_crs(boundary_gdf.crs)
    elif flood_gdf.crs != boundary_gdf.crs:
        print("\nReprojecting flood zones to match boundary CRS …")
        flood_gdf = flood_gdf.to_crs(boundary_gdf.crs)

    # ── 4. Dissolve all features in flood shapefile into one geometry ─────────
    print("\nDissolving all flood zone features …")
    flood_union = unary_union(flood_gdf.geometry)
    flood_dissolved = gpd.GeoDataFrame(geometry=[flood_union], crs=boundary_gdf.crs)
    flood_dissolved["code"] = 1
    flood_dissolved = flood_dissolved.explode(index_parts=False).reset_index(drop=True)
    flood_dissolved = flood_dissolved[
        ~flood_dissolved.geometry.is_empty & flood_dissolved.geometry.notna()
    ]
    print(f"  → {len(flood_dissolved)} feature(s) after dissolve  [code=1]")

    # ── 5. Difference: boundary − flood zones ────────────────────────────────
    print("\nComputing difference (Slovenia boundary − flood zones) …")
    boundary_union = unary_union(boundary_gdf.geometry)
    remaining_geom = boundary_union.difference(flood_union)

    remaining = gpd.GeoDataFrame(geometry=[remaining_geom], crs=boundary_gdf.crs)
    remaining["code"] = 0
    remaining = remaining.explode(index_parts=False).reset_index(drop=True)
    remaining = remaining[
        ~remaining.geometry.is_empty & remaining.geometry.notna()
    ]
    print(f"  → {len(remaining)} feature(s) after difference  [code=0]")

    # ── 6. Concatenate and save ───────────────────────────────────────────────
    final = gpd.pd.concat([flood_dissolved, remaining], ignore_index=True)
    final = gpd.GeoDataFrame(final, geometry="geometry", crs=boundary_gdf.crs)

    print(f"\nFinal summary:")
    print(f"  code=1 (flood zone) : {(final['code'] == 1).sum()} feature(s)")
    print(f"  code=0 (non-flood)  : {(final['code'] == 0).sum()} feature(s)")
    print(f"  Total               : {len(final)} feature(s)")

    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)
    final.to_file(OUTPUT_SHP)
    print(f"\nSaved → {OUTPUT_SHP}")

if __name__ == "__main__":
    main()