"""
Dissolve Flood_Hazard_Binary.shp → 2 rows (code=0 and code=1)
"""

import geopandas as gpd
from shapely.ops import unary_union

INPUT_SHP  = "/home/arora/Documents/FSM_Slovenia/For_Analysis/Vectors/Flood_Hazard_Binary.shp"
OUTPUT_SHP = "/home/arora/Documents/FSM_Slovenia/For_Analysis/Vectors/Flood_Hazard_Binary_Dissolved.shp"

gdf = gpd.read_file(INPUT_SHP)
print(f"Loaded {len(gdf)} features  |  CRS: {gdf.crs}")
print(f"  code=0 features: {(gdf['code'] == 0).sum()}")
print(f"  code=1 features: {(gdf['code'] == 1).sum()}")

dissolved = (
    gdf.groupby("code", as_index=False)
       .agg(geometry=("geometry", unary_union))
)

dissolved = gpd.GeoDataFrame(dissolved, geometry="geometry", crs=gdf.crs)
dissolved = dissolved.sort_values("code").reset_index(drop=True)

print(f"\nOutput: {len(dissolved)} rows")
print(dissolved[["code"]])

dissolved.to_file(OUTPUT_SHP)
print(f"\nSaved → {OUTPUT_SHP}")