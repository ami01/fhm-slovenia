"""
Prepares vector shapefiles for rasterization:
  1. Adds GRP_DESC column to the three 'grouped' shapefiles
  2. Adds CODE column to Rivers, Roads, and Flooded_NoFlood
  3. Rasterizes all six shapefiles at 12.5 m resolution in EPSG:3794
"""

import subprocess
import geopandas as gpd
from pathlib import Path

WORK_DIR = Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/Vector_CgFs/In Raster")
OUT_DIR  = Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/Rasters")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RES   = 12.5          # metres
CRS   = "EPSG:3794"

def add_code_column(shp_path: Path, src_col: str, out_name: str):
    """Renames src_col (or copies it) to CODE and saves.
    If src_col and CODE differ only by case, renames in-place to avoid DBF collision."""
    gdf = gpd.read_file(shp_path)
    if src_col == "CODE":
        pass  # already correct
    elif src_col.upper() == "CODE":
        # same name different case — rename to avoid DBF collision
        gdf = gdf.rename(columns={src_col: "CODE"})
    else:
        gdf["CODE"] = gdf[src_col].astype(int)
    gdf["CODE"] = gdf["CODE"].astype(int)
    out = WORK_DIR / out_name
    gdf.to_file(out, driver="ESRI Shapefile")
    print(f"  Saved: {out.name}  (CODE = {src_col})")
    return out


def rasterize(shp_path: Path, burn_col: str, out_tif: Path, is_line: bool = False):
    """Rasterizes a shapefile using gdal_rasterize at RES metres in CRS."""
    gdf = gpd.read_file(shp_path).to_crs(CRS)
    xmin, ymin, xmax, ymax = gdf.total_bounds

    cmd = [
        "gdal_rasterize",
        "-a", burn_col,
        "-tr", str(RES), str(RES),
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-a_srs", CRS,
        "-a_nodata", "-9999",
        "-ot", "Int16",
        "-co", "COMPRESS=LZW",
    ]
    if is_line:
        cmd += ["-l", shp_path.stem]
    cmd += [str(shp_path), str(out_tif)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR rasterizing {shp_path.name}:\n{result.stderr}")
    else:
        print(f"  Raster: {out_tif.name}")


# ── 2. Add CODE column to Rivers, Roads, Flooded_NoFlood ──────────────────

print("\n=== Adding CODE columns ===")

rivers_out  = add_code_column(WORK_DIR / "Slovenia_Rivers_N.shp",
                               "STRAHLER",  "Slovenia_Rivers_N.shp")


# ── 3. Rasterize all six shapefiles ───────────────────────────────────────

print("\n=== Rasterizing ===")

rasterize(rivers_out, "CODE",     OUT_DIR / "Slovenia_Rivers_N.tif",  is_line=True)

print("\nDone. Rasters saved to:", OUT_DIR)
