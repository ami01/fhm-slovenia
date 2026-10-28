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

# ── GRP_DESC lookup tables ─────────────────────────────────────────────────

# Hgk: GRP_DESC is derived dynamically from TYPE_ENG sub-values per AQUIFER group
# (see add_hgk_grp_desc below)

LITO_DESC = {
    0:  "Quaternary loose sediments",
    1:  "Organic and peat deposits",
    2:  "Anthropogenic and special deposits",
    3:  "Neogene clastic sediments",
    4:  "Miocene carbonate rocks",
    5:  "Paleogene flysch",
    6:  "Cretaceous-Paleocene carbonate rocks",
    7:  "Jurassic carbonate rocks",
    8:  "Triassic carbonate and clastic rocks",
    9:  "Paleozoic clastic and volcanic rocks",
    10: "Igneous and volcanic rocks",
    11: "Metamorphic rocks",
}

SOIL_DESC = {
    0:  "Non-soil and artificial surfaces",
    1:  "Lithosol and Regosol",
    2:  "Rendzina",
    3:  "Brown calcareous soils",
    4:  "Ranker",
    5:  "Eutric brown soils",
    6:  "Dystric brown soils",
    7:  "Leached and podzolic soils",
    8:  "Terra rossa",
    9:  "Alluvial and colluvial soils",
    10: "Rigosol",
    11: "Pseudogley",
    12: "Gley and peat soils",
}


def add_hgk_grp_desc(shp_path: Path, out_name: str):
    """Processes Hgk_grouped using TYPE_ENG as the GRP column.
    - Adds TYPE_CODE (numeric 1-N per unique TYPE_ENG, used for rasterization)
    - Adds GRP_DESC listing the LAWA sub-codes merged into each TYPE_ENG group
    - Drops AQUIFER and DESCR_ENG (not required)
    """
    gdf = gpd.read_file(shp_path)
    type_order   = sorted(gdf["TYPE_ENG"].unique())
    type_code_map = {t: i + 1 for i, t in enumerate(type_order)}
    gdf["TYPE_CODE"] = gdf["TYPE_ENG"].map(type_code_map)
    grp_desc_map = (
        gdf.groupby("TYPE_ENG")["LAWA"]
        .apply(lambda x: "; ".join(sorted(x.unique())))
        .to_dict()
    )
    gdf["GRP_DESC"] = gdf["TYPE_ENG"].map(grp_desc_map)
    gdf = gdf.drop(columns=[c for c in ["AQUIFER", "DESCR_ENG"] if c in gdf.columns])
    gdf = gdf[["LAWA", "TYPE_ENG", "TYPE_CODE", "GRP_DESC", "geometry"]]
    out = WORK_DIR / out_name
    gdf.to_file(out, driver="ESRI Shapefile")
    print(f"  Saved: {out.name}  (GRP col=TYPE_ENG, TYPE_CODE for rasterization)")
    for t, c in sorted(type_code_map.items(), key=lambda x: x[1]):
        print(f"    {c:2d}  {t:<55} LAWA: {grp_desc_map[t]}")
    return out


def add_grp_desc(shp_path: Path, grp_col: str, desc_map: dict, out_name: str):
    """Adds GRP_DESC column from a lookup dict and saves back to the shapefile."""
    gdf = gpd.read_file(shp_path)
    gdf["GRP_DESC"] = gdf[grp_col].map(desc_map)
    # Reorder so GRP_DESC sits right after grp_col
    cols = list(gdf.columns)
    cols.remove("GRP_DESC")
    idx  = cols.index(grp_col)
    cols.insert(idx + 1, "GRP_DESC")
    gdf  = gdf[cols]
    out  = WORK_DIR / out_name
    gdf.to_file(out, driver="ESRI Shapefile")
    print(f"  Saved: {out.name}  (GRP col={grp_col}, {len(desc_map)} groups)")
    return out


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


# ── 1. Add GRP_DESC to grouped shapefiles ─────────────────────────────────

print("\n=== Adding GRP_DESC columns ===")

hgk_out   = add_hgk_grp_desc(WORK_DIR / "Hgk_grouped.shp", "Hgk_grouped.shp")

lito_out  = add_grp_desc(WORK_DIR / "lito_poly_250_d96_grouped.shp",
                          "LITO_GRP", LITO_DESC, "lito_poly_250_d96_grouped.shp")

soil_out  = add_grp_desc(WORK_DIR / "Pedoloska_grouped.shp",
                          "SOIL_GRP", SOIL_DESC, "Pedoloska_grouped.shp")

# ── 2. Add CODE column to Rivers, Roads, Flooded_NoFlood ──────────────────

print("\n=== Adding CODE columns ===")

rivers_out  = add_code_column(WORK_DIR / "Slovenia_Rivers.shp",
                               "SIRINA",  "Slovenia_Rivers.shp")

roads_out   = add_code_column(WORK_DIR / "Slovenia_Roads.shp",
                               "Code",    "Slovenia_Roads.shp")

flood_out   = add_code_column(WORK_DIR / "Flooded_NoFlood.shp",
                               "Code",    "Flooded_NoFlood.shp")

# ── 3. Rasterize all six shapefiles ───────────────────────────────────────

print("\n=== Rasterizing ===")

rasterize(hgk_out,    "TYPE_CODE", OUT_DIR / "Hgk_grouped.tif")
rasterize(lito_out,   "LITO_GRP", OUT_DIR / "lito_poly_250_d96_grouped.tif")
rasterize(soil_out,   "SOIL_GRP", OUT_DIR / "Pedoloska_grouped.tif")
rasterize(rivers_out, "CODE",     OUT_DIR / "Slovenia_Rivers.tif",  is_line=True)
rasterize(roads_out,  "CODE",     OUT_DIR / "Slovenia_Roads.tif",   is_line=True)
rasterize(flood_out,  "CODE",     OUT_DIR / "Flooded_NoFlood.tif")

print("\nDone. Rasters saved to:", OUT_DIR)
