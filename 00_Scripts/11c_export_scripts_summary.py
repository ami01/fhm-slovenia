"""
export_scripts_summary.py
─────────────────────────
Generates a Word document summarising all Python scripts used in the
FSM Slovenia project, including purpose, inputs, outputs, key parameters,
and dependencies for each script.
"""

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

OUT_PATH = Path("/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/10_Documentation/Scripts_Summary.docx")

# ── Script catalogue ───────────────────────────────────────────────────────

SCRIPTS = [
    {
        "name": "prepare_rainfall.py",
        "purpose": "Extracts mean annual rainfall and annual maximum daily rainfall from E-OBS NetCDF data for Slovenia and exports them as raw GeoTIFF rasters at 0.1° resolution.",
        "inputs": [
            "rr_ens_mean_0.1deg_reg_v31.0e.nc  —  E-OBS daily rainfall ensemble mean (NetCDF)",
        ],
        "outputs": [
            "slovenia_mean_annual_rainfall.tif  —  Climatological mean annual total (mm/year)",
            "slovenia_annual_max_daily.tif      —  Annual max daily rainfall per year, 75 bands (mm/day)",
        ],
        "params": [
            "AOI: lat 45.4–46.9°N, lon 13.3–16.7°E (Slovenia bounding box)",
            "CRS: EPSG:4324 (raw E-OBS grid)",
        ],
        "notes": "Run Slovenia_Rainfall_Interpolation.py next to reproject to EPSG:3794 @ 12.5 m.",
        "deps": "xarray, rioxarray",
    },
    {
        "name": "Slovenia_Rainfall_Interpolation.py",
        "purpose": "Interpolates raw E-OBS rainfall rasters from 0.1° to 12.5 m resolution using cubic spline + Gaussian smoothing, then reprojects and clips to the Slovenia boundary in EPSG:3794.",
        "inputs": [
            "slovenia_mean_annual_rainfall.tif   —  1-band mean annual rainfall (0.1°, EPSG:4324)",
            "slovenia_annual_max_daily.tif        —  75-band annual max daily rainfall (0.1°, EPSG:4324)",
            "Slovenia_Boundary.shp               —  Slovenia boundary polygon (EPSG:3794)",
        ],
        "outputs": [
            "slovenia_mean_annual_rainfall_3794_12m5.tif",
            "slovenia_annual_max_daily_mean_3794_12m5.tif  —  Mean of annual maxima",
            "slovenia_annual_max_daily_p90_3794_12m5.tif   —  90th percentile of annual maxima",
        ],
        "params": [
            "Target CRS: EPSG:3794",
            "Target resolution: 12.5 m",
            "Interpolation: cubic spline → Gaussian smoothing (sigma=1.5)",
            "Tiling: 512×512 px blocks (memory-safe)",
        ],
        "notes": "Script 1 in the DEM conditioning factor pipeline.",
        "deps": "rasterio, geopandas, scipy, numpy",
    },
    {
        "name": "DMV0125_to_geotiff.py",
        "purpose": "Converts raw DMV0125 (12.5 m DEM) ASCII XYZ tiles to GeoTIFF by reading each tile with numpy, reconstructing the regular grid, and mosaicking all tiles into a single merged GeoTIFF.",
        "inputs": [
            "DMV0125/*.XYZ  —  Raw ASCII XYZ elevation tiles (CRS: EPSG:3794 / D96/TM)",
        ],
        "outputs": [
            "Per-tile Float32 GeoTIFFs (LZW-compressed)",
            "DMV0125_merged.tif  —  Full Slovenia mosaic",
            "Reprojected versions: EPSG:4326 and EPSG:32633 (UTM 33N)",
        ],
        "params": [
            "Source CRS: EPSG:3794 (auto-detected from coordinates)",
            "Output dtype: Float32",
        ],
        "notes": "Handles the 'ungridded dataset / change of Y direction' GDAL error by bypassing the XYZ driver entirely.",
        "deps": "rasterio, numpy, gdal",
    },
    {
        "name": "Merge_DMV_0125.py",
        "purpose": "Merges all 12.5 m DEM XYZ tiles into a single ASCII YXZ file before GeoTIFF conversion.",
        "inputs": [
            "/Raw_Data/DMV0125/*.xyz  —  12.5 m DEM tiles",
        ],
        "outputs": [
            "/Merged/DMV0125_merged.xyz",
        ],
        "params": [
            "Optional: --input-dir, --output-dir, --compress, --no-header flags",
        ],
        "notes": "Run before DMV0125_to_geotiff.py.",
        "deps": "Standard library only (os, gzip, argparse)",
    },
    {
        "name": "Merge_DMV_0050.py",
        "purpose": "Merges all 5 m DEM XYZ tiles from four quadrant folders (NW/NE/SW/SE) into a single merged ASCII YXZ file.",
        "inputs": [
            "/Raw_Data/DMV0050_SZ/, _SV/, _JZ/, _JV/  —  5 m DEM tiles by quadrant",
        ],
        "outputs": [
            "/Merged/DMV0050_merged.xyz",
        ],
        "params": [
            "Quadrant folders: SZ (NW), SV (NE), JZ (SW), JV (SE)",
        ],
        "notes": "Slovenian cardinal direction naming: S=Sever(N), J=Jug(S), Z=Zahod(W), V=Vzhod(E).",
        "deps": "Standard library only",
    },
    {
        "name": "DEM_CgFs.py",
        "purpose": "GPU-accelerated derivation of DEM-based conditioning factors for flood susceptibility mapping. Computes 9 terrain morphology factors and multiple hydrological factors from the 12.5 m DEM.",
        "inputs": [
            "DMV0125_merged.tif  —  12.5 m DEM (EPSG:3794)",
            "Slovenia_Boundary.shp",
        ],
        "outputs": [
            "01_elevation.tif, 02_slope.tif, 03_aspect.tif",
            "04_plan_curvature.tif, 05_profile_curvature.tif",
            "06_tpi.tif  (Topographic Position Index)",
            "07_tri.tif  (Terrain Ruggedness Index)",
            "08_roughness.tif, 09_valley_depth.tif",
            "Hydrological factors: flow accumulation, TWI, SPI, STI, etc.",
        ],
        "params": [
            "Resolution: 12.5 m, CRS: EPSG:3794",
            "GPU acceleration via CuPy (falls back to NumPy if no GPU)",
        ],
        "notes": "Most computationally intensive script. Requires CUDA-capable GPU for full speed.",
        "deps": "rasterio, numpy, cupy (optional), pysheds, scipy",
    },
    {
        "name": "CgFs_Reproject.py",
        "purpose": "Reprojects and resamples all conditioning factor rasters to EPSG:3794 at 12.5 m, clipped to the Slovenia boundary. Auto-selects resampling method (cubic for continuous, nearest for categorical).",
        "inputs": [
            "DEM_FACTORS_DIR/*.tif  —  Conditioning factors in any CRS/resolution",
            "Slovenia_Boundary.shp  —  EPSG:3794",
        ],
        "outputs": [
            "OUTPUT_DIR/<name>_3794_12m5.tif  —  One file per input",
        ],
        "params": [
            "Target CRS: EPSG:3794",
            "Target resolution: 12.5 m",
            "Resampling: cubic (float) / nearest (integer) — override via RESAMPLING_OVERRIDES dict",
        ],
        "notes": "Script 2 in the conditioning factor pipeline. Run after DEM_CgFs.py.",
        "deps": "rasterio, geopandas, numpy",
    },
    {
        "name": "merge_flood_zones.py",
        "purpose": "Creates a binary flood hazard shapefile by dissolving IKPN Q500 flood zones (code=1) and computing the non-flooded remainder of the Slovenia boundary (code=0).",
        "inputs": [
            "IKPN_Q500.shp          —  Q500 flood hazard polygons",
            "Slovenia_Boundary.shp  —  EPSG:3794",
        ],
        "outputs": [
            "Flood_Hazard_Binary.shp  —  2 features: code=0 (no flood), code=1 (flooded)",
        ],
        "params": [],
        "notes": "Run dissolve.py afterwards to spatially dissolve the output into clean multipart features.",
        "deps": "geopandas, shapely",
    },
    {
        "name": "dissolve.py",
        "purpose": "Dissolves all features in Flood_Hazard_Binary.shp by code value, producing exactly 2 multipart features (code=0 and code=1).",
        "inputs": [
            "Flood_Hazard_Binary.shp",
        ],
        "outputs": [
            "Flood_Hazard_Binary_Dissolved.shp",
        ],
        "params": [],
        "notes": "Run after merge_flood_zones.py.",
        "deps": "geopandas, shapely",
    },
    {
        "name": "merge_rasters.py",
        "purpose": "Mosaics multiple GeoTIFF tiles from a download folder into a single merged GeoTIFF. Intended for Terrascope or similar tile downloads.",
        "inputs": [
            "All *.tif files in the specified folder",
        ],
        "outputs": [
            "merged_output.tif  —  In the same folder as the input tiles",
        ],
        "params": [
            "folder: path to the directory containing input tiles",
        ],
        "notes": "Update the 'folder' variable before running.",
        "deps": "rasterio",
    },
    {
        "name": "mean_annual_daily_max_rain.py",
        "purpose": "Visualises mean annual rainfall and annual maximum daily rainfall for Slovenia using ARSO-style contourf maps. Uses cubic spline interpolation and Gaussian smoothing on the raw E-OBS rasters.",
        "inputs": [
            "slovenia_mean_annual_rainfall.tif   —  1 band (mm/year)",
            "slovenia_annual_max_daily.tif        —  75 bands (mm/day)",
        ],
        "outputs": [
            "PNG map figures (inline or saved)",
        ],
        "params": [
            "Interpolation grid: 0.01°",
            "Smoothing: Gaussian sigma=1.5",
        ],
        "notes": "Visualisation only — does not write rasters.",
        "deps": "rasterio, geopandas, scipy, matplotlib",
    },
    {
        "name": "prepare_rasters.py",
        "purpose": "Prepares all vector shapefiles for raster conversion: adds GRP_DESC description columns to grouped geological/soil shapefiles, adds CODE columns to Rivers/Roads/Flooded layers, and rasterizes all six layers at 12.5 m resolution in EPSG:3794.",
        "inputs": [
            "Hgk_grouped.shp               —  Hydrogeological units",
            "lito_poly_250_d96_grouped.shp  —  Lithological units",
            "Pedoloska_grouped.shp          —  Soil types",
            "Slovenia_Rivers.shp",
            "Slovenia_Roads.shp",
            "Flooded_NoFlood.shp",
        ],
        "outputs": [
            "Updated shapefiles with GRP_DESC / CODE columns",
            "Hgk_grouped.tif        —  burn: TYPE_CODE (1–10, from TYPE_ENG)",
            "lito_poly_250_d96_grouped.tif  —  burn: LITO_GRP (0–11)",
            "Pedoloska_grouped.tif  —  burn: SOIL_GRP (0–12)",
            "Slovenia_Rivers.tif    —  burn: CODE (= SIRINA, 1–8)",
            "Slovenia_Roads.tif     —  burn: CODE (0–4)",
            "Flooded_NoFlood.tif    —  burn: CODE (0=no flood, 1=flooded)",
        ],
        "params": [
            "Resolution: 12.5 m",
            "CRS: EPSG:3794",
            "NoData: -9999",
            "Compression: LZW",
        ],
        "notes": "Rivers and Roads use gdal_rasterize -l (line burn). See Raster_Code_Descriptions.docx for full code lookup tables.",
        "deps": "geopandas, subprocess (gdal_rasterize)",
    },
    {
        "name": "river_road_rasters.py",
        "purpose": "Generates three derived spatial rasters from the river and road layers: (1) River density using Gaussian KDE, (2) Euclidean distance to nearest river, (3) Euclidean distance to nearest road.",
        "inputs": [
            "Slovenia_Rivers.tif  —  burn: CODE 1–8 (9999 excluded)",
            "Slovenia_Roads.tif   —  burn: CODE 0–4",
        ],
        "outputs": [
            "River_Density.tif       —  Gaussian KDE density surface (km/km²)",
            "Distance_to_River.tif  —  Euclidean distance to river (metres)",
            "Distance_to_Road.tif   —  Euclidean distance to road (metres)",
        ],
        "params": [
            "KDE bandwidth (sigma): 2000 m (controls smoothness)",
            "Resolution: 12.5 m, CRS: EPSG:3794",
            "Distance method: gdal_proximity.py with DISTUNITS=GEO",
            "NoData: -9999",
        ],
        "notes": "KDE treats nodata as 0 so the density surface covers the full raster extent. Adjust KDE_BANDWIDTH_M to change smoothness.",
        "deps": "rasterio, numpy, scipy (gaussian_filter), subprocess (gdal_proximity.py)",
    },
    {
        "name": "export_codebook.py",
        "purpose": "Generates a formatted Word document containing code lookup tables for all six rasterized shapefiles, including burn values, descriptions, and a methodology note.",
        "inputs": [
            "Hardcoded lookup tables (no file inputs required)",
        ],
        "outputs": [
            "Raster_Code_Descriptions.docx",
        ],
        "params": [],
        "notes": "Re-run whenever codes or descriptions change.",
        "deps": "python-docx",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────

BLUE      = "2E75B6"
DARK_BLUE = "1F497D"

def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def add_bullet(doc, text, level=0):
    p    = doc.add_paragraph(style="List Bullet")
    run  = p.add_run(text)
    run.font.size = Pt(10)
    p.paragraph_format.left_indent = Cm(0.5 * (level + 1))
    return p


def add_label_value(doc, label, value, label_width=Cm(4)):
    tbl = doc.add_table(rows=1, cols=2)
    tbl.columns[0].width = label_width
    tbl.columns[1].width = Cm(12)
    lc = tbl.rows[0].cells[0]
    vc = tbl.rows[0].cells[1]
    lr = lc.paragraphs[0].add_run(label)
    lr.bold = True
    lr.font.size = Pt(10)
    lr.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    vr = vc.paragraphs[0].add_run(value)
    vr.font.size = Pt(10)
    # no borders
    for row in tbl.rows:
        for cell in row.cells:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            for side in ("top", "left", "bottom", "right"):
                tag = OxmlElement(f"w:{side}")
                tag.set(qn("w:val"), "none")
                tcBorders.append(tag)
            tcPr.append(tcBorders)


def add_script_block(doc, s, index):
    # Header bar
    hdr = doc.add_paragraph()
    hdr.paragraph_format.space_before = Pt(10)
    hdr.paragraph_format.space_after  = Pt(2)
    r = hdr.add_run(f"  {index}. {s['name']}")
    r.bold = True
    r.font.size  = Pt(12)
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    hdr.paragraph_format.left_indent = Cm(-0.5)
    # Blue background via table trick
    tbl = doc.add_table(rows=1, cols=1)
    cell = tbl.rows[0].cells[0]
    cell.paragraphs[0].clear()
    run = cell.paragraphs[0].add_run(f"{index}. {s['name']}")
    run.bold = True
    run.font.size  = Pt(12)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    set_cell_bg(cell, BLUE)
    cell.paragraphs[0].paragraph_format.space_before = Pt(4)
    cell.paragraphs[0].paragraph_format.space_after  = Pt(4)
    # Remove the stray empty paragraph added before the table
    hdr._element.getparent().remove(hdr._element)

    # Purpose
    pp = doc.add_paragraph()
    pp.paragraph_format.space_before = Pt(6)
    pp.paragraph_format.space_after  = Pt(4)
    pr = pp.add_run(s["purpose"])
    pr.font.size = Pt(10)
    pr.font.italic = True

    def section(title, items):
        if not items:
            return
        sp = doc.add_paragraph()
        sr = sp.add_run(title)
        sr.bold = True
        sr.font.size = Pt(10)
        sr.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
        sp.paragraph_format.space_before = Pt(2)
        sp.paragraph_format.space_after  = Pt(0)
        for item in items:
            bp = doc.add_paragraph(style="List Bullet")
            br = bp.add_run(item)
            br.font.size = Pt(9.5)
            bp.paragraph_format.left_indent  = Cm(0.8)
            bp.paragraph_format.space_before = Pt(0)
            bp.paragraph_format.space_after  = Pt(0)

    section("Inputs",     s["inputs"])
    section("Outputs",    s["outputs"])
    section("Parameters", s["params"])

    if s.get("notes"):
        np_ = doc.add_paragraph()
        np_.paragraph_format.space_before = Pt(4)
        np_.add_run("Note: ").bold = True
        nr = np_.add_run(s["notes"])
        nr.font.size   = Pt(9.5)
        nr.font.italic = True

    dp = doc.add_paragraph()
    dp.paragraph_format.space_before = Pt(2)
    dp.paragraph_format.space_after  = Pt(8)
    dp.add_run("Dependencies: ").bold = True
    dr = dp.add_run(s["deps"])
    dr.font.size = Pt(9.5)


# ── Build document ─────────────────────────────────────────────────────────

doc = Document()
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# Title
tp = doc.add_heading("Python Scripts — FSM Slovenia", level=0)
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

sp = doc.add_paragraph("Script reference for reuse — Flood Susceptibility Mapping, Slovenia")
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sp.runs[0].italic = True
sp.runs[0].font.size = Pt(11)
sp.runs[0].font.color.rgb = RGBColor(0x60, 0x60, 0x60)

doc.add_paragraph()

# TOC-style index
idx_hdr = doc.add_paragraph()
ir = idx_hdr.add_run("Scripts Index")
ir.bold = True
ir.font.size = Pt(11)
ir.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)

for i, s in enumerate(SCRIPTS, 1):
    ip = doc.add_paragraph()
    ip.paragraph_format.space_before = Pt(0)
    ip.paragraph_format.space_after  = Pt(1)
    ip.paragraph_format.left_indent  = Cm(0.5)
    ip.add_run(f"{i:02d}.  ").bold = True
    cr = ip.add_run(s["name"])
    cr.font.size = Pt(10)

doc.add_page_break()

# Script entries
for i, s in enumerate(SCRIPTS, 1):
    add_script_block(doc, s, i)

doc.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
