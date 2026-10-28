"""Exports the RABA -> LULC conversion process and its statistics to a Word document.

Documents the pipeline that turns the RABA 2026-06-30 land-use polygons into the
categorical LULC conditioning factor used by the flood-susceptibility models:

    RABA_20260630.shp  --raba_fractions.py-->  raba_frac_202606.tif  -->  LULC.tif

All figures below are measured from the produced rasters, not estimated.
"""

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

OUT_PATH = Path("/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/10_Documentation/RABA_LULC_Conversion.docx")

CELL_AREA_KM2 = 12.5 * 12.5 / 1e6   # 156.25 m2 per cell

# ── Measured data ─────────────────────────────────────────────────────────

N_FEATURES   = 1_698_891
GRID         = "19909 x 12996 @ 12.5 m, EPSG:3794"
FINE_RES     = 1.25
OVERSAMPLE   = 10
VALID_FRAC   = 129_749_785    # valid cells in the fraction raster
VALID_MODAL  = 129_721_159    # valid cells in LULC.tif (excludes RABA gaps)
NODATA_MODAL = 129_016_205
MAX_RESIDUAL = "6.333e-08"
RUNTIME_MIN  = 12
FRAC_SIZE_MB = 401

# RABA_ID -> (official English name, LULC group, feature count)
RABA_CODES = [
    (1100, "Arable land",                            "arable",         214_613),
    (1160, "Hop field",                              "arable",             558),
    (1180, "Permanent crops on arable land",         "arable",           1_341),
    (1211, "Vineyard",                               "permanent_crop",  41_208),
    (1212, "Nursery",                                "permanent_crop",      66),
    (1221, "Intensive orchard",                      "permanent_crop",   8_331),
    (1222, "Extensive orchard",                      "permanent_crop", 104_821),
    (1230, "Olive grove",                            "permanent_crop",   8_318),
    (1240, "Other permanent crop",                   "permanent_crop",   5_300),
    (1300, "Permanent grassland",                    "grassland",      313_678),
    (1321, "Swampy meadow",                          "grassland",        2_148),
    (1600, "Uncultivated agricultural land",         "grassland",      159_667),
    (1410, "Overgrown agricultural area",            "woody",          152_658),
    (1420, "Forest plantation",                      "woody",              924),
    (1500, "Trees and shrubs",                       "woody",          259_607),
    (1800, "Forest trees on agricultural land",      "woody",           23_535),
    (2000, "Forest",                                 "woody",          152_220),
    (3000, "Built-up area and related surface",      "built_up",       194_092),
    (1190, "Greenhouse",                             "built_up",         6_043),
    (4100, "Swamp",                                  "water_wetland",       12),
    (4210, "Reed",                                   "water_wetland",       51),
    (4220, "Other marshy area",                      "water_wetland",    1_512),
    (7000, "Water",                                  "water_wetland",   28_528),
    (5000, "Dried open area with special vegetation", "bare_open",      15_649),
    (6000, "Open area with little or no vegetation",  "bare_open",       4_011),
]

# LULC code -> (class name, modal cell count, mean areal fraction %)
LULC_CLASSES = [
    (1, "arable",         11_431_047,  8.70),
    (2, "permanent_crop",  3_314_701,  2.50),
    (3, "grassland",      24_175_069, 18.32),
    (4, "woody",          81_150_942, 62.42),
    (5, "built_up",        6_756_035,  5.76),
    (6, "water_wetland",     933_856,  0.75),
    (7, "bare_open",       1_959_509,  1.52),
]

# Areal composition from the fractional bands: (class, share %, km2)
FRAC_AREAS = [
    ("woody",          62.42, 12_655.6),
    ("grassland",      18.32,  3_714.0),
    ("arable",          8.70,  1_763.4),
    ("built_up",        5.76,  1_167.1),
    ("permanent_crop",  2.50,    507.7),
    ("bare_open",       1.52,    307.4),
    ("water_wetland",   0.75,    151.3),
]

PURITY = [
    ("Pure (max fraction >= 0.99)", 108_351_297),
    ("0.90 - 0.99",                   5_073_932),
    ("0.60 - 0.90",                  11_885_206),
    ("Mixed (< 0.60)",                4_439_350),
]

COVERAGE = [
    (">= 0.999",  129_671_561),
    ("0.9 - 0.999",    13_576),
    ("0.5 - 0.9",      24_143),
    ("< 0.5",          40_505),
]

# ── Helpers (styling matched to Raster_Code_Descriptions.docx) ────────────

HEADER_BG = "2E75B6"

def set_cell_bg(cell, hex_color: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def heading(doc, text, size=12):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)


def body(doc, text, italic=False, size=10):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.font.size = Pt(size)
    if italic:
        r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    p.paragraph_format.space_after = Pt(6)
    return p


def mono(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Cm(0.5)


def add_table(doc, headers, rows, widths=None, center_cols=(0,), bold_last_row=False):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"

    if widths:
        for i, w in enumerate(widths):
            for cell in tbl.columns[i].cells:
                cell.width = w

    for i, h in enumerate(headers):
        cell = tbl.rows[0].cells[i]
        cell.text = h
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(cell, HEADER_BG)

    for r_idx, row_data in enumerate(rows):
        is_total = bold_last_row and r_idx == len(rows) - 1
        bg = "DCE9F7" if is_total else ("F2F7FC" if r_idx % 2 == 0 else "FFFFFF")
        for c_idx, val in enumerate(row_data):
            cell = tbl.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            para = cell.paragraphs[0]
            para.runs[0].font.size = Pt(9)
            if is_total:
                para.runs[0].bold = True
            if c_idx in center_cols:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.runs[0].bold = True
            elif c_idx > 0:
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            set_cell_bg(cell, bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl


# ── Build document ────────────────────────────────────────────────────────

doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

title = doc.add_heading("RABA to LULC Conversion", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

sub = doc.add_paragraph()
sr = sub.add_run("Derivation of the LULC conditioning factor from the Slovenian "
                 "RABA land-use database (release 2026-06-30)")
sr.italic = True
sr.font.size = Pt(10)
sr.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 1. Purpose
heading(doc, "1. Purpose and rationale")
body(doc,
     "RABA (Evidenca dejanske rabe kmetijskih in gozdnih zemljisc) delineates land-use "
     "polygons from orthophotos at sub-metre precision, with a minimum mapping unit of "
     "1000 m² for agricultural land. The analysis grid is 12.5 m, i.e. 156.25 m² per cell.")
body(doc,
     "Rasterising the polygons directly at 12.5 m applies a pixel-centre rule: any feature "
     "smaller than one cell either flips to a neighbouring class or vanishes entirely, and "
     "the resulting aliasing is noise a learner will fit. The conversion therefore burns the "
     "polygons at a 10x finer grid (1.25 m) and block-aggregates to per-class areal "
     "fractions, from which the categorical LULC factor is then taken as the modal class. "
     "The intermediate fraction raster is retained so the sub-cell composition remains "
     "available.")

# 2. Input data
heading(doc, "2. Input data")
add_table(doc,
          ["Item", "Value"],
          [["Source layer", "RABA_20260630.shp (release 2026-06-30)"],
           ["Polygon features", f"{N_FEATURES:,}"],
           ["Distinct RABA_ID codes", f"{len(RABA_CODES)} (all mapped, none dropped)"],
           ["Attribute burned", "RABA_ID"],
           ["Source CRS", "EPSG:3794 (D96/TM, Slovene National Grid)"],
           ["Reference grid", "CgFs_Aligned/Flooded_NoFlood.tif"],
           ["Output grid", GRID],
           ["Intermediate burn grid", f"{FINE_RES} m ({OVERSAMPLE}x oversample, "
                                      f"{OVERSAMPLE**2} sub-cells per target cell)"]],
          widths=[Cm(5.5), Cm(11)], center_cols=())
body(doc,
     "The source layer and the reference grid share EPSG:3794, so no datum transformation "
     "was applied. The D48/GK to D96/TM conversion, which is grid-based and would otherwise "
     "introduce metre-level regional distortion on a 12.5 m grid, was not required.",
     italic=True, size=9)

# 3. Processing chain
heading(doc, "3. Processing chain")
mono(doc,
     "RABA_20260630.shp\n"
     "        |\n"
     "        |  raba_fractions.py  (burn at 1.25 m, block-aggregate 10x10)\n"
     "        v\n"
     "raba_frac_202606.tif    9 bands, float32  -  7 fractions + coverage + modal\n"
     "        |\n"
     "        |  extract modal band, nodata = 0, LZW\n"
     "        v\n"
     "CgFs_Aligned/LULC.tif   1 band, float32, codes 1-7")
body(doc,
     f"Processing is windowed over {19909 // 256 + 1} x {12996 // 256 + 1} tiles so the full "
     f"fine grid never resides in memory. Runtime was approximately {RUNTIME_MIN} minutes "
     f"single-threaded; the intermediate fraction raster is {FRAC_SIZE_MB} MB (deflate, "
     f"predictor 3).", italic=True, size=9)

# 4. Code mapping
heading(doc, "4. RABA_ID to LULC class mapping")
body(doc,
     "Classes are grouped by runoff and infiltration behaviour rather than agronomy. Three "
     "assignments therefore differ from the thematic MKGP reading: greenhouses (1190) are "
     "grouped with built-up land because a glass or plastic roof is hydrologically "
     "impervious; overgrown agricultural area (1410) and trees and shrubs (1500) are grouped "
     "as woody because their canopy interception resembles forest; and uncultivated "
     "agricultural land (1600) is grouped as grassland, being typically rank grass rather "
     "than bare soil.")
add_table(doc,
          ["RABA_ID", "Official class name", "LULC class", "Polygons"],
          [[c, name, grp, f"{cnt:,}"] for c, name, grp, cnt in RABA_CODES]
          + [["", "Total", "", f"{sum(c[3] for c in RABA_CODES):,}"]],
          widths=[Cm(1.9), Cm(7.2), Cm(4.0), Cm(2.5)],
          center_cols=(0,), bold_last_row=True)
body(doc, "Class names as published in RABA_Read_me.pdf.", italic=True, size=9)

# 5. LULC class distribution  ← the headline statistics
heading(doc, "5. LULC class distribution")
body(doc,
     "Cell counts and areal share for each class in CgFs_Aligned/LULC.tif. The modal share "
     "is the proportion of cells whose dominant class is the given class; the fractional "
     "share is the mean areal fraction across all cells, i.e. the true land-cover proportion "
     "before the modal reduction.")
tot_modal_cells = sum(r[2] for r in LULC_CLASSES)
add_table(doc,
          ["Code", "LULC class", "Cells", "Area (km²)", "Modal share", "Fractional share"],
          [[code, name, f"{cnt:,}", f"{cnt * CELL_AREA_KM2:,.1f}",
            f"{100 * cnt / tot_modal_cells:.2f}%", f"{frac:.2f}%"]
           for code, name, cnt, frac in LULC_CLASSES]
          + [["", "Total", f"{tot_modal_cells:,}",
              f"{tot_modal_cells * CELL_AREA_KM2:,.1f}", "100.00%", "99.97%"]],
          widths=[Cm(1.5), Cm(4.2), Cm(3.0), Cm(2.6), Cm(2.6), Cm(2.7)],
          center_cols=(0,), bold_last_row=True)
body(doc,
     f"Total mapped area is {tot_modal_cells * CELL_AREA_KM2:,.0f} km² against Slovenia's "
     f"official 20,271 km². Modal and fractional shares agree to within 0.6 percentage "
     f"points for every class, confirming the modal reduction introduces no systematic bias.",
     italic=True, size=9)

# 6. Areal composition
heading(doc, "6. Areal composition from the fractional bands")
add_table(doc,
          ["LULC class", "Share", "Area (km²)"],
          [[n, f"{s:.2f}%", f"{a:,.1f}"] for n, s, a in FRAC_AREAS]
          + [["Total", f"{sum(x[1] for x in FRAC_AREAS):.2f}%",
              f"{sum(x[2] for x in FRAC_AREAS):,.1f}"]],
          widths=[Cm(6.0), Cm(3.5), Cm(4.0)],
          center_cols=(), bold_last_row=True)
body(doc,
     "Woody cover at 62.4% comprises forest (2000) together with the scrub and transitional "
     "classes 1410, 1420, 1500 and 1800; the forest class alone corresponds to the commonly "
     "cited national figure of roughly 58%.", italic=True, size=9)

# 7. Sub-cell purity
heading(doc, "7. Sub-cell purity")
body(doc,
     "Distribution of the largest class fraction within each cell, quantifying what the "
     "modal reduction discards.")
tot_pure = sum(c for _, c in PURITY)
add_table(doc,
          ["Largest class fraction in cell", "Cells", "Share"],
          [[lbl, f"{c:,}", f"{100 * c / tot_pure:.2f}%"] for lbl, c in PURITY],
          widths=[Cm(7.5), Cm(3.5), Cm(2.5)], center_cols=())
body(doc,
     "83.5% of cells contain a single class, so the modal value is exact for the great "
     "majority of Slovenia. The remaining 16.5% are concentrated along land-use boundaries. "
     "Where that mixing matters, the fractional bands of raba_frac_202606.tif provide it "
     "directly.", italic=True, size=9)

# 8. Validation
heading(doc, "8. Validation")
add_table(doc,
          ["Check", "Result"],
          [["Grid identity with Flooded_NoFlood.tif",
            "Pass - transform, dimensions and CRS identical"],
           ["Sum of class fractions equals coverage band",
            f"Pass - max residual {MAX_RESIDUAL} (float32 rounding)"],
           ["Band value ranges", "Pass - all fractions within [0, 1]; modal within [0, 7]"],
           ["Unmapped RABA codes", "Pass - none; all 25 codes assigned"],
           ["Total area vs. official national area",
            f"Pass - {tot_modal_cells * CELL_AREA_KM2:,.0f} km² vs 20,271 km² "
            f"(0.02% difference)"],
           ["Valid cells in fraction raster", f"{VALID_FRAC:,}"],
           ["Valid cells in LULC.tif", f"{VALID_MODAL:,}"],
           ["Cells lost to RABA polygon gaps",
            f"{VALID_FRAC - VALID_MODAL:,} ({100 * (VALID_FRAC - VALID_MODAL) / VALID_FRAC:.3f}%)"]],
          widths=[Cm(6.5), Cm(10.0)], center_cols=())

heading(doc, "Polygon coverage per cell", size=10)
add_table(doc,
          ["Coverage band value", "Cells", "Share"],
          [[lbl, f"{c:,}", f"{100 * c / VALID_FRAC:.2f}%"] for lbl, c in COVERAGE],
          widths=[Cm(5.5), Cm(4.0), Cm(3.0)], center_cols=())
body(doc,
     "Coverage is effectively complete on 99.94% of cells. The small remainder lies on the "
     "national border, where the RABA polygon coverage ends inside the raster mask.",
     italic=True, size=9)

# 9. Outputs
heading(doc, "9. Output files")
add_table(doc,
          ["File", "Description"],
          [["For_Analysis/raba_frac_202606.tif",
            "9 bands, float32, 401 MB. Bands 1-7 are per-class areal fractions in [0, 1] "
            "in the order arable, permanent_crop, grassland, woody, built_up, "
            "water_wetland, bare_open; band 8 is polygon coverage; band 9 is the modal "
            "class id. Nodata NaN."],
           ["CgFs_Aligned/LULC.tif",
            "1 band, float32, LZW. Categorical conditioning factor, codes 1-7 as in "
            "Section 5. Nodata 0, denoting cells outside the Slovenia mask or without "
            "RABA polygon coverage."],
           ["For_Analysis/raba_fractions.py",
            "Conversion script. Group mapping, oversampling factor and reference grid are "
            "command-line configurable."]],
          widths=[Cm(5.5), Cm(11.0)], center_cols=())
body(doc,
     "The nodata convention of LULC.tif matches the existing categorical factors "
     "(Hgk_grouped, Pedoloska_grouped, lito_poly_250_d96_grouped), which likewise use 0, so "
     "the raster is handled correctly by the existing sampling and training code without "
     "modification.", italic=True, size=9)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT_PATH)
print(f"wrote {OUT_PATH}")
