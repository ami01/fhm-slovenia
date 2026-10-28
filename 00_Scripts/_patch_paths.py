"""Copy the pipeline scripts into 00_Scripts/ with stage-ordered names and
repoint their hardcoded paths at the FSM_Pipeline_LULC tree.

The originals in For_Analysis/Scripts/ are never modified, and the patched copies
are verified to contain no writable path into the original ML_Results/Collinearity.
"""
import re
import shutil
from pathlib import Path

SRC_SCRIPTS = Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/Scripts")
ROOT = Path("/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC")
DST = ROOT / "00_Scripts"
OLD_BASE = "/home/arora/Documents/FSM_Slovenia/For_Analysis"

# Longest-prefix first: specific subdirectories before the generic ML_Results.
REPLACEMENTS = [
    (f"{OLD_BASE}/CgFs_Aligned",                  f"{ROOT}/01_Input_Features"),
    (f"{OLD_BASE}/Collinearity",                  f"{ROOT}/02_Collinearity"),
    (f"{OLD_BASE}/ML_Results/samples",            f"{ROOT}/03_Samples"),
    (f"{OLD_BASE}/ML_Results/models_10x",         f"{ROOT}/04_Models"),
    (f"{OLD_BASE}/ML_Results/models",             f"{ROOT}/04_Models"),
    (f"{OLD_BASE}/ML_Results/results",            f"{ROOT}/05_Results"),
    (f"{OLD_BASE}/ML_Results/probability_maps",   f"{ROOT}/06_Probability_Maps"),
    (f"{OLD_BASE}/ML_Results/classified_maps",    f"{ROOT}/07_Classified_Maps"),
    (f"{OLD_BASE}/ML_Results/validation",         f"{ROOT}/08_Validation"),
    (f"{OLD_BASE}/ML_Results/spatial_cv",         f"{ROOT}/09_Analysis/spatial_cv"),
    (f"{OLD_BASE}/ML_Results/shap",               f"{ROOT}/09_Analysis/shap"),
    (f"{OLD_BASE}/ML_Results/analysis",           f"{ROOT}/09_Analysis"),
    (f"{OLD_BASE}/ML_Results",                    str(ROOT)),
]

# Derived os.path.join(...) / f-string subdirectory names, applied after the above
# so OUTPUT_DIR=ROOT still lands in the numbered folders.
JOIN_FIXES = [
    (r"os\.path\.join\((\w+), 'samples'\)",          r"os.path.join(\1, '03_Samples')"),
    (r"os\.path\.join\((\w+), 'models'\)",           r"os.path.join(\1, '04_Models')"),
    (r"os\.path\.join\((\w+), 'results'\)",          r"os.path.join(\1, '05_Results')"),
    (r"os\.path\.join\((\w+), 'probability_maps'\)", r"os.path.join(\1, '06_Probability_Maps')"),
    (r"os\.path\.join\((\w+), 'classified_maps'\)",  r"os.path.join(\1, '07_Classified_Maps')"),
    (r"os\.path\.join\((\w+), 'validation'\)",       r"os.path.join(\1, '08_Validation')"),
    (r"os\.path\.join\((\w+), 'spatial_cv'\)",       r"os.path.join(\1, '09_Analysis/spatial_cv')"),
    (r"os\.path\.join\((\w+), 'shap'\)",             r"os.path.join(\1, '09_Analysis/shap')"),
    (r"os\.path\.join\((\w+), 'analysis'\)",         r"os.path.join(\1, '09_Analysis')"),
    (r"os\.path\.join\((\w+), 'results_comparison'\)", r"os.path.join(\1, '05_Results/comparison')"),
]

# stage number -> (new name, source file, patch?)
# Stages 1-3 are data preparation, run once and already complete; they are copied
# for provenance but not re-executed. Stages 4+ are the LULC re-run.
PIPELINE = [
    # (order, new_name, source, patch)
    ("00", "00a_DMV0125_to_geotiff.py",        "DMV0125_to_geotiff.py",        False),
    ("00", "00b_Merge_DMV_0125.py",            "Merge_DMV_0125.py",            False),
    ("00", "00c_Merge_DMV_0050.py",            "Merge_DMV_0050.py",            False),
    ("00", "00d_merge_rasters.py",             "merge_rasters.py",             False),
    ("01", "01a_DEM_CgFs.py",                  "DEM_CgFs.py",                  False),
    ("01", "01b_extract_river_network.py",     "extract_river_network.py",     False),
    ("01", "01c_merge_flood_zones.py",         "merge_flood_zones.py",         False),
    ("01", "01d_dissolve.py",                  "dissolve.py",                  False),
    ("01", "01e_prepare_river_raster.py",      "prepare_river_raster.py",      False),
    ("01", "01f_prepare_rasters.py",           "prepare_rasters.py",           False),
    ("01", "01g_river_road_rasters.py",        "river_road_rasters.py",        False),
    ("01", "01h_Slovenia_Rainfall_Interpolation.py", "Slovenia_Rainfall_Interpolation.py", False),
    ("01", "01i_prepare_rainfall.py",          "prepare_rainfall.py",          False),
    ("01", "01j_mean_annual_daily_max_rain.py", "mean_annual_daily_max_rain.py", False),
    ("02", "02a_raba_fractions.py",            None,                           False),  # special
    ("02", "02b_make_lulc.py",                 None,                           False),  # special
    ("03", "03a_CgFs_Reproject.py",            "CgFs_Reproject.py",            False),
    ("03", "03b_mask_to_boundary.py",          "mask_to_boundary.py",          False),
    ("03", "03c_align_and_collinearity.py",    "align_and_collinearity.py",    False),
    ("04", "04_iterative_collinearity.py",     "iterative_collinearity.py",    True),
    ("05", "05a_train_models.py",              "train_models.py",              True),
    ("05", "05b_train_dnn.py",                 "train_dnn.py",                 True),
    ("05", "05c_train_models_multisample.py",  "train_models_multisample.py",  True),
    ("06", "06_predict_probability_maps.py",   "predict_probability_maps.py",  True),
    ("07", "07_threshold_classify_validate.py", "threshold_classify_validate.py", True),
    ("08", "08a_spatial_cv.py",                "spatial_cv.py",                True),
    ("08", "08b_validate_classify_spatial_cv_multiscale.py",
           "validate_classify_spatial_cv_multiscale.py",                       True),
    ("08", "08c_shap_analysis.py",             "shap_analysis.py",             True),
    ("09", "09a_classify_area_analysis.py",    "classify_area_analysis.py",    True),
    ("09", "09b_analysis_fr_area.py",          "analysis_fr_area.py",          True),
    ("09", "09c_subbasin_analysis.py",         "subbasin_analysis.py",         True),
    ("10", "10a_figures_maps.py",              "figures_maps.py",              True),
    ("10", "10b_generate_maps_all_sizes.py",   "generate_maps_all_sizes.py",   True),
    ("10", "10c_fix_comparison_maps.py",       "fix_comparison_maps.py",       True),
    ("10", "10d_setup_manuscript_folder.py",   "setup_manuscript_folder.py",   True),
    ("11", "11a_export_codebook.py",           "export_codebook.py",           False),
    ("11", "11b_export_raba_lulc_conversion.py", "export_raba_lulc_conversion.py", False),
    ("11", "11c_export_scripts_summary.py",    "export_scripts_summary.py",    False),
    ("11", "11d_generate_summary_report.py",   "generate_summary_report.py",   True),
]


def patch(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    for pat, rep in JOIN_FIXES:
        text = re.sub(pat, rep, text)
    return text


DST.mkdir(parents=True, exist_ok=True)
copied, patched, missing = [], [], []

for _, new_name, src_name, do_patch in PIPELINE:
    if src_name is None:
        continue
    src = SRC_SCRIPTS / src_name
    if not src.exists():
        missing.append(src_name)
        continue
    dst = DST / new_name
    if do_patch:
        dst.write_text(patch(src.read_text()))
        patched.append(new_name)
    else:
        shutil.copy2(src, dst)
    copied.append(new_name)

# the two conversion scripts live outside Scripts/ or were written this session
extra = [
    ("02a_raba_fractions.py",
     Path("/home/arora/Documents/FSM_Slovenia/For_Analysis/raba_fractions.py")),
]
for name, src in extra:
    if src.exists():
        shutil.copy2(src, DST / name)
        copied.append(name)

print(f"copied {len(copied)} scripts, {len(patched)} path-patched")
if missing:
    print(f"MISSING from Scripts/: {missing}")

# ── Verification: no patched script may reference the original output tree ──
print("\nverifying patched scripts contain no path into the original results ...")
bad = []
for name in patched:
    txt = (DST / name).read_text()
    for probe in ["For_Analysis/ML_Results", "For_Analysis/Collinearity",
                  "For_Analysis/CgFs_Aligned"]:
        if probe in txt:
            for ln, line in enumerate(txt.splitlines(), 1):
                if probe in line:
                    bad.append(f"  {name}:{ln}  {line.strip()[:100]}")
if bad:
    print(f"FAIL - {len(bad)} residual reference(s):")
    print("\n".join(bad))
else:
    print("PASS - no patched script writes into the original tree")
