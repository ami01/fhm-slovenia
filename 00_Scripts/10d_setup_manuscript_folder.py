#!/usr/bin/env python3
"""
setup_manuscript_folder.py
==========================
One-shot script to create a GitHub-ready Manuscript folder for the
Flood Susceptibility Mapping (FSM) Slovenia project.

Run from the For_Analysis/ directory:
    conda run -n floodcast_env python3 Scripts/setup_manuscript_folder.py
"""

import os
import shutil
import textwrap
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE     = "/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC"
ML       = BASE   # numbered stage folders sit directly under the root
SCRIPTS  = os.path.join(BASE, "Scripts")
MAN      = os.path.join(BASE, "Manuscript")        # output root
OLD_FIGS = os.path.join(MAN, "figures")            # existing Manuscript/figures

SIZES = ["1x", "2x", "3x", "4x", "10x"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
copied_ok  = []
copied_err = []
generated  = []

def safe_copy(src, dst):
    """Copy src → dst; skip silently if src missing."""
    if not os.path.isfile(src):
        print(f"  [SKIP] not found: {src}")
        copied_err.append(src)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copy2(src, dst)
        print(f"  [OK]   {os.path.basename(src)} → {os.path.relpath(dst, BASE)}")
        copied_ok.append(dst)
    except Exception as e:
        print(f"  [ERR]  {src}: {e}")
        copied_err.append(src)

def mkdirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Create directory structure
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("1. Creating directory structure …")
print("="*60)

dirs = [
    MAN,
    os.path.join(MAN, "code"),
    os.path.join(MAN, "figures", "main"),
    os.path.join(MAN, "figures", "supplementary"),
    os.path.join(MAN, "figures", "maps"),
    os.path.join(MAN, "tables", "main"),
    os.path.join(MAN, "tables", "supplementary"),
    os.path.join(MAN, "results", "validation"),
    os.path.join(MAN, "results", "spatial_cv"),
    os.path.join(MAN, "results", "shap"),
    os.path.join(MAN, "results", "area_statistics"),
    os.path.join(MAN, "docs"),
]
# per-size subdirectories inside results/
for sz in SIZES:
    dirs.append(os.path.join(MAN, "results", "validation", sz))
    dirs.append(os.path.join(MAN, "results", "spatial_cv", sz))
    dirs.append(os.path.join(MAN, "results", "shap", sz))

mkdirs(*dirs)
print(f"  Created {len(dirs)} directories.")

# ---------------------------------------------------------------------------
# 2. Copy scripts to code/
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("2. Copying scripts to code/ …")
print("="*60)

script_map = {
    "01_prepare_rasters.py":                  "prepare_rasters.py",
    "02_align_and_collinearity.py":           "align_and_collinearity.py",
    "03_train_models_multisample.py":         "train_models_multisample.py",
    "04_predict_probability_maps.py":         "predict_probability_maps.py",
    "05_predict_probability_maps_multiscale.py": "predict_probability_maps_1x_4x.py",
    "06_validate_classify_spatial_cv.py":     "validate_classify_spatial_cv_multiscale.py",
    "07_shap_analysis.py":                    "shap_analysis.py",
    "08_classify_area_analysis.py":           "classify_area_analysis.py",
    "09_generate_summary_report.py":          "generate_summary_report.py",
}

for dst_name, src_name in script_map.items():
    safe_copy(
        os.path.join(SCRIPTS, src_name),
        os.path.join(MAN, "code", dst_name),
    )

# ---------------------------------------------------------------------------
# 3. Copy figures
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("3. Copying figures …")
print("="*60)

# --- 3a. figures/main/ ---
print("  --- figures/main/ ---")
main_fig_map = [
    (os.path.join(ML, "08_Validation", "multiscale_comparison.png"),
     "Fig01_multiscale_performance_comparison.png"),
    (os.path.join(ML, "08_Validation", "10x", "roc_validation.png"),
     "Fig02_roc_curves_10x.png"),
    (os.path.join(ML, "08_Validation", "10x", "confusion_matrices.png"),
     "Fig03_confusion_matrices_10x.png"),
    (os.path.join(ML, "08_Validation", "10x", "metrics_comparison.png"),
     "Fig04_metrics_bar_chart_10x.png"),
    (os.path.join(ML, "09_Analysis/spatial_cv", "10x", "spatial_cv_comparison.png"),
     "Fig05_spatial_cv_vs_random_10x.png"),
    (os.path.join(ML, "09_Analysis/spatial_cv", "10x", "fold_map.png"),
     "Fig06_spatial_fold_assignment.png"),
    (os.path.join(ML, "09_Analysis/spatial_cv", "10x", "per_fold_performance.png"),
     "Fig07_per_fold_performance_10x.png"),
    (os.path.join(ML, "09_Analysis/shap", "10x", "shap_importance_comparison.png"),
     "Fig08_shap_importance_comparison_10x.png"),
    (os.path.join(ML, "09_Analysis/shap", "10x", "shap_beeswarm_rf.png"),
     "Fig09_shap_beeswarm_rf_10x.png"),
    (os.path.join(ML, "09_Analysis/shap", "10x", "shap_beeswarm_xgboost.png"),
     "Fig10_shap_beeswarm_xgboost_10x.png"),
    (os.path.join(ML, "09_Analysis/shap", "10x", "shap_beeswarm_catboost.png"),
     "Fig11_shap_beeswarm_catboost_10x.png"),
    (os.path.join(ML, "09_Analysis/shap", "10x", "shap_beeswarm_dnn.png"),
     "Fig12_shap_beeswarm_dnn_10x.png"),
    (os.path.join(ML, "09_Analysis/shap", "shap_rank_stability.png"),
     "Fig13_shap_rank_stability_all_sizes.png"),
    (os.path.join(ML, "09_Analysis", "class_area_comparison.png"),
     "Fig14_class_area_comparison.png"),
    (os.path.join(ML, "09_Analysis", "flood_area_comparison.png"),
     "Fig15_flood_area_comparison.png"),
    (os.path.join(ML, "09_Analysis", "threshold_probability_dist.png"),
     "Fig16_threshold_probability_distributions.png"),
    (os.path.join(OLD_FIGS, "Fig1_Probability_Maps.png"),
     "Fig17_probability_maps.png"),
    (os.path.join(OLD_FIGS, "Fig2_Susceptibility_5class.png"),
     "Fig18_susceptibility_5class_maps.png"),
    (os.path.join(OLD_FIGS, "Fig3_Binary_Maps.png"),
     "Fig19_binary_flood_maps.png"),
    (os.path.join(OLD_FIGS, "Fig4_Ensemble_Main.png"),
     "Fig20_ensemble_main_map.png"),
    (os.path.join(OLD_FIGS, "Fig5_Spatial_Agreement.png"),
     "Fig21_spatial_agreement_map.png"),
]

for src, dst_name in main_fig_map:
    safe_copy(src, os.path.join(MAN, "figures", "main", dst_name))

# --- 3b. figures/supplementary/ ---
print("  --- figures/supplementary/ ---")
supp_counter = 1
supp_dir = os.path.join(MAN, "figures", "supplementary")

for sz in SIZES:
    files = [
        (os.path.join(ML, "08_Validation", sz, "roc_validation.png"),
         f"FigS{supp_counter:02d}_roc_{sz}.png"),
        (os.path.join(ML, "08_Validation", sz, "confusion_matrices.png"),
         f"FigS{supp_counter+1:02d}_confusion_{sz}.png"),
        (os.path.join(ML, "08_Validation", sz, "csi_threshold_curves.png"),
         f"FigS{supp_counter+2:02d}_csi_curves_{sz}.png"),
        (os.path.join(ML, "09_Analysis/spatial_cv", sz, "spatial_cv_comparison.png"),
         f"FigS{supp_counter+3:02d}_spatial_cv_{sz}.png"),
        (os.path.join(ML, "09_Analysis/shap", sz, "shap_importance_comparison.png"),
         f"FigS{supp_counter+4:02d}_shap_{sz}.png"),
    ]
    for src, dst_name in files:
        safe_copy(src, os.path.join(supp_dir, dst_name))
    supp_counter += 5

# --- 3c. figures/maps/ — copy individual/ and subbasin/ as-is ---
print("  --- figures/maps/ ---")
maps_dir = os.path.join(MAN, "figures", "maps")

for sub in ["individual", "subbasin"]:
    src_dir = os.path.join(OLD_FIGS, sub)
    if not os.path.isdir(src_dir):
        print(f"  [SKIP] directory not found: {src_dir}")
        continue
    dst_sub = os.path.join(maps_dir, sub)
    os.makedirs(dst_sub, exist_ok=True)
    for fname in os.listdir(src_dir):
        safe_copy(os.path.join(src_dir, fname), os.path.join(dst_sub, fname))

# ---------------------------------------------------------------------------
# 4. Generate / copy Tables
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("4. Generating tables …")
print("="*60)

TMAIN = os.path.join(MAN, "tables", "main")
TSUPP = os.path.join(MAN, "tables", "supplementary")

# Table 01 – conditioning factors (hardcoded)
t01 = pd.DataFrame([
    (1,  "Distance to River",    "Distance_to_River",                   "Hydrological", "Euclidean distance to nearest stream network (m)"),
    (2,  "Distance to Road",     "Distance_to_Road",                    "Infrastructure","Euclidean distance to nearest road (m)"),
    (3,  "Hydrogeological Soil", "Hgk_grouped",                         "Soil",          "Grouped hydrogeological soil classification"),
    (4,  "Soil Type",            "Pedoloska_grouped",                   "Soil",          "Grouped pedological soil classification"),
    (5,  "River Density",        "River_Density",                       "Hydrological", "Stream channel density per unit area (km/km²)"),
    (6,  "Aspect",               "aspect",                              "Topographic",  "Slope facing direction (degrees from north)"),
    (7,  "Drainage Density",     "drainage_density",                    "Hydrological", "Total channel length divided by catchment area"),
    (8,  "Elevation",            "elevation",                           "Topographic",  "Digital elevation model height (m a.s.l.)"),
    (9,  "Flow Accumulation",    "flow_accumulation",                   "Hydrological", "Number of upstream contributing cells"),
    (10, "HAND",                 "hand",                                "Topographic",  "Height Above Nearest Drainage (m)"),
    (11, "Lithology",            "lito_poly_250_d96_grouped",           "Geological",   "Grouped lithological rock type classification"),
    (12, "LS Factor",            "ls_factor",                           "Topographic",  "Slope length and steepness (RUSLE) factor"),
    (13, "Plan Curvature",       "plan_curvature",                      "Topographic",  "Horizontal surface curvature (1/m)"),
    (14, "Profile Curvature",    "profile_curvature",                   "Topographic",  "Vertical surface curvature (1/m)"),
    (15, "Rainfall P90",         "slovenia_annual_max_daily_p90_3794_12m5","Climatic", "90th percentile of annual maximum daily precipitation (mm)"),
    (16, "SPI",                  "spi",                                 "Hydrological", "Stream Power Index"),
    (17, "TRI",                  "tri",                                 "Topographic",  "Terrain Ruggedness Index"),
    (18, "TWI",                  "twi",                                 "Topographic",  "Topographic Wetness Index"),
    (19, "Valley Depth",         "valley_depth",                        "Topographic",  "Depth of terrain below surrounding landscape (m)"),
], columns=["Number", "Factor", "Variable_Name", "Category", "Description"])
t01.to_csv(os.path.join(TMAIN, "Table01_conditioning_factors.csv"), index=False)
print("  [OK] Table01_conditioning_factors.csv")
generated.append("Table01_conditioning_factors.csv")

# Table 02 – sample sizes (hardcoded)
t02 = pd.DataFrame([
    ("1x",  50000,  100000,  9957,  19957, "Base configuration"),
    ("2x", 100000,  200000,  9957,  39914, "Double base"),
    ("3x", 150000,  300000,  9957,  59871, "Triple base"),
    ("4x", 200000,  400000,  9957,  79828, "Quadruple base"),
    ("10x",500000, 1000000, 99575, 199495, "Ten times base"),
], columns=["Size", "Samples_per_class", "Total_train",
            "Test_per_class", "Total_test", "Notes"])
t02.to_csv(os.path.join(TMAIN, "Table02_sample_sizes.csv"), index=False)
print("  [OK] Table02_sample_sizes.csv")
generated.append("Table02_sample_sizes.csv")

# Table 03 – validation metrics all sizes (copy)
src03 = os.path.join(ML, "08_Validation", "multiscale_validation.csv")
dst03 = os.path.join(TMAIN, "Table03_validation_metrics_all_sizes.csv")
safe_copy(src03, dst03)

# Table 04 – spatial CV results (copy)
src04 = os.path.join(ML, "09_Analysis/spatial_cv", "multiscale_cv_summary.csv")
dst04 = os.path.join(TMAIN, "Table04_spatial_cv_results.csv")
safe_copy(src04, dst04)

# Table 05 – threshold summary all sizes (merge)
frames05 = []
for sz in SIZES:
    p = os.path.join(ML, "08_Validation", sz, "threshold_summary.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p)
        df.insert(0, "SampleSize", sz)
        frames05.append(df)
    else:
        print(f"  [SKIP] {p}")

if frames05:
    pd.concat(frames05, ignore_index=True).to_csv(
        os.path.join(TMAIN, "Table05_threshold_summary_all_sizes.csv"), index=False)
    print("  [OK] Table05_threshold_summary_all_sizes.csv")
    generated.append("Table05_threshold_summary_all_sizes.csv")
else:
    print("  [WARN] No threshold_summary files found; Table05 skipped.")

# Table 06 – class area statistics (copy)
safe_copy(os.path.join(ML, "09_Analysis", "class_area_statistics.csv"),
          os.path.join(TMAIN, "Table06_class_area_statistics.csv"))

# Table 07 – binary flood statistics (copy)
safe_copy(os.path.join(ML, "09_Analysis", "binary_flood_statistics.csv"),
          os.path.join(TMAIN, "Table07_binary_flood_statistics.csv"))

# Table 08 – SHAP top-5 per model per size
src08 = os.path.join(ML, "09_Analysis/shap", "shap_importance_all_sizes.csv")
if os.path.isfile(src08):
    df08 = pd.read_csv(src08)
    rows = []
    for (sz, mdl), grp in df08.groupby(["SampleSize", "Model"], sort=False):
        top5 = grp.nlargest(5, "MeanAbsSHAP").reset_index(drop=True)
        for rank, row in top5.iterrows():
            rows.append({
                "SampleSize":   sz,
                "Model":        mdl,
                "Rank":         rank + 1,
                "Feature":      row["Feature"],
                "MeanAbsSHAP":  row["MeanAbsSHAP"],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(TMAIN, "Table08_shap_importance_top5.csv"), index=False)
    print("  [OK] Table08_shap_importance_top5.csv")
    generated.append("Table08_shap_importance_top5.csv")
else:
    print(f"  [SKIP] {src08}")

# Table 09 – model configurations (hardcoded)
t09 = pd.DataFrame([
    ("Random Forest",    "Ensemble (bagging)",
     "n_estimators=500; max_depth=25; min_samples_leaf=3; class_weight=balanced; n_jobs=-1",
     "SHAP TreeExplainer"),
    ("XGBoost",          "Gradient Boosting",
     "n_estimators=1000; max_depth=8; learning_rate=0.05; subsample=0.8; device=cuda; early_stopping=50",
     "SHAP native pred_contribs"),
    ("CatBoost",         "Gradient Boosting",
     "iterations=1000; depth=8; learning_rate=0.05; task_type=GPU; early_stopping=50",
     "SHAP TreeExplainer"),
    ("DNN",              "Deep Neural Network",
     "128->64->32->16->1; BatchNorm+ReLU+Dropout(0.3); Adam; CosineAnnealingLR; max_epochs=200; patience=20; batch=4096",
     "SHAP GradientExplainer"),
    ("Ensemble Mean",    "Model Averaging",
     "Unweighted mean of RF+XGBoost+CatBoost+DNN probabilities",
     "N/A"),
], columns=["Model", "Type", "Key_Hyperparameters", "Explainer"])
t09.to_csv(os.path.join(TMAIN, "Table09_model_configurations.csv"), index=False)
print("  [OK] Table09_model_configurations.csv")
generated.append("Table09_model_configurations.csv")

# --- Supplementary tables ---
print("  --- Supplementary tables ---")
for i, sz in enumerate(SIZES, start=1):
    # per-fold metrics
    safe_copy(
        os.path.join(ML, "09_Analysis/spatial_cv", sz, "per_fold_metrics.csv"),
        os.path.join(TSUPP, f"TableS{i:02d}_per_fold_metrics_{sz}.csv"),
    )
    # per-size validation metrics
    safe_copy(
        os.path.join(ML, "08_Validation", sz, "validation_metrics.csv"),
        os.path.join(TSUPP, f"TableS{i+5:02d}_validation_metrics_{sz}.csv"),
    )

# ---------------------------------------------------------------------------
# 5. Copy results CSVs
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("5. Copying results CSVs …")
print("="*60)

RES = os.path.join(MAN, "results")

safe_copy(os.path.join(ML, "08_Validation", "multiscale_validation.csv"),
          os.path.join(RES, "validation", "multiscale_validation.csv"))
safe_copy(os.path.join(ML, "09_Analysis/spatial_cv", "multiscale_cv_summary.csv"),
          os.path.join(RES, "spatial_cv", "multiscale_cv_summary.csv"))
safe_copy(os.path.join(ML, "09_Analysis/shap", "shap_importance_all_sizes.csv"),
          os.path.join(RES, "shap", "shap_importance_all_sizes.csv"))
safe_copy(os.path.join(ML, "09_Analysis", "binary_flood_statistics.csv"),
          os.path.join(RES, "area_statistics", "binary_flood_statistics.csv"))
safe_copy(os.path.join(ML, "09_Analysis", "class_area_statistics.csv"),
          os.path.join(RES, "area_statistics", "class_area_statistics.csv"))
safe_copy(os.path.join(ML, "09_Analysis", "frequency_ratio.csv"),
          os.path.join(RES, "area_statistics", "frequency_ratio.csv"))

for sz in SIZES:
    safe_copy(os.path.join(ML, "08_Validation", sz, "validation_metrics.csv"),
              os.path.join(RES, "validation", sz, "validation_metrics.csv"))
    safe_copy(os.path.join(ML, "08_Validation", sz, "threshold_summary.csv"),
              os.path.join(RES, "validation", sz, "threshold_summary.csv"))
    safe_copy(os.path.join(ML, "09_Analysis/spatial_cv", sz, "mean_metrics.csv"),
              os.path.join(RES, "spatial_cv", sz, "mean_metrics.csv"))
    safe_copy(os.path.join(ML, "09_Analysis/spatial_cv", sz, "per_fold_metrics.csv"),
              os.path.join(RES, "spatial_cv", sz, "per_fold_metrics.csv"))
    safe_copy(os.path.join(ML, "09_Analysis/shap", sz, "shap_importance_all_models.csv"),
              os.path.join(RES, "shap", sz, "shap_importance_all_models.csv"))

# ---------------------------------------------------------------------------
# 6. Generate workflow diagram
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("6. Generating workflow diagram …")
print("="*60)

steps = [
    ("Data Collection & Preparation",
     "19 conditioning factors | EPSG:3794 | 12.5 m"),
    ("Collinearity Reduction",
     "VIF analysis  →  19 selected features"),
    ("Sample Generation",
     "5 sizes: 1x (50k) → 10x (500k) per class\nBalanced flood / non-flood"),
    ("Model Training",
     "RF | XGBoost | CatBoost | DNN\nPer sample size  →  models_{size}/"),
    ("Probability Map Generation",
     "Full Slovenia raster (strip-based)\n5 models × 5 sizes = 25 maps"),
    ("Threshold Optimisation",
     "CSI-optimal & Youden thresholds"),
    ("Map Classification",
     "Binary (flood / no-flood) maps\n5-class susceptibility maps"),
    ("Validation",
     "Standard: AUC / F1 / CSI / Kappa\nSpatial Block CV: 5-fold 50 km blocks"),
    ("SHAP Explainability",
     "TreeExplainer | GradientExplainer\nBeeswarm | Bar | Dependence | Rank Stability"),
    ("Output & Reporting",
     "Probability maps | Classified maps\nValidation plots | Summary report"),
]

BOX_W  = 5.6
BOX_H  = 0.78
GAP    = 0.42
FIG_W  = 8.0
FIG_H  = 14.0

N = len(steps)
total_h = N * BOX_H + (N - 1) * GAP
x_center = FIG_W / 2

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# title
ax.text(x_center, FIG_H - 0.45, "FSM Slovenia — Analytical Workflow",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a2e")

# colours
BOX_COL  = "#1b3a6b"   # deep blue
TEXT_COL = "white"
ARROW_COL = "#555555"

top_start = FIG_H - 1.0   # y of top edge of first box

box_centres_y = []
for i, (title, subtitle) in enumerate(steps):
    y_top    = top_start - i * (BOX_H + GAP)
    y_centre = y_top - BOX_H / 2
    box_centres_y.append(y_centre)

    rect = FancyBboxPatch(
        (x_center - BOX_W / 2, y_centre - BOX_H / 2),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.04",
        linewidth=1.2,
        edgecolor="#3a6fbf",
        facecolor=BOX_COL,
    )
    ax.add_patch(rect)

    # step number badge
    badge_x = x_center - BOX_W / 2 + 0.18
    ax.text(badge_x, y_centre + 0.06,
            f"{i+1:02d}",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="#aac4f7")

    # title text
    ax.text(x_center + 0.12, y_centre + 0.14,
            title,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=TEXT_COL)

    # subtitle text
    ax.text(x_center + 0.12, y_centre - 0.16,
            subtitle,
            ha="center", va="center",
            fontsize=7.5, color="#c8d8f0",
            linespacing=1.4)

# arrows between boxes
for i in range(len(steps) - 1):
    y_bottom_current = box_centres_y[i] - BOX_H / 2
    y_top_next       = box_centres_y[i + 1] + BOX_H / 2
    ax.annotate("",
        xy=(x_center, y_top_next + 0.01),
        xytext=(x_center, y_bottom_current - 0.01),
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COL,
            lw=1.5,
            mutation_scale=14,
        ),
    )

plt.tight_layout(pad=0.3)
diag_path = os.path.join(MAN, "docs", "workflow_diagram.png")
fig.savefig(diag_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"  [OK] docs/workflow_diagram.png")
generated.append("docs/workflow_diagram.png")

# ---------------------------------------------------------------------------
# 7. Copy summary report
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("7. Copying summary report …")
print("="*60)

safe_copy(
    os.path.join(ML, "FSM_Slovenia_Summary_Report.docx"),
    os.path.join(MAN, "docs", "FSM_Slovenia_Summary_Report.docx"),
)

# ---------------------------------------------------------------------------
# 8. Write README.md
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("8. Writing README.md …")
print("="*60)

readme = textwrap.dedent("""\
# Flood Susceptibility Mapping of Slovenia using Multi-Scale Machine Learning

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![CRS EPSG:3794](https://img.shields.io/badge/CRS-EPSG%3A3794-orange)

## Abstract

This study presents a multi-scale flood susceptibility mapping (FSM) framework for
Slovenia (20,254 km²) using four machine learning algorithms — Random Forest (RF),
XGBoost, CatBoost, and a Deep Neural Network (DNN) — trained on five sample sizes
ranging from 50,000 to 500,000 flood/non-flood points per class at 12.5 m spatial
resolution in EPSG:3794.
Model performance was assessed through standard hold-out validation (AUC, F1, CSI,
Cohen's Kappa) and a spatially-aware 5-fold block cross-validation with 50 km blocks
to quantify overfitting.
SHAP (SHapley Additive exPlanations) values were computed for all models and sample
sizes to rank and interpret the 19 conditioning factors and assess rank stability
across configurations.

---

## Study Area

| Property | Value |
|---|---|
| Country | Slovenia |
| Area | ~20,254 km² |
| Resolution | 12.5 m |
| Coordinate Reference System | EPSG:3794 (D96/TM) |
| Flood inventory | Historical flood extents (merged) |

---

## Repository Structure

```
Manuscript/
├── README.md                  ← this file
├── .gitignore
├── LICENSE                    ← MIT 2026
├── environment.yml            ← conda environment
├── requirements.txt           ← pip packages
├── code/
│   ├── 01_prepare_rasters.py
│   ├── 02_align_and_collinearity.py
│   ├── 03_train_models_multisample.py
│   ├── 04_predict_probability_maps.py
│   ├── 05_predict_probability_maps_multiscale.py
│   ├── 06_validate_classify_spatial_cv.py
│   ├── 07_shap_analysis.py
│   ├── 08_classify_area_analysis.py
│   └── 09_generate_summary_report.py
├── figures/
│   ├── main/                  ← Figures 01–21 (manuscript main body)
│   ├── supplementary/         ← Figures S01–S25 (per-size validation)
│   └── maps/
│       ├── individual/        ← per-model raster map exports
│       └── subbasin/          ← sub-basin level analysis plots
├── tables/
│   ├── main/                  ← Tables 01–09 (CSV)
│   └── supplementary/         ← per-fold & per-size metrics (CSV)
├── results/
│   ├── validation/            ← validation metrics per size
│   ├── spatial_cv/            ← spatial block CV results per size
│   ├── shap/                  ← SHAP importance CSVs per size
│   └── area_statistics/       ← flood / class area statistics
└── docs/
    ├── workflow_diagram.png   ← auto-generated pipeline flowchart
    └── FSM_Slovenia_Summary_Report.docx
```

---

## Conditioning Factors (19)

| # | Factor | Category |
|---|--------|----------|
| 1 | Distance to River | Hydrological |
| 2 | Distance to Road | Infrastructure |
| 3 | Hydrogeological Soil | Soil |
| 4 | Soil Type | Soil |
| 5 | River Density | Hydrological |
| 6 | Aspect | Topographic |
| 7 | Drainage Density | Hydrological |
| 8 | Elevation | Topographic |
| 9 | Flow Accumulation | Hydrological |
| 10 | HAND | Topographic |
| 11 | Lithology | Geological |
| 12 | LS Factor | Topographic |
| 13 | Plan Curvature | Topographic |
| 14 | Profile Curvature | Topographic |
| 15 | Rainfall P90 | Climatic |
| 16 | SPI | Hydrological |
| 17 | TRI | Topographic |
| 18 | TWI | Topographic |
| 19 | Valley Depth | Topographic |

> **Note:** Raw raster files (~tens of GB) are **not** included in this repository.
> Download them from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) *(placeholder — update before submission)*

---

## Methods Summary

### Models
| Model | Type | Key settings |
|---|---|---|
| Random Forest | Ensemble (bagging) | 500 trees, max_depth=25, balanced weights |
| XGBoost | Gradient Boosting | 1000 estimators, lr=0.05, GPU |
| CatBoost | Gradient Boosting | 1000 iterations, depth=8, GPU |
| DNN | Deep Neural Network | 128→64→32→16→1, BatchNorm+Dropout |
| Ensemble Mean | Model Averaging | Unweighted mean of all four |

### Sample Sizes
Five training configurations: **1x** (50k), **2x** (100k), **3x** (150k), **4x** (200k),
**10x** (500k) flood and non-flood points each (balanced).

### Validation
- **Hold-out**: 20% stratified split; metrics: AUC, Average Precision, F1, Accuracy,
  Cohen's Kappa, CSI, Sensitivity, Specificity.
- **Spatial Block CV**: 5-fold, 50 km blocks to avoid spatial autocorrelation leakage.
- **Threshold optimisation**: CSI-optimal and Youden-J thresholds per model per size.

### SHAP Explainability
- TreeExplainer for RF and CatBoost.
- Native `pred_contribs` for XGBoost.
- GradientExplainer for DNN.
- Beeswarm, bar, dependence, and rank-stability plots across all sizes.

---

## Results Summary

### Best model performance (10x sample size — hold-out test set)

| Model | AUC | F1 | CSI | Kappa |
|---|---|---|---|---|
| XGBoost | 0.987 | 0.953 | 0.909 | 0.903 |
| CatBoost | 0.986 | 0.950 | 0.905 | 0.898 |
| Ensemble | 0.985 | 0.948 | 0.902 | 0.894 |
| RF | 0.982 | 0.940 | 0.887 | 0.879 |

*(Full metrics available in `results/validation/` and `tables/main/Table03_validation_metrics_all_sizes.csv`)*

---

## Usage

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate floodcast_env
```

### 2. Run the analytical pipeline (in order)

```bash
conda activate floodcast_env

python code/01_prepare_rasters.py
python code/02_align_and_collinearity.py
python code/03_train_models_multisample.py
python code/04_predict_probability_maps.py
python code/05_predict_probability_maps_multiscale.py
python code/06_validate_classify_spatial_cv.py
python code/07_shap_analysis.py
python code/08_classify_area_analysis.py
python code/09_generate_summary_report.py
```

> Each script writes outputs to `ML_Results/`. Adjust path constants at the top of
> each script to match your local data location.

---

## Citation

If you use this code or data in your research, please cite:

```
Arora, A. et al. (2026). Flood Susceptibility Mapping of Slovenia using
Multi-Scale Machine Learning at 12.5 m Resolution. [Journal Name].
https://doi.org/10.XXXX/XXXXXX  ← update before submission
```

---

## License

This project is released under the [MIT License](LICENSE).
© 2026 Aman Arora
""")

with open(os.path.join(MAN, "README.md"), "w", encoding="utf-8") as fh:
    fh.write(readme)
print("  [OK] README.md")
generated.append("README.md")

# ---------------------------------------------------------------------------
# 9. Write .gitignore
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("9. Writing .gitignore …")
print("="*60)

gitignore = textwrap.dedent("""\
# Large raster files
*.tif
*.tiff
*.img

# Large model files
*.pkl
*.cbm
*.pth

# Sample data (large CSVs)
ML_Results/samples/

# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Jupyter
*.ipynb

# Temp
*.tmp
*.log
""")

with open(os.path.join(MAN, ".gitignore"), "w", encoding="utf-8") as fh:
    fh.write(gitignore)
print("  [OK] .gitignore")
generated.append(".gitignore")

# ---------------------------------------------------------------------------
# 10. Write environment.yml
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("10. Writing environment.yml …")
print("="*60)

env_yml = textwrap.dedent("""\
name: floodcast_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - rasterio
  - geopandas
  - gdal
  - pip
  - pip:
    - xgboost
    - catboost
    - torch
    - torchvision
    - shap
    - joblib
    - python-docx
    - openpyxl
""")

with open(os.path.join(MAN, "environment.yml"), "w", encoding="utf-8") as fh:
    fh.write(env_yml)
print("  [OK] environment.yml")
generated.append("environment.yml")

# ---------------------------------------------------------------------------
# 11. Write requirements.txt
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("11. Writing requirements.txt …")
print("="*60)

req_txt = textwrap.dedent("""\
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
xgboost>=2.0
catboost>=1.2
torch>=2.0
shap>=0.43
joblib>=1.3
python-docx>=1.0
openpyxl>=3.1
rasterio>=1.3
""")

with open(os.path.join(MAN, "requirements.txt"), "w", encoding="utf-8") as fh:
    fh.write(req_txt)
print("  [OK] requirements.txt")
generated.append("requirements.txt")

# ---------------------------------------------------------------------------
# 12. Write LICENSE
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("12. Writing LICENSE …")
print("="*60)

license_txt = textwrap.dedent("""\
MIT License

Copyright (c) 2026 Aman Arora

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")

with open(os.path.join(MAN, "LICENSE"), "w", encoding="utf-8") as fh:
    fh.write(license_txt)
print("  [OK] LICENSE")
generated.append("LICENSE")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Files copied successfully : {len(copied_ok)}")
print(f"  Files skipped / not found: {len(copied_err)}")
print(f"  Files generated          : {len(generated)}")
print(f"\n  Manuscript root: {MAN}")
if copied_err:
    print("\n  Skipped sources (review if needed):")
    for p in copied_err:
        print(f"    - {p}")
print("\nDone.")
