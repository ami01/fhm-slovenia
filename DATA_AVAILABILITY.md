# Data Availability

This repository contains **code, metrics, figures and reports** — not the raster data.
The input stack is 13 GB and the model outputs 16 GB, so neither is version-controlled.
Everything excluded is either a public download or regenerable from the scripts here.

## Source datasets (public, not redistributed here)

| Dataset | Provider | Notes |
|---|---|---|
| DMV0125 — 12.5 m DEM | GURS (Surveying and Mapping Authority of Slovenia) | ASCII XYZ tiles; converted by `00a_DMV0125_to_geotiff.py` |
| RABA — actual land use | MKGP (Ministry of Agriculture, Forestry and Food) | Release used: **2026-06-30**, 1,698,891 polygons. Monthly releases from 2015 onward |
| DTM_SLO hydrography / roads | GURS | Watercourses (537,172 lines), roads (162,435 lines) |
| IKPN Q500 flood hazard | DRSV (Slovenian Water Agency) | 500-year return period; the target variable |
| Pedological (soil) map | — *provider unverified in the scripts* | 7,976 polygons |
| Lithology 1:250,000 | — *provider unverified in the scripts* | 4,551 polygons, ESRI:102109 |
| Hydrogeological units | — *provider unverified in the scripts* | 2,418 polygons, LAWA sub-codes, EPSG:4326 |
| E-OBS gridded rainfall | ECMWF / Copernicus | `rr_ens_mean_0.1deg` |

Provider names marked *unverified* are not stated in the generating scripts; confirm
before citing.

## Regenerating what is excluded

Run stages in the order given in `README.md`. Each stage writes into its numbered folder.

| Excluded | Regenerate with | Cost |
|---|---|---|
| `01_Input_Features/*.tif` (13 GB) | stages 00–03 on the source downloads | hours |
| `LULC.tif`, `raba_frac_202606.tif` | `02a_raba_fractions.py` + `02b_make_lulc.py` | 35 min |
| `02_Collinearity/samples_balanced_masked.csv` | `04_iterative_collinearity.py` | 11 min |
| `03_Samples/` (1.5 GB) | `05a` / `05c` | included in training |
| `04_Models/` (1.5 GB) | `05a` (1x), `05c` (all sizes) | 2h 54m |
| `06_Probability_Maps/` (16 GB) | `06b_predict_probability_maps_all_sizes.py` | 4h 06m |
| `07_Classified_Maps/` (670 MB) | `07_threshold_classify_validate.py` | 6 min |
| `**/fold_assignments.csv` (508 MB) | `08b` / `08c` | 26 min / 9h 37m |

Full per-stage timings: `99_Logs/stage_timings.csv`.

## What *is* in the repository

- **`00_Scripts/`** — all 41 pipeline scripts, numbered in execution order, plus the
  helper scripts that patched paths, verified the dependency graph and split `train_dnn.py`
- **`02_Collinearity/`** — VIF elimination log, final VIF table, correlation matrix,
  `selected_features.txt`
- **`05_Results/`** — per-size metrics, the baseline comparison, the DNN seed-variation and
  LULC-encoding studies, importance/ROC/PR plots
- **`08_Validation/`** — validation metrics, confusion matrices, threshold curves, reports
- **`09_Analysis/`** — spatial-CV summaries (per fold and per size), SHAP importances
- **`10_Documentation/`** — `RABA_LULC_Conversion.docx`, `dataset_inventory.md`/`.csv`
- **`99_Logs/`** — stdout of every stage and `stage_timings.csv`

Grid metadata for every raster, including the ones not shipped, is in
`10_Documentation/dataset_inventory.md` and `.csv`.
