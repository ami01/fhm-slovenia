# FSM Slovenia — Pipeline (LULC run)

Flood susceptibility mapping for Slovenia using Random Forest, XGBoost, CatBoost and a
PyTorch MLP, at 12.5 m across the whole country (258.8 M cells, EPSG:3794).

Code and results for: Arora, A., Petan, S. *Beyond the training inventory: transferability and independent
validation of national machine-learning flood susceptibility maps for Slovenia* (submitted to the Journal of Hydrology).
The manuscript itself is not part of this repository. The maps, conditioning factors, trained models and samples are
on Zenodo: https://doi.org/10.5281/zenodo.23726819 (CC BY 4.0). Code licence: MIT.

### Start here

| Document | What it covers |
|---|---|
| **[PROGRESS.md](PROGRESS.md)** | What was done, all results, defects found, compute cost, open decisions |
| **[10_Documentation/dataset_inventory.md](10_Documentation/dataset_inventory.md)** | Every dataset: type, resolution, rows × cols, source, derivation |
| **[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)** | Where the source data comes from and how to regenerate what isn't shipped |
| **[99_Logs/stage_timings.csv](99_Logs/stage_timings.csv)** | Measured duration of every stage, failures included |
| This file | Execution order, folder layout, dependency graph, caveats |

Reproduce the environment with `requirements.txt` or `environment.yml`. Note that
`statsmodels` and `shap` were missing from the original environment and are required by
stages 04 and 08d.

Scripts in `00_Scripts/` are numbered **in execution order**: `00a` runs first,
`11d` runs last.

This tree is a *new* run that adds the **LULC** conditioning factor derived from
RABA 2026-06-30. The original `For_Analysis/ML_Results/` and
`For_Analysis/Collinearity/` are **not touched** — the previous 19-feature results
remain intact for comparison.

---

## Folder layout

| Folder | Contents |
|---|---|
| `00_Scripts/` | All pipeline scripts, numbered in execution order |
| `01_Input_Features/` | Conditioning-factor rasters (symlinks to `CgFs_Aligned/`, 12.5 m, EPSG:3794) + `LULC.tif` |
| `02_Collinearity/` | VIF elimination log, correlation matrix, `selected_features.txt` |
| `03_Samples/` | Balanced train/test sample CSVs and shapefiles |
| `04_Models/` | Trained model files (RF, XGBoost, CatBoost, DNN) |
| `05_Results/` | Metrics, importance plots, per-model results |
| `06_Probability_Maps/` | Continuous susceptibility probability rasters |
| `07_Classified_Maps/` | Thresholded 5-class susceptibility rasters |
| `08_Validation/` | Validation metrics, ROC/PR curves |
| `09_Analysis/` | Area statistics, frequency ratio, spatial CV, SHAP, sub-basin |
| `10_Documentation/` | Word documents (codebook, RABA→LULC conversion, scripts summary) |
| `99_Logs/` | stdout/stderr of every stage |
| `supp_tables/` | Complete machine-readable tables summarised in the Supplementary Material (hold-out and per-fold results, transferability folds, frequency ratios, resolution analysis, fold composition) |

`01_Input_Features/` holds **symlinks**, not copies — the rasters are 13 GB and
duplicating them serves no purpose. Delete the folder and re-link if the source moves.

---

## Execution order

### Stage 00 — Raw DEM assembly *(already complete, not re-run)*
| # | Script | Purpose |
|---|---|---|
| 00a | `DMV0125_to_geotiff.py` | DMV0125 ASCII XYZ tiles → GeoTIFF |
| 00b | `Merge_DMV_0125.py` | Merge 12.5 m DEM tiles |
| 00c | `Merge_DMV_0050.py` | Merge 5 m DEM tiles |
| 00d | `merge_rasters.py` | Mosaic GeoTIFF tiles |

### Stage 01 — Conditioning factor derivation *(already complete)*
| # | Script | Purpose |
|---|---|---|
| 01a | `DEM_CgFs.py` | GPU derivation of slope, aspect, TWI, SPI, TPI, TRI, HAND, curvature, LS-factor, valley depth |
| 01b | `extract_river_network.py` | Stream network + Strahler order (PySheds) |
| 01c | `merge_flood_zones.py` | IKPN Q500 flood zones → binary hazard layer |
| 01d | `dissolve.py` | Dissolve hazard layer to 2 classes |
| 01e | `prepare_river_raster.py` | River shapefile prep |
| 01f | `prepare_rasters.py` | Add GRP_DESC/CODE columns, rasterise the grouped geology/soil layers |
| 01g | `river_road_rasters.py` | River density, road distance, drainage density |
| 01h | `Slovenia_Rainfall_Interpolation.py` | Interpolate + reproject + clip rainfall |
| 01i | `prepare_rainfall.py` | Extract E-OBS rainfall statistics |
| 01j | `mean_annual_daily_max_rain.py` | Annual max daily rainfall maps |

### Stage 02 — LULC derivation *(NEW — this run)*
| # | Script | Purpose |
|---|---|---|
| 02a | `raba_fractions.py` | RABA polygons → per-class areal fractions (burn at 1.25 m, aggregate 10×10) |
| 02b | `make_lulc.py` | Extract modal band → `LULC.tif` (codes 1–7, nodata 0) |
| 02c | `qa_raba_fractions.py` | Verify grid identity, fraction/coverage identity, areal totals |

See `10_Documentation/RABA_LULC_Conversion.docx` for the full method and statistics.

### Stage 03 — Grid alignment *(already complete)*
| # | Script | Purpose |
|---|---|---|
| 03a | `CgFs_Reproject.py` | Reproject/resample all factors → EPSG:3794 @ 12.5 m |
| 03b | `mask_to_boundary.py` | Clip all rasters to the Slovenia boundary |
| 03c | `align_and_collinearity.py` | Align to common grid + first-pass collinearity |

### Stage 04 — Feature selection ← **re-run starts here**
| # | Script | Purpose |
|---|---|---|
| 04 | `iterative_collinearity.py` | Balanced sampling + iterative VIF elimination (>10) + correlation screen (\|r\|>0.8) → `selected_features.txt` |

Discovers features by listing `01_Input_Features/*.tif`, so `LULC.tif` is included
automatically. **Writes `02_Collinearity/selected_features.txt`, which every later
stage reads.**

### Stage 05 — Model training
| # | Script | Purpose |
|---|---|---|
| 05a | `train_models.py` | Random Forest + XGBoost + CatBoost, 50k samples/class |
| 05b | `train_dnn.py` | PyTorch MLP 128→64→32→16→1 — trains and saves `model_dnn.pth` |
| 05c | `train_models_multisample.py` | Optional: repeat at 1×/2×/3×/4× sample sizes |

`05b` must precede stage 06, because `predict_probability_maps.py` loads
`model_dnn.pth` to build the DNN probability map alongside the tree models.

### Stage 06 — Prediction
| # | Script | Purpose |
|---|---|---|
| 06 | `predict_probability_maps.py` | Strip-based full-Slovenia probability rasters (259 M cells × each model) |

### Stage 07 — Classification
| # | Script | Purpose |
|---|---|---|
| 07 | `threshold_classify_validate.py` | Threshold optimisation → 5-class maps → validation metrics |

### Stage 08 — Validation and interpretation
| # | Script | Purpose |
|---|---|---|
| 08a | `dnn_merge_validation.py` | Merge the DNN row into `validation_metrics.csv`, redraw the 5-model comparison plots |
| 08b | `spatial_cv.py` | Spatial block cross-validation |
| 08c | `validate_classify_spatial_cv_multiscale.py` | Multi-scale validation across sample sizes |
| 08d | `shap_analysis.py` | SHAP feature attribution |

`08a` is the second half of the original `train_dnn.py` — see *Split scripts* below.

### Stage 09 — Spatial analysis
| # | Script | Purpose |
|---|---|---|
| 09a | `classify_area_analysis.py` | Area per susceptibility class |
| 09b | `analysis_fr_area.py` | Frequency ratio + class area statistics |
| 09c | `subbasin_analysis.py` | Zonal statistics by sub-basin |

### Stage 10 — Figures
| # | Script | Purpose |
|---|---|---|
| 10a | `figures_maps.py` | Publication figures |
| 10b | `generate_maps_all_sizes.py` | Figures for every sample size |
| 10c | `fix_comparison_maps.py` | Fix legend overlap in comparison figures |
| 10d | `setup_manuscript_folder.py` | Assemble the manuscript folder |

### Stage 11 — Documentation *(run last)*
| # | Script | Purpose |
|---|---|---|
| 11a | `export_codebook.py` | Raster code descriptions → docx |
| 11b | `export_raba_lulc_conversion.py` | RABA→LULC conversion + statistics → docx |
| 11c | `export_scripts_summary.py` | Script summary → docx |
| 11d | `generate_summary_report.py` | Final summary report → docx |

---

## Dependency chain

```
01_Input_Features/*.tif  (incl. LULC.tif)
        |
        v
[04] iterative_collinearity  ->  02_Collinearity/selected_features.txt
        |
        +--> [05a] train_models  ->  03_Samples/  04_Models/  05_Results/
        +--> [05b] train_dnn     ->  04_Models/model_dnn.pth
        |                            08_Validation/dnn_metrics.json  (hand-off)
        v
[06] predict_probability_maps    ->  06_Probability_Maps/
     (loads RF + XGB + CatBoost + DNN)
        |
        v
[07] threshold_classify_validate ->  07_Classified_Maps/
                                     08_Validation/validation_metrics.csv
        |
        v
[08a] dnn_merge_validation       ->  08_Validation/  (DNN row + comparison plots)
        |
        +--> [08b-d] spatial_cv, multiscale, SHAP  ->  09_Analysis/
        +--> [09]    area / frequency-ratio / sub-basin  ->  09_Analysis/
                    |
                    v
             [10] figures  ->  Manuscript figures
                    |
                    v
             [11] documentation  ->  10_Documentation/*.docx
```

Two hard constraints:
- **04 precedes everything** — `selected_features.txt` is the contract every
  downstream script reads.
- **05b precedes 06, and 08a follows 07** — the DNN's training and its validation
  merge sit on opposite sides of the prediction/classification stages.

Verify the ordering at any time with `python3 00_Scripts/_analyse_deps.py`, which
derives the graph from what each script actually reads and writes and reports any
script numbered before something it depends on.

---

## Notes and caveats

**Split scripts.** The original `train_dnn.py` welded together two phases that
belong on opposite sides of the pipeline: it trained the network (which stage 06
depends on) *and* merged its metrics into `validation_metrics.csv` (which stage 07
produces). As one file it was unorderable — a genuine circular dependency, and it
crashed with `FileNotFoundError` on a clean run. It is split into:

- `05b_train_dnn.py` — trains, saves `model_dnn.pth`, writes its own probability
  and classified maps, then persists `dnn_metrics.json` and
  `dnn_test_predictions.npz` as a hand-off.
- `08a_dnn_merge_validation.py` — loads the hand-off plus stage 07's
  `validation_metrics.csv`, merges the DNN row, redraws the 5-model comparison
  plots. Exits with a clear message if run before 05b or 07.

The original unsplit file is preserved at
`For_Analysis/Scripts/train_dnn.py` (untouched).

**Helper scripts** (underscore-prefixed, not pipeline stages):
`_patch_paths.py` rebuilds `00_Scripts/` from the originals;
`_analyse_deps.py` verifies the numbering against the real dependency graph;
`_split_dnn.py` performs the train_dnn split.

**Paths.** Scripts in stages 04–11 were path-patched to write into this tree; the
originals in `For_Analysis/Scripts/` are unmodified. Stages 00–03 are copied
verbatim for provenance and still reference the original data locations — they are
not re-run.

**Model directory collision.** The original layout separated `models/` (50k
samples/class) from `models_10x/` (500k). Both map to `04_Models/` here, so stage 06
consumes whatever stage 05 produced. Stage 05a trains at 50k/class, matching the
original 1× configuration — the right baseline for judging whether LULC helps.
If you want the 10× comparison, run 05c first.

**LULC is nominal, encoded ordinally.** `train_models.py` passes features as plain
float columns, with no one-hot encoding and no CatBoost `cat_features`. Codes 1–7
therefore carry an ordering that has no meaning. This matches the existing treatment
of `Hgk_grouped`, `Pedoloska_grouped` and `lito_poly_250_d96_grouped`, so it is
consistent — but it is a modelling compromise, and it costs the DNN more than the
tree models. `raba_frac_202606.tif` holds the continuous fractional alternative if
you want to test it.

**VIF on a nominal categorical** is not strictly meaningful either, for the same
reason and again consistently with the three existing grouped rasters.

**Environment.** All stages run under `floodcast_env`:
`conda run -n floodcast_env python3 00_Scripts/<script>`.
`statsmodels` was missing and installed for stage 04.
