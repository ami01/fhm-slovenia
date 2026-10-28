# FSM Slovenia — LULC Run Progress

**Period:** 2026-07-27 → 2026-07-30
**Goal:** add a land-use/land-cover conditioning factor derived from RABA 2026-06-30 to
the flood-susceptibility pipeline, reorganise the pipeline into a documented execution
order, and re-run training / prediction / validation at every sample size.

Original results in `For_Analysis/ML_Results/` were left untouched throughout; everything
here is a parallel tree.

---

## 1. What was built

### LULC factor
RABA polygons (1,698,891 features, EPSG:3794) → per-class areal fractions burned at
1.25 m and block-aggregated 10×10 → `raba_frac_202606.tif` (9 bands) → modal class →
`CgFs_Aligned/LULC.tif` (7 classes, nodata 0).

Validated: grid identical to the feature stack, `max|Σfractions − coverage| = 6.3e-08`,
coverage ≥ 0.999 on 99.94 % of 129.7 M valid cells, group areas total **20,266 km²**
against Slovenia's 20,271 km² (0.02 % difference). All 25 RABA codes mapped, none dropped.

Method and statistics: `10_Documentation/RABA_LULC_Conversion.docx`.

### Ordered pipeline
41 scripts renumbered `00a` → `11d` across 12 stages in `00_Scripts/`, outputs in
numbered folders `01_Input_Features` → `10_Documentation`. Execution order, dependency
diagram and caveats in `README.md`. `_analyse_deps.py` derives the dependency graph from
what each script reads and writes and verifies the numbering.

---

## 2. Results

### LULC helps the tree models, slightly
Paired comparison (same seed, same split, only the feature set differs), 50k samples/class:

| Model | AUC 19-feat | AUC +LULC | Δ AUC | Δ CSI | Δ Kappa |
|---|---|---|---|---|---|
| XGBoost | 0.9871 | 0.9877 | +0.0006 | +0.0013 | +0.0017 |
| CatBoost | 0.9859 | 0.9864 | +0.0005 | +0.0012 | +0.0014 |
| Random Forest | 0.9815 | 0.9821 | +0.0006 | +0.0010 | +0.0012 |

LULC survived VIF screening at **1.308** and the selected set is exactly the original 19
features plus LULC — nothing displaced, and the same four features eliminated as before
(`slope`, `tpi`, and two rainfall layers).
Ranks 10th of 20 by native importance, 5th–6th by SHAP on the boosted models.

### The DNN needed a seeded study to say anything
`train_dnn.py` set no random seed. Two identical runs differed by ~0.0005 AUC and
~0.004 Kappa — the same magnitude as the effects being measured. Seeded it, then ran
10 seeds per configuration.

| variant | AUC | CSI | F1 | Kappa |
|---|---|---|---|---|
| `baseline19` | 0.9770 ±0.0003 | 0.8803 ±0.0017 | 0.9364 ±0.0010 | 0.8694 ±0.0023 |
| `ordinal20` (codes 1–7) | 0.9779 ±0.0003 | 0.8798 ±0.0008 | 0.9361 ±0.0004 | 0.8686 ±0.0010 |
| `onehot` (6 dummies) | 0.9780 ±0.0002 | 0.8801 ±0.0007 | 0.9362 ±0.0004 | 0.8684 ±0.0013 |
| **`fractions`** (6 continuous) | **0.9783 ±0.0003** | **0.8811 ±0.0016** | **0.9368 ±0.0009** | **0.8698 ±0.0019** |

Every encoding improves AUC significantly (3.1–4.2 sd); none changes CSI/F1/Kappa beyond
noise. **`fractions` is best on all four metrics** — an argument for using the continuous
fraction bands rather than the categorical raster.

Encoding is *not* the deciding factor: `onehot` and `ordinal20` are statistically
indistinguishable.

### Sample size helps on random splits and not at all spatially
Random split, held-out pixels — every model improves monotonically:

| Size | XGBoost | CatBoost | RF | DNN |
|---|---|---|---|---|
| 1x (50k/class) | 0.9877 | 0.9864 | 0.9821 | 0.9788 |
| 10x (500k/class) | **0.9941** | 0.9910 | 0.9903 | 0.9860 |

Spatial block CV, 5 folds, 50 km blocks — flat or *declining*:

| Model | 1x | 2x | 3x | 4x | 10x | trend |
|---|---|---|---|---|---|---|
| **RF** | 0.9663 | 0.9668 | 0.9667 | 0.9668 | 0.9665 | flat, best at every size |
| XGBoost | 0.9636 | 0.9613 | 0.9604 | 0.9595 | **0.9583** | declining, 5/5 |
| CatBoost | 0.9647 | 0.9633 | 0.9624 | 0.9615 | 0.9620 | declining |
| DNN | 0.9623 | 0.9617 | 0.9629 | 0.9617 | 0.9616 | flat |

XGBoost CSI declines 0.8149 → 0.8025 over the same range.

**This is the most consequential finding of the run.** The gains from larger samples on
random splits appear to be largely an artifact of spatial autocorrelation: more pixels
from the same flood polygons put more near-duplicate neighbours on both sides of a random
split. Generalisation to unseen *regions* does not improve, and XGBoost — the most
flexible learner — degrades. For a map intended for areas without flood records, the
honest figures are **AUC ≈ 0.964, CSI ≈ 0.82**, not CSI 0.94.

Caveat: per-fold spread (±0.016–0.024 AUC) exceeds the gaps between models, so no single
pairwise ranking is individually significant. The XGBoost decline across 5 of 5
independent size runs is a consistent pattern; RF's flatness is the robust part.

### Validation, full Slovenia, 10x
| Model | Threshold | AUC | CSI | F1 | Kappa |
|---|---|---|---|---|---|
| XGBoost | 0.5369 | 0.9941 | 0.9406 | 0.9694 | 0.9381 |
| Ensemble_mean | 0.5566 | 0.9921 | 0.9280 | 0.9627 | 0.9244 |
| CatBoost | 0.5222 | 0.9910 | 0.9255 | 0.9613 | 0.9216 |
| RF | 0.5172 | 0.9903 | 0.9186 | 0.9576 | 0.9138 |
| DNN | 0.5320 | 0.9860 | 0.9085 | 0.9520 | 0.9019 |

199,495 held-out test samples. LULC version gains +0.0003 AUC / +0.0007 CSI / +0.0008
Kappa over the original 10x run.

---

## 3. Defects found and fixed

| Issue | Consequence if unfixed |
|---|---|
| `DEM_Filled_Slovenia.tif` is 12.46 m, not 12.5 m, with a different origin | LULC raster would not have aligned with the feature stack |
| `train_dnn.py` welded two phases with a circular dependency (trains before stage 06, merges after stage 07) | crashed on any clean run; unorderable as one file. Split into `05b` + `08a` |
| Path patching missed f-string-derived paths | `09c`/`10a`/`10b`/`10c` would have **overwritten the original manuscript figures**; three `.docx` generators would have overwritten the originals |
| `05c` wrote `models_{size}/` into the pipeline root | bypassed the numbered structure |
| `08d` still expected the legacy `models_1x/` layout | failed on missing `scaler.pkl` |
| Stage 07/08a read probability maps from a flat path after `06b` wrote per-size folders | would have silently used a stale 1x DNN map and missed the other models |
| Stage 07 already emits a DNN row; `08a` would append another | duplicate DNN row mixing 1x and 10x metrics in one table. Guarded |
| `train_dnn.py` set no random seed | single-run DNN numbers unusable; caused two wrong conclusions before the seeded study |
| `statsmodels` and `shap` missing from `floodcast_env` | stages 04 and 08d could not run |

Two conclusions were reported and later corrected, both from comparing a seeded 10-run
mean against a single unseeded baseline run. The single run happened to sit ~2.7 sd above
the true baseline mean, first hiding and then manufacturing an effect. Only the symmetric
10-seed-vs-10-seed comparison in `05e` is reliable.

---

## 4. Compute

Recorded per stage in `99_Logs/stage_timings.csv`, failures included.

| Stage | Duration |
|---|---|
| 02a raba_fractions | 34m 05s |
| 04 collinearity | 11m 02s |
| 05a train (1x) | 6m 26s |
| 05b DNN | 16m 44s |
| 05d DNN seed study | 1h 05m 42s |
| 05c multisample (5 sizes) | 2h 47m 08s |
| 06b prediction (25 maps, 16 GB) | 4h 05m 49s |
| 05e encoding study (4×10 seeds) | 4h 50m 21s |
| 07 threshold/classify/validate | 6m 19s |
| 08a DNN merge | 1m 42s |
| 08b spatial CV (10x) | 26m 10s |
| 08c multiscale (all sizes) | **9h 36m 54s** |
| **Total** | **≈ 24.5 h** |

---

## 5. Status

**Pipeline complete end to end.** Stages 02, 04, 05a–e, 06b, 07, 08a–d, 09a–c, 10a–d,
11a–d all finished. Total recorded compute **30 h 49 m across 34 stage runs**
(`99_Logs/stage_timings.csv`, failures included).

Outputs: 25 probability maps, 10 classified maps, 235 figures, a structured
`Manuscript/` folder, four `.docx` reports, and a manuscript in Markdown and
Copernicus LaTeX.

### Late corrections worth knowing

Two defects were found *after* the stages had "succeeded", both in `09b`:

- The frequency-ratio analysis ran on **6 pixels instead of 1.3 M** and wrote a table
  of zeros, because of the `read(1, out_shape=(1,h,w))[0]` bug at two further lines
  beyond the one already fixed. A degeneracy guard now aborts rather than writing
  meaningless values. Re-run gave physically coherent results (HAND FR 4.52 → 0.00).
- `LULC` was missing from `CATEGORICAL_CGFS`, so a nominal 7-class variable was binned
  into 10 quantiles as though continuous.

Both are the class of failure a static preflight cannot catch — the stage exits 0 and
produces a well-formed file. Check numbers, not just exit codes.

### Open items

See `CLAUDE.md` §"Current state" for the authoritative list. In brief: author
self-citations, the missing study-area figure, three unverified data provenances, the
first-author affiliation, `copernicus.cls`, and BibTeX conversion.

**Most promising untested improvement:** the continuous LULC fraction bands
(`raba_frac_202606.tif`) beat the categorical encoding on all four DNN metrics in the
seed study, but were never used in the main models.
