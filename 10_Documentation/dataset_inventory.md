# Dataset Inventory — FSM Slovenia

All grid metadata below is **measured** from the files with `rasterio` / `pyogrio`, not
transcribed from documentation. Source attributions are from the generating scripts in
`00_Scripts/`; where a script did not name an origin it is marked *unverified*.

Common analysis grid: **12996 × 19909 @ 12.5 m, EPSG:3794** (D96/TM, Slovene National
Grid) — 258.8 M cells, 129.7 M of them inside the national boundary.

---

## 1. Target variable

| Raster | Type | Res | rows × cols | NoData | Size | Origin |
|---|---|---|---|---|---|---|
| `Flooded_NoFlood.tif` | int16, binary 0/1 | 12.5 m | 12996 × 19909 | −9999 | 3 MB | Rasterised from DRSV **IKPN Q500** flood-hazard polygons (500-year return period), dissolved to two classes |

Class balance: 7,326,493 flood cells / 122,423,292 no-flood cells.

---

## 2. Conditioning factors — in the final model (20)

`D` = derived in this project, `P` = primary/acquired dataset.

| # | Raster | Type | Res | rows × cols | NoData | Size | D/P | Origin / derivation |
|---|---|---|---|---|---|---|---|---|
| 1 | `elevation.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 574 MB | D | DMV0125 12.5 m DEM (GURS), merged from ASCII XYZ tiles, sink-filled |
| 2 | `aspect.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 626 MB | D | DEM derivative (`01a_DEM_CgFs.py`, GPU) |
| 3 | `plan_curvature.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 691 MB | D | DEM derivative |
| 4 | `profile_curvature.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 691 MB | D | DEM derivative |
| 5 | `tri.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 634 MB | D | Terrain Ruggedness Index, DEM derivative |
| 6 | `twi.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 846 MB | D | Topographic Wetness Index, DEM derivative |
| 7 | `spi.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 668 MB | D | Stream Power Index, DEM derivative |
| 8 | `ls_factor.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 913 MB | D | Slope-length factor, DEM derivative |
| 9 | `hand.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 671 MB | D | Height Above Nearest Drainage, DEM + stream network |
| 10 | `valley_depth.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 696 MB | D | DEM derivative |
| 11 | `flow_accumulation.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 901 MB | D | DEM derivative (D8/D∞) |
| 12 | `drainage_density.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 178 MB | D | Stream length per unit area, from extracted network |
| 13 | `Distance_to_River.tif` | float32 | 12.5 m | 12996 × 19909 | NaN | 1056 MB | D | Euclidean distance to DTM_SLO hydrography lines (537,172 features) |
| 14 | `River_Density.tif` | float32 | 12.5 m | 12996 × 19909 | −9999 | 745 MB | D | Gaussian KDE density of river network (km/km²) |
| 15 | `Distance_to_Road.tif` | float32 | 12.5 m | 12996 × 19909 | NaN | 988 MB | D | Euclidean distance to DTM_SLO roads (162,435 lines) |
| 16 | `slovenia_annual_max_daily_p90_3794_12m5.tif` | float32 | 12.5 m | 12996 × 19909 | NaN | 477 MB | D | 90th-percentile annual max daily rainfall, interpolated from **E-OBS** (ECMWF `rr_ens_mean_0.1deg`) |
| 17 | `Hgk_grouped.tif` | float32, 10 classes | 12.5 m | 12996 × 19909 | **0** | 12 MB | P→D | Hydrogeological units, grouped from LAWA sub-codes (2,418 polygons) |
| 18 | `Pedoloska_grouped.tif` | int16, 13 classes | 12.5 m | 12996 × 19909 | −9999 | 6 MB | P→D | Pedological (soil) map, grouped to 13 classes (7,976 polygons) |
| 19 | `lito_poly_250_d96_grouped.tif` | int16, 12 classes | 12.5 m | 12996 × 19909 | −9999 | 4 MB | P→D | Lithology 1:250,000, grouped to 12 classes (4,551 polygons) |
| 20 | **`LULC.tif`** | float32, 7 classes | 12.5 m | 12996 × 19909 | **0** | 20 MB | D | **NEW** — modal class of the RABA areal-fraction raster; see §4 |

---

## 3. Conditioning factors — computed but VIF-eliminated (4)

Removed by `04_iterative_collinearity.py` at VIF > 10. Retained on disk.

| Raster | Type | Res | rows × cols | Size | VIF at removal | Origin |
|---|---|---|---|---|---|---|
| `slovenia_annual_max_daily_mean_3794_12m5.tif` | float32 | 12.5 m | 12996 × 19909 | 474 MB | 100.11 | E-OBS interpolated |
| `tpi.tif` | float32 | 12.5 m | 12996 × 19909 | 690 MB | 73.02 | DEM derivative |
| `slope.tif` | float32 | 12.5 m | 12996 × 19909 | 633 MB | 44.32 | DEM derivative |
| `slovenia_mean_annual_rainfall_3794_12m5.tif` | float32 | 12.5 m | 12996 × 19909 | 470 MB | 11.67 | E-OBS interpolated |

---

## 4. LULC intermediate

| File | Type | Res | rows × cols | Bands | Size | Origin |
|---|---|---|---|---|---|---|
| `raba_frac_202606.tif` | float32 | 12.5 m | 12996 × 19909 | **9** | 420 MB | Per-class areal fractions from RABA polygons, burned at 1.25 m (10× oversample) then block-aggregated |

Bands 1–7 `frac_{arable, permanent_crop, grassland, woody, built_up, water_wetland, bare_open}`,
band 8 `coverage`, band 9 `modal_gid`. Band 9 becomes `LULC.tif`.
Full method and statistics: `RABA_LULC_Conversion.docx`.

---

## 5. Elevation source rasters (different grid — do not mix)

| File | Type | Res | rows × cols | Size | Note |
|---|---|---|---|---|---|
| `DMV0125_EPSG3794_merged.tif` | float32 | **12.46 m** | 13484 × 20226 | 510 MB | Merged DMV0125 tiles, pre-clip |
| `DEM_Filled_Slovenia.tif` | float32 | **12.46 m** | 13038 × 19883 | 1037 MB | Sink-filled DEM |

⚠ Both are on a **12.46 m** grid with a different origin from the 12.5 m analysis stack.
They are the *source* for `elevation.tif`, not interchangeable with it. Using either as a
rasterisation reference produces output that will not align with the feature stack.

---

## 6. Vector inputs

| Shapefile | Geometry | Features | CRS | Size | Role |
|---|---|---|---|---|---|
| `RABA_2026_06_30/RABA_20260630.shp` | Polygon | 1,698,891 | 3794 | 1246 MB | Land use, release 2026-06-30 (MKGP) |
| `DTM_SLO_..._TEKOCEVODE_L_line.shp` | LineString | 537,172 | 3794 | 424 MB | Watercourses (full) |
| `In Raster/Slovenia_Rivers_N.shp` | LineString | 10,545 | 3794 | 15 MB | Named/major rivers |
| `In Raster/Slovenia_Roads.shp` | LineString | 162,435 | 3794 | 199 MB | Road network |
| `DTM_SLO_..._POKRITOSTTAL_P_poligon.shp` | Polygon | 894,494 | 3794 | 2334 MB | Land cover (alternative to RABA; unused) |
| `DTM_SLO_..._MOKROTNEPOVRSINE_P_poligon.shp` | Polygon | 1,772 | 3794 | 6 MB | Wetlands |
| `Pedoloska_grouped.shp` | Polygon | 7,976 | 3794 | 34 MB | Soil map |
| `lito_poly_250_d96_grouped.shp` | Polygon | 4,551 | **ESRI:102109** | 7 MB | Lithology — note non-EPSG CRS, reprojected on rasterisation |
| `Hgk_grouped.shp` | Polygon | 2,418 | **4326** | 5 MB | Hydrogeology — geographic CRS, reprojected |
| `Flooded_NoFlood.shp` | MultiPolygon | 2 | 3794 | 44 MB | Dissolved flood hazard (target) |

---

## 7. Model outputs

| Artefact | Count | Type | Res | Size |
|---|---|---|---|---|
| Probability maps | 25 (5 models × 5 sizes) | float32 | 12.5 m | 16 GB total |
| Classified maps | binary + 5-class per model per size | int16 | 12.5 m | ~22 MB each |
| Trained models | 4 per size × 5 sizes + 1x set | pkl/json/cbm/pth | — | 109 MB (1x) – 563 MB (10x) |
| Sample CSVs | 5 sizes × train/test | CSV + SHP | point | 77 MB |

Models: `model_rf.pkl`, `model_xgb.json`, `model_cat.cbm`, `model_dnn.pth`, `scaler.pkl`.
Sample sizes: 1x = 50k/class … 10x = 500k/class.

---

## Notes on data hygiene

**Two NoData conventions coexist.** Categorical rasters (`Hgk_grouped`, `LULC`) use
**0**; continuous rasters use **−9999**; three use **NaN** (`Distance_to_River`,
`Distance_to_Road`, the rainfall set). The training code reads each raster's declared
nodata and maps it to NaN, so this is handled — but a raster added without a correct
nodata tag would silently inject sentinel values as real data.

**Mixed source CRS.** `Hgk_grouped.shp` is EPSG:4326 and `lito_poly_250_d96_grouped.shp`
is ESRI:102109 (ETRS89 Slovenia TM), not EPSG:3794. Both are reprojected during
rasterisation. ESRI:102109 and EPSG:3794 share the same projection parameters but
different datum authorities, so the transformation is near-identity — worth stating in a
methods section rather than leaving implicit.

**`slope` was VIF-eliminated but `tri` retained.** Both describe local relief; the screen
kept the ruggedness index over slope itself. `tri` then ranks 2nd by SHAP importance, so
this is not a case of an important variable being dropped.
