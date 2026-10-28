"""
12a_zenodo_package.py — assemble the Zenodo data deposit for the published (v2) results.

Builds Zenodo_data/ at the repository root:
    data/<folder>/...      the files under plain folder names, file names unchanged. GeoTIFFs are re-encoded
                           losslessly (DEFLATE level 9, floating-point predictor for float rasters; every cell is
                           checked against the original); other files are hard-linked.
    upload/<folder>.zip    one archive per folder, ready to upload (stored: the GeoTIFFs are compressed inside,
                           so zip compression would add nothing)
    upload/README.md       description of every file, thresholds, CRS, nodata, citation
    upload/checksums_md5.txt

Rasters are copied losslessly. Only the final (v2) products behind the manuscript are included; the superseded
first-pipeline outputs (06_Probability_Maps/, 07_Classified_Maps/) are not.

Usage:  python3 00_Scripts/12a_zenodo_package.py [--skip FOLDER ...]
"""
import argparse
import hashlib
import json
import os
import shutil
import zipfile

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy

ROOT = '/home/arora/manuscripts/FSM_Pipeline_LULC'
OUT = os.path.join(ROOT, 'Zenodo_data')
DATA, UPLOAD = os.path.join(OUT, 'data'), os.path.join(OUT, 'upload')

MODELS = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]

# folder name in the deposit -> list of source files (relative to ROOT)
SOURCES = {
    'probability_maps': [f'06_Probability_Maps_v2/v2/FSM_{m}_probability.tif' for m in MODELS],
    'binary_maps_screening': [f'07_Classified_Maps_v2/v2/FSM_{m}_binary.tif' for m in MODELS],
    'binary_maps_prevalence': [f'07_Classified_Maps_v2/v2_prev/FSM_{m}_binary.tif' for m in MODELS],
    # the layer behind the 8.8 / 87.6 / 3.6 % in the text (09k run on the v2 map set)
    'model_agreement': ['09_Analysis/overflag_v2_v2/agreement4_binary.tif', '09_Analysis/overflag_v2_v2/agreement4.csv'],
    'area_of_applicability': ['09_Analysis/aoa_v2/AOA_100m.tif', '09_Analysis/aoa_v2/DI_100m.tif',
                              '09_Analysis/aoa_v2/summary.json'],
    'warning_units': ['09_Analysis/unit_share_v2/warning_units_flood_prone.csv'],
    'conditioning_factors': [f'01_Input_Features_v2/{f}.tif' for f in FEATURES] + ['01_Input_Features_v2/Flooded_NoFlood.tif'],
    'models': [f'04_Models_v2/v2/{f}' for f in ('model_rf.pkl', 'model_xgb.json', 'model_cat.cbm', 'model_dnn.pt',
                                                'scaler.pkl', 'thresholds.json')],
    'samples': ['03_Samples_v2/v2/samples_train.parquet', '03_Samples_v2/v2/samples_test.parquet'],
}

SCREEN = json.load(open(os.path.join(ROOT, '04_Models_v2', 'v2', 'thresholds.json')))['thresholds']
PREV = json.load(open(os.path.join(ROOT, '09_Analysis', 'threshold_prevalence', 'v2_thresholds.json')))


def md5(path, buf=1 << 24):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''):
            h.update(b)
    return h.hexdigest()


def link(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:          # different filesystem: fall back to a copy
        shutil.copy2(src, dst)


def recompress(src, dst):
    """Lossless re-encode: DEFLATE 9, floating-point predictor for float rasters (none for the small integer
    class maps, where a predictor makes them larger); verified cell by cell."""
    if os.path.exists(dst):
        os.remove(dst)
    with rasterio.open(src) as s:
        pred = 3 if np.dtype(s.dtypes[0]).kind == 'f' else 1
    rio_copy(src, dst, driver='GTiff', COMPRESS='DEFLATE', ZLEVEL=9, PREDICTOR=pred, TILED='YES',
             BIGTIFF='IF_SAFER', NUM_THREADS='ALL_CPUS')
    with rasterio.open(src) as a, rasterio.open(dst) as b:
        same_nodata = a.nodata == b.nodata or (np.isnan(a.nodata) and np.isnan(b.nodata))
        assert a.crs == b.crs and a.transform == b.transform and same_nodata, f'{dst}: georeferencing changed'
        for _, w in a.block_windows(1):
            if not np.array_equal(a.read(1, window=w), b.read(1, window=w), equal_nan=True):
                raise ValueError(f'{dst}: values differ from {src} in {w}')


def mb(n):
    return f'{n / 1e6:,.0f} MB' if n >= 1e6 else f'{n / 1e3:,.0f} kB'


def readme(sizes):
    thr = '\n'.join(f'| {m} | {SCREEN[m]} | {PREV[m]} |' for m in MODELS)
    rows = '\n'.join(f'| `{k}.zip` | {mb(v)} |' for k, v in sizes.items())
    return f"""# National machine-learning flood susceptibility maps for Slovenia at 12.5 m: data

Data deposit for: Arora, A., Petan, S. *Beyond the training inventory: transferability and independent
validation of national machine-learning flood susceptibility maps for Slovenia.* Submitted to the Journal of Hydrology.

DOI of this deposit: https://doi.org/10.5281/zenodo.23726819

Code: https://github.com/ami01/fhm-slovenia (archived separately on Zenodo; see the related identifiers).

## Common properties of the rasters

- Coordinate system: Slovenia 1996 / Slovene National Grid (D96/TM), **EPSG:3794**
- Grid: 12.5 m, 19 909 x 12 996 cells (area of applicability: 100 m), identical origin for all 12.5 m layers
- Format: GeoTIFF, tiled, DEFLATE compressed with predictor (lossless). Opens in QGIS, ArcGIS, GDAL, rasterio.
- Cells outside Slovenia are nodata.

## Files

| Archive | Size |
|---|---|
{rows}

**probability_maps/**: flood susceptibility probability (float32, 0-1, nodata -9999) from Random Forest, XGBoost,
CatBoost, the deep neural network (DNN) and the ensemble mean of the four.

**binary_maps_screening/**: flood-prone (1) / not flood-prone (0), uint8, nodata 255, at each model's screening
threshold (CSI-optimal on the balanced validation set). Recommended for national screening.

**binary_maps_prevalence/**: as above, at the threshold that matches the national flood base rate (5.6 %).
Recommended for estimating the mapped flood-prone area.

| Model | Screening threshold | Prevalence threshold |
|---|---|---|
{thr}

**model_agreement/**: `agreement4_binary.tif`, number of the four learners (0-4) whose screening map flags the
cell (uint8, nodata 255); `agreement4.csv`, national area and share per count.

**area_of_applicability/**: `AOA_100m.tif`, 1 inside / 0 outside the models' area of applicability (uint8, nodata
255, 100 m); `DI_100m.tif`, dissimilarity index (float32); `summary.json`, threshold and shares.

**warning_units/**: flood-prone share of each ARSO hydrological warning unit (screening ensemble).

**conditioning_factors/**: the 20 predictors used by the models and the flood-inventory target
(`Flooded_NoFlood.tif`: 1 flooded, 0 not flooded). Flow accumulation, SPI and LS factor are stored untransformed;
the models use log(1 + x). Categorical class codes are documented in the code repository (`10_Documentation/`).
These layers are derived from data of the Surveying and Mapping Authority of the Republic of Slovenia (GURS), the
Geological Survey of Slovenia (GeoZS), the Ministry of Agriculture, Forestry and Food (MKGP), the Slovenian Water
Agency (DRSV) and E-OBS (Copernicus Climate Change Service); the providers' terms apply to the source data.

**models/**: the trained national models (scikit-learn RF pickle, XGBoost JSON, CatBoost CBM, PyTorch state dict,
feature scaler for the DNN, thresholds). Load them with the code repository's environment (`environment.yml`).

**samples/**: training and test samples (parquet; label, D96/TM coordinates and the 20 predictors).

`checksums_md5.txt` lists the MD5 checksum of every file inside the archives.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0). Please cite the article and this deposit.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip', nargs='*', default=[], help='folders to leave out, e.g. conditioning_factors')
    args = ap.parse_args()
    os.makedirs(UPLOAD, exist_ok=True)

    sizes, sums = {}, []
    for folder, files in SOURCES.items():
        if folder in args.skip:
            print(f'skip {folder}')
            continue
        d = os.path.join(DATA, folder)
        os.makedirs(d, exist_ok=True)
        for rel in files:
            src = os.path.join(ROOT, rel)
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            dst = os.path.join(d, os.path.basename(rel))
            if rel.endswith('.tif'):
                recompress(src, dst)
            else:
                link(src, dst)
        z = os.path.join(UPLOAD, folder + '.zip')
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name in sorted(os.listdir(d)):
                zf.write(os.path.join(d, name), f'{folder}/{name}')
                sums.append(f'{md5(os.path.join(d, name))}  {folder}/{name}')
        sizes[folder] = os.path.getsize(z)
        print(f'{folder:24s} {len(files):3d} files  {mb(sizes[folder])}')

    open(os.path.join(UPLOAD, 'checksums_md5.txt'), 'w').write('\n'.join(sums) + '\n')
    open(os.path.join(UPLOAD, 'README.md'), 'w').write(readme(sizes))
    print(f'total upload {mb(sum(sizes.values()))} -> {UPLOAD}')


if __name__ == '__main__':
    main()
