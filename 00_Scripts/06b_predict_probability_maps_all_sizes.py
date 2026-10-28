"""
Probability Maps — all sample sizes (1x through 10x)
=========================================
Generates strip-based probability maps for all four sample sizes
(1x=50k, 2x=100k, 3x=150k, 4x=200k per class) using the trained
RF, XGBoost, CatBoost, and DNN models for each size, plus a
4-model ensemble mean.

Models loaded from: ML_Results/models_{size}/
Outputs saved to:   ML_Results/probability_maps/{size}/
  FSM_RF_probability.tif
  FSM_XGBoost_probability.tif
  FSM_CatBoost_probability.tif
  FSM_DNN_probability.tif
  FSM_Ensemble_mean_probability.tif

Skips a size if all 5 output maps already exist.
"""

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features'
FEATURES_FILE = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
BASE_DIR      = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
STRIP_ROWS    = 200
NODATA_OUT    = -9999.0
DNN_DROPOUT   = 0.3

SIZES = ['1x', '2x', '3x', '4x', '10x']

import os, warnings, gc, time
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.windows import Window
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {DEVICE}\n")

# ── Feature list ──────────────────────────────────────────────────────────────
with open(FEATURES_FILE) as f:
    FEATURES = [l.strip() for l in f if l.strip()]
print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}\n")

# ── Reference grid ────────────────────────────────────────────────────────────
feat_paths = {}
for fname in os.listdir(ALIGNED_DIR):
    stem = os.path.splitext(fname)[0]
    if stem in FEATURES:
        feat_paths[stem] = os.path.join(ALIGNED_DIR, fname)

missing = [f for f in FEATURES if f not in feat_paths]
if missing:
    raise FileNotFoundError(f"Missing aligned rasters: {missing}")

ref_path = feat_paths[FEATURES[0]]
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_h, ref_w = ref.height, ref.width
    ref_crs = ref.crs

print(f"Grid: {ref_w} × {ref_h} px | EPSG:{ref_crs.to_epsg()}")
print(f"Total pixels: {ref_w * ref_h:,}\n")

out_meta = ref_meta.copy()
out_meta.update(dtype='float32', nodata=NODATA_OUT, count=1,
                compress='lzw', tiled=True, blockxsize=512, blockysize=512)

# ── DNN architecture (must match training) ────────────────────────────────────
class FloodDNN(nn.Module):
    def __init__(self, input_dim, dropout=DNN_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(32, 16),         nn.BatchNorm1d(16),  nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x)

# ── Open feature rasters once (shared across sizes) ───────────────────────────
feat_ds  = {stem: rasterio.open(path) for stem, path in feat_paths.items()}
feat_nds = {stem: ds.nodata for stem, ds in feat_ds.items()}

MODEL_NAMES = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']

# ── Main loop over sample sizes ───────────────────────────────────────────────
total_t0 = time.time()

for size in SIZES:
    print()
    print("=" * 65)
    print(f"SAMPLE SIZE: {size}")
    print("=" * 65)

    mdl_dir = os.path.join(BASE_DIR, '04_Models', size)
    out_dir = os.path.join(BASE_DIR, '06_Probability_Maps', size)
    os.makedirs(out_dir, exist_ok=True)

    out_paths = {name: os.path.join(out_dir, f'FSM_{name}_probability.tif')
                 for name in MODEL_NAMES}

    # Skip if all maps already exist
    if all(os.path.exists(p) for p in out_paths.values()):
        sizes_mb = {n: os.path.getsize(p)/1e6 for n, p in out_paths.items()}
        print(f"  All maps exist — skipping. "
              f"({', '.join(f'{n}:{s:.0f}MB' for n,s in sizes_mb.items())})")
        continue

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"Loading models from {mdl_dir} ...")
    scaler = joblib.load(os.path.join(mdl_dir, 'scaler.pkl'))

    rf_model = joblib.load(os.path.join(mdl_dir, 'model_rf.pkl'))
    print("  RF loaded")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(mdl_dir, 'model_xgb.json'))
    print("  XGBoost loaded")

    cat_model = CatBoostClassifier()
    cat_model.load_model(os.path.join(mdl_dir, 'model_cat.cbm'))
    print("  CatBoost loaded")

    dnn_model = FloodDNN(len(FEATURES)).to(DEVICE)
    dnn_model.load_state_dict(torch.load(os.path.join(mdl_dir, 'model_dnn.pth'),
                                          map_location=DEVICE))
    dnn_model.eval()
    print("  DNN loaded\n")

    # ── Open output files ─────────────────────────────────────────────────────
    out_ds = {name: rasterio.open(path, 'w', **out_meta) for name, path in out_paths.items()}

    # ── Strip-based prediction ────────────────────────────────────────────────
    n_strips = (ref_h + STRIP_ROWS - 1) // STRIP_ROWS
    t0 = time.time()

    print(f"Predicting {n_strips} strips × {STRIP_ROWS} rows ...")
    print(f"{'Strip':<7} {'Rows':<14} {'Valid px':>12}  {'RF':>6} {'XGB':>6} {'CAT':>6} {'DNN':>6}  {'Time':>7}")
    print('-' * 72)

    for s in range(n_strips):
        row_start = s * STRIP_ROWS
        row_end   = min(row_start + STRIP_ROWS, ref_h)
        n_rows    = row_end - row_start
        win       = Window(0, row_start, ref_w, n_rows)
        n_pix     = n_rows * ref_w

        X_strip = np.full((n_pix, len(FEATURES)), np.nan, dtype=np.float32)
        for fi, stem in enumerate(FEATURES):
            band = feat_ds[stem].read(1, window=win).flatten().astype(np.float32)
            nd   = feat_nds[stem]
            if nd is not None:
                mask = np.isnan(band) if (isinstance(nd, float) and np.isnan(nd)) else (band == nd)
                band[mask] = np.nan
            X_strip[:, fi] = band

        valid   = ~np.any(np.isnan(X_strip), axis=1)
        n_valid = int(valid.sum())

        out_strips = {name: np.full(n_pix, NODATA_OUT, dtype=np.float32) for name in MODEL_NAMES}

        if n_valid > 0:
            Xv    = X_strip[valid]
            Xv_sc = scaler.transform(Xv).astype(np.float32)

            p_rf  = rf_model.predict_proba(Xv)[:, 1].astype(np.float32)
            p_xgb = xgb_model.get_booster().predict(xgb.DMatrix(Xv)).astype(np.float32)
            p_cat = cat_model.predict_proba(Xv)[:, 1].astype(np.float32)

            with torch.no_grad():
                chunks = [dnn_model(torch.tensor(Xv_sc[i:i+8192]).to(DEVICE)).cpu()
                          for i in range(0, len(Xv_sc), 8192)]
            p_dnn = torch.sigmoid(torch.cat(chunks)).numpy().flatten().astype(np.float32)

            p_ens = ((p_rf + p_xgb + p_cat + p_dnn) / 4.0).astype(np.float32)

            out_strips['RF'][valid]            = p_rf
            out_strips['XGBoost'][valid]       = p_xgb
            out_strips['CatBoost'][valid]      = p_cat
            out_strips['DNN'][valid]           = p_dnn
            out_strips['Ensemble_mean'][valid] = p_ens

        for name in MODEL_NAMES:
            out_ds[name].write(out_strips[name].reshape(n_rows, ref_w), 1, window=win)

        elapsed = time.time() - t0
        def _avg(arr): return arr[valid].mean() if n_valid > 0 else float('nan')
        print(f"{s+1:<7} {row_start}–{row_end:<7} {n_valid:>12,}  "
              f"{_avg(out_strips['RF']):>6.3f} {_avg(out_strips['XGBoost']):>6.3f} "
              f"{_avg(out_strips['CatBoost']):>6.3f} {_avg(out_strips['DNN']):>6.3f}  "
              f"{elapsed:>6.1f}s", flush=True)

        del X_strip, out_strips, Xv, Xv_sc
        gc.collect()

    for ds in out_ds.values():
        ds.close()

    elapsed_size = time.time() - t0
    print(f"\n[{size}] Done in {elapsed_size/60:.1f} min")

    # Quick stats
    print(f"\nMap statistics [{size}]:")
    print(f"  {'Model':<20} {'Size (MB)':>10}  {'Valid px':>14}  {'Min':>6}  {'Max':>6}  {'Mean':>6}")
    for name, path in out_paths.items():
        size_mb = os.path.getsize(path) / 1e6
        with rasterio.open(path) as ds:
            arr = ds.read(1)
        vp = arr[arr != NODATA_OUT]
        print(f"  {name:<20} {size_mb:>9.1f}M  {len(vp):>14,}  "
              f"{vp.min():>6.3f}  {vp.max():>6.3f}  {vp.mean():>6.3f}")

    # Free model memory before next size
    del rf_model, xgb_model, cat_model, dnn_model, scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

# ── Close feature rasters ─────────────────────────────────────────────────────
for ds in feat_ds.values():
    ds.close()

print()
print("=" * 65)
print(f"ALL DONE  ({(time.time()-total_t0)/60:.1f} min total)")
print("=" * 65)
for size in SIZES:
    print(f"  probability_maps/{size}/  — 5 maps")
