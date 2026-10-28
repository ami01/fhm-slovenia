"""
Flood Susceptibility Probability Maps — Slovenia
=================================================
Generates strip-based probability maps using all four best models
(10x sample size: 500k per class) and a 4-model ensemble mean.

Models loaded from: ML_Results/models_10x/
  model_rf.pkl | model_xgb.json | model_cat.cbm | model_dnn.pth | scaler.pkl

Outputs → ML_Results/probability_maps/
  FSM_RF_probability.tif
  FSM_XGBoost_probability.tif
  FSM_CatBoost_probability.tif
  FSM_DNN_probability.tif
  FSM_Ensemble_mean_probability.tif   ← mean of all four
"""

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features'
FEATURES_FILE = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
# 05c writes per-size models to 04_Models/<tag>/. Stage 07 validates against the
# 10x test split, so predict from the 10x models. Override with MODEL_SIZE=<tag>.
import os as _os
MODELS_DIR    = _os.path.join(
    '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/04_Models',
    _os.environ.get('MODEL_SIZE', '10x'))
OUTPUT_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/06_Probability_Maps'
STRIP_ROWS    = 200
NODATA_OUT    = -9999.0
DNN_DROPOUT   = 0.3

import os, warnings, gc, time
warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models from:", MODELS_DIR)

scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
print("  Scaler loaded")

rf_model = joblib.load(os.path.join(MODELS_DIR, 'model_rf.pkl'))
print("  RF loaded")

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(MODELS_DIR, 'model_xgb.json'))
print("  XGBoost loaded")

cat_model = CatBoostClassifier()
cat_model.load_model(os.path.join(MODELS_DIR, 'model_cat.cbm'))
print("  CatBoost loaded")

dnn_model = FloodDNN(len(FEATURES)).to(DEVICE)
dnn_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'model_dnn.pth'),
                                      map_location=DEVICE))
dnn_model.eval()
print("  DNN loaded\n")

# ── Open feature rasters ──────────────────────────────────────────────────────
feat_ds  = {stem: rasterio.open(path) for stem, path in feat_paths.items()}
feat_nds = {stem: ds.nodata for stem, ds in feat_ds.items()}

# ── Open output files ─────────────────────────────────────────────────────────
MODEL_NAMES = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
out_paths = {name: os.path.join(OUTPUT_DIR, f'FSM_{name}_probability.tif')
             for name in MODEL_NAMES}
out_ds = {name: rasterio.open(path, 'w', **out_meta) for name, path in out_paths.items()}

# ── Strip-based prediction ────────────────────────────────────────────────────
n_strips = (ref_h + STRIP_ROWS - 1) // STRIP_ROWS
t0 = time.time()

print(f"Predicting in {n_strips} strips × {STRIP_ROWS} rows ...")
print(f"{'Strip':<7} {'Rows':<14} {'Valid px':>12}  {'RF':>6} {'XGB':>6} {'CAT':>6} {'DNN':>6}  {'Time':>7}")
print('-' * 72)

for s in range(n_strips):
    row_start = s * STRIP_ROWS
    row_end   = min(row_start + STRIP_ROWS, ref_h)
    n_rows    = row_end - row_start
    win       = Window(0, row_start, ref_w, n_rows)
    n_pix     = n_rows * ref_w

    # Read features
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

        # RF
        p_rf = rf_model.predict_proba(Xv)[:, 1].astype(np.float32)

        # XGBoost
        p_xgb = xgb_model.get_booster().predict(xgb.DMatrix(Xv)).astype(np.float32)

        # CatBoost
        p_cat = cat_model.predict_proba(Xv)[:, 1].astype(np.float32)

        # DNN
        with torch.no_grad():
            chunks = [dnn_model(torch.tensor(Xv_sc[i:i+8192]).to(DEVICE)).cpu()
                      for i in range(0, len(Xv_sc), 8192)]
        p_dnn = torch.sigmoid(torch.cat(chunks)).numpy().flatten().astype(np.float32)

        # Ensemble mean of all four
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

# ── Close all handles ─────────────────────────────────────────────────────────
for ds in feat_ds.values():
    ds.close()
for ds in out_ds.values():
    ds.close()

total = time.time() - t0
print(f"\nDone in {total/60:.1f} min")

# ── Output stats ──────────────────────────────────────────────────────────────
print("\nOutput map statistics:")
print(f"  {'Model':<20} {'Size (MB)':>10}  {'Valid px':>14}  {'Min':>6}  {'Max':>6}  {'Mean':>6}")
print(f"  {'-'*20} {'-'*10}  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*6}")
for name, path in out_paths.items():
    size_mb = os.path.getsize(path) / 1e6
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    valid_px = arr[arr != NODATA_OUT]
    print(f"  {name:<20} {size_mb:>9.1f}M  {len(valid_px):>14,}  "
          f"{valid_px.min():>6.3f}  {valid_px.max():>6.3f}  {valid_px.mean():>6.3f}")

print(f"\nSaved to: {OUTPUT_DIR}")
