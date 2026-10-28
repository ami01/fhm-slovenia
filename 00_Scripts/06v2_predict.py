"""
06v2 — National prediction with the models of 05v2_train (review 2026-10-09).
=============================================================================
    FEATURE_SET=v2 python 00_Scripts/06v2_predict.py
Reads the 20 selected layers from 01_Input_Features[_v2] in row blocks, applies the same log1p transform as
05v2_train (flow accumulation, SPI, LS factor), and writes per model:
    06_Probability_Maps_v2/<set>/FSM_<model>_probability.tif   (float32, nodata -9999)
    07_Classified_Maps_v2/<set>/FSM_<model>_binary.tif          (uint8 0/1, nodata 255; threshold from the validation split)
Valid cells = inventory raster valid (inside Slovenia) and every feature finite.
"""
import json, os, time, warnings
warnings.filterwarnings('ignore')
import joblib, numpy as np, rasterio, torch, torch.nn as nn
from rasterio.windows import Window
import xgboost as xgb
from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = os.environ.get('FEATURE_SET', 'v2')
TAG = os.environ.get('MODEL_TAG', SET)                 # model folder, e.g. v2_hardneg
FDIR = os.path.join(ROOT, '01_Input_Features' if SET == 'v1' else '01_Input_Features_v2')
MDL = os.path.join(ROOT, '04_Models_v2', TAG)
PDIR, CDIR = os.path.join(ROOT, '06_Probability_Maps_v2', TAG), os.path.join(ROOT, '07_Classified_Maps_v2', TAG)
for d in (PDIR, CDIR): os.makedirs(d, exist_ok=True)
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
meta = json.load(open(os.path.join(MDL, 'thresholds.json'))); THR, LOG = meta['thresholds'], meta['log_features']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']


class FloodDNN(nn.Module):
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)


rf = joblib.load(os.path.join(MDL, 'model_rf.pkl')); rf.set_params(n_jobs=-1, verbose=0)
xm = xgb.XGBClassifier(); xm.load_model(os.path.join(MDL, 'model_xgb.json')); xm.set_params(device='cuda')
cm = CatBoostClassifier(); cm.load_model(os.path.join(MDL, 'model_cat.cbm'))
sc = joblib.load(os.path.join(MDL, 'scaler.pkl'))
dnn = FloodDNN(len(FEATURES)).to(DEVICE); dnn.load_state_dict(torch.load(os.path.join(MDL, 'model_dnn.pt'), map_location=DEVICE)); dnn.eval()

srcs = {f: rasterio.open(os.path.join(FDIR, f + '.tif')) for f in FEATURES}
ref = rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif'))
H, W = ref.shape
pprof = ref.profile | dict(dtype='float32', nodata=-9999.0, compress='deflate', tiled=True, blockxsize=512, blockysize=512, BIGTIFF='YES')
bprof = ref.profile | dict(dtype='uint8', nodata=255, compress='deflate', tiled=True, blockxsize=512, blockysize=512)
outs_p = {m: rasterio.open(os.path.join(PDIR, f'FSM_{m}_probability.tif'), 'w', **pprof) for m in MODELS}
outs_b = {m: rasterio.open(os.path.join(CDIR, f'FSM_{m}_binary.tif'), 'w', **bprof) for m in MODELS}
t0, STEP = time.time(), 512
for r0 in range(0, H, STEP):
    win = Window(0, r0, W, min(STEP, H - r0))
    ok = ref.read(1, window=win) != ref.nodata
    cols = []
    for f in FEATURES:
        a = srcs[f].read(1, window=win).astype('float64'); nd = srcs[f].nodata
        if nd is not None and not np.isnan(nd): a[a == nd] = np.nan
        a = np.log1p(np.maximum(a, 0)) if f in LOG else a
        cols.append(a); ok &= np.isfinite(a)
    shape = ok.shape
    P = {m: np.full(shape, -9999.0, np.float32) for m in MODELS}
    if ok.any():
        X = np.stack([c[ok] for c in cols], 1).astype(np.float32)
        p = {'RF': rf.predict_proba(X)[:, 1], 'XGBoost': xm.predict_proba(X)[:, 1], 'CatBoost': cm.predict_proba(X)[:, 1]}
        Xs = torch.tensor(sc.transform(X).astype(np.float32))
        with torch.no_grad():
            p['DNN'] = torch.sigmoid(torch.cat([dnn(Xs[i:i + 262144].to(DEVICE)).cpu() for i in range(0, len(Xs), 262144)])).numpy().ravel()
        p['Ensemble_mean'] = np.mean([p[k] for k in ('RF', 'XGBoost', 'CatBoost', 'DNN')], 0)
        for m in MODELS:
            P[m][ok] = p[m]
    for m in MODELS:
        outs_p[m].write(P[m], 1, window=win)
        outs_b[m].write(np.where(P[m] == -9999.0, 255, (P[m] >= THR[m]).astype(np.uint8)).astype(np.uint8), 1, window=win)
    if (r0 // STEP) % 5 == 0:
        print(f'rows {r0 + shape[0]:,}/{H:,}  {time.time() - t0:.0f} s', flush=True)
for o in list(outs_p.values()) + list(outs_b.values()):
    o.close()
print(f'done in {(time.time() - t0) / 60:.1f} min -> {PDIR}, {CDIR}')
