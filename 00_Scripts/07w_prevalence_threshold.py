"""
07w — Prevalence-correct threshold chosen WITHOUT the test data (review 2026-10-09).
====================================================================================
    MODEL_TAG=v2 python 00_Scripts/07w_prevalence_threshold.py
The balanced 1:1 threshold over-flags at the national 5.6 % base rate (FBI 1.7-2.1). Here each model's threshold
maximises CSI on the training VALIDATION split (same 10 % split as 05v2_train), with non-flood rows reweighted so the
split has the national prevalence (π = 0.0563). The test set and the national maps are not used for the choice.
The binary maps are then rebuilt from the v2 probability maps and verified at the base rate against the inventory.
Outputs → 07_Classified_Maps_v2/<tag>_prev/FSM_<model>_binary.tif; 09_Analysis/threshold_prevalence/<tag>.csv
"""
import json, os, joblib, numpy as np, pandas as pd, rasterio, torch, torch.nn as nn, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from rasterio.windows import Window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = os.environ.get('MODEL_TAG', 'v2'); PI = 0.0563; SEED = 42
MDL = os.path.join(ROOT, '04_Models_v2', TAG)
PM = os.path.join(ROOT, '06_Probability_Maps_v2', TAG)
OUTB = os.path.join(ROOT, '07_Classified_Maps_v2', f'{TAG}_prev'); os.makedirs(OUTB, exist_ok=True)
OUT = os.path.join(ROOT, '09_Analysis', 'threshold_prevalence'); os.makedirs(OUT, exist_ok=True)
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
MODELS = ['XGBoost', 'Ensemble_mean', 'CatBoost', 'RF', 'DNN']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FloodDNN(nn.Module):
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)


tr = pd.read_parquet(os.path.join(ROOT, '03_Samples_v2', TAG, 'samples_train.parquet'))
X, y = tr[FEATURES].values.astype(np.float32), tr.label.values.astype(np.float32)
_, X_va, _, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)   # identical split to 05v2
rf = joblib.load(os.path.join(MDL, 'model_rf.pkl'))
xm = xgb.XGBClassifier(); xm.load_model(os.path.join(MDL, 'model_xgb.json'))
cm = CatBoostClassifier(); cm.load_model(os.path.join(MDL, 'model_cat.cbm'))
sc = joblib.load(os.path.join(MDL, 'scaler.pkl'))
dnn = FloodDNN(len(FEATURES)).to(DEVICE); dnn.load_state_dict(torch.load(os.path.join(MDL, 'model_dnn.pt'), map_location=DEVICE)); dnn.eval()
with torch.no_grad():
    pd_ = torch.sigmoid(dnn(torch.tensor(sc.transform(X_va).astype(np.float32)).to(DEVICE))).cpu().numpy().ravel()
P = {'RF': rf.predict_proba(X_va)[:, 1], 'XGBoost': xm.predict_proba(X_va)[:, 1], 'CatBoost': cm.predict_proba(X_va)[:, 1], 'DNN': pd_}
P['Ensemble_mean'] = np.mean([P[k] for k in ('RF', 'XGBoost', 'CatBoost', 'DNN')], 0)
w = np.where(y_va == 1, 1.0, (1 - PI) / PI * (y_va == 1).sum() / (y_va == 0).sum())   # reweight to prevalence π
thr = {}
for m, p in P.items():
    best = (0.5, -1)
    for t in np.round(np.arange(0.30, 0.995, 0.005), 3):
        q = p >= t; tp = (w * (q & (y_va == 1))).sum(); fp = (w * (q & (y_va == 0))).sum(); fn = (w * (~q & (y_va == 1))).sum()
        c = tp / (tp + fp + fn)
        if c > best[1]: best = (float(t), c)
    thr[m] = best[0]
print('prevalence-correct thresholds (validation split):', thr, flush=True)

with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
    inv = d.read(1); ok0 = inv != d.nodata; prof = d.profile; px = abs(d.transform.a) ** 2 / 1e6
rows = []
for m in MODELS:
    with rasterio.open(os.path.join(PM, f'FSM_{m}_probability.tif')) as d:
        p = d.read(1)
    ok = ok0 & (p != -9999)
    b = np.where(ok, (p >= thr[m]).astype(np.uint8), 255).astype(np.uint8)
    with rasterio.open(os.path.join(OUTB, f'FSM_{m}_binary.tif'), 'w', **(prof | dict(dtype='uint8', nodata=255, compress='deflate'))) as d:
        d.write(b, 1)
    f, o = (b == 1)[ok], (inv == 1)[ok]; n = ok.sum()
    tp = (f & o).sum(); fp = (f & ~o).sum(); fn = (~f & o).sum(); hr = (tp + fp) * (tp + fn) / n
    rows.append(dict(model=m, threshold=thr[m], POD=tp / (tp + fn), FAR=fp / (tp + fp), FBI=(tp + fp) / (tp + fn),
                     CSI=tp / (tp + fp + fn), ETS=(tp - hr) / (tp + fp + fn - hr), flagged_km2=(tp + fp) * px))
R = pd.DataFrame(rows).round(4); R.to_csv(os.path.join(OUT, f'{TAG}.csv'), index=False)
json.dump(thr, open(os.path.join(OUT, f'{TAG}_thresholds.json'), 'w'), indent=2)
print(R.to_string(index=False))
