"""
05v2 — Retrain the four learners + ensemble with corrected data and a leak-free protocol (review 2026-10-09).
==========================================================================================================
    FEATURE_SET=v2 python 00_Scripts/05v2_train.py      (v1 = original rasters, same corrected protocol)

Same sample points, labels and train/test split as the production 10x run (03_Samples/samples_{train,test}_10x.csv).
Features are re-extracted at those points from 01_Input_Features (v1) or 01_Input_Features_v2 (v2: flow-path HAND,
re-routed flow accumulation, fixed hydrogeology), so the only differences between runs are the data and protocol.

Protocol fixes vs 05c (production):
  * 05c used the TEST set as eval_set for XGBoost/CatBoost early stopping (CatBoost use_best_model=True) and as the
    DNN's validation set for early stopping / best-epoch selection, and 07 chose the CSI threshold on the test set.
    Here a stratified 10 % validation split of the TRAINING data does all of that; the test set is touched once.
  * Heavy-tailed predictors (flow accumulation, SPI, LS factor; max 5e7-1e8) enter as log1p; trees are invariant to
    this, the DNN is not.
Learner settings are those of 05c (RF 300 trees depth 20 leaf 5; XGBoost/CatBoost 1000 rounds lr 0.05 depth 8,
early stopping 30; DNN 128-64-32-16 BN dropout 0.3, Adam 1e-3, cosine, batch 4096, 200 epochs, patience 20).
Outputs → 04_Models_v2/<set>/ (models, scaler, thresholds.json), 05_Results_v2/<set>/metrics_holdout.csv,
          03_Samples_v2/<set>/samples_{train,test}.parquet
"""
import gc, json, os, time, warnings
warnings.filterwarnings('ignore')
import joblib, numpy as np, pandas as pd, rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = os.environ.get('FEATURE_SET', 'v2')
SIZE = os.environ.get('MAP_SIZE', '10x')            # 1x (50 k per class) … 10x (500 k per class), nested samples
TAG = os.environ.get('TAG') or (SET if SIZE == '10x' else f'{SET}_{SIZE}')
TRAIN_CSV = os.environ.get('TRAIN_CSV')                # e.g. samples_train_10x_hardneg.csv (03h)
FDIR = os.path.join(ROOT, '01_Input_Features' if SET == 'v1' else '01_Input_Features_v2')
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
DROP = [f for f in os.environ.get('DROP_FEATURES', '').split(',') if f]   # ablation, e.g. DROP_FEATURES=LULC TAG=v2_1x_noLULC
ALL_FEATURES = FEATURES                     # extraction and NaN-dropping use all, so ablation rows are paired
FEATURES = [f for f in FEATURES if f not in DROP]
LOG = ['flow_accumulation', 'spi', 'ls_factor']
SEED = 42
MDL, RES, SMP = (os.path.join(ROOT, d, TAG) for d in ('04_Models_v2', '05_Results_v2', '03_Samples_v2'))
for d in (MDL, RES, SMP): os.makedirs(d, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(SEED); torch.manual_seed(SEED)
t_all = time.time()


def extract(df):
    out = df[['label', 'x_3794', 'y_3794']].copy()
    for f in ALL_FEATURES:
        with rasterio.open(os.path.join(FDIR, f + '.tif')) as d:
            r, c = rasterio.transform.rowcol(d.transform, df.x_3794.values, df.y_3794.values)
            a = d.read(1); v = a[np.asarray(r), np.asarray(c)].astype('float64')
            if d.nodata is not None and not np.isnan(d.nodata): v[v == d.nodata] = np.nan
        out[f] = np.log1p(np.maximum(v, 0)) if f in LOG else v
    return out


_sfx = '' if SIZE == '1x' else f'_{SIZE}'
tr = extract(pd.read_csv(os.path.join(ROOT, '03_Samples', TRAIN_CSV or f'samples_train{_sfx}.csv')))
te = extract(pd.read_csv(os.path.join(ROOT, '03_Samples', f'samples_test{_sfx}.csv')))
for n, d in (('train', tr), ('test', te)):
    bad = d[FEATURES].isna().any(axis=1)
    print(f'{SET} {n}: {len(d):,} samples, {bad.sum()} with missing features (dropped)', flush=True)
tr = tr.dropna(); te = te.dropna()
tr.to_parquet(os.path.join(SMP, 'samples_train.parquet'), index=False); te.to_parquet(os.path.join(SMP, 'samples_test.parquet'), index=False)

X, y = tr[FEATURES].values.astype(np.float32), tr.label.values.astype(np.float32)
X_te, y_te = te[FEATURES].values.astype(np.float32), te.label.values.astype(np.float32)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
scale_pos = float((y_tr == 0).sum()) / float((y_tr == 1).sum())


def csi_threshold(yy, p):
    ts = np.round(np.arange(0.05, 0.951, 0.005), 3); best = (0.5, -1)
    for t in ts:
        q = p >= t; tp = (q & (yy == 1)).sum(); c = tp / max(tp + (q & (yy == 0)).sum() + (~q & (yy == 1)).sum(), 1)
        if c > best[1]: best = (t, c)
    return float(best[0])


def metr(yy, p, t, name):
    q = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yy, q, labels=[0, 1]).ravel()
    return dict(Model=name, threshold=t, AUC=roc_auc_score(yy, p), CSI=tp / (tp + fp + fn), F1=f1_score(yy, q),
                Kappa=cohen_kappa_score(yy, q), Sensitivity=tp / (tp + fn), Specificity=tn / (tn + fp))


class FloodDNN(nn.Module):
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)


P_va, P_te = {}, {}
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=5, max_features='sqrt',
                            class_weight='balanced', n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
P_va['RF'], P_te['RF'] = rf.predict_proba(X_va)[:, 1], rf.predict_proba(X_te)[:, 1]
joblib.dump(rf, os.path.join(MDL, 'model_rf.pkl')); del rf; print(f'RF {time.time() - t0:.0f} s', flush=True)
t0 = time.time()
xm = xgb.XGBClassifier(n_estimators=1000, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       scale_pos_weight=scale_pos, tree_method='hist', device='cuda', eval_metric='auc',
                       early_stopping_rounds=30, random_state=SEED, verbosity=0)
xm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
P_va['XGBoost'], P_te['XGBoost'] = xm.predict_proba(X_va)[:, 1], xm.predict_proba(X_te)[:, 1]
xm.save_model(os.path.join(MDL, 'model_xgb.json')); print(f'XGBoost best round {xm.best_iteration}, {time.time() - t0:.0f} s', flush=True); del xm
t0 = time.time()
cm = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3, auto_class_weights='Balanced',
                        task_type='GPU', devices='0', eval_metric='AUC', early_stopping_rounds=30, random_seed=SEED, verbose=False)
cm.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
P_va['CatBoost'], P_te['CatBoost'] = cm.predict_proba(X_va)[:, 1], cm.predict_proba(X_te)[:, 1]
cm.save_model(os.path.join(MDL, 'model_cat.cbm')); print(f'CatBoost best round {cm.best_iteration_}, {time.time() - t0:.0f} s', flush=True); del cm
t0 = time.time()
sc = StandardScaler().fit(X_tr); joblib.dump(sc, os.path.join(MDL, 'scaler.pkl'))
a, v, t_ = (sc.transform(z).astype(np.float32) for z in (X_tr, X_va, X_te))
pw = torch.tensor([scale_pos], dtype=torch.float32).to(DEVICE)
dnn = FloodDNN(a.shape[1]).to(DEVICE); crit = nn.BCEWithLogitsLoss(pos_weight=pw)
opt = optim.Adam(dnn.parameters(), lr=1e-3, weight_decay=1e-4)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)
dl = DataLoader(TensorDataset(torch.tensor(a), torch.tensor(y_tr).unsqueeze(1)), batch_size=4096, shuffle=True)
pred = lambda Z: torch.sigmoid(torch.cat([dnn(torch.tensor(Z[i:i + 16384]).to(DEVICE)).cpu()
                                          for i in range(0, len(Z), 16384)])).numpy().ravel()
best, best_ep, state = 0, 0, None
for ep in range(1, 201):
    dnn.train()
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad(); l = crit(dnn(xb), yb); l.backward()
        nn.utils.clip_grad_norm_(dnn.parameters(), 1.0); opt.step()
    dnn.eval()
    with torch.no_grad():
        au = roc_auc_score(y_va, pred(v))
    sch.step()
    if au > best: best, best_ep, state = au, ep, {k: q.cpu().clone() for k, q in dnn.state_dict().items()}
    if ep - best_ep >= 20: break
dnn.load_state_dict(state); dnn.eval()
with torch.no_grad():
    P_va['DNN'], P_te['DNN'] = pred(v), pred(t_)
torch.save(state, os.path.join(MDL, 'model_dnn.pt')); print(f'DNN best epoch {best_ep}, {time.time() - t0:.0f} s', flush=True)
P_va['Ensemble_mean'] = np.mean([P_va[k] for k in ('RF', 'XGBoost', 'CatBoost', 'DNN')], 0)
P_te['Ensemble_mean'] = np.mean([P_te[k] for k in ('RF', 'XGBoost', 'CatBoost', 'DNN')], 0)
thr = {k: csi_threshold(y_va, p) for k, p in P_va.items()}
json.dump(dict(thresholds=thr, feature_set=SET, size=SIZE, log_features=LOG, n_train=len(X_tr), n_val=len(X_va), n_test=len(X_te),
               dnn_best_epoch=best_ep, minutes=(time.time() - t_all) / 60), open(os.path.join(MDL, 'thresholds.json'), 'w'), indent=2)
M = pd.DataFrame([metr(y_te, P_te[k], thr[k], k) for k in P_te]).round(4)
M.to_csv(os.path.join(RES, 'metrics_holdout.csv'), index=False)
pd.DataFrame({'label': y_te, **P_te}).to_parquet(os.path.join(RES, 'holdout_predictions.parquet'), index=False)
print(M.to_string(index=False)); print(f'done in {(time.time() - t_all) / 60:.1f} min')
