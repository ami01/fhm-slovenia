"""
08f — Transferability tests on the v2 data (review 2026-10-09).
==============================================================
    MODE=regime  python 00_Scripts/08f_transfer_tests.py   # leave-one-regime-out
    MODE=buffer  BUFFER_M=5000 python 00_Scripts/08f_transfer_tests.py   # 50 km block CV with a training buffer

regime: the 26 ARSO hydrological warning units (geo_static.json) are grouped a priori into five groups —
  Alpine, Dinaric karst, sub-Pannonian, sub-Mediterranean, and central basins (Ljubljana Moor, Gradaščica, middle
  Sava; between regimes, so not forced into one) — and each group is predicted by models trained on the other four.
buffer: the 08e folds (50 km blocks, checkerboard), but training samples within BUFFER_M of any test sample are removed,
  so no training point touches the test area (08e's blocks share edges).
Learners and settings as 08e (RF 200 trees; XGBoost/CatBoost 500 rounds with early stopping on a 10 % validation split of
the training data; DNN 128-64-32-16). Scores at the model's national threshold (04_Models_v2/v2/thresholds.json) and at
a threshold from the training validation split; AUC is threshold-free. Samples: 03_Samples_v2/v2 (v2 features).
Outputs → 09_Analysis/transfer_v2/<mode>/{per_fold_metrics.csv, mean_metrics.csv, fold_predictions.parquet}
"""
import gc, json, os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE, BUF = os.environ.get('MODE', 'regime'), float(os.environ.get('BUFFER_M', 5000))
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
THR = json.load(open(os.path.join(ROOT, '04_Models_v2', 'v2', 'thresholds.json')))['thresholds']
OUT = os.path.join(ROOT, '09_Analysis', 'transfer_v2', MODE if MODE == 'regime' else f'buffer{int(BUF)}'); os.makedirs(OUT, exist_ok=True)
SEED = 42; DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); np.random.seed(SEED); torch.manual_seed(SEED)
REGIMES = {
    'Alpine': ['Zgornja Soča', 'Srednja Soča', 'Idrijca', 'Zgornja Sava', 'Sora', 'Kamniška Bistrica', 'Savinja', 'Zgornja Drava'],
    'Dinaric karst': ['Kras', 'Reka', 'Kraška Ljubljanica', 'Krka', 'Kolpa'],
    'sub-Pannonian': ['Mura', 'Ledava in Velika Krka', 'Pesnica in Ščavnica', 'Dravinja', 'Spodnja Drava', 'Sotla', 'Spodnja Sava'],
    'sub-Mediterranean': ['Morje', 'Vipava', 'Rižana in Dragonja'],
    'central basins': ['Ljubljanica', 'Gradaščica', 'Srednja Sava']}

df = pd.concat([pd.read_parquet(os.path.join(ROOT, '03_Samples_v2', 'v2', f'samples_{k}.parquet')) for k in ('train', 'test')],
               ignore_index=True)
if MODE == 'regime':
    u = gpd.read_file(os.path.join(ROOT, 'geo_static.json')).to_crs(3794)
    u['regime'] = u.ime.map({n: r for r, ns in REGIMES.items() for n in ns})
    assert u.regime.notna().all(), u.ime[u.regime.isna()].tolist()
    pts = gpd.GeoDataFrame(df[['x_3794', 'y_3794']], geometry=gpd.points_from_xy(df.x_3794, df.y_3794), crs=3794)
    j = gpd.sjoin(pts, u[['regime', 'geometry']], how='left', predicate='within')
    df['fold'] = j.groupby(level=0).regime.first().reindex(df.index).values
    df = df[df.fold.notna()]
else:
    bx = ((df.x_3794 - df.x_3794.min()) // 50_000).astype(int); by = ((df.y_3794 - df.y_3794.min()) // 50_000).astype(int)
    df['fold'] = ((bx * 2 + by) % 5) + 1
print(MODE, df.groupby('fold').label.agg(['size', 'mean']).round(3).to_string(), flush=True)


class FloodDNN(nn.Module):
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)


def csi_t(y, p):
    best = (0.5, -1)
    for t in np.linspace(0.05, 0.95, 91):
        q = p >= t; tp = (q & (y == 1)).sum(); c = tp / max(tp + (q & (y == 0)).sum() + (~q & (y == 1)).sum(), 1)
        if c > best[1]: best = (t, c)
    return best[0]


def scores(y, p, t):
    q = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return dict(CSI=tp / max(tp + fp + fn, 1), F1=f1_score(y, q), Kappa=cohen_kappa_score(y, q),
                POD=tp / max(tp + fn, 1), Specificity=tn / max(tn + fp, 1))


def dnn_fit(Xa, ya, Xv, yv, Xt, sp):
    sc = StandardScaler().fit(Xa); a, v, t_ = (sc.transform(z).astype(np.float32) for z in (Xa, Xv, Xt))
    m = FloodDNN(a.shape[1]).to(DEVICE); crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([sp], dtype=torch.float32).to(DEVICE))
    opt = optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-5)
    dl = DataLoader(TensorDataset(torch.tensor(a), torch.tensor(ya).unsqueeze(1)), batch_size=2048, shuffle=True)
    pr = lambda Z: torch.sigmoid(torch.cat([m(torch.tensor(Z[i:i + 16384]).to(DEVICE)).cpu() for i in range(0, len(Z), 16384)])).numpy().ravel()
    best, be, st = 0, 0, None
    for ep in range(1, 151):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad(); l = crit(m(xb), yb); l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            au = roc_auc_score(yv, pr(v))
        sch.step()
        if au > best: best, be, st = au, ep, {k: q.cpu().clone() for k, q in m.state_dict().items()}
        if ep - be >= 15: break
    m.load_state_dict(st); m.eval()
    with torch.no_grad():
        return pr(t_), pr(v)


rows, preds = [], []
t_all = time.time()
for fold in sorted(df.fold.unique(), key=str):
    te, tr = df[df.fold == fold], df[df.fold != fold]
    if MODE != 'regime':
        d, _ = cKDTree(te[['x_3794', 'y_3794']].values).query(tr[['x_3794', 'y_3794']].values, k=1)
        n0 = len(tr); tr = tr[d >= BUF]; print(f'fold {fold}: buffer {BUF:.0f} m removed {n0 - len(tr):,} training samples', flush=True)
    if te.label.nunique() < 2:
        print(f'fold {fold}: single class in test, skipped'); continue
    X, y = tr[FEATURES].values.astype(np.float32), tr.label.values.astype(np.float32)
    Xt, yt = te[FEATURES].values.astype(np.float32), te.label.values.astype(int)
    Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
    sp = float((ya == 0).sum()) / max(float((ya == 1).sum()), 1)
    P, T = {}, {}; t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
                                n_jobs=-1, random_state=SEED, oob_score=True).fit(X, y)
    P['RF'] = rf.predict_proba(Xt)[:, 1]; T['RF'] = csi_t(y, rf.oob_decision_function_[:, 1]); del rf
    xm = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sp,
                           tree_method='hist', device='cuda', early_stopping_rounds=20, eval_metric='auc', random_state=SEED, verbosity=0)
    xm.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False); P['XGBoost'] = xm.predict_proba(Xt)[:, 1]; T['XGBoost'] = csi_t(yv, xm.predict_proba(Xv)[:, 1]); del xm
    cm = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8, auto_class_weights='Balanced', task_type='GPU', devices='0',
                            early_stopping_rounds=20, eval_metric='AUC', random_seed=SEED, verbose=False)
    cm.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True); P['CatBoost'] = cm.predict_proba(Xt)[:, 1]; T['CatBoost'] = csi_t(yv, cm.predict_proba(Xv)[:, 1]); del cm
    P['DNN'], pv = dnn_fit(Xa, ya, Xv, yv, Xt, sp); T['DNN'] = csi_t(yv, pv)
    P['Ensemble_mean'] = np.mean([P[k] for k in ('RF', 'XGBoost', 'CatBoost', 'DNN')], 0); T['Ensemble_mean'] = THR['Ensemble_mean']
    for m, p in P.items():
        for rule, t in (('national', THR[m]), ('trainfold', T[m])):
            rows.append(dict(fold=fold, model=m, rule=rule, threshold=t, AUC=roc_auc_score(yt, p), n_test=len(yt),
                             flood_share=yt.mean(), n_train=len(y), **scores(yt, p, t)))
    preds.append(pd.DataFrame({'fold': fold, 'x_3794': te.x_3794.values, 'y_3794': te.y_3794.values, 'label': yt,
                               **{k: v.astype(np.float32) for k, v in P.items()}}))
    gc.collect(); torch.cuda.empty_cache()
    print(f'{fold}: ' + ', '.join(f'{k} AUC {roc_auc_score(yt, v):.3f}' for k, v in P.items()) + f'  [{time.time() - t0:.0f} s]', flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'per_fold_metrics.csv'), index=False)
pd.concat(preds).to_parquet(os.path.join(OUT, 'fold_predictions.parquet'), index=False)
M = R.groupby(['model', 'rule'])[['AUC', 'CSI', 'Kappa', 'POD']].agg(['mean', 'std']).round(4)
M.columns = ['_'.join(c) for c in M.columns]; M.reset_index().to_csv(os.path.join(OUT, 'mean_metrics.csv'), index=False)
print(R[R.rule == 'national'].pivot_table(index='fold', columns='model', values='AUC').round(3).to_string())
print(M.to_string()); print(f'done in {(time.time() - t_all) / 60:.1f} min -> {OUT}')
