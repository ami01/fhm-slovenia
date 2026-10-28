"""
08e — Spatial block CV, v2 (review 2026-10-09): honest thresholds, ensemble score, substrate ablation.
=====================================================================================================
Same folds (50 km blocks, checkerboard (2·bx + by) mod 5), same learners and settings as 08b, at the
operational baseline (10x). Differences from 08b:

  * 08b chose each fold's CSI threshold on the TEST fold's own labels (optimistic for CSI/F1/κ). Here every
    fold is scored at three thresholds:
        national   — the model's CSI-optimal threshold from the national hold-out (08_Validation/<size>/
                     threshold_summary.csv): what a user applies to the delivered map;
        trainfold  — CSI-optimal on data the fold model was not fitted to but that come from the training folds
                     (internal 10 % validation split for XGBoost/CatBoost/DNN, out-of-bag for RF);
        oracle     — CSI-optimal on the test fold (08b's choice), kept for comparison only.
  * Per-sample fold predictions are saved, so the ensemble (unweighted mean of the four fold models, threshold
    from the national hold-out) gets a spatial-CV score.
  * RF ablation without the three substrate maps (soil, lithology, hydrogeology).
Note: training and test blocks share edges (no buffer), so this is spatial blocking, not a minimum distance.

SAMPLE_SET=v1|v2 uses the 05v2_train samples and thresholds (output folder <size>_<set>).
BLOCK_OFFSET_M=<m> shifts the block-grid origin by that distance in x and y (sensitivity to the grid origin, co-author
review 2026-10-22; output folder gets the suffix _shift<m>). Default 0 = the published folds.
Outputs → 09_Analysis/spatial_cv_v2/<size>[_<set>]/: per_fold_metrics.csv, mean_metrics.csv, fold_predictions.parquet,
          ablation_rf.csv, run_meta.json
"""
import gc, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = os.environ.get('MAP_SIZE', '10x')
SAMPLES_TRAIN = os.path.join(ROOT, '03_Samples', f'samples_train_{SIZE}.csv' if SIZE != '1x' else 'samples_train.csv')
SAMPLES_TEST = os.path.join(ROOT, '03_Samples', f'samples_test_{SIZE}.csv' if SIZE != '1x' else 'samples_test.csv')
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
SET = os.environ.get('SAMPLE_SET')               # None = production samples/thresholds; 'v1'/'v2' = 05v2_train outputs
if SET:
    import json as _json
    THR = pd.Series(_json.load(open(os.path.join(ROOT, '04_Models_v2', SET, 'thresholds.json')))['thresholds'])
    OUT = os.path.join(ROOT, '09_Analysis', 'spatial_cv_v2', f'{SIZE}_{SET}')
else:
    THR = pd.read_csv(os.path.join(ROOT, '08_Validation', SIZE, 'threshold_summary.csv')).set_index('Model').CSI_threshold
    OUT = os.path.join(ROOT, '09_Analysis', 'spatial_cv_v2', SIZE)
OFFSET = float(os.environ.get('BLOCK_OFFSET_M', 0))
if OFFSET: OUT += f'_shift{int(OFFSET)}'
os.makedirs(OUT, exist_ok=True)
BLOCK_SIZE, N_FOLDS, SEED = 50_000, 5, 42
DNN_EPOCHS, DNN_PATIENCE, DNN_BATCH = 150, 15, 2048
SUBSTRATE = ['Pedoloska_grouped', 'lito_poly_250_d96_grouped', 'Hgk_grouped']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(SEED); torch.manual_seed(SEED)

df = (pd.concat([pd.read_parquet(os.path.join(ROOT, '03_Samples_v2', SET, f'samples_{k}.parquet')) for k in ('train', 'test')],
                ignore_index=True) if SET else pd.concat([pd.read_csv(SAMPLES_TRAIN), pd.read_csv(SAMPLES_TEST)], ignore_index=True))
df['bx'] = ((df.x_3794 - df.x_3794.min() + OFFSET) // BLOCK_SIZE).astype(int)
df['by'] = ((df.y_3794 - df.y_3794.min() + OFFSET) // BLOCK_SIZE).astype(int)
df['fold'] = ((df.bx * 2 + df.by) % N_FOLDS) + 1
print(f'{SIZE}: {len(df):,} samples, {df.groupby(["bx", "by"]).ngroups} blocks, flood {df.label.mean():.3f}', flush=True)


class FloodDNN(nn.Module):                      # identical to 08b
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)


def csi_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 100); best_t, best_c = 0.5, -1
    for t in ts:
        q = p >= t; tp = (q & (y == 1)).sum(); c = tp / max(tp + (q & (y == 0)).sum() + (~q & (y == 1)).sum(), 1)
        if c > best_c: best_c, best_t = c, t
    return best_t


def scores(y, p, t):
    q = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return dict(CSI=tp / (tp + fp + fn), F1=f1_score(y, q), Kappa=cohen_kappa_score(y, q),
                Sensitivity=tp / (tp + fn), Specificity=tn / (tn + fp))


def dnn_fit_predict(X_tr2, y_tr2, X_val, y_val, X_te, scale_pos):
    sc = StandardScaler(); a = sc.fit_transform(X_tr2).astype(np.float32)
    v = sc.transform(X_val).astype(np.float32); te = sc.transform(X_te).astype(np.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([scale_pos], dtype=torch.float32).to(DEVICE))
    m = FloodDNN(a.shape[1]).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DNN_EPOCHS, eta_min=1e-5)
    dl = DataLoader(TensorDataset(torch.tensor(a), torch.tensor(y_tr2).unsqueeze(1)), batch_size=DNN_BATCH, shuffle=True)
    pred = lambda X: torch.sigmoid(torch.cat([m(torch.tensor(X[i:i + 8192]).to(DEVICE)).cpu()
                                              for i in range(0, len(X), 8192)])).numpy().ravel()
    best, best_ep, state = 0, 0, None
    for ep in range(1, DNN_EPOCHS + 1):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad(); l = crit(m(xb), yb); l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            au = roc_auc_score(y_val, pred(v))
        sch.step()
        if au > best: best, best_ep, state = au, ep, {k: t.cpu().clone() for k, t in m.state_dict().items()}
        if ep - best_ep >= DNN_PATIENCE: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return pred(te), pred(v)


rows, preds, abl = [], [], []
t_start = time.time()
for fold in range(1, N_FOLDS + 1):
    tr, te = df[df.fold != fold], df[df.fold == fold]
    X_tr, y_tr = tr[FEATURES].values.astype(np.float32), tr.label.values.astype(np.float32)
    X_te, y_te = te[FEATURES].values.astype(np.float32), te.label.values.astype(np.float32)
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED)
    scale_pos = float((y_tr2 == 0).sum()) / max(float((y_tr2 == 1).sum()), 1)
    P, Tt = {}, {}
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, max_features='sqrt',
                                class_weight='balanced', n_jobs=-1, random_state=SEED, oob_score=True).fit(X_tr, y_tr)
    P['RF'] = rf.predict_proba(X_te)[:, 1]; Tt['RF'] = csi_threshold(y_tr, rf.oob_decision_function_[:, 1]); del rf
    xm = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=scale_pos, tree_method='hist', device='cuda', early_stopping_rounds=20,
                           eval_metric='auc', random_state=SEED, verbosity=0)
    xm.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
    P['XGBoost'] = xm.predict_proba(X_te)[:, 1]; Tt['XGBoost'] = csi_threshold(y_val, xm.predict_proba(X_val)[:, 1]); del xm
    cm = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8, auto_class_weights='Balanced', task_type='GPU',
                            devices='0', early_stopping_rounds=20, eval_metric='AUC', random_seed=SEED, verbose=False)
    cm.fit(X_tr2, y_tr2, eval_set=(X_val, y_val), use_best_model=True)
    P['CatBoost'] = cm.predict_proba(X_te)[:, 1]; Tt['CatBoost'] = csi_threshold(y_val, cm.predict_proba(X_val)[:, 1]); del cm
    P['DNN'], pv = dnn_fit_predict(X_tr2, y_tr2, X_val, y_val, X_te, scale_pos); Tt['DNN'] = csi_threshold(y_val, pv)
    P['Ensemble_mean'] = np.mean([P[k] for k in ('RF', 'XGBoost', 'CatBoost', 'DNN')], 0)
    Tt['Ensemble_mean'] = np.nan                         # no per-fold validation predictions for the mean; use national
    for mname, p in P.items():
        auc = roc_auc_score(y_te, p)
        for tname, t in (('national', THR[mname]), ('trainfold', Tt[mname]), ('oracle', csi_threshold(y_te, p))):
            if np.isnan(t): continue
            rows.append(dict(Model=mname, Fold=fold, threshold_rule=tname, threshold=round(float(t), 4), AUC=auc,
                             **scores(y_te, p, t), N_test=len(y_te), Flood_pct=y_te.mean() * 100))
    preds.append(pd.DataFrame({'fold': fold, 'x_3794': te.x_3794.values, 'y_3794': te.y_3794.values, 'label': y_te,
                               **{k: v.astype(np.float32) for k, v in P.items()}}))
    # RF ablation without substrate
    cols = [f for f in FEATURES if f not in SUBSTRATE]
    rfa = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, max_features='sqrt',
                                 class_weight='balanced', n_jobs=-1, random_state=SEED).fit(tr[cols].values, y_tr)
    abl.append(dict(fold=fold, AUC_all=roc_auc_score(y_te, P['RF']),
                    AUC_no_substrate=roc_auc_score(y_te, rfa.predict_proba(te[cols].values)[:, 1]))); del rfa
    gc.collect(); torch.cuda.empty_cache()
    print(f'fold {fold}: ' + ', '.join(f'{k} {roc_auc_score(y_te, v):.4f}' for k, v in P.items())
          + f'  [{time.time() - t0:.0f} s]', flush=True)

R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'per_fold_metrics.csv'), index=False)
M = (R.groupby(['Model', 'threshold_rule'])[['AUC', 'CSI', 'F1', 'Kappa', 'Sensitivity', 'Specificity', 'threshold']]
     .agg(['mean', 'std']).round(4))
M.columns = ['_'.join(c) for c in M.columns]; M.reset_index().to_csv(os.path.join(OUT, 'mean_metrics.csv'), index=False)
pd.concat(preds).to_parquet(os.path.join(OUT, 'fold_predictions.parquet'), index=False)
A = pd.DataFrame(abl); A.to_csv(os.path.join(OUT, 'ablation_rf.csv'), index=False)
json.dump(dict(size=SIZE, n_samples=len(df), block_offset_m=OFFSET, blocks=int(df.groupby(['bx', 'by']).ngroups), seed=SEED,
               minutes=(time.time() - t_start) / 60, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
          open(os.path.join(OUT, 'run_meta.json'), 'w'), indent=2)
print(M[['AUC_mean', 'AUC_std', 'CSI_mean', 'CSI_std', 'threshold_mean']].to_string())
print('ablation RF: all %.4f ± %.4f | no substrate %.4f ± %.4f' % (A.AUC_all.mean(), A.AUC_all.std(), A.AUC_no_substrate.mean(),
                                                                  A.AUC_no_substrate.std()))
print(f'done in {(time.time() - t_start) / 60:.1f} min -> {OUT}')
