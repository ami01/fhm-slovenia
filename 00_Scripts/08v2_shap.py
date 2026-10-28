"""
08v2 — SHAP for the v2 models at all five sample sizes (review 2026-10-09). Same recipe as 08d:
RF TreeExplainer (tree_path_dependent, 1,000 test samples), XGBoost native pred_contribs (3,000), CatBoost
TreeExplainer (3,000), DNN GradientExplainer (1,000; 300 background). Positive-class attributions, mean |SHAP|.
    python 00_Scripts/08v2_shap.py   -> 09_Analysis/shap_v2/shap_importance_all_sizes.csv, rank_comparison.csv
"""
import json, os, warnings
warnings.filterwarnings('ignore')
import joblib, numpy as np, pandas as pd, shap, torch, torch.nn as nn, xgboost as xgb
from catboost import CatBoostClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '09_Analysis', 'shap_v2'); os.makedirs(OUT, exist_ok=True)
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
SIZES, SEED = ['1x', '2x', '3x', '4x', '10x'], 42
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


def pos(v):          # TreeExplainer may return (n, f, 2) or a list per class
    if isinstance(v, list): return v[1]
    v = np.asarray(v); return v[..., 1] if v.ndim == 3 else v


rows = []
for size in SIZES:
    tag = 'v2' if size == '10x' else f'v2_{size}'
    mdl = os.path.join(ROOT, '04_Models_v2', tag)
    te = pd.read_parquet(os.path.join(ROOT, '03_Samples_v2', tag, 'samples_test.parquet'))
    X = te[FEATURES].values.astype(np.float32); rng = np.random.default_rng(SEED)
    i1k, i3k, ibg = (rng.choice(len(X), n, replace=False) for n in (1000, 3000, 300))
    rf = joblib.load(os.path.join(mdl, 'model_rf.pkl'))
    s_rf = pos(shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent').shap_values(X[i1k], check_additivity=False))
    bst = xgb.XGBClassifier(); bst.load_model(os.path.join(mdl, 'model_xgb.json'))
    s_xgb = bst.get_booster().predict(xgb.DMatrix(X[i3k]), pred_contribs=True)[:, :-1]
    cm = CatBoostClassifier(); cm.load_model(os.path.join(mdl, 'model_cat.cbm'))
    s_cat = pos(shap.TreeExplainer(cm).shap_values(X[i3k]))
    sc = joblib.load(os.path.join(mdl, 'scaler.pkl'))
    dnn = FloodDNN(len(FEATURES)).to(DEVICE); dnn.load_state_dict(torch.load(os.path.join(mdl, 'model_dnn.pt'), map_location=DEVICE)); dnn.eval()
    tt = lambda Z: torch.tensor(sc.transform(Z).astype(np.float32)).to(DEVICE)
    s_dnn = np.asarray(shap.GradientExplainer(dnn, tt(X[ibg])).shap_values(tt(X[i1k])))
    s_dnn = s_dnn.reshape(len(i1k), -1)[:, :len(FEATURES)] if s_dnn.ndim != 2 else s_dnn
    for name, sv in (('RF', s_rf), ('XGBoost', s_xgb), ('CatBoost', s_cat), ('DNN', s_dnn)):
        for f, v in zip(FEATURES, np.abs(sv).mean(0)):
            rows.append(dict(SampleSize=size, Model=name, Feature=f, MeanAbsSHAP=float(v)))
    print(size, 'done', flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, 'shap_importance_all_sizes.csv'), index=False)
D['rank'] = D.groupby(['SampleSize', 'Model']).MeanAbsSHAP.rank(ascending=False)
R = D.groupby('Feature')['rank'].agg(['mean', 'min', 'max']).sort_values('mean').round(2)
R.to_csv(os.path.join(OUT, 'rank_summary.csv')); print(R.to_string())
