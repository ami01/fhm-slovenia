"""
SHAP Analysis — Slovenia Flood Susceptibility (all sizes × all models)
=======================================================================
Runs SHAP for all 4 models (RF, XGBoost, CatBoost, DNN) at each
sample size (1x, 2x, 3x, 4x, 10x) using the corresponding trained
models and held-out test samples.

Explainers used:
  RF        : TreeExplainer (tree_path_dependent)   | 1 000 test samples
  XGBoost   : Native pred_contribs (Booster API)    | 3 000 test samples
  CatBoost  : TreeExplainer                         | 3 000 test samples
  DNN       : GradientExplainer                     | 1 000 test samples
                                                      300 background samples

Outputs → ML_Results/shap/{size}/
  shap_importance_all_models.csv      mean |SHAP| per feature per model
  shap_beeswarm_{model}.png           beeswarm summary plot
  shap_bar_{model}.png                bar importance plot
  shap_importance_comparison.png      multi-model comparison
  shap_dependence_top3_{model}.png    dependence plots — top-3 features

Also saved:
  ML_Results/shap/shap_importance_all_sizes.csv   cross-size comparison
  ML_Results/shap/shap_rank_stability.png         feature rank stability
"""

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
ML_BASE     = BASE
FEAT_FILE   = f'{BASE}/02_Collinearity/selected_features.txt'
SAMPLES_DIR = f'{BASE}/03_Samples'
OUT_BASE    = f'{BASE}/09_Analysis/shap'

SIZES = ['1x', '2x', '3x', '4x', '10x']

N_RF  = 1000    # test samples for RF (slow TreeExplainer)
N_BST = 3000    # test samples for XGB / CatBoost
N_DNN = 1000    # test samples for DNN
N_BG  = 300     # background samples for DNN GradientExplainer

SEED      = 42
DNN_DROP  = 0.3

import warnings; warnings.filterwarnings('ignore')
import os, gc
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, shap
import torch, torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier

os.makedirs(OUT_BASE, exist_ok=True)
rng    = np.random.default_rng(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}\n")

# ── Feature names ─────────────────────────────────────────────────────────────
with open(FEAT_FILE) as f:
    FEATURES = [l.strip() for l in f if l.strip()]
N_FEAT = len(FEATURES)
print(f"Features ({N_FEAT}): {FEATURES}\n")

SHORT = {
    'Distance_to_River': 'Dist_River',   'Distance_to_Road': 'Dist_Road',
    'Hgk_grouped':       'Hydro_Soil',   'Pedoloska_grouped': 'Soil_Type',
    'River_Density':     'River_Den',    'aspect':            'Aspect',
    'drainage_density':  'Drain_Den',    'elevation':         'Elevation',
    'flow_accumulation': 'Flow_Acc',     'hand':              'HAND',
    'lito_poly_250_d96_grouped': 'Lithology',
    'ls_factor':         'LS_Factor',    'plan_curvature':    'Plan_Curv',
    'profile_curvature': 'Prof_Curv',
    'slovenia_annual_max_daily_p90_3794_12m5': 'Rain_P90',
    'spi':               'SPI',          'tri':               'TRI',
    'twi':               'TWI',          'valley_depth':      'Valley_Dep',
}
FEAT_LABELS = [SHORT.get(f, f) for f in FEATURES]

MODEL_COLORS = {'RF':'#2196F3','XGBoost':'#FF5722','CatBoost':'#4CAF50','DNN':'#FF9800'}

# ── DNN architecture ──────────────────────────────────────────────────────────
class FloodDNN(nn.Module):
    def __init__(self, n_in=N_FEAT, dropout=DNN_DROP):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64), nn.BatchNorm1d( 64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear( 64,  32), nn.BatchNorm1d( 32),  nn.ReLU(), nn.Dropout(dropout*0.7),
            nn.Linear( 32,  16), nn.BatchNorm1d( 16),  nn.ReLU(),
            nn.Linear( 16,   1),
        )
    def forward(self, x): return self.net(x)

# ── Helper ────────────────────────────────────────────────────────────────────
def savefig(path, dpi=160):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close('all')
    print(f"  → {os.path.basename(path)}")

# ── Accumulate cross-size importance ─────────────────────────────────────────
all_imp_records = []   # for cross-size comparison

# =============================================================================
# MAIN LOOP OVER SIZES
# =============================================================================
for size in SIZES:
    print()
    print("=" * 65)
    print(f"SHAP ANALYSIS — {size}")
    print("=" * 65)

    suffix  = '' if size == '1x' else f'_{size}'
    mdl_dir = os.path.join(ML_BASE, '04_Models', size)
    out_dir = os.path.join(OUT_BASE, size)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load test samples ─────────────────────────────────────────────────────
    te_path     = os.path.join(SAMPLES_DIR, f'samples_test{suffix}.csv')
    tr_path     = os.path.join(SAMPLES_DIR, f'samples_train{suffix}.csv')
    df_te       = pd.read_csv(te_path)
    df_tr       = pd.read_csv(tr_path)
    X_te_raw    = df_te[FEATURES].values.astype(np.float32)
    X_tr_raw    = df_tr[FEATURES].values.astype(np.float32)
    print(f"Test samples : {len(X_te_raw):,}")
    print(f"Train samples: {len(X_tr_raw):,}")

    # Subsample indices
    n_te        = len(X_te_raw)
    n_tr        = len(X_tr_raw)
    idx_rf      = rng.choice(n_te, min(N_RF,  n_te), replace=False)
    idx_bst     = rng.choice(n_te, min(N_BST, n_te), replace=False)
    idx_dnn     = rng.choice(n_te, min(N_DNN, n_te), replace=False)
    idx_bg      = rng.choice(n_tr, min(N_BG,  n_tr), replace=False)

    # Scale for DNN
    scaler      = joblib.load(os.path.join(mdl_dir, 'scaler.pkl'))
    X_te_sc     = scaler.transform(X_te_raw).astype(np.float32)
    X_tr_sc     = scaler.transform(X_tr_raw).astype(np.float32)
    X_dnn_test  = X_te_sc[idx_dnn]
    X_dnn_bg    = X_tr_sc[idx_bg]

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\nLoading models from {mdl_dir} ...")
    model_rf  = joblib.load(os.path.join(mdl_dir, 'model_rf.pkl'))

    model_cat = CatBoostClassifier()
    model_cat.load_model(os.path.join(mdl_dir, 'model_cat.cbm'))

    model_dnn = FloodDNN().to(DEVICE)
    model_dnn.load_state_dict(torch.load(os.path.join(mdl_dir, 'model_dnn.pth'),
                                          map_location=DEVICE, weights_only=False))
    model_dnn.eval()
    print("  RF, CatBoost, DNN loaded")

    # ── SHAP computation ──────────────────────────────────────────────────────
    shap_dict = {}

    # RF — TreeExplainer
    print(f"\n[1/4] RF TreeExplainer (n={len(idx_rf)}) ...")
    exp_rf = shap.TreeExplainer(model_rf, feature_perturbation='tree_path_dependent')
    sv_rf  = exp_rf.shap_values(X_te_raw[idx_rf], check_additivity=False)
    if isinstance(sv_rf, list):   sv_rf = sv_rf[1]
    elif sv_rf.ndim == 3:         sv_rf = sv_rf[:, :, 1]
    shap_dict['RF'] = (sv_rf, X_te_raw[idx_rf])
    print(f"  shape: {sv_rf.shape}")
    del exp_rf; gc.collect()

    # XGBoost — native pred_contribs (avoids shap/xgb3 base_score bug)
    print(f"[2/4] XGBoost pred_contribs (n={len(idx_bst)}) ...")
    bst_xgb = xgb.Booster()
    bst_xgb.load_model(os.path.join(mdl_dir, 'model_xgb.json'))
    dm_xgb  = xgb.DMatrix(X_te_raw[idx_bst], feature_names=FEATURES)
    sv_xgb  = bst_xgb.predict(dm_xgb, pred_contribs=True)[:, :-1].astype(np.float32)
    shap_dict['XGBoost'] = (sv_xgb, X_te_raw[idx_bst])
    print(f"  shape: {sv_xgb.shape}")
    del bst_xgb, dm_xgb; gc.collect()

    # CatBoost — TreeExplainer
    print(f"[3/4] CatBoost TreeExplainer (n={len(idx_bst)}) ...")
    exp_cat = shap.TreeExplainer(model_cat)
    sv_cat  = exp_cat.shap_values(X_te_raw[idx_bst])
    if isinstance(sv_cat, list): sv_cat = sv_cat[1]
    elif sv_cat.ndim == 3:       sv_cat = sv_cat[:, :, 1]
    shap_dict['CatBoost'] = (sv_cat, X_te_raw[idx_bst])
    print(f"  shape: {sv_cat.shape}")
    del exp_cat; gc.collect()

    # DNN — GradientExplainer
    print(f"[4/4] DNN GradientExplainer (n={len(idx_dnn)}, bg={len(idx_bg)}) ...")
    bg_t    = torch.tensor(X_dnn_bg,   dtype=torch.float32).to(DEVICE)
    te_t    = torch.tensor(X_dnn_test, dtype=torch.float32).to(DEVICE)
    exp_dnn = shap.GradientExplainer(model_dnn, bg_t)
    sv_dnn  = exp_dnn.shap_values(te_t)
    if isinstance(sv_dnn, list):         sv_dnn = sv_dnn[0]
    if isinstance(sv_dnn, torch.Tensor): sv_dnn = sv_dnn.detach().cpu().numpy()
    if sv_dnn.ndim == 3:                 sv_dnn = sv_dnn[:, :, 0]
    shap_dict['DNN'] = (sv_dnn, X_dnn_test)
    print(f"  shape: {sv_dnn.shape}")
    del exp_dnn; gc.collect()

    # ── Mean |SHAP| importance ─────────────────────────────────────────────────
    imp = {name: np.abs(sv).mean(axis=0) for name, (sv, _) in shap_dict.items()}
    df_imp = pd.DataFrame(imp, index=FEAT_LABELS)
    df_imp.to_csv(os.path.join(out_dir, 'shap_importance_all_models.csv'))
    print(f"\nTop-5 features per model [{size}]:")
    for m in df_imp.columns:
        top5 = df_imp[m].sort_values(ascending=False).head(5)
        print(f"  {m}: {list(top5.index)}")

    # Accumulate for cross-size
    for model in df_imp.columns:
        for feat, val in df_imp[model].items():
            all_imp_records.append({'SampleSize': size, 'Model': model,
                                    'Feature': feat, 'MeanAbsSHAP': val})

    # ── a) Beeswarm plots ──────────────────────────────────────────────────────
    print("\nBeeswarm plots ...")
    for name, (sv, Xsub) in shap_dict.items():
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, Xsub, feature_names=FEAT_LABELS,
                          plot_type='dot', max_display=19, show=False)
        plt.title(f'SHAP Beeswarm — {name}  [{size}]', fontsize=14, fontweight='bold', pad=14)
        savefig(os.path.join(out_dir, f'shap_beeswarm_{name.lower()}.png'))

    # ── b) Bar plots ───────────────────────────────────────────────────────────
    print("Bar plots ...")
    for name, (sv, Xsub) in shap_dict.items():
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, Xsub, feature_names=FEAT_LABELS,
                          plot_type='bar', max_display=19, show=False)
        plt.title(f'SHAP Mean |SHAP| — {name}  [{size}]', fontsize=14, fontweight='bold', pad=14)
        for patch in plt.gca().patches:
            patch.set_facecolor(MODEL_COLORS[name])
        savefig(os.path.join(out_dir, f'shap_bar_{name.lower()}.png'))

    # ── c) Multi-model importance comparison ───────────────────────────────────
    print("Importance comparison plot ...")
    df_s = df_imp.copy()
    df_s['mean_all'] = df_s.mean(axis=1)
    df_s = df_s.sort_values('mean_all', ascending=True).drop(columns='mean_all')

    n, n_m = len(FEATURES), len(shap_dict)
    bh  = 0.18
    offs = np.linspace(-(n_m-1)/2, (n_m-1)/2, n_m) * bh
    y   = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, 10))
    for i, (mname, color) in enumerate(MODEL_COLORS.items()):
        if mname not in df_s.columns: continue
        ax.barh(y + offs[i], df_s[mname].values, bh,
                label=mname, color=color, alpha=0.85, edgecolor='white')
    ax.set_yticks(y); ax.set_yticklabels(df_s.index, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title(f'SHAP Feature Importance — All Models  [{size}]',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11); ax.grid(axis='x', alpha=0.3)
    savefig(os.path.join(out_dir, 'shap_importance_comparison.png'))

    # ── d) Dependence plots — top-3 features (XGBoost ranking) ────────────────
    print("Dependence plots ...")
    top3 = df_imp['XGBoost'].sort_values(ascending=False).head(3).index.tolist()
    print(f"  Top-3 (XGBoost): {top3}")

    for model_name, (sv, Xsub) in shap_dict.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'SHAP Dependence — {model_name}  [{size}] — Top-3 XGBoost Features',
                     fontsize=13, fontweight='bold')
        for col, feat_label in enumerate(top3):
            ax = axes[col]
            orig_feats = [k for k, v in SHORT.items() if v == feat_label]
            feat_idx   = FEATURES.index(orig_feats[0]) if orig_feats else FEAT_LABELS.index(feat_label)
            fv     = Xsub[:, feat_idx]
            sv_col = sv[:, feat_idx]
            sc = ax.scatter(fv, sv_col, c=sv_col, cmap='RdYlBu_r',
                            alpha=0.35, s=5, linewidths=0)
            ax.axhline(0, color='k', lw=0.8, ls='--')
            ax.set_xlabel(feat_label, fontsize=10)
            ax.set_ylabel('SHAP value' if col == 0 else '', fontsize=9)
            ax.set_title(feat_label, fontsize=10, fontweight='bold')
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        savefig(os.path.join(out_dir, f'shap_dependence_top3_{model_name.lower()}.png'), dpi=150)

    # Free memory
    del model_rf, model_cat, model_dnn, shap_dict, scaler
    gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    print(f"\n[{size}] SHAP done → {out_dir}")

# =============================================================================
# CROSS-SIZE COMPARISON
# =============================================================================
print()
print("=" * 65)
print("CROSS-SIZE SHAP IMPORTANCE COMPARISON")
print("=" * 65)

all_imp_df = pd.DataFrame(all_imp_records)
all_imp_df.to_csv(os.path.join(OUT_BASE, 'shap_importance_all_sizes.csv'), index=False)

# Feature rank stability: XGBoost across sizes
print("\nXGBoost feature rank across sizes:")
xgb_imp = all_imp_df[all_imp_df.Model=='XGBoost'].pivot_table(
    values='MeanAbsSHAP', index='Feature', columns='SampleSize')
xgb_imp['mean'] = xgb_imp.mean(axis=1)
xgb_imp = xgb_imp.sort_values('mean', ascending=False)
print(xgb_imp[[s for s in SIZES if s in xgb_imp.columns]].round(4).to_string())

# Rank stability heatmap
fig, axes = plt.subplots(1, len(MODEL_COLORS), figsize=(24, 10))
for ax, (model_name, color) in zip(axes, MODEL_COLORS.items()):
    sub = all_imp_df[all_imp_df.Model==model_name].pivot_table(
        values='MeanAbsSHAP', index='Feature', columns='SampleSize')
    if sub.empty:
        ax.set_visible(False); continue
    sub = sub[[s for s in SIZES if s in sub.columns]]
    # Rank (1=most important)
    ranked = sub.rank(ascending=False).astype(int)
    im = ax.imshow(ranked.values, aspect='auto', cmap='RdYlGn_r', vmin=1, vmax=N_FEAT)
    ax.set_xticks(range(len(sub.columns))); ax.set_xticklabels(sub.columns, fontsize=9)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_title(model_name, fontsize=11, fontweight='bold', color=color)
    for i in range(len(sub)):
        for j in range(len(sub.columns)):
            ax.text(j, i, str(ranked.values[i, j]),
                    ha='center', va='center', fontsize=8,
                    color='black' if ranked.values[i,j] < N_FEAT//2 else 'white')
    plt.colorbar(im, ax=ax, label='Rank', fraction=0.046, pad=0.04)

plt.suptitle('Feature Importance Rank Stability Across Sample Sizes',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_BASE, 'shap_rank_stability.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f"\nSaved: shap_rank_stability.png")

print()
print("=" * 65)
print("ALL SHAP DONE")
print("=" * 65)
for size in SIZES:
    out_dir = os.path.join(OUT_BASE, size)
    if os.path.isdir(out_dir):
        files = [f for f in os.listdir(out_dir) if f.endswith('.png') or f.endswith('.csv')]
        print(f"  shap/{size}/  — {len(files)} files")
print(f"  shap/shap_importance_all_sizes.csv")
print(f"  shap/shap_rank_stability.png")
