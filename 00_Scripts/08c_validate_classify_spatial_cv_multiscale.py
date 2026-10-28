"""
Multi-Scale Validation + Classification + Spatial CV
=====================================================
For each sample size (1x, 2x, 3x, 4x):

  PART A — Standard validation
    1. Load held-out test samples for that size.
    2. Extract probabilities from probability maps.
    3. Find CSI-optimal & Youden thresholds.
    4. Write binary (flood/no-flood) and 5-class susceptibility maps.
    5. Compute AUC, F1, CSI, Kappa, Sensitivity, Specificity.

  PART B — Spatial block cross-validation
    6. Combine train + test samples for that size.
    7. Partition into 5 folds by 50 km spatial blocks.
    8. Retrain RF / XGBoost / CatBoost / DNN from scratch per fold.
    9. Report mean ± std across folds.
    10. Compare random-split vs spatial-CV metrics.

Outputs → ML_Results/
  probability_maps/{size}/           (inputs — must exist)
  classified_maps/{size}/
    FSM_{model}_binary.tif
    FSM_{model}_5class.tif
  validation/{size}/
    threshold_summary.csv
    validation_metrics.csv
    confusion_matrices.png
    roc_validation.png
    csi_threshold_curves.png
    metrics_comparison.png
  spatial_cv/{size}/
    fold_assignments.csv
    per_fold_metrics.csv
    mean_metrics.csv
    fold_map.png
    spatial_cv_comparison.png
    per_fold_performance.png
  validation/
    multiscale_comparison.csv
    multiscale_comparison.png
"""

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
SAMPLES_DIR = f'{BASE}/03_Samples'
NODATA      = -9999.0
STRIP_ROWS  = 300
BLOCK_SIZE  = 50_000   # 50 km
N_FOLDS     = 5
RANDOM_SEED = 42
PIXEL_KM2   = (12.5 * 12.5) / 1e6

# DNN hyperparams (reduced for CV runtime)
DNN_EPOCHS   = 150
DNN_PATIENCE = 15
DNN_BATCH    = 2048
DNN_DROPOUT  = 0.3

SIZES        = ['1x', '2x', '3x', '4x', '10x']
MODELS       = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
CLASS_LABELS = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
COLORS       = {'RF': '#2196F3', 'XGBoost': '#FF5722', 'CatBoost': '#4CAF50',
                'DNN': '#FF9800', 'Ensemble_mean': '#9C27B0'}

import os, warnings, gc, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc as sk_auc,
    confusion_matrix, f1_score, accuracy_score,
    cohen_kappa_score, average_precision_score
)
import xgboost as xgb
from catboost import CatBoostClassifier
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(RANDOM_SEED)

# ── DNN architecture ──────────────────────────────────────────────────────────
class FloodDNN(nn.Module):
    def __init__(self, n_in, dropout=DNN_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64), nn.BatchNorm1d( 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear( 64,  32), nn.BatchNorm1d( 32), nn.ReLU(), nn.Dropout(dropout*0.7),
            nn.Linear( 32,  16), nn.BatchNorm1d( 16), nn.ReLU(),
            nn.Linear( 16,   1)
        )
    def forward(self, x): return self.net(x)

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_csi_threshold(y_true, y_prob, n=200):
    best_t, best_c = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, n):
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        c = tp / (tp + fp + fn + 1e-10)
        if c > best_c:
            best_c, best_t = c, t
    return best_t, best_c

def find_youden_threshold(y_true, y_prob, n=200):
    best_t, best_j = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, n):
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        j = tp/(tp+fn+1e-10) + tn/(tn+fp+1e-10) - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j

def full_metrics(y_true, y_prob, thresh, name):
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model':       name,
        'Threshold':   round(thresh, 4),
        'AUC':         round(roc_auc_score(y_true, y_prob),           4),
        'Avg_Prec':    round(average_precision_score(y_true, y_prob), 4),
        'F1':          round(f1_score(y_true, y_pred),                4),
        'Accuracy':    round(accuracy_score(y_true, y_pred),          4),
        'Kappa':       round(cohen_kappa_score(y_true, y_pred),       4),
        'CSI':         round(tp/(tp+fp+fn+1e-10),                     4),
        'Sensitivity': round(tp/(tp+fn+1e-10),                        4),
        'Specificity': round(tn/(tn+fp+1e-10),                        4),
        'Precision':   round(tp/(tp+fp+1e-10),                        4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }

def cv_metrics(y_true, y_prob, name, fold, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model': name, 'Fold': fold,
        'AUC':         round(roc_auc_score(y_true, y_prob), 4),
        'CSI':         round(tp/(tp+fp+fn+1e-10),           4),
        'F1':          round(f1_score(y_true, y_pred),       4),
        'Kappa':       round(cohen_kappa_score(y_true, y_pred), 4),
        'Sensitivity': round(tp/(tp+fn+1e-10),               4),
        'Specificity': round(tn/(tn+fp+1e-10),               4),
        'N_test':      len(y_true),
    }

# ── Accumulate cross-size results ─────────────────────────────────────────────
all_size_val   = []   # standard validation metrics across sizes
all_size_cv    = []   # spatial CV mean metrics across sizes

# =============================================================================
# MAIN LOOP OVER SIZES
# =============================================================================
for size in SIZES:
    suffix    = '' if size == '1x' else f'_{size}'
    prob_dir  = os.path.join(BASE, '06_Probability_Maps', size)
    clsf_dir  = os.path.join(BASE, '07_Classified_Maps',  size)
    vald_dir  = os.path.join(BASE, '08_Validation',        size)
    scv_dir   = os.path.join(BASE, '09_Analysis/spatial_cv',        size)
    for d in [clsf_dir, vald_dir, scv_dir]:
        os.makedirs(d, exist_ok=True)

    print()
    print("=" * 65)
    print(f"SIZE: {size}")
    print("=" * 65)

    # ── Check probability maps exist ──────────────────────────────────────────
    missing_maps = [m for m in MODELS
                    if not os.path.exists(os.path.join(prob_dir, f'FSM_{m}_probability.tif'))]
    if missing_maps:
        print(f"  WARNING: missing probability maps for {size}: {missing_maps}")
        print(f"  Skipping {size} — run predict_probability_maps_1x_4x.py first.")
        continue

    # ── Skip if already fully done ────────────────────────────────────────────
    done_val = os.path.exists(os.path.join(vald_dir, 'validation_metrics.csv'))
    done_cv  = os.path.exists(os.path.join(scv_dir,  'mean_metrics.csv'))
    if done_val and done_cv:
        print(f"  Already complete — loading results for cross-size summary.")
        _v = pd.read_csv(os.path.join(vald_dir, 'validation_metrics.csv'))
        for _, row in _v.iterrows():
            row_dict = row.to_dict(); row_dict['SampleSize'] = size
            all_size_val.append(row_dict)
        _cv = pd.read_csv(os.path.join(scv_dir, 'mean_metrics.csv'))
        for _, row in _cv.iterrows():
            all_size_cv.append(row.to_dict())
        continue

    # ─────────────────────────────────────────────────────────────────────────
    # PART A — STANDARD VALIDATION + CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"PART A — Standard validation  [{size}]")
    print(f"{'─'*65}")

    # Load test samples
    te_path = os.path.join(SAMPLES_DIR, f'samples_test{suffix}.csv')
    df_te   = pd.read_csv(te_path)
    y_true  = df_te['label'].values.astype(int)
    xs, ys  = df_te['x_3794'].values, df_te['y_3794'].values
    print(f"Test: {len(df_te):,}  flood={y_true.sum():,}  no-flood={(y_true==0).sum():,}")

    # Extract probabilities from maps
    prob_maps = {}
    for model in MODELS:
        path = os.path.join(prob_dir, f'FSM_{model}_probability.tif')
        with rasterio.open(path) as ds:
            tf   = ds.transform
            cols = np.clip(np.floor((xs - tf.c) / tf.a).astype(int), 0, ds.width-1)
            rows = np.clip(np.floor((ys - tf.f) / tf.e).astype(int), 0, ds.height-1)
            arr  = ds.read(1)
            probs = arr[rows, cols].astype(np.float64)
            probs[probs == NODATA] = np.nan
        prob_maps[model] = probs

    all_valid = np.all([~np.isnan(prob_maps[m]) for m in MODELS], axis=0)
    y_val = y_true[all_valid]
    print(f"Samples with all models valid: {all_valid.sum():,}")

    # Optimal thresholds
    thresh_records  = []
    optimal_thresh  = {}
    quant_breaks    = {}

    print(f"\n  {'Model':<20} {'CSI t':>8} {'CSI val':>9} {'Youden t':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*9} {'-'*10}")
    for model in MODELS:
        probs = prob_maps[model][all_valid]
        csi_t, csi_v = find_csi_threshold(y_val, probs)
        you_t, you_v = find_youden_threshold(y_val, probs)
        optimal_thresh[model] = csi_t
        thresh_records.append({'Model': model,
                                'CSI_threshold': round(csi_t, 4), 'CSI_value': round(csi_v, 4),
                                'Youden_threshold': round(you_t, 4), 'Youden_value': round(you_v, 4)})
        print(f"  {model:<20} {csi_t:>8.4f} {csi_v:>9.4f} {you_t:>10.4f}")

    pd.DataFrame(thresh_records).to_csv(os.path.join(vald_dir, 'threshold_summary.csv'), index=False)

    # Quantile breaks for 5-class maps
    for model in MODELS:
        path = os.path.join(prob_dir, f'FSM_{model}_probability.tif')
        with rasterio.open(path) as ds:
            scale  = 10
            arr_ds = ds.read(1, out_shape=(ds.height//scale, ds.width//scale),
                             resampling=rasterio.enums.Resampling.nearest)
        vp = arr_ds[arr_ds != NODATA].flatten()
        quant_breaks[model] = np.percentile(vp, [20, 40, 60, 80])

    # Write binary + 5-class rasters
    print(f"\n  Writing classified maps to {clsf_dir} ...")
    for model in MODELS:
        prob_path = os.path.join(prob_dir, f'FSM_{model}_probability.tif')
        bin_path  = os.path.join(clsf_dir, f'FSM_{model}_binary.tif')
        cls_path  = os.path.join(clsf_dir, f'FSM_{model}_5class.tif')
        thresh    = optimal_thresh[model]
        qb        = quant_breaks[model]

        with rasterio.open(prob_path) as src:
            meta_bin = src.meta.copy()
            meta_bin.update(dtype='int16', nodata=-9999,
                            compress='lzw', tiled=True, blockxsize=512, blockysize=512)
            meta_cls = meta_bin.copy()

            with rasterio.open(bin_path, 'w', **meta_bin) as bds, \
                 rasterio.open(cls_path, 'w', **meta_cls) as cds:
                for r0 in range(0, src.height, STRIP_ROWS):
                    r1  = min(r0 + STRIP_ROWS, src.height)
                    win = Window(0, r0, src.width, r1 - r0)
                    prob = src.read(1, window=win)
                    nd   = prob == NODATA

                    binary = np.where(nd, -9999, (prob >= thresh).astype(np.int16))
                    cls    = np.full_like(prob, -9999, dtype=np.int16)
                    v      = ~nd
                    cls[v & (prob <  qb[0])]                        = 1
                    cls[v & (prob >= qb[0]) & (prob < qb[1])]       = 2
                    cls[v & (prob >= qb[1]) & (prob < qb[2])]       = 3
                    cls[v & (prob >= qb[2]) & (prob < qb[3])]       = 4
                    cls[v & (prob >= qb[3])]                         = 5
                    bds.write(binary, 1, window=win)
                    cds.write(cls,    1, window=win)

        print(f"    {model:<20} thresh={thresh:.4f}  → binary + 5-class saved")

    # Validation metrics
    val_results = []
    cm_data     = {}
    print(f"\n  {'Model':<20} {'AUC':>7} {'F1':>7} {'CSI':>7} {'Kappa':>7} {'Sens':>7} {'Spec':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for model in MODELS:
        probs = prob_maps[model][all_valid]
        t     = optimal_thresh[model]
        m     = full_metrics(y_val, probs, t, model)
        val_results.append(m)
        cm_data[model] = confusion_matrix(y_val, (probs >= t).astype(int))
        print(f"  {model:<20} {m['AUC']:>7.4f} {m['F1']:>7.4f} {m['CSI']:>7.4f} "
              f"{m['Kappa']:>7.4f} {m['Sensitivity']:>7.4f} {m['Specificity']:>7.4f}")

    val_df = pd.DataFrame(val_results).sort_values('AUC', ascending=False)
    val_df.to_csv(os.path.join(vald_dir, 'validation_metrics.csv'), index=False)

    # Add to cross-size accumulator
    for m in val_results:
        m['SampleSize'] = size
        all_size_val.append(m)

    # ── Plots ─────────────────────────────────────────────────────────────────
    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0,1],[0,1],'k--', lw=1.2, alpha=0.5, label='Random chance')
    for model in MODELS:
        probs = prob_maps[model][all_valid]
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc_val = sk_auc(fpr, tpr)
        t = optimal_thresh[model]
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp_ = confusion_matrix(y_val, pred).ravel()
        ax.plot(fpr, tpr, lw=2, color=COLORS[model],
                label=f'{model}  (AUC={auc_val:.4f}, t={t:.3f})')
        ax.plot(1-tn/(tn+fp+1e-10), tp_/(tp_+fn+1e-10), 'o', color=COLORS[model], ms=9)
    ax.set_xlabel('1 − Specificity', fontsize=12); ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title(f'ROC Curves [{size}]', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(vald_dir, 'roc_validation.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Confusion matrices (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, model in enumerate(MODELS):
        ax  = axes.flat[i]
        cm  = cm_data[model]
        pct = cm.astype(float) / cm.sum() * 100
        ann = np.array([[f'{cm[r,c]}\n({pct[r,c]:.1f}%)' for c in range(2)] for r in range(2)])
        sns.heatmap(cm, annot=ann, fmt='', cmap='Blues', ax=ax,
                    xticklabels=['No Flood','Flood'], yticklabels=['No Flood','Flood'],
                    linewidths=1, linecolor='white', cbar=False)
        m_ = val_df[val_df.Model==model].iloc[0]
        ax.set_title(f"{model}\nAUC={m_['AUC']:.4f}  CSI={m_['CSI']:.4f}  t={optimal_thresh[model]:.3f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=9); ax.set_ylabel('Actual', fontsize=9)
    axes.flat[-1].set_visible(False)
    plt.suptitle(f'Confusion Matrices [{size}] — CSI-Optimal Thresholds', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(vald_dir, 'confusion_matrices.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # CSI-threshold curves (2×3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    threshs = np.linspace(0.01, 0.99, 200)
    for i, model in enumerate(MODELS):
        ax    = axes.flat[i]
        probs = prob_maps[model][all_valid]
        csi_v = []
        for t in threshs:
            pred = (probs >= t).astype(int)
            tn, fp, fn, tp_ = confusion_matrix(y_val, pred).ravel()
            csi_v.append(tp_/(tp_+fp+fn+1e-10))
        opt_t = optimal_thresh[model]
        ax.plot(threshs, csi_v, color=COLORS[model], lw=2)
        ax.axvline(opt_t, color='red', ls='--', lw=1.5, label=f't={opt_t:.3f}')
        ax.plot(opt_t, max(csi_v), 'ro', ms=8)
        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.set_xlabel('Threshold', fontsize=10); ax.set_ylabel('CSI', fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    axes.flat[-1].set_visible(False)
    plt.suptitle(f'CSI vs Threshold [{size}]', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(vald_dir, 'csi_threshold_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Metrics comparison bar
    met_cols = ['AUC','F1','CSI','Sensitivity','Specificity','Kappa']
    x  = np.arange(len(met_cols)); w = 0.15
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (_, row) in enumerate(val_df.iterrows()):
        vals   = [row[m] for m in met_cols]
        offset = (i - 2) * w
        bars   = ax.bar(x + offset, vals, w, label=row['Model'],
                        color=COLORS.get(row['Model'], 'gray'), alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(met_cols, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison [{size}]', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(vald_dir, 'metrics_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved → {vald_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART B — SPATIAL CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"PART B — Spatial CV  [{size}]")
    print(f"{'─'*65}")

    # Load features list
    feat_file = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
    with open(feat_file) as f:
        FEATURES = [l.strip() for l in f if l.strip()]

    # Combine train + test for this size
    tr_path = os.path.join(SAMPLES_DIR, f'samples_train{suffix}.csv')
    df_all  = pd.concat([pd.read_csv(tr_path), pd.read_csv(te_path)], ignore_index=True)
    print(f"Combined samples: {len(df_all):,}  flood={int((df_all.label==1).sum()):,}")

    # Spatial block assignment
    df_all['bx']   = ((df_all.x_3794 - df_all.x_3794.min()) // BLOCK_SIZE).astype(int)
    df_all['by']   = ((df_all.y_3794 - df_all.y_3794.min()) // BLOCK_SIZE).astype(int)
    df_all['fold'] = ((df_all['bx'] * 2 + df_all['by']) % N_FOLDS) + 1
    df_all.to_csv(os.path.join(scv_dir, 'fold_assignments.csv'), index=False)

    print(f"Spatial grid: {df_all.bx.max()+1} cols × {df_all.by.max()+1} rows  |  "
          f"Block size: {BLOCK_SIZE/1000:.0f} km")
    print(f"\n  {'Fold':<5} {'N':>8} {'Flood':>8} {'No-flood':>10} {'Flood%':>8}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for fld in range(1, N_FOLDS+1):
        sub = df_all[df_all.fold==fld]
        fl  = (sub.label==1).sum(); nfl = (sub.label==0).sum()
        print(f"  {fld:<5} {len(sub):>8,} {fl:>8,} {nfl:>10,} {fl/len(sub)*100:>7.1f}%")

    # Fold map
    FOLD_COLORS = {1:'#2196F3', 2:'#FF5722', 3:'#4CAF50', 4:'#9C27B0', 5:'#FF9800'}
    fig, ax = plt.subplots(figsize=(10, 7))
    for fld in range(1, N_FOLDS+1):
        sub = df_all[df_all.fold==fld]
        ax.scatter(sub.x_3794/1000, sub.y_3794/1000, s=0.3, alpha=0.4,
                   color=FOLD_COLORS[fld], rasterized=True)
    patches = [mpatches.Patch(color=FOLD_COLORS[f], label=f'Fold {f}') for f in range(1,6)]
    ax.legend(handles=patches, fontsize=11, loc='lower right')
    ax.set_xlabel('Easting (km)', fontsize=11); ax.set_ylabel('Northing (km)', fontsize=11)
    ax.set_title(f'Spatial Fold Assignment [{size}] — 50 km blocks', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3); ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(scv_dir, 'fold_map.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # CV loop
    cv_results = []
    t_cv_start = time.time()

    for fold in range(1, N_FOLDS+1):
        print(f"\n  ── Fold {fold}/{N_FOLDS} ──")
        df_tr_f = df_all[df_all.fold != fold].copy()
        df_te_f = df_all[df_all.fold == fold].copy()

        X_tr = df_tr_f[FEATURES].values.astype(np.float32)
        y_tr = df_tr_f['label'].values.astype(np.float32)
        X_te = df_te_f[FEATURES].values.astype(np.float32)
        y_te = df_te_f['label'].values.astype(np.float32)
        print(f"     Train: {len(X_tr):,}  Test: {len(X_te):,}  flood%={y_te.mean()*100:.1f}%")

        # Internal val split for early stopping
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=RANDOM_SEED)
        scale_pos = float((y_tr2==0).sum()) / max(float((y_tr2==1).sum()), 1)

        sc = StandardScaler()
        X_tr2_sc = sc.fit_transform(X_tr2).astype(np.float32)
        X_val_sc  = sc.transform(X_val).astype(np.float32)
        X_te_sc   = sc.transform(X_te).astype(np.float32)

        # RF
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                                     max_features='sqrt', class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_SEED)
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_te)[:, 1]
        t_rf, _ = find_csi_threshold(y_te, p_rf, n=100)
        cv_results.append(cv_metrics(y_te, p_rf, 'RF', fold, t_rf))
        print(f"     RF      AUC={cv_results[-1]['AUC']:.4f}  CSI={cv_results[-1]['CSI']:.4f}  [{time.time()-t0:.0f}s]")
        del rf; gc.collect()

        # XGBoost
        t0 = time.time()
        xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    scale_pos_weight=scale_pos, tree_method='hist', device='cuda',
                                    early_stopping_rounds=20, eval_metric='auc',
                                    random_state=RANDOM_SEED, verbosity=0)
        xgb_m.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
        p_xgb = xgb_m.predict_proba(X_te)[:, 1]
        t_xgb, _ = find_csi_threshold(y_te, p_xgb, n=100)
        cv_results.append(cv_metrics(y_te, p_xgb, 'XGBoost', fold, t_xgb))
        print(f"     XGBoost AUC={cv_results[-1]['AUC']:.4f}  CSI={cv_results[-1]['CSI']:.4f}  [{time.time()-t0:.0f}s]")
        del xgb_m; gc.collect()

        # CatBoost
        t0 = time.time()
        cat_m = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8,
                                    auto_class_weights='Balanced', task_type='GPU', devices='0',
                                    early_stopping_rounds=20, eval_metric='AUC',
                                    random_seed=RANDOM_SEED, verbose=False)
        cat_m.fit(X_tr2, y_tr2, eval_set=(X_val, y_val), use_best_model=True)
        p_cat = cat_m.predict_proba(X_te)[:, 1]
        t_cat, _ = find_csi_threshold(y_te, p_cat, n=100)
        cv_results.append(cv_metrics(y_te, p_cat, 'CatBoost', fold, t_cat))
        print(f"     CatBoost AUC={cv_results[-1]['AUC']:.4f}  CSI={cv_results[-1]['CSI']:.4f}  [{time.time()-t0:.0f}s]")
        del cat_m; gc.collect()

        # DNN
        t0 = time.time()
        pos_w    = torch.tensor([scale_pos], dtype=torch.float32)
        crit     = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(DEVICE))
        crit_cpu = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        Xtr_t = torch.tensor(X_tr2_sc); ytr_t = torch.tensor(y_tr2).unsqueeze(1)
        Xva_t = torch.tensor(X_val_sc); yva_t = torch.tensor(y_val).unsqueeze(1)
        Xte_t = torch.tensor(X_te_sc)

        dnn = FloodDNN(len(FEATURES)).to(DEVICE)
        opt = optim.Adam(dnn.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DNN_EPOCHS, eta_min=1e-5)
        dl  = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=DNN_BATCH, shuffle=True)

        best_auc_dnn, best_ep, best_state = 0.0, 0, None
        for ep in range(1, DNN_EPOCHS+1):
            dnn.train()
            for Xb, yb in dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(dnn(Xb), yb); loss.backward()
                nn.utils.clip_grad_norm_(dnn.parameters(), 1.0); opt.step()
            dnn.eval()
            with torch.no_grad():
                lv = torch.cat([dnn(Xva_t[i:i+4096].to(DEVICE)).cpu()
                                for i in range(0, len(Xva_t), 4096)])
                pv = torch.sigmoid(lv).numpy().flatten()
            va_auc = roc_auc_score(y_val, pv); sch.step()
            if va_auc > best_auc_dnn:
                best_auc_dnn = va_auc; best_ep = ep
                best_state = {k: v.cpu().clone() for k, v in dnn.state_dict().items()}
            if ep - best_ep >= DNN_PATIENCE:
                break

        dnn.load_state_dict(best_state); dnn.eval()
        with torch.no_grad():
            lt = torch.cat([dnn(Xte_t[i:i+4096].to(DEVICE)).cpu()
                            for i in range(0, len(Xte_t), 4096)])
        p_dnn = torch.sigmoid(lt).numpy().flatten()
        t_dnn, _ = find_csi_threshold(y_te, p_dnn, n=100)
        cv_results.append(cv_metrics(y_te, p_dnn, 'DNN', fold, t_dnn))
        print(f"     DNN     AUC={cv_results[-1]['AUC']:.4f}  CSI={cv_results[-1]['CSI']:.4f}  "
              f"[{time.time()-t0:.0f}s, best_ep={best_ep}]")
        del dnn; gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    print(f"\n  Total CV time [{size}]: {(time.time()-t_cv_start)/60:.1f} min")

    # Aggregate CV results
    cv_df     = pd.DataFrame(cv_results)
    agg_cols  = ['AUC','CSI','F1','Kappa','Sensitivity','Specificity']
    mean_rows = []
    for model in ['RF','XGBoost','CatBoost','DNN']:
        sub = cv_df[cv_df.Model==model]
        row = {'Model': model}
        for c in agg_cols:
            row[f'{c}_mean'] = round(sub[c].mean(), 4)
            row[f'{c}_std']  = round(sub[c].std(),  4)
        row['SampleSize'] = size
        mean_rows.append(row)
        all_size_cv.append(row)

    mean_df = pd.DataFrame(mean_rows)
    cv_df.to_csv(os.path.join(scv_dir, 'per_fold_metrics.csv'),  index=False)
    mean_df.to_csv(os.path.join(scv_dir, 'mean_metrics.csv'), index=False)

    print(f"\n  Spatial CV mean ± std [{size}]:")
    print(f"  {'Model':<12} {'AUC':>14} {'CSI':>14} {'F1':>14} {'Kappa':>14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    for _, r in mean_df.iterrows():
        print(f"  {r.Model:<12} "
              f"{r.AUC_mean:.4f}±{r.AUC_std:.4f}  "
              f"{r.CSI_mean:.4f}±{r.CSI_std:.4f}  "
              f"{r.F1_mean:.4f}±{r.F1_std:.4f}  "
              f"{r.Kappa_mean:.4f}±{r.Kappa_std:.4f}")

    # Comparison plot: random vs spatial CV
    TREE_MODELS = ['RF','XGBoost','CatBoost','DNN']
    met_show    = ['AUC','CSI','F1','Kappa']
    rand_df     = val_df[val_df.Model.isin(TREE_MODELS)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for ax, metric in zip(axes, met_show):
        x = np.arange(len(TREE_MODELS)); w = 0.35
        rand_vals = [rand_df[rand_df.Model==m][metric].values[0] for m in TREE_MODELS]
        cv_means  = [mean_df[mean_df.Model==m][f'{metric}_mean'].values[0] for m in TREE_MODELS]
        cv_stds   = [mean_df[mean_df.Model==m][f'{metric}_std'].values[0]  for m in TREE_MODELS]

        ax.bar(x - w/2, rand_vals, w, label='Random split',
               color=[COLORS[m] for m in TREE_MODELS], alpha=0.4, hatch='//', edgecolor='gray')
        ax.bar(x + w/2, cv_means, w, yerr=cv_stds, capsize=4, label='Spatial CV',
               color=[COLORS[m] for m in TREE_MODELS], alpha=0.9, edgecolor='black')
        for i, (rv, cv, ce) in enumerate(zip(rand_vals, cv_means, cv_stds)):
            ax.text(i-w/2, rv+0.003, f'{rv:.3f}', ha='center', va='bottom', fontsize=7, color='gray')
            ax.text(i+w/2, cv+ce+0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(TREE_MODELS, fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.12); ax.grid(axis='y', alpha=0.3)

    hand = [mpatches.Patch(facecolor='gray', alpha=0.4, hatch='//', label='Random split'),
            mpatches.Patch(facecolor='dimgray', alpha=0.9, label='Spatial CV (5-fold, 50 km)')]
    fig.legend(handles=hand, loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.01))
    plt.suptitle(f'Random Split vs Spatial CV [{size}] — Slovenia FSM',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(scv_dir, 'spatial_cv_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Per-fold performance
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, ['AUC','CSI','F1','Kappa']):
        for model in TREE_MODELS:
            sub   = cv_df[cv_df.Model==model]
            folds = sub['Fold'].values
            vals  = sub[metric].values
            ax.plot(folds, vals, 'o-', color=COLORS[model], lw=1.8, ms=7,
                    label=f"{model} (μ={vals.mean():.3f})")
            ax.axhline(vals.mean(), color=COLORS[model], ls=':', lw=1.2, alpha=0.6)
        ax.set_xlabel('Fold', fontsize=10); ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_xticks(range(1, N_FOLDS+1)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle(f'Per-Fold Performance [{size}]', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(scv_dir, 'per_fold_performance.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Spatial CV plots saved → {scv_dir}")

# =============================================================================
# CROSS-SIZE SUMMARY
# =============================================================================
print()
print("=" * 65)
print("CROSS-SIZE COMPARISON")
print("=" * 65)

val_all_df = pd.DataFrame(all_size_val)
val_all_df.to_csv(os.path.join(BASE, '08_Validation', 'multiscale_validation.csv'), index=False)

cv_all_df  = pd.DataFrame(all_size_cv)
cv_all_df.to_csv(os.path.join(BASE, '09_Analysis/spatial_cv', 'multiscale_cv_summary.csv'), index=False)

# Cross-size comparison plot (AUC per model across sizes)
TREE_MODELS = ['RF','XGBoost','CatBoost','DNN']
SIZE_LABELS  = SIZES
met_show     = ['AUC','CSI','F1','Kappa']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
style_map = {'RF': '-o', 'XGBoost': '-s', 'CatBoost': '-^', 'DNN': '-D'}

for ax, metric in zip(axes.flat, met_show):
    for model in TREE_MODELS:
        # Random split
        rand_vals = [val_all_df[(val_all_df.SampleSize==s) & (val_all_df.Model==model)][metric].values[0]
                     for s in SIZE_LABELS if len(val_all_df[(val_all_df.SampleSize==s) & (val_all_df.Model==model)]) > 0]
        ax.plot(SIZE_LABELS[:len(rand_vals)], rand_vals, style_map[model],
                color=COLORS[model], lw=2, ms=8, label=f'{model}', alpha=0.9)
        for i, v in enumerate(rand_vals):
            ax.annotate(f'{v:.4f}', (i, v), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=7)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Size', fontsize=10); ax.set_ylabel('Score', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('Effect of Sample Size on Validation Performance — Slovenia FSM (1x–10x)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(BASE, '08_Validation', 'multiscale_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\nCross-size comparison saved → ML_Results/validation/multiscale_comparison.png")
print(f"\nALL DONE")
print(f"  validation/{{size}}/   — metrics, plots, thresholds")
print(f"  classified_maps/{{size}}/ — binary + 5-class TIFs")
print(f"  spatial_cv/{{size}}/   — fold metrics, CV comparison")
