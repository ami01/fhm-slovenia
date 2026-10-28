"""
DNN Validation Merge — Flood Susceptibility Mapping (Slovenia)
=============================================================
Phase B of the former train_dnn.py. Merges the DNN row into
validation_metrics.csv and regenerates the 5-model comparison plots.

MUST run after stage 07 (threshold_classify_validate.py), which creates
validation_metrics.csv. Reads the hand-off files written by 05b_train_dnn.py.

Inputs
  08_Validation/validation_metrics.csv        from stage 07
  08_Validation/dnn_metrics.json              from stage 05b
  08_Validation/dnn_test_predictions.npz      from stage 05b
Outputs
  08_Validation/validation_metrics.csv        updated, now including DNN
  08_Validation/  comparison ROC / PR / bar plots across all 5 models
"""

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features'
FEATURES_FILE = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
SAMPLES_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/03_Samples'
MODELS_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/04_Models'
# Same as stage 07: 06b writes maps per sample size, so read the 10x set rather than
# the parent directory (which holds only a stale 1x DNN map from stage 05b).
# `import os as _os` because this config block runs before the main import block.
import os as _os
PROB_DIR      = _os.path.join(
    '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/06_Probability_Maps',
    _os.environ.get('MAP_SIZE', '10x'))
CLSF_DIR      = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/07_Classified_Maps'
VALD_DIR      = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/08_Validation'

EPOCHS        = 200
BATCH_SIZE    = 4096
LR            = 1e-3
PATIENCE      = 20          # early stopping patience on val AUC
DROPOUT       = 0.3
STRIP_ROWS    = 200
NODATA_OUT    = -9999.0

import os, warnings, gc, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import rasterio
from rasterio.windows import Window
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, f1_score, accuracy_score, cohen_kappa_score,
    average_precision_score, precision_recall_curve
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ── Load the hand-off from stage 05b ──────────────────────────────────────────
import json as _json

_metrics_path = os.path.join(VALD_DIR, 'dnn_metrics.json')
_preds_path   = os.path.join(VALD_DIR, 'dnn_test_predictions.npz')
for _p in (_metrics_path, _preds_path):
    if not os.path.exists(_p):
        raise SystemExit(
            f"Missing {_p}\n"
            "Run 05b_train_dnn.py first (it writes the hand-off files)."
        )
if not os.path.exists(os.path.join(VALD_DIR, 'validation_metrics.csv')):
    raise SystemExit(
        f"Missing {os.path.join(VALD_DIR, 'validation_metrics.csv')}\n"
        "Run stage 07 (07_threshold_classify_validate.py) first - it creates this file."
    )

with open(_metrics_path) as _fh:
    dnn_metrics = _json.load(_fh)
_z = np.load(_preds_path)
y_te, probs_test, best_thresh = _z['y_te'], _z['probs_test'], float(_z['best_thresh'])
print(f"Loaded DNN hand-off: AUC={dnn_metrics['AUC']}  threshold={best_thresh:.4f}")

# Reconstruct the intermediates the plotting section below expects. These were
# computed in phase A (05b) and are cheap to redo from the handoff rather than
# persisting four more arrays. The sweep MUST match 05b's exactly (np.linspace over
# 0.01..0.99 in 200 steps) or the CSI curve will not line up with the saved
# threshold. Derived, not guessed: see 05b_train_dnn.py.
thresholds = np.linspace(0.01, 0.99, 200)
csi_curve, best_csi = [], 0.0
for _t in thresholds:
    _tn, _fp, _fn, _tp = confusion_matrix(y_te, (probs_test >= _t).astype(int)).ravel()
    _c = _tp / (_tp + _fp + _fn + 1e-10)
    csi_curve.append(_c)
    best_csi = max(best_csi, _c)
y_pred = (probs_test >= best_thresh).astype(int)

_chk = float(np.max(csi_curve))
if abs(_chk - dnn_metrics['CSI']) > 5e-4:
    print(f"WARNING: reconstructed peak CSI {_chk:.4f} differs from the handoff's "
          f"{dnn_metrics['CSI']:.4f} - the sweep may no longer match 05b.")
else:
    print(f"Reconstructed CSI curve agrees with handoff (peak {_chk:.4f}).")

# ── Merge with existing validation CSV ───────────────────────────────────────
val_csv = os.path.join(VALD_DIR, 'validation_metrics.csv')
val_df  = pd.read_csv(val_csv)

# Stage 07 already evaluates all five models, DNN included, using the full-Slovenia
# 10x probability map. Appending the handoff row here would add a SECOND DNN row
# carrying 1x metrics, silently mixing sample sizes in one table. Only append when
# stage 07 did not produce a DNN row.
if 'DNN' in val_df['Model'].astype(str).values:
    print("\nvalidation_metrics.csv already contains a DNN row from stage 07 "
          "(evaluated on the 10x probability map).")
    print("Skipping the append - the stage 07 row supersedes the 05b handoff, which "
          "is 1x only.\nRegenerating the comparison plots from the existing table.\n")
else:
    val_df = pd.concat([val_df, pd.DataFrame([dnn_metrics])], ignore_index=True)
    val_df = val_df.sort_values('AUC', ascending=False)
    val_df.to_csv(val_csv, index=False)
    print(f"\nUpdated validation_metrics.csv  (now includes DNN)\n")

val_df = val_df.sort_values('AUC', ascending=False)

# ── Updated comparison plots ──────────────────────────────────────────────────
COLORS = {
    'RF': '#2196F3', 'XGBoost': '#FF5722',
    'CatBoost': '#4CAF50', 'Ensemble_mean': '#9C27B0', 'DNN': '#FF9800'
}
ALL_MODELS = list(val_df['Model'])

# ROC curves — all 5 models
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1],'k--', lw=1.2, alpha=0.5, label='Random chance')

# Load prob maps for all previous models + DNN test probs
prob_maps_all = {}
df_te_csv = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_test.csv'))
xs = df_te_csv['x_3794'].values; ys_coord = df_te_csv['y_3794'].values

for m in ['RF', 'XGBoost', 'CatBoost', 'Ensemble_mean']:
    path = os.path.join(PROB_DIR, f'FSM_{m}_probability.tif')
    with rasterio.open(path) as ds:
        tf = ds.transform
        cs = np.floor((xs - tf.c) / tf.a).astype(int)
        rs = np.floor((ys_coord - tf.f) / tf.e).astype(int)
        rs = np.clip(rs, 0, ds.height-1); cs = np.clip(cs, 0, ds.width-1)
        arr = ds.read(1)
        prbs = arr[rs, cs].astype(np.float64)
        prbs[prbs == -9999.0] = np.nan
    prob_maps_all[m] = prbs

prob_maps_all['DNN'] = probs_test
y_true_all = df_te_csv['label'].values.astype(int)

for model in ['XGBoost', 'CatBoost', 'Ensemble_mean', 'RF', 'DNN']:
    prbs = prob_maps_all[model]
    valid = ~np.isnan(prbs)
    fpr, tpr, _ = roc_curve(y_true_all[valid], prbs[valid])
    auc_val = auc(fpr, tpr)
    row = val_df[val_df['Model']==model].iloc[0]
    t = row['Threshold']; sens = row['Sensitivity']; spec = row['Specificity']
    ls = '--' if model == 'DNN' else '-'
    lw = 2.8 if model == 'DNN' else 2.0
    ax.plot(fpr, tpr, lw=lw, ls=ls, color=COLORS[model],
            label=f'{model}  (AUC={auc_val:.4f})')
    ax.plot(1-spec, sens, 'o', color=COLORS[model], ms=8, zorder=5)

ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves — All Models (DNN highlighted)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'roc_all_models.png'), dpi=200, bbox_inches='tight')
plt.close()

# Metrics bar comparison — all 5 models
met_cols = ['AUC','F1','CSI','Sensitivity','Specificity','Kappa']
x = np.arange(len(met_cols))
n_mod = len(val_df)
w = 0.15
fig, ax = plt.subplots(figsize=(15, 6))
for i, (_, row) in enumerate(val_df.iterrows()):
    vals   = [row[m] for m in met_cols]
    offset = (i - n_mod/2 + 0.5) * w
    bars   = ax.bar(x + offset, vals, w, label=row['Model'],
                    color=COLORS.get(row['Model'], 'gray'),
                    alpha=0.9 if row['Model']=='DNN' else 0.78,
                    edgecolor='black' if row['Model']=='DNN' else 'white',
                    linewidth=1.2 if row['Model']=='DNN' else 0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
ax.set_xticks(x); ax.set_xticklabels(met_cols, fontsize=11)
ax.set_ylim(0, 1.18); ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison — All Models including DNN', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'metrics_all_models.png'), dpi=200, bbox_inches='tight')
plt.close()

# Confusion matrix — DNN only
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_te, y_pred)
cm_pct = cm.astype(float) / cm.sum() * 100
annot  = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)])
sns.heatmap(cm, annot=annot, fmt='', cmap='Oranges', ax=ax,
            xticklabels=['No Flood','Flood'], yticklabels=['No Flood','Flood'],
            linewidths=1, linecolor='white', cbar=False)
ax.set_title(f"DNN  |  AUC={dnn_metrics['AUC']}  CSI={dnn_metrics['CSI']}  t={best_thresh:.3f}",
             fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=10); ax.set_ylabel('Actual', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'confusion_matrix_DNN.png'), dpi=200, bbox_inches='tight')
plt.close()

# CSI-threshold curve — DNN
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, csi_curve, color='#FF9800', lw=2.2)
ax.axvline(best_thresh, color='red', ls='--', lw=1.5, label=f'Optimal t={best_thresh:.3f}')
ax.plot(best_thresh, best_csi, 'ro', ms=9)
ax.set_xlabel('Threshold'); ax.set_ylabel('CSI')
ax.set_title('DNN — CSI vs Threshold', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'csi_threshold_dnn.png'), dpi=200, bbox_inches='tight')
plt.close()

print("Plots saved: roc_all_models.png, metrics_all_models.png,")
print("             confusion_matrix_DNN.png, csi_threshold_dnn.png")

# ── Final comparison table ────────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL COMPARISON — ALL MODELS")
print("=" * 65)
disp_cols = ['Model','Threshold','AUC','CSI','F1','Sensitivity','Specificity','Kappa']
print(val_df[disp_cols].to_string(index=False))
print()
best_row = val_df.iloc[0]
print(f"Best overall: {best_row['Model']}  "
      f"(AUC={best_row['AUC']}  CSI={best_row['CSI']}  Kappa={best_row['Kappa']})")
print(f"\nAll outputs in: {VALD_DIR}")
