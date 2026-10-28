"""
Slovenia Flood Susceptibility — Multi-Sample ML Training Pipeline
=================================================================
Trains RF | XGBoost (GPU) | CatBoost (GPU) | DNN on 5 sample sizes:
  1x  : 50 000 per class  (uses existing samples_train/test.csv)
  2x  : 100 000 per class
  3x  : 150 000 per class
  4x  : 200 000 per class
  10x : 500 000 per class

Skip logic:
  - All 4 models trained (XGB + DNN exist) → load metrics.csv, skip entirely.
  - Tree models exist but DNN missing → reload tree models for inference,
    train DNN only, then update metrics.csv and plots.
  - Nothing trained → train all 4 from scratch.

Outputs saved in ML_Results/
  samples/
    samples_train.csv / samples_test.csv          ← 1x (pre-existing)
    samples_train_{N}x.csv / samples_test_{N}x.csv
  models_{size}/  model_rf.pkl | model_xgb.json | model_cat.cbm
                  model_dnn.pth | scaler.pkl
  results_{size}/ metrics.csv | roc_curves.png | pr_curves.png
                  feature_importance_*.png | model_comparison.png
                  dnn_training_history.png
  results_comparison/
    sample_size_comparison.csv
    sample_size_comparison.png
    auc_heatmap.png
"""

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features'
TARGET_RASTER = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features/Flooded_NoFlood.tif'
FEATURES_FILE = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
OUTPUT_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'

BASE_N_SAMPLES = 50_000   # per class for 1x; multiplied for larger sizes
TEST_SIZE      = 0.20
RANDOM_SEED    = 42

# DNN hyperparameters
DNN_EPOCHS      = 200
DNN_BATCH_SIZE  = 4096
DNN_LR          = 1e-3
DNN_PATIENCE    = 20
DNN_DROPOUT     = 0.3

SAMPLE_CONFIGS = [
    {'tag': '1x',  'multiplier': 1,  'use_existing': True},
    {'tag': '2x',  'multiplier': 2,  'use_existing': False},
    {'tag': '3x',  'multiplier': 3,  'use_existing': False},
    {'tag': '4x',  'multiplier': 4,  'use_existing': False},
    {'tag': '10x', 'multiplier': 10, 'use_existing': False},
]

# ── Setup ─────────────────────────────────────────────────────────────────────
import os, warnings, gc, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, accuracy_score,
    cohen_kappa_score
)
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {DEVICE}  "
      f"({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})\n")

np.random.seed(RANDOM_SEED)

# The tree models already used RANDOM_SEED, but torch was left unseeded, making the
# DNN column of the sample-size comparison non-reproducible. Seed it to match.
import random as _random
_random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SAMPLES_DIR = os.path.join(OUTPUT_DIR, '03_Samples')
COMP_DIR    = os.path.join(OUTPUT_DIR, '05_Results/comparison')
for d in [SAMPLES_DIR, COMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Load feature list ─────────────────────────────────────────────────────────
with open(FEATURES_FILE) as f:
    FEATURES = [l.strip() for l in f if l.strip()]
print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}\n")

# ── Pre-load raster data (flood index arrays) ─────────────────────────────────
print("=" * 65)
print("Pre-loading target raster ...")
print("=" * 65)

def is_nodata(arr, nd):
    if nd is None:
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nd, float) and np.isnan(nd):
        return np.isnan(arr)
    return arr == nd

with rasterio.open(TARGET_RASTER) as ds:
    label  = ds.read(1).astype(np.float32)
    lbl_nd = ds.nodata
    tf     = ds.transform

lbl_mask   = is_nodata(label, lbl_nd)
flood_idx  = np.argwhere((label == 1) & ~lbl_mask)
nflood_idx = np.argwhere((label == 0) & ~lbl_mask)
print(f"Available flood pixels   : {len(flood_idx):,}")
print(f"Available no-flood pixels: {len(nflood_idx):,}\n")

# ── Helper: sample pixels from rasters ───────────────────────────────────────
def sample_pixels(n_per_class, seed):
    rng = np.random.RandomState(seed)
    n_fl  = min(n_per_class, len(flood_idx))
    n_nfl = min(n_per_class, len(nflood_idx))

    sel_fl  = flood_idx [rng.choice(len(flood_idx),  n_fl,  replace=False)]
    sel_nfl = nflood_idx[rng.choice(len(nflood_idx), n_nfl, replace=False)]
    all_idx = np.vstack([sel_fl, sel_nfl])
    rows, cols = all_idx[:, 0], all_idx[:, 1]

    xs = tf.c + (cols + 0.5) * tf.a
    ys = tf.f + (rows + 0.5) * tf.e

    labels_arr = np.array([1] * n_fl + [0] * n_nfl)

    feat_data = {'x_3794': xs, 'y_3794': ys}
    for fname in sorted(os.listdir(ALIGNED_DIR)):
        stem = os.path.splitext(fname)[0]
        if stem not in FEATURES:
            continue
        with rasterio.open(os.path.join(ALIGNED_DIR, fname)) as ds:
            arr = ds.read(1)
            nd  = ds.nodata
        vals = arr[rows, cols].astype(np.float64)
        vals[is_nodata(vals, nd)] = np.nan
        feat_data[stem] = vals

    df = pd.DataFrame(feat_data)
    df.insert(0, 'label', labels_arr)
    n_before = len(df)
    df.dropna(subset=FEATURES, inplace=True)
    print(f"  Sampled {n_fl+n_nfl:,} → kept {len(df):,} after NaN drop (dropped {n_before-len(df):,})")
    return df

# ── Helper: save samples as CSV + SHP ─────────────────────────────────────────
def save_samples(df_tr, df_te, size_tag):
    suffix = '' if size_tag == '1x' else f'_{size_tag}'
    for df_s, split in [(df_tr, 'train'), (df_te, 'test')]:
        csv_path = os.path.join(SAMPLES_DIR, f'samples_{split}{suffix}.csv')
        df_s.to_csv(csv_path, index=False)
        coords = df_s[['x_3794', 'y_3794']].values
        gdf = gpd.GeoDataFrame(
            df_s,
            geometry=[Point(x, y) for x, y in coords],
            crs='EPSG:3794'
        )
        shp_path = os.path.join(SAMPLES_DIR, f'samples_{split}{suffix}.shp')
        gdf.to_file(shp_path)
        print(f"  {split:<6} {size_tag}: {len(df_s):,} samples → {csv_path}")

# ── Helper: compute metrics ───────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, name, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model'      : name,
        'AUC'        : round(roc_auc_score(y_true, y_prob),           4),
        'Avg_Prec'   : round(average_precision_score(y_true, y_prob), 4),
        'F1'         : round(f1_score(y_true, y_pred),                4),
        'Accuracy'   : round(accuracy_score(y_true, y_pred),          4),
        'Kappa'      : round(cohen_kappa_score(y_true, y_pred),       4),
        'CSI'        : round(tp / (tp + fp + fn + 1e-10),             4),
        'Sensitivity': round(tp / (tp + fn + 1e-10),                  4),
        'Specificity': round(tn / (tn + fp + 1e-10),                  4),
        'Precision'  : round(tp / (tp + fp + 1e-10),                  4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }

# ── Helper: feature importance plot ──────────────────────────────────────────
def plot_importance(imp_arr, feat_names, model_name, save_path):
    idx    = np.argsort(imp_arr)[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(feat_names)))
    bars    = ax.barh(
        [feat_names[i] for i in idx[::-1]],
        imp_arr[idx[::-1]],
        color=colors
    )
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Feature Importance — {model_name}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, imp_arr[idx[::-1]]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

# ── DNN architecture ─────────────────────────────────────────────────────────
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

def train_dnn(X_tr_sc, y_tr, X_te_sc, y_te, mdl_dir, res_dir, size_tag):
    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    crit_cpu   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr,    dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_te_sc, dtype=torch.float32)
    y_te_t = torch.tensor(y_te,    dtype=torch.float32).unsqueeze(1)

    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=DNN_BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)

    model     = FloodDNN(X_tr_sc.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=DNN_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DNN_EPOCHS, eta_min=1e-5)

    best_auc, best_epoch, best_state = 0.0, 0, None
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_auc': []}

    print(f"\n[{size_tag}] DNN training  (device={DEVICE})")
    print(f"  {'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} {'Val AUC':>9}")
    print('  ' + '-' * 44)

    for epoch in range(1, DNN_EPOCHS + 1):
        model.train()
        running = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(Xb)
        train_loss = running / len(X_tr_t)

        model.eval()
        with torch.no_grad():
            chunks = [model(X_te_t[i:i+8192].to(DEVICE)).cpu()
                      for i in range(0, len(X_te_t), 8192)]
            logits_val = torch.cat(chunks)
            val_loss   = crit_cpu(logits_val, y_te_t).item()
            probs_val  = torch.sigmoid(logits_val).numpy().flatten()

        val_auc = roc_auc_score(y_te, probs_val)
        scheduler.step()

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        if epoch % 10 == 0 or epoch == 1:
            marker = ' ← best' if val_auc > best_auc else ''
            print(f"  {epoch:>6} {train_loss:>11.5f} {val_loss:>10.5f} {val_auc:>9.5f}{marker}")

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= DNN_PATIENCE:
            print(f"  Early stopping at epoch {epoch}  (best={best_epoch}, AUC={best_auc:.5f})")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(mdl_dir, 'model_dnn.pth'))
    print(f"  Saved: model_dnn.pth  (best epoch={best_epoch})")

    # Training history plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(history['epoch'], history['train_loss'], label='Train loss', color='#2196F3', lw=2)
    ax1.plot(history['epoch'], history['val_loss'],   label='Val loss',   color='#FF5722', lw=2)
    ax1.axvline(best_epoch, color='green', ls='--', lw=1.5, label=f'Best epoch {best_epoch}')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['epoch'], history['val_auc'], color='#4CAF50', lw=2)
    ax2.axvline(best_epoch, color='green', ls='--', lw=1.5, label=f'Best AUC={best_auc:.4f}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(f'DNN Training History [{size_tag}]', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(res_dir, 'dnn_training_history.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Final inference on test set
    model.eval()
    with torch.no_grad():
        chunks = [model(X_te_t[i:i+8192].to(DEVICE)).cpu()
                  for i in range(0, len(X_te_t), 8192)]
    probs_test = torch.sigmoid(torch.cat(chunks)).numpy().flatten()
    del model; gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return probs_test

# ── Helper: ROC + PR + comparison plots ──────────────────────────────────────
COLORS = {'Random Forest': '#2196F3', 'XGBoost': '#FF5722',
          'CatBoost': '#4CAF50', 'DNN': '#FF9800'}
LINES  = {'Random Forest': '-', 'XGBoost': '--', 'CatBoost': '-.', 'DNN': ':'}

def make_plots(predictions, y_te, results_dir, size_tag, metrics_df):
    # ROC
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random chance')
    for name, probs in predictions.items():
        fpr, tpr, _ = roc_curve(y_te, probs)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, color=COLORS[name], ls=LINES[name],
                label=f'{name}  (AUC = {auc_val:.4f})')
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(f'ROC Curves — Slovenia FSM  [{size_tag}]', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # PR
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, probs in predictions.items():
        prec, rec, _ = precision_recall_curve(y_te, probs)
        ap = average_precision_score(y_te, probs)
        ax.plot(rec, prec, lw=2.2, color=COLORS[name], ls=LINES[name],
                label=f'{name}  (AP = {ap:.4f})')
    ax.axhline(y_te.mean(), color='k', ls='--', lw=1.2, alpha=0.5,
               label=f'Baseline (prevalence={y_te.mean():.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'PR Curves — Slovenia FSM  [{size_tag}]', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'pr_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Comparison bar chart
    met_plot = ['AUC', 'F1', 'CSI', 'Sensitivity', 'Specificity', 'Kappa']
    x        = np.arange(len(met_plot))
    n_models = len(metrics_df)
    width    = 0.8 / n_models
    fig, ax  = plt.subplots(figsize=(13, 6))
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        vals   = [row[m] for m in met_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=row['Model'],
                        color=COLORS.get(row['Model'], 'gray'), alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=45)
    ax.set_xticks(x)
    ax.set_xticklabels(met_plot, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison — Slovenia FSM  [{size_tag}]',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved in {results_dir}")

# ── Main loop: iterate over sample sizes ──────────────────────────────────────
all_results = []   # accumulate across sample sizes for final comparison

for cfg in SAMPLE_CONFIGS:
    size_tag   = cfg['tag']
    multiplier = cfg['multiplier']
    n_per_class = BASE_N_SAMPLES * multiplier
    suffix     = '' if size_tag == '1x' else f'_{size_tag}'

    print()
    print("=" * 65)
    print(f"SAMPLE SIZE: {size_tag}  ({n_per_class:,} per class)")
    print("=" * 65)

    # ── Directories for this size ─────────────────────────────────────────
    mdl_dir = os.path.join(OUTPUT_DIR, '04_Models', size_tag)
    res_dir = os.path.join(OUTPUT_DIR, '05_Results', size_tag)
    for d in [mdl_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Skip / partial-skip logic ─────────────────────────────────────────
    metrics_csv  = os.path.join(res_dir, 'metrics.csv')
    xgb_sentinel = os.path.join(mdl_dir, 'model_xgb.json')
    dnn_sentinel = os.path.join(mdl_dir, 'model_dnn.pth')

    # Full skip: all 4 models including DNN are done
    if os.path.exists(xgb_sentinel) and os.path.exists(dnn_sentinel) and os.path.exists(metrics_csv):
        print(f"  All models (incl. DNN) already exist — loading {metrics_csv}")
        existing = pd.read_csv(metrics_csv)
        sfx  = '' if size_tag == '1x' else f'_{size_tag}'
        n_tr = len(pd.read_csv(os.path.join(SAMPLES_DIR, f'samples_train{sfx}.csv')))
        n_te = len(pd.read_csv(os.path.join(SAMPLES_DIR, f'samples_test{sfx}.csv')))
        for _, row in existing.iterrows():
            m = row.to_dict(); m['SampleSize'] = size_tag
            m['N_train'] = n_tr; m['N_test'] = n_te
            all_results.append(m)
        print(f"  Loaded {len(existing)} results  (train={n_tr:,}  test={n_te:,})")
        print(f"[{size_tag}] Skipped (already done).\n")
        continue

    # Partial skip: tree models exist, DNN missing — reload trees for inference
    tree_models_exist = os.path.exists(xgb_sentinel)

    # ── Load or generate samples ──────────────────────────────────────────
    if tree_models_exist:
        # Tree models already trained — just load the CSV samples
        print(f"  Tree models exist — loading samples from CSV for DNN training ...")
        sfx   = '' if size_tag == '1x' else f'_{size_tag}'
        df_tr = pd.read_csv(os.path.join(SAMPLES_DIR, f'samples_train{sfx}.csv'))
        df_te = pd.read_csv(os.path.join(SAMPLES_DIR, f'samples_test{sfx}.csv'))
        print(f"  Train: {len(df_tr):,}  |  Test: {len(df_te):,}")
    elif cfg['use_existing']:
        print(f"Loading pre-existing samples ...")
        df_tr = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_train.csv'))
        df_te = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_test.csv'))
        print(f"  Train: {len(df_tr):,}  |  Test: {len(df_te):,}")
    else:
        print(f"Sampling {n_per_class:,} per class from rasters ...")
        t0  = time.time()
        df  = sample_pixels(n_per_class, seed=RANDOM_SEED)

        X   = df[FEATURES].values.astype(np.float32)
        y   = df['label'].values.astype(np.float32)
        coords = df[['x_3794', 'y_3794']].values

        X_tr_arr, X_te_arr, y_tr_arr, y_te_arr, idx_tr, idx_te = train_test_split(
            X, y, np.arange(len(df)),
            test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
        )

        def _build_df(X_arr, y_arr, coord_arr):
            out = pd.DataFrame(X_arr, columns=FEATURES)
            out.insert(0, 'label', y_arr.astype(int))
            out.insert(1, 'x_3794', coord_arr[:, 0])
            out.insert(2, 'y_3794', coord_arr[:, 1])
            return out

        df_tr = _build_df(X_tr_arr, y_tr_arr, coords[idx_tr])
        df_te = _build_df(X_te_arr, y_te_arr, coords[idx_te])
        print(f"  Train: {len(df_tr):,}  |  Test: {len(df_te):,}  ({time.time()-t0:.1f}s)")

        print("Saving samples ...")
        save_samples(df_tr, df_te, size_tag)

    # ── Prepare arrays ────────────────────────────────────────────────────
    X_tr = df_tr[FEATURES].values.astype(np.float32)
    y_tr = df_tr['label'].values.astype(np.float32)
    X_te = df_te[FEATURES].values.astype(np.float32)
    y_te = df_te['label'].values.astype(np.float32)

    print(f"  Train flood: {int(y_tr.sum()):,}  no-flood: {int((y_tr==0).sum()):,}")
    print(f"  Test  flood: {int(y_te.sum()):,}  no-flood: {int((y_te==0).sum()):,}")

    # ── Scale ─────────────────────────────────────────────────────────────
    if tree_models_exist:
        # Load the scaler fitted during tree model training
        scaler = joblib.load(os.path.join(mdl_dir, 'scaler.pkl'))
    else:
        scaler = StandardScaler()
        scaler.fit(X_tr)
        joblib.dump(scaler, os.path.join(mdl_dir, 'scaler.pkl'))
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # ── Train models ──────────────────────────────────────────────────────
    predictions  = {}
    feature_imps = {}

    if tree_models_exist:
        # Load existing tree models and run inference only
        print(f"\n[{size_tag}] Loading tree models for inference ...")
        rf_loaded = joblib.load(os.path.join(mdl_dir, 'model_rf.pkl'))
        predictions['Random Forest'] = rf_loaded.predict_proba(X_te)[:, 1]
        feature_imps['Random Forest'] = rf_loaded.feature_importances_
        del rf_loaded; gc.collect()

        xgb_loaded = xgb.XGBClassifier()
        xgb_loaded.load_model(os.path.join(mdl_dir, 'model_xgb.json'))
        predictions['XGBoost'] = xgb_loaded.predict_proba(X_te)[:, 1]
        feature_imps['XGBoost'] = xgb_loaded.feature_importances_
        del xgb_loaded; gc.collect()

        cat_loaded = CatBoostClassifier()
        cat_loaded.load_model(os.path.join(mdl_dir, 'model_cat.cbm'))
        predictions['CatBoost'] = cat_loaded.predict_proba(X_te)[:, 1]
        feature_imps['CatBoost'] = cat_loaded.get_feature_importance()
        del cat_loaded; gc.collect()
        print(f"  Tree model inference done.")

    else:
        # Random Forest
        print(f"\n[{size_tag}] Random Forest ...")
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced',
            n_jobs=-1, random_state=RANDOM_SEED
        )
        rf.fit(X_tr, y_tr)
        predictions['Random Forest'] = rf.predict_proba(X_te)[:, 1]
        feature_imps['Random Forest'] = rf.feature_importances_
        joblib.dump(rf, os.path.join(mdl_dir, 'model_rf.pkl'))
        print(f"  Saved: model_rf.pkl")
        del rf; gc.collect()

        # XGBoost
        print(f"[{size_tag}] XGBoost (GPU hist) ...")
        scale_pos = float((y_tr == 0).sum()) / float((y_tr == 1).sum())
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            tree_method='hist', device='cuda', eval_metric='auc',
            early_stopping_rounds=30, random_state=RANDOM_SEED, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        predictions['XGBoost'] = xgb_model.predict_proba(X_te)[:, 1]
        feature_imps['XGBoost'] = xgb_model.feature_importances_
        xgb_model.save_model(os.path.join(mdl_dir, 'model_xgb.json'))
        print(f"  Best round: {xgb_model.best_iteration}  Saved: model_xgb.json")
        del xgb_model; gc.collect()

        # CatBoost
        print(f"[{size_tag}] CatBoost (GPU) ...")
        cat_model = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
            auto_class_weights='Balanced', task_type='GPU', devices='0',
            eval_metric='AUC', early_stopping_rounds=30,
            random_seed=RANDOM_SEED, verbose=False
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_te, y_te), use_best_model=True)
        predictions['CatBoost'] = cat_model.predict_proba(X_te)[:, 1]
        feature_imps['CatBoost'] = cat_model.get_feature_importance()
        cat_model.save_model(os.path.join(mdl_dir, 'model_cat.cbm'))
        print(f"  Best round: {cat_model.best_iteration_}  Saved: model_cat.cbm")
        del cat_model; gc.collect()

    # DNN (always trained fresh if not already saved)
    dnn_probs = train_dnn(X_tr_sc, y_tr, X_te_sc, y_te, mdl_dir, res_dir, size_tag)
    predictions['DNN'] = dnn_probs

    # ── Metrics ───────────────────────────────────────────────────────────
    results = []
    print(f"\n[{size_tag}] Metrics:")
    for name, probs in predictions.items():
        m = compute_metrics(y_te, probs, name)
        results.append(m)
        print(f"  {name}: AUC={m['AUC']}  F1={m['F1']}  CSI={m['CSI']}  "
              f"Sens={m['Sensitivity']}  Spec={m['Specificity']}  Kappa={m['Kappa']}")

    metrics_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    metrics_df.to_csv(os.path.join(res_dir, 'metrics.csv'), index=False)

    # ── Feature importance plots ───────────────────────────────────────────
    for name, imp in feature_imps.items():
        fname = f"feature_importance_{name.lower().replace(' ', '_')}.png"
        plot_importance(imp, FEATURES, f'{name} [{size_tag}]',
                        os.path.join(res_dir, fname))

    # ── ROC / PR / comparison plots ────────────────────────────────────────
    make_plots(predictions, y_te, res_dir, size_tag, metrics_df)

    # ── Accumulate for cross-size comparison ──────────────────────────────
    for m in results:
        m['SampleSize'] = size_tag
        m['N_train']    = len(y_tr)
        m['N_test']     = len(y_te)
        all_results.append(m)

    print(f"\n[{size_tag}] Done.\n")

# ── Cross-size comparison ─────────────────────────────────────────────────────
print("=" * 65)
print("CROSS-SAMPLE-SIZE COMPARISON")
print("=" * 65)

comp_df = pd.DataFrame(all_results)
comp_df.to_csv(os.path.join(COMP_DIR, 'sample_size_comparison.csv'), index=False)
print("Saved: sample_size_comparison.csv")
print(comp_df[['SampleSize', 'N_train', 'Model', 'AUC', 'F1', 'CSI',
               'Sensitivity', 'Specificity', 'Kappa']].to_string(index=False))

# ── Multi-panel comparison plot ───────────────────────────────────────────────
SIZES   = ['1x', '2x', '3x', '4x', '10x']
MODELS  = ['Random Forest', 'XGBoost', 'CatBoost', 'DNN']
METRICS = ['AUC', 'F1', 'CSI', 'Sensitivity', 'Specificity', 'Kappa']

size_colors  = {'1x': '#1565C0', '2x': '#2E7D32', '3x': '#E65100', '4x': '#6A1B9A', '10x': '#AD1457'}
model_styles = {'Random Forest': '-o', 'XGBoost': '-s', 'CatBoost': '-^', 'DNN': '-D'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax_i, metric in enumerate(METRICS):
    ax = axes[ax_i]
    for model in MODELS:
        sub = comp_df[comp_df['Model'] == model].set_index('SampleSize')
        vals = [sub.loc[s, metric] if s in sub.index else np.nan for s in SIZES]
        ax.plot(SIZES, vals, model_styles[model], lw=2, ms=8,
                label=model, color=COLORS[model])
        for x_pos, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(f'{v:.4f}', (x_pos, v),
                            textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=7.5)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Size', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=max(0, comp_df[metric].min() - 0.02))

plt.suptitle('Effect of Sample Size on Model Performance — Slovenia FSM',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(COMP_DIR, 'sample_size_comparison.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("Saved: sample_size_comparison.png")

# ── Heatmap: AUC by model × sample size ──────────────────────────────────────
pivot = comp_df.pivot_table(values='AUC', index='Model', columns='SampleSize')[SIZES]
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(pivot.values, aspect='auto', cmap='YlGn', vmin=0.95, vmax=1.0)
ax.set_xticks(range(len(SIZES)));    ax.set_xticklabels(SIZES, fontsize=11)
ax.set_yticks(range(len(pivot)));    ax.set_yticklabels(pivot.index, fontsize=11)
ax.set_xlabel('Sample Size', fontsize=12); ax.set_ylabel('Model', fontsize=12)
ax.set_title('AUC — Model × Sample Size', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='AUC')
for i in range(len(pivot)):
    for j in range(len(SIZES)):
        ax.text(j, i, f'{pivot.values[i, j]:.4f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='black' if pivot.values[i, j] < 0.98 else 'white')
plt.tight_layout()
fig.savefig(os.path.join(COMP_DIR, 'auc_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: auc_heatmap.png")

print()
print("=" * 65)
print("ALL DONE")
print("=" * 65)
print(f"Samples   → {SAMPLES_DIR}")
print(f"Comparison→ {COMP_DIR}")
for tag in ['1x', '2x', '3x', '4x', '10x']:
    print(f"  models_{tag}/   results_{tag}/")
