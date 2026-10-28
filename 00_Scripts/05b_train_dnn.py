"""
Deep Neural Network — Flood Susceptibility Mapping (Slovenia)
=============================================================
Architecture : MLP with 4 hidden layers, BatchNorm, Dropout
               128 → 64 → 32 → 16 → 1 (sigmoid)
Training     : GPU | Adam + CosineAnnealingLR | BCEWithLogitsLoss
               with pos_weight for class imbalance
               Early stopping on validation AUC (patience=20)
Outputs
  ML_Results/models/model_dnn.pth       trained weights
  ML_Results/models/scaler_dnn.pkl      StandardScaler fitted on train
  ML_Results/probability_maps/FSM_DNN_probability.tif
  ML_Results/classified_maps/FSM_DNN_binary.tif
  ML_Results/classified_maps/FSM_DNN_5class.tif
  08_Validation/dnn_metrics.json          metrics handed to stage 08a
  08_Validation/dnn_test_predictions.npz  test-set preds handed to stage 08a

NOTE: this script stops after computing DNN metrics. The merge into
validation_metrics.csv and the 5-model comparison plots require stage 07 and
live in 08a_dnn_merge_validation.py.
"""

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features'
FEATURES_FILE = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
SAMPLES_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/03_Samples'
MODELS_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/04_Models'
PROB_DIR      = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/06_Probability_Maps'
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

# ── Reproducibility ───────────────────────────────────────────────────────────
# The original script set no seed, so two identical runs differed by ~0.0005 AUC
# and ~0.004 Kappa - the same magnitude as the effects being measured. Seed
# everything so DNN results are comparable across runs like the tree models
# (which already use RANDOM_SEED=42). Override with SEED=<n> in the environment
# to sample the run-to-run distribution deliberately.
import random as _random
RANDOM_SEED = int(os.environ.get('SEED', 42))
_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Random seed: {RANDOM_SEED}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})\n")

# ── Feature list ──────────────────────────────────────────────────────────────
with open(FEATURES_FILE) as f:
    FEATURES = [l.strip() for l in f if l.strip()]
print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}\n")

# ── Load train / test samples ─────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading samples")
print("=" * 65)
df_tr = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_train.csv'))
df_te = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_test.csv'))

X_tr_raw = df_tr[FEATURES].values.astype(np.float32)
y_tr     = df_tr['label'].values.astype(np.float32)
X_te_raw = df_te[FEATURES].values.astype(np.float32)
y_te     = df_te['label'].values.astype(np.float32)

print(f"Train : {len(X_tr_raw):,}  (flood={int(y_tr.sum()):,}  no-flood={int((y_tr==0).sum()):,})")
print(f"Test  : {len(X_te_raw):,}  (flood={int(y_te.sum()):,}  no-flood={int((y_te==0).sum()):,})")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float32)
X_te   = scaler.transform(X_te_raw).astype(np.float32)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_dnn.pkl'))
print("StandardScaler fitted and saved.\n")

# ── DNN architecture ──────────────────────────────────────────────────────────
class FloodDNN(nn.Module):
    def __init__(self, input_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Block 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            # Block 4
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Output
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# ── Training ──────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 2 — Training DNN")
print("=" * 65)

pos_weight    = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()],
                              dtype=torch.float32)
criterion     = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
criterion_cpu = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # for val loss on CPU

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=True)

model     = FloodDNN(len(FEATURES)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_auc    = 0.0
best_epoch  = 0
best_state  = None
history     = {'epoch':[], 'train_loss':[], 'val_auc':[], 'val_loss':[]}

print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} {'Val AUC':>9} {'LR':>10}")
print('-' * 55)

for epoch in range(1, EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * len(Xb)
    train_loss = running_loss / len(X_tr_t)

    # ── Validate ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits_val = []
        for i in range(0, len(X_te_t), 8192):
            logits_val.append(model(X_te_t[i:i+8192].to(DEVICE)).cpu())
        logits_val = torch.cat(logits_val)
        val_loss   = criterion_cpu(logits_val, y_te_t).item()
        probs_val  = torch.sigmoid(logits_val).numpy().flatten()

    val_auc = roc_auc_score(y_te, probs_val)
    scheduler.step()
    cur_lr  = optimizer.param_groups[0]['lr']

    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)

    if epoch % 10 == 0 or epoch == 1:
        marker = ' ← best' if val_auc > best_auc else ''
        print(f"{epoch:>6} {train_loss:>11.5f} {val_loss:>10.5f} {val_auc:>9.5f} {cur_lr:>10.2e}{marker}")

    if val_auc > best_auc:
        best_auc   = val_auc
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch - best_epoch >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (best epoch={best_epoch}, best AUC={best_auc:.5f})")
        break

model.load_state_dict(best_state)
torch.save(best_state, os.path.join(MODELS_DIR, 'model_dnn.pth'))
print(f"\nBest model (epoch {best_epoch}, AUC={best_auc:.5f}) saved: model_dnn.pth")

# ── Training history plot ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(history['epoch'], history['train_loss'], label='Train loss', color='#2196F3', lw=2)
ax1.plot(history['epoch'], history['val_loss'],   label='Val loss',   color='#FF5722', lw=2)
ax1.axvline(best_epoch, color='green', ls='--', lw=1.5, label=f'Best epoch {best_epoch}')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE Loss')
ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(history['epoch'], history['val_auc'], color='#4CAF50', lw=2, label='Val AUC')
ax2.axvline(best_epoch, color='green', ls='--', lw=1.5, label=f'Best AUC={best_auc:.4f}')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC')
ax2.set_title('Validation AUC'); ax2.legend(); ax2.grid(alpha=0.3)
plt.suptitle('DNN Training History — Slovenia Flood Susceptibility', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'dnn_training_history.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Training history plot saved.\n")

# ── Final test-set probabilities ──────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits_all = []
    for i in range(0, len(X_te_t), 8192):
        logits_all.append(model(X_te_t[i:i+8192].to(DEVICE)).cpu())
probs_test = torch.sigmoid(torch.cat(logits_all)).numpy().flatten()

# ── Probability map — strip-based inference ────────────────────────────────────
print("=" * 65)
print("STEP 3 — Generating DNN probability map")
print("=" * 65)

feat_paths = {}
for fname in os.listdir(ALIGNED_DIR):
    stem = os.path.splitext(fname)[0]
    if stem in FEATURES:
        feat_paths[stem] = os.path.join(ALIGNED_DIR, fname)

ref_path = feat_paths[FEATURES[0]]
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_h, ref_w = ref.height, ref.width

out_meta = ref_meta.copy()
out_meta.update(dtype='float32', nodata=NODATA_OUT, count=1,
                compress='lzw', tiled=True, blockxsize=512, blockysize=512)

feat_ds  = {stem: rasterio.open(path) for stem, path in feat_paths.items()}
feat_nds = {stem: ds.nodata for stem, ds in feat_ds.items()}

prob_map_path = os.path.join(PROB_DIR, 'FSM_DNN_probability.tif')
n_strips = (ref_h + STRIP_ROWS - 1) // STRIP_ROWS
t0 = time.time()

print(f"Strips: {n_strips} × {STRIP_ROWS} rows  |  Grid: {ref_w}×{ref_h}")
print(f"{'Strip':>6}  {'Rows':<14}  {'Valid px':>12}  {'Mean prob':>10}  {'Time':>7}")
print('-' * 58)

model.eval()
with rasterio.open(prob_map_path, 'w', **out_meta) as dst:
    for s in range(n_strips):
        row_start = s * STRIP_ROWS
        row_end   = min(row_start + STRIP_ROWS, ref_h)
        n_rows    = row_end - row_start
        win       = Window(0, row_start, ref_w, n_rows)
        n_pix     = n_rows * ref_w

        X_strip = np.full((n_pix, len(FEATURES)), np.nan, dtype=np.float32)
        for fi, stem in enumerate(FEATURES):
            band = feat_ds[stem].read(1, window=win).flatten().astype(np.float32)
            nd   = feat_nds[stem]
            if nd is not None:
                mask = np.isnan(band) if (isinstance(nd, float) and np.isnan(nd)) else (band == nd)
                band[mask] = np.nan
            X_strip[:, fi] = band

        valid    = ~np.any(np.isnan(X_strip), axis=1)
        out_strip = np.full(n_pix, NODATA_OUT, dtype=np.float32)

        if valid.sum() > 0:
            Xv_sc = scaler.transform(X_strip[valid]).astype(np.float32)
            preds  = []
            with torch.no_grad():
                for i in range(0, len(Xv_sc), 8192):
                    batch = torch.tensor(Xv_sc[i:i+8192]).to(DEVICE)
                    p = torch.sigmoid(model(batch)).cpu().numpy().flatten()
                    preds.append(p)
            out_strip[valid] = np.concatenate(preds).astype(np.float32)

        dst.write(out_strip.reshape(n_rows, ref_w), 1, window=win)

        elapsed = time.time() - t0
        mean_p  = out_strip[valid].mean() if valid.sum() > 0 else float('nan')
        print(f"{s+1:>6}  {row_start}–{row_end:<8}  {valid.sum():>12,}  {mean_p:>10.4f}  {elapsed:>6.1f}s",
              flush=True)
        gc.collect()

for ds in feat_ds.values(): ds.close()
print(f"\nDNN probability map saved: {prob_map_path}")

# ── Classify DNN map ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 4 — Threshold optimisation & classification")
print("=" * 65)

# CSI-optimal threshold on test set
thresholds = np.linspace(0.01, 0.99, 200)
best_csi, best_thresh = 0.0, 0.5
csi_curve = []
for t in thresholds:
    pred = (probs_test >= t).astype(int)
    tn, fp, fn, tp_ = confusion_matrix(y_te, pred).ravel()
    c = tp_ / (tp_ + fp + fn + 1e-10)
    csi_curve.append(c)
    if c > best_csi:
        best_csi, best_thresh = c, t

print(f"CSI-optimal threshold : {best_thresh:.4f}  (CSI = {best_csi:.4f})")

# Quantile breaks from downsampled probability map
with rasterio.open(prob_map_path) as ds:
    arr_ds = ds.read(1, out_shape=(ds.height//10, ds.width//10),
                     resampling=rasterio.enums.Resampling.nearest)
valid_pix = arr_ds[arr_ds != NODATA_OUT].flatten()
qb = np.percentile(valid_pix, [20, 40, 60, 80])
print(f"5-class quantile breaks: {qb[0]:.4f} | {qb[1]:.4f} | {qb[2]:.4f} | {qb[3]:.4f}")

def get_out_meta_from(prob_path, dtype, nodata):
    with rasterio.open(prob_path) as s:
        meta = s.meta.copy()
    meta.update(dtype=dtype, nodata=nodata,
                compress='lzw', tiled=True, blockxsize=512, blockysize=512)
    return meta

bin_path = os.path.join(CLSF_DIR, 'FSM_DNN_binary.tif')
cls_path = os.path.join(CLSF_DIR, 'FSM_DNN_5class.tif')

with rasterio.open(prob_map_path) as src, \
     rasterio.open(bin_path, 'w', **get_out_meta_from(prob_map_path, 'int16', -9999)) as bds, \
     rasterio.open(cls_path, 'w', **get_out_meta_from(prob_map_path, 'int16', -9999)) as cds:
    for row_start in range(0, src.height, STRIP_ROWS):
        row_end = min(row_start + STRIP_ROWS, src.height)
        win  = Window(0, row_start, src.width, row_end - row_start)
        prob = src.read(1, window=win)
        nd   = prob == NODATA_OUT
        binary = np.where(nd, -9999, (prob >= best_thresh).astype(np.int16))
        cls    = np.full_like(prob, -9999, dtype=np.int16)
        v = ~nd
        cls[v & (prob <  qb[0])]                    = 1
        cls[v & (prob >= qb[0]) & (prob < qb[1])]   = 2
        cls[v & (prob >= qb[1]) & (prob < qb[2])]   = 3
        cls[v & (prob >= qb[2]) & (prob < qb[3])]   = 4
        cls[v & (prob >= qb[3])]                     = 5
        bds.write(binary, 1, window=win)
        cds.write(cls,    1, window=win)

print("Binary and 5-class maps saved.")

# ── Full validation ───────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 5 — Validation metrics")
print("=" * 65)

y_pred = (probs_test >= best_thresh).astype(int)
tn, fp, fn, tp_ = confusion_matrix(y_te, y_pred).ravel()
dnn_metrics = {
    'Model'      : 'DNN',
    'Threshold'  : round(best_thresh, 4),
    'AUC'        : round(roc_auc_score(y_te, probs_test), 4),
    'CSI'        : round(tp_ / (tp_ + fp + fn + 1e-10),   4),
    'F1'         : round(f1_score(y_te, y_pred),           4),
    'Accuracy'   : round(accuracy_score(y_te, y_pred),     4),
    'Kappa'      : round(cohen_kappa_score(y_te, y_pred),  4),
    'Sensitivity': round(tp_ / (tp_ + fn + 1e-10),        4),
    'Specificity': round(tn  / (tn  + fp + 1e-10),        4),
    'Precision'  : round(tp_ / (tp_ + fp + 1e-10),        4),
    'TP': int(tp_), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
}

print(f"\n  DNN Results:")
print(f"  AUC={dnn_metrics['AUC']}  CSI={dnn_metrics['CSI']}  F1={dnn_metrics['F1']}")
print(f"  Sensitivity={dnn_metrics['Sensitivity']}  Specificity={dnn_metrics['Specificity']}  Kappa={dnn_metrics['Kappa']}")


# ── Hand-off to stage 08a ─────────────────────────────────────────────────────
# The merge into validation_metrics.csv and the 5-model comparison plots need
# stage 07 to have run, so they live in 08a_dnn_merge_validation.py. Persist what
# that script needs.
import json as _json
os.makedirs(VALD_DIR, exist_ok=True)
with open(os.path.join(VALD_DIR, 'dnn_metrics.json'), 'w') as _fh:
    _json.dump(dnn_metrics, _fh, indent=2)
np.savez(os.path.join(VALD_DIR, 'dnn_test_predictions.npz'),
         y_te=np.asarray(y_te), probs_test=np.asarray(probs_test),
         best_thresh=np.float64(best_thresh))
print("\nSaved dnn_metrics.json and dnn_test_predictions.npz for stage 08a.")
print("Run 08a_dnn_merge_validation.py after stage 07 to merge the comparison.")
