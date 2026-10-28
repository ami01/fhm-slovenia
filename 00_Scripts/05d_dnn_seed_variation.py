"""DNN seed-variation study — quantify run-to-run noise and test LULC significance.

A single DNN run cannot tell you whether a metric difference is real, because the
network is stochastic. This trains the DNN N times with different seeds and reports
mean +- sd per metric, so any claimed improvement can be judged against the spread.

Run AFTER 05b (needs 03_Samples and 02_Collinearity in place). Cheap relative to
full training because it reuses the existing sample CSVs and skips map generation.

Outputs -> 05_Results/dnn_seed_variation.csv
           05_Results/dnn_seed_variation.png
"""
import os, json, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
SAMPLES_DIR = os.path.join(ROOT, '03_Samples')
FEATURES_FILE = os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')
RESULTS_DIR = os.path.join(ROOT, '05_Results')

N_SEEDS = int(os.environ.get('N_SEEDS', 10))
EPOCHS = 200
BATCH_SIZE = 4096
LR = 1e-3
PATIENCE = 20
DROPOUT = 0.3

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix, f1_score,
                             accuracy_score, cohen_kappa_score)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = [l.strip() for l in open(FEATURES_FILE) if l.strip()]
df_tr = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_train.csv'))
df_te = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_test.csv'))
X_tr, y_tr = df_tr[FEATURES].values.astype(np.float32), df_tr['label'].values.astype(np.float32)
X_te, y_te = df_te[FEATURES].values.astype(np.float32), df_te['label'].values.astype(np.float32)
print(f"{len(FEATURES)} features | train {len(X_tr):,} | test {len(X_te):,} | {N_SEEDS} seeds\n")


class FloodDNN(nn.Module):
    def __init__(self, d, dropout=DROPOUT):
        super().__init__()
        layers, prev = [], d
        for h in (128, 64, 32, 16):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    sc = StandardScaler().fit(X_tr)
    Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.float32))
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, drop_last=False)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)

    model = FloodDNN(len(FEATURES)).to(DEVICE)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte_t)).cpu().numpy()
        auc = roc_auc_score(y_te, p)
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).cpu().numpy()

    # CSI-optimal threshold, matching 05b
    best_t, best_csi = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        yp = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
        csi = tp / (tp + fp + fn + 1e-10)
        if csi > best_csi:
            best_csi, best_t = csi, t

    yp = (probs >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
    return {
        'seed': seed, 'threshold': round(float(best_t), 4),
        'AUC': roc_auc_score(y_te, probs),
        'CSI': tp / (tp + fp + fn + 1e-10),
        'F1': f1_score(y_te, yp),
        'Accuracy': accuracy_score(y_te, yp),
        'Kappa': cohen_kappa_score(y_te, yp),
        'Sensitivity': tp / (tp + fn + 1e-10),
        'Specificity': tn / (tn + fp + 1e-10),
        'epochs_run': ep + 1,
    }


rows = []
for i, seed in enumerate(range(42, 42 + N_SEEDS), 1):
    r = run(seed)
    rows.append(r)
    print(f"  [{i}/{N_SEEDS}] seed={seed:3d}  AUC={r['AUC']:.4f}  CSI={r['CSI']:.4f}  "
          f"F1={r['F1']:.4f}  Kappa={r['Kappa']:.4f}  ({r['epochs_run']} epochs)")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, 'dnn_seed_variation.csv'), index=False)

METRICS = ['AUC', 'CSI', 'F1', 'Kappa', 'Sensitivity', 'Specificity']
print("\n" + "=" * 72)
print(f"DNN run-to-run variation over {N_SEEDS} seeds")
print("=" * 72)
print(f"{'metric':<14}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}{'range':>10}")
summary = []
for m in METRICS:
    v = df[m]
    print(f"{m:<14}{v.mean():10.4f}{v.std():10.4f}{v.min():10.4f}{v.max():10.4f}{v.max()-v.min():10.4f}")
    summary.append({'metric': m, 'mean': round(v.mean(), 4), 'sd': round(v.std(), 4),
                    'min': round(v.min(), 4), 'max': round(v.max(), 4),
                    'range': round(v.max() - v.min(), 4)})
pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, 'dnn_seed_variation_summary.csv'), index=False)

# Baseline comparison: is the LULC effect larger than the noise?
BASELINE_19 = {'AUC': 0.9783, 'CSI': 0.8849, 'F1': 0.9389, 'Kappa': 0.8746}
print("\n" + "=" * 72)
print("LULC effect vs run-to-run noise (19-feature baseline was a single run)")
print("=" * 72)
print(f"{'metric':<12}{'baseline':>10}{'mean(20f)':>12}{'delta':>10}{'sd':>10}{'|d|/sd':>9}  verdict")
for m, b in BASELINE_19.items():
    v = df[m]; d = v.mean() - b; ratio = abs(d) / v.std() if v.std() > 0 else float('inf')
    verdict = 'within noise' if ratio < 2 else 'exceeds noise'
    print(f"{m:<12}{b:10.4f}{v.mean():12.4f}{d:+10.4f}{v.std():10.4f}{ratio:9.2f}  {verdict}")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, m in zip(axes.ravel(), METRICS):
    ax.plot(df['seed'], df[m], 'o-', color='#FF9800', lw=1.5, ms=5)
    ax.axhline(df[m].mean(), color='#2196F3', ls='--', lw=1.2,
               label=f"mean {df[m].mean():.4f}")
    ax.fill_between(df['seed'], df[m].mean() - df[m].std(), df[m].mean() + df[m].std(),
                    color='#2196F3', alpha=0.15, label=f"+-1 sd ({df[m].std():.4f})")
    if m in BASELINE_19:
        ax.axhline(BASELINE_19[m], color='#D32F2F', ls=':', lw=1.4, label='19-feat baseline')
    ax.set_title(m, fontweight='bold'); ax.set_xlabel('seed'); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle(f'DNN run-to-run variation over {N_SEEDS} seeds (20 features incl. LULC)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'dnn_seed_variation.png'), dpi=200, bbox_inches='tight')
print(f"\nSaved dnn_seed_variation.csv / _summary.csv / .png -> {RESULTS_DIR}")
