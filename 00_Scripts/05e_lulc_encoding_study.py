"""LULC encoding study — does the ordinal coding explain the DNN degradation?

LULC helps the tree models (+0.0006 AUC) but measurably HURTS the DNN
(CSI -0.0046, ~4 sd). The hypothesis is the encoding, not the data: codes 1..7 are
nominal, but a dense layer reads them as a continuous magnitude where bare_open (7)
is "seven times" arable (1). Trees can split around that; an MLP cannot.

Trains the DNN across N seeds under four feature sets and compares distributions:

  baseline19  19 features, no LULC                        (the control)
  ordinal20   19 + LULC as an integer code 1..7           (current pipeline)
  onehot      19 + 6 binary indicator columns             (nominal, no false order)
  fractions   19 + 6 continuous areal fractions           (sub-cell composition,
                                                           from raba_frac_202606.tif)

bare_open is the dropped reference category in both onehot and fractions, so no
variant is perfectly collinear. Everything is derived from the existing sample CSVs
plus the fraction raster - no new rasters and no retraining of the tree models.

Outputs -> 05_Results/lulc_encoding_study.csv         per-seed, per-variant
           05_Results/lulc_encoding_summary.csv       mean/sd per variant
           05_Results/lulc_encoding_study.png
"""
import os, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
SAMPLES_DIR = os.path.join(ROOT, '03_Samples')
FEATURES_FILE = os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')
RESULTS_DIR = os.path.join(ROOT, '05_Results')
FRAC_TIF = '/home/arora/Documents/FSM_Slovenia/For_Analysis/raba_frac_202606.tif'

N_SEEDS = int(os.environ.get('N_SEEDS', 10))
EPOCHS = int(os.environ.get('EPOCHS', 200))     # lower it for a smoke test
BATCH_SIZE, LR, PATIENCE, DROPOUT = 4096, 1e-3, 20, 0.3
GPU_MEM_FRACTION = float(os.environ.get('GPU_MEM_FRACTION', 0.15))

CLASSES = ['arable', 'permanent_crop', 'grassland', 'woody',
           'built_up', 'water_wetland', 'bare_open']
REFERENCE = 'bare_open'          # dropped category (smallest, 1.5% of area)

import random
import numpy as np
import pandas as pd
import rasterio
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
if DEVICE.type == 'cuda':
    # 06b may be running concurrently on the same 4 GB card. Cap this process so a
    # runaway allocation here cannot OOM a multi-hour inference job.
    torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION)
    print(f"GPU capped at {GPU_MEM_FRACTION:.0%} of device memory")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_FEATURES = [l.strip() for l in open(FEATURES_FILE) if l.strip()]
BASE19 = [f for f in ALL_FEATURES if f != 'LULC']
assert len(BASE19) == 19, f"expected 19 base features, got {len(BASE19)}"

df_tr = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_train.csv'))
df_te = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_test.csv'))
print(f"train {len(df_tr):,} | test {len(df_te):,} | base features {len(BASE19)}")


def sample_fractions(dfs):
    """Read the 7 areal-fraction bands at each sample's coordinates.

    rasterio's ds.sample() is unusably slow here: the raster is tiled and deflate
    compressed, so 100k scattered points decompress a 256x256 block each. Reading a
    whole band (1.03 GB) and fancy-indexing it is ~50x faster; bands are read one at
    a time so peak memory stays near a single band rather than all seven.
    """
    with rasterio.open(FRAC_TIF) as ds:
        names = list(ds.descriptions)
        inv = ~ds.transform
        rc = []
        for df in dfs:
            cols, rows = inv * (df['x_3794'].values, df['y_3794'].values)
            rows = np.clip(rows.astype(np.int64), 0, ds.height - 1)
            cols = np.clip(cols.astype(np.int64), 0, ds.width - 1)
            rc.append((rows, cols))

        out = [np.empty((len(df), len(CLASSES)), dtype=np.float32) for df in dfs]
        for j, c in enumerate(CLASSES):
            band = ds.read(names.index(f'frac_{c}') + 1)
            for i, (rows, cols) in enumerate(rc):
                out[i][:, j] = band[rows, cols]
            del band
            print(f"  read frac_{c}")

    cols_out = [f'frac_{c}' for c in CLASSES]
    return [pd.DataFrame(np.nan_to_num(v, nan=0.0), columns=cols_out, index=df.index)
            for v, df in zip(out, dfs)]


print("sampling fraction raster at sample coordinates ...")
frac_tr, frac_te = sample_fractions([df_tr, df_te])
print(f"  mean coverage train={frac_tr.sum(axis=1).mean():.4f} "
      f"test={frac_te.sum(axis=1).mean():.4f}")


def onehot(df):
    out = {}
    for i, c in enumerate(CLASSES, 1):
        if c == REFERENCE:
            continue
        out[f'lulc_{c}'] = (df['LULC'].values == i).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


keep_frac = [f'frac_{c}' for c in CLASSES if c != REFERENCE]

VARIANTS = {
    'baseline19': (df_tr[BASE19], df_te[BASE19]),
    'ordinal20':  (df_tr[BASE19 + ['LULC']], df_te[BASE19 + ['LULC']]),
    'onehot':     (pd.concat([df_tr[BASE19], onehot(df_tr)], axis=1),
                   pd.concat([df_te[BASE19], onehot(df_te)], axis=1)),
    'fractions':  (pd.concat([df_tr[BASE19], frac_tr[keep_frac]], axis=1),
                   pd.concat([df_te[BASE19], frac_te[keep_frac]], axis=1)),
}
y_tr = df_tr['label'].values.astype(np.float32)
y_te = df_te['label'].values.astype(np.float32)
for k, (a, _) in VARIANTS.items():
    print(f"  {k:<12} {a.shape[1]} columns")


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


def run(Xtr_df, Xte_df, seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    sc = StandardScaler().fit(Xtr_df.values)
    Xtr = sc.transform(Xtr_df.values).astype(np.float32)
    Xte = sc.transform(Xte_df.values).astype(np.float32)

    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(y_tr)),
                    batch_size=BATCH_SIZE, shuffle=True, generator=g)
    Xte_t = torch.tensor(Xte).to(DEVICE)

    model = FloodDNN(Xtr.shape[1]).to(DEVICE)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte_t)).cpu().numpy()
        auc = roc_auc_score(y_te, p)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte_t)).cpu().numpy()

    best_t, best_csi = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        tn, fp, fn, tp = confusion_matrix(y_te, (probs >= t).astype(int)).ravel()
        csi = tp / (tp + fp + fn + 1e-10)
        if csi > best_csi:
            best_csi, best_t = csi, t
    yp = (probs >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
    del model, Xte_t
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return {'AUC': roc_auc_score(y_te, probs), 'CSI': tp / (tp + fp + fn + 1e-10),
            'F1': f1_score(y_te, yp), 'Accuracy': accuracy_score(y_te, yp),
            'Kappa': cohen_kappa_score(y_te, yp),
            'Sensitivity': tp / (tp + fn + 1e-10), 'Specificity': tn / (tn + fp + 1e-10),
            'threshold': round(float(best_t), 4), 'epochs': ep + 1}


rows = []
for vname, (Xtr_df, Xte_df) in VARIANTS.items():
    print(f"\n=== {vname} ({Xtr_df.shape[1]} features) ===")
    for i, seed in enumerate(range(42, 42 + N_SEEDS), 1):
        r = run(Xtr_df, Xte_df, seed)
        r.update(variant=vname, seed=seed, n_features=Xtr_df.shape[1])
        rows.append(r)
        print(f"  [{i}/{N_SEEDS}] seed={seed:3d}  AUC={r['AUC']:.4f}  "
              f"CSI={r['CSI']:.4f}  F1={r['F1']:.4f}  Kappa={r['Kappa']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'lulc_encoding_study.csv'),
                              index=False)   # checkpoint after each variant

df = pd.DataFrame(rows)
METRICS = ['AUC', 'CSI', 'F1', 'Kappa']
summary = df.groupby('variant')[METRICS].agg(['mean', 'std'])
summary.to_csv(os.path.join(RESULTS_DIR, 'lulc_encoding_summary.csv'))

print("\n" + "=" * 78)
print(f"LULC encoding comparison — DNN over {N_SEEDS} seeds per variant")
print("=" * 78)
print(f"{'variant':<13}" + "".join(f"{m:>17}" for m in METRICS))
for v in VARIANTS:
    s = df[df.variant == v]
    print(f"{v:<13}" + "".join(f"{s[m].mean():>11.4f}+-{s[m].std():.4f}" for m in METRICS))

base = df[df.variant == 'baseline19']
print("\n" + "=" * 78)
print("Each encoding vs baseline19 (no LULC).  |d|/sd_pooled >= 2 => real effect")
print("=" * 78)
print(f"{'variant':<13}{'metric':<10}{'delta':>10}{'sd_pooled':>11}{'|d|/sd':>9}  verdict")
for v in ['ordinal20', 'onehot', 'fractions']:
    s = df[df.variant == v]
    for m in METRICS:
        d = s[m].mean() - base[m].mean()
        sd = np.sqrt((s[m].std() ** 2 + base[m].std() ** 2) / 2)
        if not np.isfinite(sd) or sd <= 0:
            # <2 seeds: no spread estimate, so no significance claim is possible
            print(f"{v:<13}{m:<10}{d:+10.4f}{'n/a':>11}{'n/a':>9}  need >=2 seeds")
            continue
        r = abs(d) / sd
        verdict = ('better' if d > 0 else 'WORSE') if r >= 2 else 'within noise'
        print(f"{v:<13}{m:<10}{d:+10.4f}{sd:11.4f}{r:9.2f}  {verdict}")
    print()

fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 5))
order = ['baseline19', 'ordinal20', 'onehot', 'fractions']
colors = ['#9E9E9E', '#D32F2F', '#2196F3', '#4CAF50']
for ax, m in zip(axes, METRICS):
    data = [df[df.variant == v][m].values for v in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.axhline(base[m].mean(), color='#9E9E9E', ls='--', lw=1.2, label='baseline19 mean')
    ax.set_title(m, fontweight='bold'); ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=30); ax.legend(fontsize=7)
fig.suptitle(f'LULC encoding effect on the DNN ({N_SEEDS} seeds per variant)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'lulc_encoding_study.png'), dpi=200,
            bbox_inches='tight')
print(f"Saved lulc_encoding_study.csv / _summary.csv / .png -> {RESULTS_DIR}")
