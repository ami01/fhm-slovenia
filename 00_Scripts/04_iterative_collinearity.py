"""
Iterative VIF Elimination + Correlation Screening
==================================================
1. Sample balanced flood/no-flood pixels from masked CgFs_Aligned rasters.
2. Iteratively remove the feature with the highest VIF (> VIF_THRESHOLD)
   and log each step until all remaining features are below the threshold.
3. After VIF is clean, flag any remaining correlated pairs (|r| > CORR_THRESHOLD).
4. Export full elimination log, final VIF table, correlation heatmap, and
   a clean feature list.

Outputs → Collinearity/
  samples_balanced_masked.csv       re-sampled from masked aligned rasters
  vif_elimination_log.csv           one row per removal step
  vif_final.csv                     VIF of retained features
  correlation_final.csv             Pearson matrix of retained features
  correlation_heatmap_final.png     heatmap
  collinearity_report_final.txt     human-readable summary
  selected_features.txt             final clean feature list (one per line)
"""

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features'
COLLLIN_DIR   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity'

N_SAMPLES     = 50_000   # per class
RANDOM_SEED   = 42
VIF_THRESHOLD  = 10.0    # remove features above this
CORR_THRESHOLD = 0.80    # flag pairs above this

# Features to skip entirely (non-predictors)
EXCLUDE = {'flow_direction'}

import os, warnings
warnings.filterwarnings('ignore')
os.makedirs(COLLLIN_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.enums import Resampling
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(RANDOM_SEED)

# ── Step 1: Discover files ────────────────────────────────────────────────────
all_files = sorted(
    f for f in os.listdir(ALIGNED_DIR)
    if f.endswith('.tif') and os.path.splitext(f)[0] not in EXCLUDE
)
target_file = 'Flooded_NoFlood.tif'
feat_files  = [f for f in all_files if f != target_file]

print("=" * 65)
print("STEP 1 — Balanced sampling from masked aligned rasters")
print("=" * 65)
print(f"Target : {target_file}")
print(f"Features ({len(feat_files)}): {', '.join(os.path.splitext(f)[0] for f in feat_files)}\n")

# ── Step 2: Sample balanced pixels ───────────────────────────────────────────
def read_nodata(nd, arr):
    """Return boolean mask: True where pixel is nodata."""
    if nd is None:
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nd, float) and np.isnan(nd):
        return np.isnan(arr)
    return arr == nd

with rasterio.open(os.path.join(ALIGNED_DIR, target_file)) as ds:
    label = ds.read(1).astype(np.float32)
    lbl_nd = ds.nodata

lbl_mask   = read_nodata(lbl_nd, label)
flood_idx  = np.argwhere((label == 1) & ~lbl_mask)
nflood_idx = np.argwhere((label == 0) & ~lbl_mask)

n_flood  = min(N_SAMPLES, len(flood_idx))
n_nflood = min(N_SAMPLES, len(nflood_idx))

if n_flood < N_SAMPLES:
    print(f"WARNING: only {len(flood_idx):,} flood pixels — using all.")

sel_fl  = flood_idx [np.random.choice(len(flood_idx),  n_flood,  replace=False)]
sel_nfl = nflood_idx[np.random.choice(len(nflood_idx), n_nflood, replace=False)]
all_idx = np.vstack([sel_fl, sel_nfl])
rows, cols = all_idx[:,0], all_idx[:,1]
labels = np.array([1]*n_flood + [0]*n_nflood)

print(f"Flood pixels available   : {len(flood_idx):>10,}")
print(f"No-flood pixels available: {len(nflood_idx):>10,}")
print(f"Sampled per class        : flood={n_flood:,}  no-flood={n_nflood:,}")
print(f"\nReading {len(feat_files)} feature rasters at {len(all_idx):,} locations ...")

feat_data = {}
for fname in feat_files:
    stem = os.path.splitext(fname)[0]
    path = os.path.join(ALIGNED_DIR, fname)
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nd  = ds.nodata
    vals = arr[rows, cols].astype(np.float64)
    vals[read_nodata(nd, vals)] = np.nan
    feat_data[stem] = vals

df = pd.DataFrame(feat_data)
df.insert(0, 'label', labels)
n_before = len(df)
df.dropna(inplace=True)
print(f"Samples after NaN removal: {len(df):,}  (dropped {n_before-len(df):,})")
print(f"  Flood (1)   : {(df.label==1).sum():,}")
print(f"  No-flood (0): {(df.label==0).sum():,}")

df.to_csv(os.path.join(COLLLIN_DIR, 'samples_balanced_masked.csv'), index=False)
print(f"Saved: samples_balanced_masked.csv\n")

# ── Step 3: Iterative VIF elimination ─────────────────────────────────────────
print("=" * 65)
print("STEP 2 — Iterative VIF elimination")
print(f"         Threshold : VIF > {VIF_THRESHOLD}")
print("=" * 65)

feat_cols = [c for c in df.columns if c != 'label']
remaining = list(feat_cols)
elim_log  = []   # (step, removed_feature, vif_at_removal, remaining_count)

def compute_vif(df_sub, cols):
    """Return Series of VIF values, indexed by feature name."""
    X = df_sub[cols].copy()
    X = (X - X.mean()) / X.std()   # standardise to reduce numerical issues
    vifs = {}
    for i, col in enumerate(cols):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.inf
        vifs[col] = round(float(v), 3)
    return pd.Series(vifs).sort_values(ascending=False)

step = 0
while True:
    vif = compute_vif(df, remaining)
    max_vif  = vif.iloc[0]
    max_feat = vif.index[0]

    print(f"\n--- Round {step+1} | Features remaining: {len(remaining)} ---")
    print(vif.to_string())

    if max_vif <= VIF_THRESHOLD:
        print(f"\nAll VIFs ≤ {VIF_THRESHOLD}. Stopping.")
        break

    print(f"\n  → Removing '{max_feat}'  (VIF = {max_vif:.2f})")
    elim_log.append({
        'step'           : step + 1,
        'removed_feature': max_feat,
        'vif_at_removal' : max_vif,
        'features_before': len(remaining),
        'features_after' : len(remaining) - 1,
    })
    remaining.remove(max_feat)
    step += 1

print(f"\nElimination complete. Steps taken: {step}")
print(f"Features retained: {len(remaining)}")

# ── Step 4: Save elimination log ──────────────────────────────────────────────
log_df = pd.DataFrame(elim_log)
log_path = os.path.join(COLLLIN_DIR, 'vif_elimination_log.csv')
log_df.to_csv(log_path, index=False)
print(f"Saved: vif_elimination_log.csv")

# ── Step 5: Final VIF table ───────────────────────────────────────────────────
vif_final = compute_vif(df, remaining).reset_index()
vif_final.columns = ['Feature', 'VIF']

def vif_label(v):
    if np.isinf(v): return "PERFECT multicollinearity"
    if v > 10:       return "SEVERE"
    if v > 5:        return "HIGH"
    if v > 2.5:      return "MODERATE"
    return "low"

vif_final['Interpretation'] = vif_final['VIF'].apply(vif_label)
vif_path = os.path.join(COLLLIN_DIR, 'vif_final.csv')
vif_final.to_csv(vif_path, index=False)
print(f"Saved: vif_final.csv")

# ── Step 6: Correlation of retained features ──────────────────────────────────
print()
print("=" * 65)
print("STEP 3 — Correlation check on retained features")
print(f"         Flagging pairs with |r| > {CORR_THRESHOLD}")
print("=" * 65)

X_final = df[remaining]
corr    = X_final.corr(method='pearson')
corr.round(4).to_csv(os.path.join(COLLLIN_DIR, 'correlation_final.csv'))

high_corr = []
for i in range(len(remaining)):
    for j in range(i+1, len(remaining)):
        r = corr.iloc[i, j]
        if abs(r) > CORR_THRESHOLD:
            high_corr.append((remaining[i], remaining[j], round(r, 4)))
high_corr.sort(key=lambda x: -abs(x[2]))

print(f"\nHigh-correlation pairs (|r| > {CORR_THRESHOLD}): {len(high_corr)}")
if high_corr:
    for f1, f2, r in high_corr:
        print(f"  {f1:<45} ↔ {f2:<45}  r = {r:+.4f}")
else:
    print("  None — all pairs within acceptable range.")

# Heatmap
n = len(remaining)
fig_size = max(14, n * 0.85)
fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask_upper, annot=True, fmt='.2f',
    cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor='white',
    annot_kws={'size': 9 if n <= 16 else 7},
    ax=ax, square=True,
    cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}
)
ax.set_title(
    f'Pearson Correlation — Final Retained Features (n={n})\n'
    f'Slovenia Flood Susceptibility | VIF threshold = {VIF_THRESHOLD}',
    fontsize=13, fontweight='bold', pad=12
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
hmap_path = os.path.join(COLLLIN_DIR, 'correlation_heatmap_final.png')
fig.savefig(hmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Heatmap saved: correlation_heatmap_final.png")

# ── Step 7: Human-readable report ────────────────────────────────────────────
report_path = os.path.join(COLLLIN_DIR, 'collinearity_report_final.txt')
with open(report_path, 'w') as rpt:
    rpt.write("ITERATIVE VIF ELIMINATION REPORT — Slovenia CgFs\n")
    rpt.write("=" * 70 + "\n\n")
    rpt.write(f"VIF threshold       : {VIF_THRESHOLD}\n")
    rpt.write(f"Correlation flag    : |r| > {CORR_THRESHOLD}\n")
    rpt.write(f"Samples per class   : {N_SAMPLES:,}\n")
    rpt.write(f"Total samples used  : {len(df):,}\n")
    rpt.write(f"Initial features    : {len(feat_cols)}\n")
    rpt.write(f"Features removed    : {len(elim_log)}\n")
    rpt.write(f"Features retained   : {len(remaining)}\n\n")

    rpt.write("─" * 70 + "\n")
    rpt.write("ELIMINATION LOG\n")
    rpt.write("─" * 70 + "\n")
    rpt.write(f"  {'Step':<6} {'Removed Feature':<45} {'VIF':>8}  {'Remaining'}\n")
    rpt.write(f"  {'─'*6} {'─'*45} {'─'*8}  {'─'*9}\n")
    for _, row in log_df.iterrows():
        rpt.write(f"  {int(row.step):<6} {row.removed_feature:<45} {row.vif_at_removal:>8.2f}  {int(row.features_after)}\n")

    rpt.write("\n" + "─" * 70 + "\n")
    rpt.write("FINAL VIF TABLE (retained features)\n")
    rpt.write("─" * 70 + "\n")
    rpt.write(f"  {'Feature':<45} {'VIF':>8}  Interpretation\n")
    rpt.write(f"  {'─'*45} {'─'*8}  {'─'*20}\n")
    for _, row in vif_final.iterrows():
        rpt.write(f"  {row.Feature:<45} {row.VIF:>8.3f}  {row.Interpretation}\n")

    rpt.write("\n" + "─" * 70 + "\n")
    rpt.write(f"HIGH CORRELATION PAIRS (|r| > {CORR_THRESHOLD})\n")
    rpt.write("─" * 70 + "\n")
    if high_corr:
        for f1, f2, r in high_corr:
            rpt.write(f"  {f1:<45} ↔ {f2:<43}  r = {r:+.4f}\n")
    else:
        rpt.write("  None.\n")

    rpt.write("\n" + "─" * 70 + "\n")
    rpt.write("FINAL RETAINED FEATURE SET\n")
    rpt.write("─" * 70 + "\n")
    for i, feat in enumerate(sorted(remaining), 1):
        rpt.write(f"  {i:2d}. {feat}\n")

print(f"Report saved: collinearity_report_final.txt")

# ── Step 8: Clean feature list ────────────────────────────────────────────────
feat_list_path = os.path.join(COLLLIN_DIR, 'selected_features.txt')
with open(feat_list_path, 'w') as f:
    f.write('\n'.join(sorted(remaining)) + '\n')
print(f"Feature list saved: selected_features.txt")

# ── Summary to console ────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\nRemoved ({len(elim_log)}):")
for _, row in log_df.iterrows():
    print(f"  Step {int(row.step):2d}: {row.removed_feature:<45}  VIF = {row.vif_at_removal:.2f}")

print(f"\nRetained ({len(remaining)}):")
print(f"\n  {'Feature':<45} {'VIF':>8}  Interpretation")
print(f"  {'─'*45} {'─'*8}  {'─'*20}")
for _, row in vif_final.iterrows():
    print(f"  {row.Feature:<45} {row.VIF:>8.3f}  {row.Interpretation}")

if high_corr:
    print(f"\nRemaining high-correlation pairs (|r| > {CORR_THRESHOLD}):")
    for f1, f2, r in high_corr:
        print(f"  {f1} ↔ {f2}  r={r:+.4f}")
else:
    print(f"\nNo remaining high-correlation pairs (|r| > {CORR_THRESHOLD}).")

print(f"\nAll outputs in: {COLLLIN_DIR}")
