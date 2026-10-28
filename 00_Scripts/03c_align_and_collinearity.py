"""
Align CgFs to common grid + Balanced Sampling + Collinearity Analysis
======================================================================
Step 1 — Warp all CgFs to the Flooded_NoFlood.tif reference grid
         (EPSG:3794, 12.5 m, same extent & pixel alignment).
Step 2 — Extract N_SAMPLES balanced samples per class (flood=1 / no-flood=0).
Step 3 — Pearson correlation matrix + Variance Inflation Factor (VIF).

Outputs
-------
  CgFs_Aligned/         aligned rasters (one per input CgF)
  Collinearity/
    samples_balanced.csv
    correlation_matrix.csv
    correlation_heatmap.png
    vif_table.csv
    collinearity_report.txt
"""

# ── Configuration ─────────────────────────────────────────────────────────────
import os, sys, warnings
warnings.filterwarnings('ignore')

CGFS_DIR      = '/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs'
TARGET_RASTER = os.path.join(CGFS_DIR, 'Flooded_NoFlood.tif')   # reference grid
ALIGNED_DIR   = '/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs_Aligned'
COLLLIN_DIR   = '/home/arora/Documents/FSM_Slovenia/For_Analysis/Collinearity'

N_SAMPLES     = 50_000   # per class (flood / no-flood)
RANDOM_SEED   = 42

# Features to EXCLUDE from collinearity (auxiliary / non-predictor)
EXCLUDE_STEMS = {'flow_direction', 'Flooded_NoFlood', 'CgFs_Dataset_Summary'}

os.makedirs(ALIGNED_DIR, exist_ok=True)
os.makedirs(COLLLIN_DIR,  exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as RS
import rasterio.windows
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(RANDOM_SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
_INT_DTYPES = {'uint8', 'uint16', 'int16', 'uint32', 'int32', 'int8', 'uint64', 'int64'}

def resampling_for(dtype):
    return RS.nearest if dtype in _INT_DTYPES else RS.bilinear

def is_nodata(arr, nd):
    if nd is None:
        return np.zeros(arr.shape, dtype=bool)
    if isinstance(nd, float) and np.isnan(nd):
        return np.isnan(arr)
    return arr == nd

# ── Step 1: Read reference grid ───────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Aligning all CgFs to reference grid")
print("=" * 65)

with rasterio.open(TARGET_RASTER) as ref:
    ref_crs       = ref.crs
    ref_transform = ref.transform
    ref_width     = ref.width
    ref_height    = ref.height
    ref_nodata    = ref.nodata
    ref_dtype     = ref.dtypes[0]

print(f"Reference : {os.path.basename(TARGET_RASTER)}")
print(f"CRS       : {ref_crs.to_epsg()}")
print(f"Size      : {ref_width} × {ref_height} px")
print(f"Pixel     : {ref_transform.a:.4f} m")
xmin = ref_transform.c
ymax = ref_transform.f
xmax = xmin + ref_width  * ref_transform.a
ymin = ymax + ref_height * ref_transform.e
print(f"Extent    : xmin={xmin:.1f}  ymin={ymin:.1f}  xmax={xmax:.1f}  ymax={ymax:.1f}")
print()

# ── Step 2: Warp each CgF ────────────────────────────────────────────────────
src_files = sorted(
    f for f in os.listdir(CGFS_DIR)
    if f.endswith('.tif') and os.path.splitext(f)[0] not in EXCLUDE_STEMS
)

aligned_paths = {}   # stem -> aligned path

for i, fname in enumerate(src_files, 1):
    stem      = os.path.splitext(fname)[0]
    src_path  = os.path.join(CGFS_DIR, fname)
    out_path  = os.path.join(ALIGNED_DIR, fname)
    aligned_paths[stem] = out_path

    with rasterio.open(src_path) as src:
        same_grid = (
            src.crs == ref_crs
            and src.transform.almost_equals(ref_transform, precision=1e-3)
            and src.width  == ref_width
            and src.height == ref_height
        )

    if same_grid and os.path.exists(out_path):
        print(f"[{i:2d}/{len(src_files)}] {fname:<50}  already aligned — skip")
        continue

    with rasterio.open(src_path) as src:
        dtype   = src.dtypes[0]
        nodata  = src.nodata
        rsmp    = resampling_for(dtype)

        # fill value
        fill = nodata if nodata is not None else (
            -9999 if dtype in _INT_DTYPES else np.nan
        )

        out_meta = dict(
            driver='GTiff', dtype=dtype,
            nodata=fill,
            width=ref_width, height=ref_height, count=1,
            crs=ref_crs, transform=ref_transform,
            compress='lzw', tiled=True,
            blockxsize=512, blockysize=512,
        )

        dest = np.full((ref_height, ref_width), fill, dtype=dtype)

        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=rsmp,
            src_nodata=nodata,
            dst_nodata=fill,
        )

        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(dest, 1)

    print(f"[{i:2d}/{len(src_files)}] {fname:<50}  aligned ({rsmp.name})")

print(f"\nAligned rasters saved to: {ALIGNED_DIR}\n")

# ── Step 3: Balanced sampling ─────────────────────────────────────────────────
print("=" * 65)
print("STEP 2 — Extracting balanced samples")
print("=" * 65)

# Read target label
with rasterio.open(TARGET_RASTER) as ds:
    label = ds.read(1).astype(np.float32)
    lbl_nd = ds.nodata

label_mask = is_nodata(label, lbl_nd)

flood_idx    = np.argwhere((label == 1) & ~label_mask)
noflood_idx  = np.argwhere((label == 0) & ~label_mask)

print(f"Flood pixels    : {len(flood_idx):>10,}")
print(f"No-flood pixels : {len(noflood_idx):>10,}")
print(f"Sampling        : {N_SAMPLES:,} per class")

if len(flood_idx) < N_SAMPLES:
    print(f"WARNING: Only {len(flood_idx):,} flood pixels — using all of them.")
    N_SAMPLES = len(flood_idx)

sel_flood   = flood_idx[np.random.choice(len(flood_idx),   N_SAMPLES, replace=False)]
sel_noflood = noflood_idx[np.random.choice(len(noflood_idx), N_SAMPLES, replace=False)]

all_idx = np.vstack([sel_flood, sel_noflood])      # shape (2*N, 2): [row, col]
all_lbl = np.array([1]*N_SAMPLES + [0]*N_SAMPLES)

rows, cols = all_idx[:, 0], all_idx[:, 1]

print(f"\nReading {len(aligned_paths)} feature rasters at {2*N_SAMPLES:,} sample locations ...")

feat_data   = {}
skipped     = []

for stem, apath in sorted(aligned_paths.items()):
    if not os.path.exists(apath):
        skipped.append(stem)
        continue
    with rasterio.open(apath) as ds:
        arr  = ds.read(1)
        nd   = ds.nodata
    vals = arr[rows, cols].astype(np.float64)
    nd_mask = is_nodata(vals, nd)
    vals[nd_mask] = np.nan
    feat_data[stem] = vals

if skipped:
    print(f"  Skipped (file missing): {skipped}")

# Build DataFrame
df = pd.DataFrame(feat_data)
df.insert(0, 'label', all_lbl)

# Drop rows where ANY feature is NaN
n_before = len(df)
df.dropna(inplace=True)
n_after  = len(df)
print(f"Samples after dropping NaN rows: {n_after:,} (dropped {n_before - n_after:,})")

# Class balance after NaN removal
vc = df['label'].value_counts()
print(f"  Flood (1)    : {vc.get(1,0):,}")
print(f"  No-flood (0) : {vc.get(0,0):,}")

samples_path = os.path.join(COLLLIN_DIR, 'samples_balanced.csv')
df.to_csv(samples_path, index=False)
print(f"\nSamples saved: {samples_path}")

# ── Step 4: Collinearity analysis ─────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 3 — Collinearity analysis")
print("=" * 65)

feat_cols = [c for c in df.columns if c != 'label']
X = df[feat_cols].copy()

# -- 4a. Pearson correlation matrix ------------------------------------------
print("\n[4a] Pearson correlation matrix ...")
corr = X.corr(method='pearson')
corr_path = os.path.join(COLLLIN_DIR, 'correlation_matrix.csv')
corr.round(4).to_csv(corr_path)
print(f"     Saved: {corr_path}")

# Heatmap
fig, ax = plt.subplots(figsize=(20, 17))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle (keep lower)
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
    linewidths=0.4, linecolor='white',
    annot_kws={'size': 7},
    ax=ax, square=True,
    cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}
)
ax.set_title(
    'Pearson Correlation Matrix — CgFs (Slovenia Flood Susceptibility)',
    fontsize=14, fontweight='bold', pad=12
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
heatmap_path = os.path.join(COLLLIN_DIR, 'correlation_heatmap.png')
fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"     Heatmap saved: {heatmap_path}")

# Highly correlated pairs (|r| > 0.8)
high_corr = []
for i in range(len(feat_cols)):
    for j in range(i+1, len(feat_cols)):
        r = corr.iloc[i, j]
        if abs(r) > 0.80:
            high_corr.append((feat_cols[i], feat_cols[j], round(r, 4)))
high_corr.sort(key=lambda x: -abs(x[2]))

print(f"\n     Pairs with |r| > 0.80: {len(high_corr)}")
for f1, f2, r in high_corr:
    print(f"       {f1:<40} ↔ {f2:<40}  r = {r:+.4f}")

# -- 4b. VIF ------------------------------------------------------------------
print("\n[4b] Variance Inflation Factor (VIF) ...")

# Standardise (z-score) to avoid numerical blow-up
X_std = (X - X.mean()) / X.std()

vif_records = []
for idx, col in enumerate(feat_cols):
    try:
        vif = variance_inflation_factor(X_std.values, idx)
    except Exception as e:
        vif = np.inf
    vif_records.append({'Feature': col, 'VIF': round(vif, 3)})

vif_df = pd.DataFrame(vif_records).sort_values('VIF', ascending=False).reset_index(drop=True)
vif_path = os.path.join(COLLLIN_DIR, 'vif_table.csv')
vif_df.to_csv(vif_path, index=False)
print(f"     Saved: {vif_path}")

print("\n     VIF Table (sorted descending):")
print(f"     {'Feature':<45} {'VIF':>8}  Interpretation")
print(f"     {'-'*45} {'-'*8}  {'-'*20}")
for _, row in vif_df.iterrows():
    v = row['VIF']
    if   np.isinf(v):      note = "PERFECT multicollinearity"
    elif v > 10:            note = "SEVERE — consider removing"
    elif v > 5:             note = "HIGH — monitor"
    elif v > 2.5:           note = "MODERATE"
    else:                   note = "low"
    print(f"     {row['Feature']:<45} {v:>8.2f}  {note}")

# ── Step 5: Write text report ─────────────────────────────────────────────────
report_path = os.path.join(COLLLIN_DIR, 'collinearity_report.txt')
with open(report_path, 'w') as rpt:
    rpt.write("COLLINEARITY REPORT — Slovenia CgFs\n")
    rpt.write(f"Samples per class : {N_SAMPLES:,} (flood=1, no-flood=0)\n")
    rpt.write(f"Total samples used: {len(df):,}\n")
    rpt.write(f"Features analysed : {len(feat_cols)}\n\n")

    rpt.write("─" * 70 + "\n")
    rpt.write("HIGHLY CORRELATED PAIRS (|Pearson r| > 0.80)\n")
    rpt.write("─" * 70 + "\n")
    if high_corr:
        for f1, f2, r in high_corr:
            rpt.write(f"  {f1:<40} ↔ {f2:<38}  r = {r:+.4f}\n")
    else:
        rpt.write("  None\n")

    rpt.write("\n" + "─" * 70 + "\n")
    rpt.write("VIF TABLE (sorted descending)\n")
    rpt.write("─" * 70 + "\n")
    rpt.write(f"  {'Feature':<45} {'VIF':>8}  Interpretation\n")
    rpt.write(f"  {'-'*45} {'-'*8}  {'-'*25}\n")
    for _, row in vif_df.iterrows():
        v = row['VIF']
        if   np.isinf(v):  note = "PERFECT multicollinearity"
        elif v > 10:        note = "SEVERE — consider removing"
        elif v > 5:         note = "HIGH — monitor"
        elif v > 2.5:       note = "MODERATE"
        else:               note = "low"
        rpt.write(f"  {row['Feature']:<45} {v:>8.2f}  {note}\n")

print(f"\nReport saved: {report_path}")

print()
print("=" * 65)
print("ALL DONE")
print(f"  Aligned rasters : {ALIGNED_DIR}")
print(f"  Collinearity    : {COLLLIN_DIR}")
print("=" * 65)
