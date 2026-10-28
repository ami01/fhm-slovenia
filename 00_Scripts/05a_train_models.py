"""
Slovenia Flood Susceptibility — ML Training Pipeline
=====================================================
Models   : Random Forest | XGBoost (GPU) | CatBoost (GPU)
Features : 19 VIF-selected, collinearity-free CgFs (EPSG:3794, 12.5 m)
Target   : Flooded_NoFlood.tif (binary 0/1)

Outputs → ML_Results/
  samples/
    samples_train.csv / samples_test.csv
    samples_train.shp / samples_test.shp   ← EPSG:3794 point shapefiles
  models/
    model_rf.pkl | model_xgb.json | model_cat.cbm
  results/
    metrics_all_models.csv
    roc_curves.png
    pr_curves.png
    feature_importance_rf.png
    feature_importance_xgb.png
    feature_importance_cat.png
"""

# ── Config ────────────────────────────────────────────────────────────────────
ALIGNED_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features'
TARGET_RASTER  = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features/Flooded_NoFlood.tif'
FEATURES_FILE  = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
OUTPUT_DIR     = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'

N_SAMPLES      = 50_000   # per class
TEST_SIZE      = 0.20     # 80/20 stratified split
RANDOM_SEED    = 42

# ── Setup ─────────────────────────────────────────────────────────────────────
import os, warnings, gc
warnings.filterwarnings('ignore')

SAMPLES_DIR = os.path.join(OUTPUT_DIR, '03_Samples')
MODELS_DIR  = os.path.join(OUTPUT_DIR, '04_Models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, '05_Results')
for d in [SAMPLES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

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
    cohen_kappa_score, classification_report
)
import xgboost as xgb
from catboost import CatBoostClassifier

np.random.seed(RANDOM_SEED)

# ── Step 1: Load selected features ────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading feature list & rasters")
print("=" * 65)

with open(FEATURES_FILE) as f:
    FEATURES = [l.strip() for l in f if l.strip()]

print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}\n")

# ── Step 2: Sample balanced pixels WITH coordinates ──────────────────────────
print("STEP 2 — Balanced sampling with coordinates")
print("=" * 65)

def is_nodata(arr, nd):
    if nd is None:           return np.zeros(arr.shape, dtype=bool)
    if isinstance(nd, float) and np.isnan(nd): return np.isnan(arr)
    return arr == nd

with rasterio.open(TARGET_RASTER) as ds:
    label   = ds.read(1).astype(np.float32)
    lbl_nd  = ds.nodata
    tf      = ds.transform

lbl_mask   = is_nodata(label, lbl_nd)
flood_idx  = np.argwhere((label == 1) & ~lbl_mask)
nflood_idx = np.argwhere((label == 0) & ~lbl_mask)

n_fl  = min(N_SAMPLES, len(flood_idx))
n_nfl = min(N_SAMPLES, len(nflood_idx))

sel_fl  = flood_idx [np.random.choice(len(flood_idx),  n_fl,  replace=False)]
sel_nfl = nflood_idx[np.random.choice(len(nflood_idx), n_nfl, replace=False)]
all_idx = np.vstack([sel_fl, sel_nfl])
rows, cols = all_idx[:, 0], all_idx[:, 1]

# Compute real-world coordinates (centre of each pixel)
xs = tf.c + (cols + 0.5) * tf.a
ys = tf.f + (rows + 0.5) * tf.e

labels_arr = np.array([1]*n_fl + [0]*n_nfl)

print(f"Flood    : {n_fl:,}  |  No-flood : {n_nfl:,}  |  Total : {n_fl+n_nfl:,}")
print(f"Reading {len(FEATURES)} feature rasters ...")

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
    print(f"  {stem}")

df = pd.DataFrame(feat_data)
df.insert(0, 'label', labels_arr)

# Verify all features loaded
missing_feats = [f for f in FEATURES if f not in df.columns]
if missing_feats:
    print(f"WARNING: Missing features in data: {missing_feats}")

n_before = len(df)
df.dropna(subset=FEATURES, inplace=True)
print(f"\nSamples after NaN drop: {len(df):,}  (dropped {n_before-len(df):,})")
print(f"  Flood (1)   : {(df.label==1).sum():,}")
print(f"  No-flood (0): {(df.label==0).sum():,}")

# ── Step 3: Train / test split ────────────────────────────────────────────────
print()
print("STEP 3 — Train/test split (80/20 stratified)")
print("=" * 65)

X = df[FEATURES].values.astype(np.float32)
y = df['label'].values.astype(np.float32)
coords = df[['x_3794','y_3794']].values

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y, np.arange(len(df)),
    test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
print(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}")
print(f"  Train flood ratio: {y_tr.mean():.3f}  |  Test flood ratio: {y_te.mean():.3f}")

# ── Step 4: Save samples as CSV + SHP ────────────────────────────────────────
print()
print("STEP 4 — Saving samples (CSV + SHP)")
print("=" * 65)

def save_samples(X_arr, y_arr, coord_arr, feat_names, tag):
    df_out = pd.DataFrame(X_arr, columns=feat_names)
    df_out.insert(0, 'label', y_arr.astype(int))
    df_out.insert(1, 'x_3794', coord_arr[:, 0])
    df_out.insert(2, 'y_3794', coord_arr[:, 1])

    csv_path = os.path.join(SAMPLES_DIR, f'samples_{tag}.csv')
    df_out.to_csv(csv_path, index=False)

    gdf = gpd.GeoDataFrame(
        df_out,
        geometry=[Point(x, y) for x, y in zip(coord_arr[:, 0], coord_arr[:, 1])],
        crs='EPSG:3794'
    )
    shp_path = os.path.join(SAMPLES_DIR, f'samples_{tag}.shp')
    gdf.to_file(shp_path)
    print(f"  {tag:<8}: {len(df_out):,} samples → {csv_path}")
    print(f"           shapefile → {shp_path}")

save_samples(X_tr, y_tr, coords[idx_tr], FEATURES, 'train')
save_samples(X_te, y_te, coords[idx_te], FEATURES, 'test')

# ── Step 5: Scale (for reference — tree models use raw) ──────────────────────
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

# ── Step 6: Train models ──────────────────────────────────────────────────────
print()
print("STEP 5 — Model training")
print("=" * 65)

predictions   = {}   # name → predicted probabilities on test set
feature_imps  = {}   # name → importance array

# ─── Model 1: Random Forest ──────────────────────────────────────────────────
print("\n[1/3] Random Forest (sklearn) ...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=RANDOM_SEED
)
rf.fit(X_tr, y_tr)
predictions['Random Forest'] = rf.predict_proba(X_te)[:, 1]
feature_imps['Random Forest'] = rf.feature_importances_
joblib.dump(rf, os.path.join(MODELS_DIR, 'model_rf.pkl'))
print(f"  Saved: model_rf.pkl")
del rf; gc.collect()

# ─── Model 2: XGBoost (GPU) ──────────────────────────────────────────────────
print("\n[2/3] XGBoost (GPU hist) ...")
scale_pos = float((y_tr == 0).sum()) / float((y_tr == 1).sum())

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    tree_method='hist',
    device='cuda',
    eval_metric='auc',
    early_stopping_rounds=30,
    random_state=RANDOM_SEED,
    verbosity=0
)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    verbose=False
)
predictions['XGBoost'] = xgb_model.predict_proba(X_te)[:, 1]
feature_imps['XGBoost'] = xgb_model.feature_importances_
xgb_model.save_model(os.path.join(MODELS_DIR, 'model_xgb.json'))
print(f"  Best round : {xgb_model.best_iteration}")
print(f"  Saved: model_xgb.json")
del xgb_model; gc.collect()

# ─── Model 3: CatBoost (GPU) ─────────────────────────────────────────────────
print("\n[3/3] CatBoost (GPU) ...")
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    auto_class_weights='Balanced',
    task_type='GPU',
    devices='0',
    eval_metric='AUC',
    early_stopping_rounds=30,
    random_seed=RANDOM_SEED,
    verbose=False
)
cat_model.fit(
    X_tr, y_tr,
    eval_set=(X_te, y_te),
    use_best_model=True
)
predictions['CatBoost'] = cat_model.predict_proba(X_te)[:, 1]
feature_imps['CatBoost'] = cat_model.get_feature_importance()
cat_model.save_model(os.path.join(MODELS_DIR, 'model_cat.cbm'))
print(f"  Best round : {cat_model.best_iteration_}")
print(f"  Saved: model_cat.cbm")
del cat_model; gc.collect()

# ── Step 7: Metrics ───────────────────────────────────────────────────────────
print()
print("STEP 6 — Evaluation metrics")
print("=" * 65)

def compute_metrics(y_true, y_prob, name, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model'      : name,
        'AUC'        : round(roc_auc_score(y_true, y_prob),        4),
        'Avg_Prec'   : round(average_precision_score(y_true, y_prob), 4),
        'F1'         : round(f1_score(y_true, y_pred),              4),
        'Accuracy'   : round(accuracy_score(y_true, y_pred),        4),
        'Kappa'      : round(cohen_kappa_score(y_true, y_pred),     4),
        'CSI'        : round(tp / (tp + fp + fn + 1e-10),          4),
        'Sensitivity': round(tp / (tp + fn + 1e-10),               4),
        'Specificity': round(tn / (tn + fp + 1e-10),               4),
        'Precision'  : round(tp / (tp + fp + 1e-10),               4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }

results = []
for name, probs in predictions.items():
    m = compute_metrics(y_te, probs, name)
    results.append(m)
    print(f"\n{name}")
    print(f"  AUC={m['AUC']}  F1={m['F1']}  CSI={m['CSI']}  "
          f"Sensitivity={m['Sensitivity']}  Specificity={m['Specificity']}  Kappa={m['Kappa']}")

metrics_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
metrics_csv = os.path.join(RESULTS_DIR, 'metrics_all_models.csv')
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nMetrics saved: metrics_all_models.csv")

# ── Step 8: ROC curves ────────────────────────────────────────────────────────
print()
print("STEP 7 — Generating plots")
print("=" * 65)

COLORS = {'Random Forest': '#2196F3', 'XGBoost': '#FF5722', 'CatBoost': '#4CAF50'}
LINES  = {'Random Forest': '-',       'XGBoost': '--',       'CatBoost': '-.'}

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1],'k--', lw=1.2, alpha=0.5, label='Random chance')
for name, probs in predictions.items():
    fpr, tpr, _ = roc_curve(y_te, probs)
    auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.2, color=COLORS[name], ls=LINES[name],
            label=f'{name}  (AUC = {auc_val:.4f})')
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves — Slovenia Flood Susceptibility', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: roc_curves.png")

# ── Step 9: Precision-Recall curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for name, probs in predictions.items():
    prec, rec, _ = precision_recall_curve(y_te, probs)
    ap = average_precision_score(y_te, probs)
    ax.plot(rec, prec, lw=2.2, color=COLORS[name], ls=LINES[name],
            label=f'{name}  (AP = {ap:.4f})')
ax.axhline(y_te.mean(), color='k', ls='--', lw=1.2, alpha=0.5, label=f'Baseline (prevalence={y_te.mean():.3f})')
ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — Slovenia Flood Susceptibility', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'pr_curves.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: pr_curves.png")

# ── Step 10: Feature importance plots ─────────────────────────────────────────
def plot_importance(imp_arr, feat_names, model_name, fname):
    idx   = np.argsort(imp_arr)[::-1]
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
    # Add value labels
    for bar, val in zip(bars, imp_arr[idx[::-1]]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()

for name, imp in feature_imps.items():
    fname = f"feature_importance_{name.lower().replace(' ','_')}.png"
    plot_importance(imp, FEATURES, name, fname)
    print(f"  Saved: {fname}")

# ── Step 11: Metrics summary bar chart ───────────────────────────────────────
met_plot = ['AUC', 'F1', 'CSI', 'Sensitivity', 'Specificity', 'Kappa']
x       = np.arange(len(met_plot))
width   = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
for i, (_, row) in enumerate(metrics_df.iterrows()):
    vals = [row[m] for m in met_plot]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=row['Model'],
                  color=list(COLORS.values())[i], alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
ax.set_xticks(x)
ax.set_xticklabels(met_plot, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison — Slovenia Flood Susceptibility', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: model_comparison.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL RESULTS")
print("=" * 65)
print(metrics_df[['Model','AUC','F1','CSI','Sensitivity','Specificity','Kappa']].to_string(index=False))
print()
print(f"Models  → {MODELS_DIR}")
print(f"Samples → {SAMPLES_DIR}")
print(f"Plots   → {RESULTS_DIR}")
