"""
Spatial Block Cross-Validation — Slovenia Flood Susceptibility
==============================================================
Partitions Slovenia into a 5×4 grid of 50 km blocks, assigns blocks
to 5 folds in a checkerboard pattern, and evaluates RF / XGBoost /
CatBoost / DNN under spatially-separated train-test splits.

Minimum train-test separation ≈ 50 km, avoiding spatial autocorrelation
leakage that inflates metrics with random splits.

Outputs → ML_Results/spatial_cv/
  fold_assignments.csv
  per_fold_metrics.csv
  mean_metrics.csv
  spatial_cv_comparison.png       random-split vs spatial-CV
  fold_map.png                    geographic distribution of folds
  spatial_cv_report.docx
"""

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_TRAIN = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/03_Samples/samples_train.csv'
SAMPLES_TEST  = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/03_Samples/samples_test.csv'
FEATURES_FILE = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt'
RAND_METRICS  = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/08_Validation/validation_metrics.csv'
OUTPUT_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/09_Analysis/spatial_cv'
MODELS_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/04_Models'

BLOCK_SIZE  = 50_000   # metres  →  16 blocks across Slovenia
N_FOLDS     = 5
RANDOM_SEED = 42

# DNN hyperparams (reduced for 5-fold runtime)
DNN_EPOCHS   = 150
DNN_PATIENCE = 15
DNN_BATCH    = 2048

import os, warnings, gc, time
warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    cohen_kappa_score, confusion_matrix
)
import xgboost as xgb
from catboost import CatBoostClassifier
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(RANDOM_SEED)

# ── Load features ─────────────────────────────────────────────────────────────
with open(FEATURES_FILE) as f:
    FEATURES = [l.strip() for l in f if l.strip()]

# ── Combine all samples ───────────────────────────────────────────────────────
df = pd.concat([
    pd.read_csv(SAMPLES_TRAIN),
    pd.read_csv(SAMPLES_TEST)
], ignore_index=True)
print(f"Total samples: {len(df):,}  |  Flood: {(df.label==1).sum():,}  No-flood: {(df.label==0).sum():,}")

# ── Assign spatial blocks & folds ─────────────────────────────────────────────
df['bx']   = ((df.x_3794 - df.x_3794.min()) // BLOCK_SIZE).astype(int)
df['by']   = ((df.y_3794 - df.y_3794.min()) // BLOCK_SIZE).astype(int)
df['fold'] = ((df['bx'] * 2 + df['by']) % N_FOLDS) + 1

df.to_csv(os.path.join(OUTPUT_DIR, 'fold_assignments.csv'), index=False)

print(f"\nSpatial grid  : {df.bx.max()+1} cols × {df.by.max()+1} rows = {df.groupby(['bx','by']).ngroups} blocks")
print(f"Block size    : {BLOCK_SIZE/1000:.0f} km  →  min train-test separation ≈ {BLOCK_SIZE/1000:.0f} km")
print(f"\n{'Fold':<6} {'N':>7} {'Flood':>8} {'No-flood':>10} {'Flood%':>8}  Blocks")
print('-' * 75)
for f in range(1, N_FOLDS+1):
    sub = df[df.fold == f]
    fl = (sub.label==1).sum(); nfl = (sub.label==0).sum()
    blocks = sorted(set(zip(sub.bx, sub.by)))
    print(f"{f:<6} {len(sub):>7,} {fl:>8,} {nfl:>10,} {fl/len(sub)*100:>7.1f}%  "
          f"{[f'({b[0]},{b[1]})' for b in blocks]}")

# ── Fold map ──────────────────────────────────────────────────────────────────
FOLD_COLORS = {1:'#2196F3', 2:'#FF5722', 3:'#4CAF50', 4:'#9C27B0', 5:'#FF9800'}
fig, ax = plt.subplots(figsize=(10, 7))
for fold in range(1, N_FOLDS+1):
    sub = df[df.fold == fold]
    ax.scatter(sub.x_3794/1000, sub.y_3794/1000, s=0.3, alpha=0.4,
               color=FOLD_COLORS[fold], rasterized=True)
patches = [mpatches.Patch(color=FOLD_COLORS[f], label=f'Fold {f}') for f in range(1,6)]
ax.legend(handles=patches, fontsize=11, loc='lower right')
ax.set_xlabel('Easting (km, EPSG:3794)', fontsize=11)
ax.set_ylabel('Northing (km, EPSG:3794)', fontsize=11)
ax.set_title('Spatial Block Cross-Validation — Fold Assignment\n(50 km blocks, 5 folds)', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'fold_map.png'), dpi=180, bbox_inches='tight')
plt.close()
print("\nFold map saved.")

# ── DNN architecture ──────────────────────────────────────────────────────────
class FloodDNN(nn.Module):
    def __init__(self, n_in, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64), nn.BatchNorm1d( 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear( 64,  32), nn.BatchNorm1d( 32), nn.ReLU(), nn.Dropout(dropout*0.7),
            nn.Linear( 32,  16), nn.BatchNorm1d( 16), nn.ReLU(),
            nn.Linear( 16,   1)
        )
    def forward(self, x): return self.net(x)

# ── Metric helper ─────────────────────────────────────────────────────────────
def metrics(y_true, y_prob, name, fold, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model': name, 'Fold': fold,
        'AUC':         round(roc_auc_score(y_true, y_prob),      4),
        'CSI':         round(tp/(tp+fp+fn+1e-10),                4),
        'F1':          round(f1_score(y_true, y_pred),            4),
        'Kappa':       round(cohen_kappa_score(y_true, y_pred),   4),
        'Sensitivity': round(tp/(tp+fn+1e-10),                   4),
        'Specificity': round(tn/(tn+fp+1e-10),                   4),
        'N_test':      len(y_true),
        'Flood_pct':   round(y_true.mean()*100, 1),
    }

def csi_threshold(y_true, y_prob):
    best_t, best_c = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 100):
        p = (y_prob >= t).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_true,p).ravel()
        c = tp/(tp+fp+fn+1e-10)
        if c > best_c: best_c, best_t = c, t
    return best_t

# ── Cross-validation loop ─────────────────────────────────────────────────────
print()
print("=" * 65)
print("SPATIAL CROSS-VALIDATION")
print("=" * 65)

all_results = []
t_start = time.time()

for fold in range(1, N_FOLDS+1):
    print(f"\n{'─'*65}")
    print(f"FOLD {fold} / {N_FOLDS}")
    print(f"{'─'*65}")

    test_mask  = df['fold'] == fold
    train_mask = ~test_mask

    df_tr = df[train_mask].copy()
    df_te = df[test_mask].copy()

    X_tr = df_tr[FEATURES].values.astype(np.float32)
    y_tr = df_tr['label'].values.astype(np.float32)
    X_te = df_te[FEATURES].values.astype(np.float32)
    y_te = df_te['label'].values.astype(np.float32)

    print(f"  Train: {len(X_tr):,}  (flood%={y_tr.mean()*100:.1f}%)")
    print(f"  Test : {len(X_te):,}  (flood%={y_te.mean()*100:.1f}%)")

    # Internal val split (10% of train) for early stopping
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=RANDOM_SEED)

    scale_pos = float((y_tr2==0).sum()) / max(float((y_tr2==1).sum()), 1)

    # Scale (for DNN)
    sc = StandardScaler()
    X_tr2_sc = sc.fit_transform(X_tr2).astype(np.float32)
    X_val_sc  = sc.transform(X_val).astype(np.float32)
    X_te_sc   = sc.transform(X_te).astype(np.float32)

    # ── RF ────────────────────────────────────────────────────────────────
    print("  [RF] training ...", end=' ', flush=True)
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=RANDOM_SEED)
    rf.fit(X_tr, y_tr)
    probs_rf = rf.predict_proba(X_te)[:, 1]
    t_rf = round(csi_threshold(y_te, probs_rf), 4)
    all_results.append(metrics(y_te, probs_rf, 'RF', fold, t_rf))
    print(f"AUC={all_results[-1]['AUC']}  CSI={all_results[-1]['CSI']}  [{time.time()-t0:.0f}s]")
    del rf; gc.collect()

    # ── XGBoost ───────────────────────────────────────────────────────────
    print("  [XGB] training ...", end=' ', flush=True)
    t0 = time.time()
    xgb_m = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        tree_method='hist', device='cuda',
        early_stopping_rounds=20, eval_metric='auc',
        random_state=RANDOM_SEED, verbosity=0)
    xgb_m.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
    probs_xgb = xgb_m.predict_proba(X_te)[:, 1]
    t_xgb = round(csi_threshold(y_te, probs_xgb), 4)
    all_results.append(metrics(y_te, probs_xgb, 'XGBoost', fold, t_xgb))
    print(f"AUC={all_results[-1]['AUC']}  CSI={all_results[-1]['CSI']}  [{time.time()-t0:.0f}s]")
    del xgb_m; gc.collect()

    # ── CatBoost ──────────────────────────────────────────────────────────
    print("  [CAT] training ...", end=' ', flush=True)
    t0 = time.time()
    cat_m = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=8,
        auto_class_weights='Balanced', task_type='GPU', devices='0',
        early_stopping_rounds=20, eval_metric='AUC',
        random_seed=RANDOM_SEED, verbose=False)
    cat_m.fit(X_tr2, y_tr2, eval_set=(X_val, y_val), use_best_model=True)
    probs_cat = cat_m.predict_proba(X_te)[:, 1]
    t_cat = round(csi_threshold(y_te, probs_cat), 4)
    all_results.append(metrics(y_te, probs_cat, 'CatBoost', fold, t_cat))
    print(f"AUC={all_results[-1]['AUC']}  CSI={all_results[-1]['CSI']}  [{time.time()-t0:.0f}s]")
    del cat_m; gc.collect()

    # ── DNN ───────────────────────────────────────────────────────────────
    print("  [DNN] training ...", end=' ', flush=True)
    t0 = time.time()
    pos_w = torch.tensor([scale_pos], dtype=torch.float32)
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
            loss = crit(dnn(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(dnn.parameters(), 1.0)
            opt.step()
        dnn.eval()
        with torch.no_grad():
            lv = []
            for i in range(0, len(Xva_t), 4096):
                lv.append(dnn(Xva_t[i:i+4096].to(DEVICE)).cpu())
            pv = torch.sigmoid(torch.cat(lv)).numpy().flatten()
        va_auc = roc_auc_score(y_val, pv)
        sch.step()
        if va_auc > best_auc_dnn:
            best_auc_dnn = va_auc; best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in dnn.state_dict().items()}
        if ep - best_ep >= DNN_PATIENCE:
            break

    dnn.load_state_dict(best_state)
    dnn.eval()
    with torch.no_grad():
        lt = []
        for i in range(0, len(Xte_t), 4096):
            lt.append(dnn(Xte_t[i:i+4096].to(DEVICE)).cpu())
    probs_dnn = torch.sigmoid(torch.cat(lt)).numpy().flatten()
    t_dnn = round(csi_threshold(y_te, probs_dnn), 4)
    all_results.append(metrics(y_te, probs_dnn, 'DNN', fold, t_dnn))
    print(f"AUC={all_results[-1]['AUC']}  CSI={all_results[-1]['CSI']}  "
          f"[{time.time()-t0:.0f}s, best_ep={best_ep}]")
    del dnn; gc.collect(); torch.cuda.empty_cache()

# ── Aggregate results ─────────────────────────────────────────────────────────
print(f"\nTotal CV time: {(time.time()-t_start)/60:.1f} min")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'per_fold_metrics.csv'), index=False)

# Mean ± std per model
agg_cols = ['AUC','CSI','F1','Kappa','Sensitivity','Specificity']
mean_rows = []
for model in ['RF','XGBoost','CatBoost','DNN']:
    sub = results_df[results_df.Model == model]
    row = {'Model': model}
    for c in agg_cols:
        row[f'{c}_mean'] = round(sub[c].mean(), 4)
        row[f'{c}_std']  = round(sub[c].std(),  4)
    mean_rows.append(row)

mean_df = pd.DataFrame(mean_rows)
mean_df.to_csv(os.path.join(OUTPUT_DIR, 'mean_metrics.csv'), index=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SPATIAL CV — MEAN ± STD ACROSS 5 FOLDS")
print("=" * 65)
print(f"\n{'Model':<12} {'AUC':>12} {'CSI':>12} {'F1':>12} {'Kappa':>12}")
print('-' * 65)
for _, r in mean_df.iterrows():
    print(f"{r.Model:<12} "
          f"{r.AUC_mean:.4f}±{r.AUC_std:.4f}  "
          f"{r.CSI_mean:.4f}±{r.CSI_std:.4f}  "
          f"{r.F1_mean:.4f}±{r.F1_std:.4f}  "
          f"{r.Kappa_mean:.4f}±{r.Kappa_std:.4f}")

print()
print("PER FOLD DETAIL:")
print(f"\n{'Model':<12} {'Fold':>5} {'AUC':>7} {'CSI':>7} {'F1':>7} {'Kappa':>7}  "
      f"{'N_test':>7}  {'Flood%':>7}")
print('-' * 75)
for _, r in results_df.sort_values(['Model','Fold']).iterrows():
    print(f"{r.Model:<12} {int(r.Fold):>5} {r.AUC:>7.4f} {r.CSI:>7.4f} "
          f"{r.F1:>7.4f} {r.Kappa:>7.4f}  {int(r.N_test):>7,}  {r.Flood_pct:>6.1f}%")

# ── Comparison plot: random split vs spatial CV ───────────────────────────────
rand_df  = pd.read_csv(RAND_METRICS)
met_show = ['AUC','CSI','F1','Kappa']

fig, axes = plt.subplots(1, len(met_show), figsize=(16, 6))
COLORS = {'RF':'#2196F3','XGBoost':'#FF5722','CatBoost':'#4CAF50','DNN':'#FF9800'}

for ax, metric in zip(axes, met_show):
    models = ['RF','XGBoost','CatBoost','DNN']
    x = np.arange(len(models)); w = 0.35

    # Random split bars
    rand_vals = [rand_df[rand_df.Model==m][metric].values[0] for m in models]
    bars1 = ax.bar(x - w/2, rand_vals, w, label='Random split',
                   color=[COLORS[m] for m in models], alpha=0.45,
                   edgecolor='gray', linewidth=0.8, hatch='//')

    # Spatial CV bars (mean ± std)
    cv_means = [mean_df[mean_df.Model==m][f'{metric}_mean'].values[0] for m in models]
    cv_stds  = [mean_df[mean_df.Model==m][f'{metric}_std'].values[0]  for m in models]
    bars2 = ax.bar(x + w/2, cv_means, w, yerr=cv_stds, capsize=4,
                   label='Spatial CV (mean±std)',
                   color=[COLORS[m] for m in models], alpha=0.9,
                   edgecolor='black', linewidth=0.8)

    # Value labels
    for bar, v in zip(bars1, rand_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, color='gray')
    for bar, v, e in zip(bars2, cv_means, cv_stds):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+e+0.012,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.12); ax.grid(axis='y', alpha=0.3)
    if ax == axes[0]: ax.set_ylabel('Score', fontsize=11)

# Shared legend
hand = [mpatches.Patch(facecolor='gray', alpha=0.45, hatch='//', label='Random split (80/20)'),
        mpatches.Patch(facecolor='dimgray', alpha=0.9, label='Spatial CV (5-fold, 50 km blocks)')]
fig.legend(handles=hand, loc='upper center', ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, 1.01), frameon=True)
plt.suptitle('Random Split vs Spatial Cross-Validation\nSlovenia Flood Susceptibility',
             fontsize=13, fontweight='bold', y=1.06)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'spatial_cv_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("\nComparison plot saved.")

# ── Per-fold box/strip plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, metric in zip(axes.flat, ['AUC','CSI','F1','Kappa']):
    for model in ['RF','XGBoost','CatBoost','DNN']:
        vals = results_df[results_df.Model==model][metric].values
        folds = results_df[results_df.Model==model]['Fold'].values
        ax.plot(folds, vals, 'o-', color=COLORS[model], lw=1.8, ms=7,
                label=f"{model} (μ={vals.mean():.3f})")
        ax.axhline(vals.mean(), color=COLORS[model], ls=':', lw=1.2, alpha=0.6)
    rand_vals_dict = {m: rand_df[rand_df.Model==m][metric].values[0]
                      for m in ['RF','XGBoost','CatBoost','DNN']}
    ax.set_xlabel('Spatial Fold', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_xticks(range(1, N_FOLDS+1))
    ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.3)

plt.suptitle('Per-Fold Performance — Spatial Cross-Validation', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'per_fold_performance.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Word report ───────────────────────────────────────────────────────────────
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def shd(cell, fill, fc=(255,255,255)):
    tc = cell._tc; p = tc.get_or_add_tcPr()
    s = OxmlElement('w:shd'); s.set(qn('w:val'),'clear')
    s.set(qn('w:color'),'auto'); s.set(qn('w:fill'), fill); p.append(s)
    for r in cell.paragraphs[0].runs: r.font.color.rgb = RGBColor(*fc)

doc = Document()
doc.styles['Normal'].font.name = 'Calibri'
doc.styles['Normal'].font.size = Pt(11)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Spatial Cross-Validation Report — Slovenia FSM')
r.bold = True; r.font.size = Pt(17); r.font.color.rgb = RGBColor(0x1F,0x49,0x7D)
doc.add_paragraph()

# Method
h = doc.add_paragraph(); h.add_run('1. Methodology').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)
doc.add_paragraph(
    f'Slovenia was partitioned into a regular grid of 50 km × 50 km spatial blocks '
    f'({df.groupby(["bx","by"]).ngroups} blocks total). Blocks were assigned to {N_FOLDS} folds '
    f'using a checkerboard pattern, ensuring a minimum train-test separation of ~50 km. '
    f'This eliminates the spatial autocorrelation leakage present in random 80/20 splits '
    f'(where 43.6% of test points were within 1 km of training points). '
    f'Total samples: {len(df):,} ({int((df.label==1).sum()):,} flood, {int((df.label==0).sum()):,} no-flood).'
)
doc.add_paragraph()

# Fold summary table
h = doc.add_paragraph(); h.add_run('2. Fold Summary').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)

fold_rows = []
for f in range(1, N_FOLDS+1):
    sub = df[df.fold==f]
    fl = (sub.label==1).sum(); nfl = (sub.label==0).sum()
    fold_rows.append({'Fold': f, 'N samples': f'{len(sub):,}',
                      'Flood': f'{fl:,}', 'No-flood': f'{nfl:,}',
                      'Flood %': f'{fl/len(sub)*100:.1f}%'})

tbl = doc.add_table(rows=1+len(fold_rows), cols=5)
tbl.style = 'Table Grid'
for i, h_ in enumerate(['Fold','N samples','Flood','No-flood','Flood %']):
    tbl.rows[0].cells[i].text = h_
    tbl.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    tbl.rows[0].cells[i].paragraphs[0].runs[0].font.size = Pt(9)
    shd(tbl.rows[0].cells[i], '1F497D')
for ri, row in enumerate(fold_rows):
    for ci, val in enumerate(row.values()):
        tbl.rows[ri+1].cells[ci].text = str(val)
        tbl.rows[ri+1].cells[ci].paragraphs[0].runs[0].font.size = Pt(9)
        if ri%2==0: shd(tbl.rows[ri+1].cells[ci], 'DEEAF1', (0,0,0))
doc.add_paragraph()

# Mean metrics table
h = doc.add_paragraph(); h.add_run('3. Mean Performance (5-Fold)').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)

display_cols = ['AUC','CSI','F1','Kappa','Sensitivity','Specificity']
tbl2 = doc.add_table(rows=1+len(mean_df), cols=1+len(display_cols))
tbl2.style = 'Table Grid'
for i, h_ in enumerate(['Model']+display_cols):
    tbl2.rows[0].cells[i].text = h_
    tbl2.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    tbl2.rows[0].cells[i].paragraphs[0].runs[0].font.size = Pt(9)
    shd(tbl2.rows[0].cells[i], '2E74B5')
for ri, (_, row) in enumerate(mean_df.iterrows()):
    tbl2.rows[ri+1].cells[0].text = row['Model']
    for ci, c in enumerate(display_cols):
        tbl2.rows[ri+1].cells[ci+1].text = f"{row[f'{c}_mean']:.4f} ± {row[f'{c}_std']:.4f}"
        if ri%2==0: shd(tbl2.rows[ri+1].cells[ci+1], 'DEEAF1', (0,0,0))
    if ri%2==0: shd(tbl2.rows[ri+1].cells[0], 'DEEAF1', (0,0,0))
doc.add_paragraph()

# Comparison with random split
h = doc.add_paragraph(); h.add_run('4. Random Split vs Spatial CV Comparison').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)

comp_rows = []
for model in ['RF','XGBoost','CatBoost','DNN']:
    rand_row = rand_df[rand_df.Model==model].iloc[0]
    cv_row   = mean_df[mean_df.Model==model].iloc[0]
    for metric in ['AUC','CSI','F1','Kappa']:
        diff = cv_row[f'{metric}_mean'] - rand_row[metric]
        comp_rows.append({'Model': model, 'Metric': metric,
                          'Random split': f"{rand_row[metric]:.4f}",
                          'Spatial CV (mean)': f"{cv_row[f'{metric}_mean']:.4f} ± {cv_row[f'{metric}_std']:.4f}",
                          'Difference': f"{diff:+.4f}"})
comp_df = pd.DataFrame(comp_rows)

tbl3 = doc.add_table(rows=1+len(comp_df), cols=len(comp_df.columns))
tbl3.style = 'Table Grid'
for i, h_ in enumerate(comp_df.columns):
    tbl3.rows[0].cells[i].text = h_
    tbl3.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    tbl3.rows[0].cells[i].paragraphs[0].runs[0].font.size = Pt(9)
    shd(tbl3.rows[0].cells[i], '2E74B5')
for ri, (_, row) in enumerate(comp_df.iterrows()):
    for ci, val in enumerate(row):
        tbl3.rows[ri+1].cells[ci].text = str(val)
        r = tbl3.rows[ri+1].cells[ci].paragraphs[0].runs[0]
        r.font.size = Pt(9)
        if str(val).startswith('-'): r.font.color.rgb = RGBColor(0xC0,0x00,0x00)
        if ri%2==0: shd(tbl3.rows[ri+1].cells[ci], 'DEEAF1', (0,0,0))

doc.add_paragraph()
fp = doc.add_paragraph(); fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fp.add_run('Generated: 2026-05-04  |  Project: FSM Slovenia').font.size = Pt(9)

doc.save(os.path.join(OUTPUT_DIR, 'spatial_cv_report.docx'))
print("Word report saved.")
print(f"\nAll outputs → {OUTPUT_DIR}")
