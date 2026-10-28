"""
Threshold Optimisation → Classification → Validation
=====================================================
Uses 10x models (500k per class) probability maps and the 10x test split.

1. Find CSI-optimal & Youden-optimal thresholds per model from test samples.
2. Apply optimal threshold → binary flood map (Int16: 1=flood, 0=no-flood).
3. Apply quantile classification → 5-class susceptibility map
   (1=Very Low … 5=Very High) based on all valid pixels in Slovenia.
4. Validate classified maps against test samples:
   AUC, CSI, F1, Sensitivity, Specificity, Kappa, Accuracy.
5. Save confusion matrices, ROC comparison, metrics tables, and a
   summary Word document.

Models: RF | XGBoost | CatBoost | DNN | Ensemble_mean

Outputs → ML_Results/
  classified_maps/
    FSM_{model}_binary.tif          (Int16, 0/1)
    FSM_{model}_5class.tif          (Int16, 1–5)
  validation/
    threshold_summary.csv
    validation_metrics.csv
    confusion_matrix_{model}.csv
    roc_validation.png
    metrics_comparison.png
    confusion_matrices.png
    csi_threshold_curves.png
    validation_report.docx
"""

# ── Config ────────────────────────────────────────────────────────────────────
# 06b writes probability maps per sample size into 06_Probability_Maps/<tag>/.
# This stage validates against the 10x test split, so read the 10x maps. Pointing at
# the parent directory would miss RF/XGBoost/CatBoost entirely and silently pick up
# the stale top-level 1x DNN map left by stage 05b. Override with MAP_SIZE=<tag>.
import os as _os
PROB_DIR    = _os.path.join(
    '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/06_Probability_Maps',
    _os.environ.get('MAP_SIZE', '10x'))
SAMPLES_DIR = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/03_Samples'
OUTPUT_DIR  = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
NODATA_IN   = -9999.0

CLASS_LABELS = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
STRIP_ROWS   = 300

import os, warnings
warnings.filterwarnings('ignore')
CLSF_DIR = os.path.join(OUTPUT_DIR, '07_Classified_Maps')
VALD_DIR = os.path.join(OUTPUT_DIR, '08_Validation')
for d in [CLSF_DIR, VALD_DIR]: os.makedirs(d, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.windows import Window
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, f1_score, accuracy_score,
    cohen_kappa_score
)

MODELS = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
COLORS = {'RF': '#2196F3', 'XGBoost': '#FF5722', 'CatBoost': '#4CAF50',
          'DNN': '#FF9800', 'Ensemble_mean': '#9C27B0'}

# ── Step 1: Load test samples + extract prob values from maps ─────────────────
print("=" * 65)
print("STEP 1 — Loading test samples & extracting map probabilities")
print("=" * 65)

df_test = pd.read_csv(os.path.join(SAMPLES_DIR, 'samples_test_10x.csv'))
y_true  = df_test['label'].values.astype(int)
xs      = df_test['x_3794'].values
ys      = df_test['y_3794'].values
print(f"Test samples (10x): {len(df_test):,}  |  Flood: {y_true.sum():,}  |  No-flood: {(y_true==0).sum():,}")

prob_maps = {}
for model in MODELS:
    path = os.path.join(PROB_DIR, f'FSM_{model}_probability.tif')
    with rasterio.open(path) as ds:
        tf = ds.transform
        # Convert coordinates to row/col
        cols = np.floor((xs - tf.c) / tf.a).astype(int)
        rows = np.floor((ys - tf.f) / tf.e).astype(int)
        # Clip to valid range
        rows = np.clip(rows, 0, ds.height - 1)
        cols = np.clip(cols, 0, ds.width  - 1)
        arr  = ds.read(1)
        probs = arr[rows, cols].astype(np.float64)
        probs[probs == NODATA_IN] = np.nan
    prob_maps[model] = probs
    valid = ~np.isnan(probs)
    print(f"  {model:<20}  extracted {valid.sum():,} valid probs  "
          f"| mean={probs[valid].mean():.4f}  min={probs[valid].min():.4f}  max={probs[valid].max():.4f}")

# Keep only rows where all models have valid probs
all_valid = np.all([~np.isnan(prob_maps[m]) for m in MODELS], axis=0)
print(f"\nSamples with all models valid: {all_valid.sum():,}")
y_val = y_true[all_valid]

# ── Step 2: Find optimal thresholds ──────────────────────────────────────────
print()
print("=" * 65)
print("STEP 2 — Optimal threshold search (CSI & Youden)")
print("=" * 65)

def find_optimal_threshold(y_true, y_prob):
    thresholds  = np.linspace(0.01, 0.99, 200)
    best_csi    = {'thresh': 0.5, 'val': 0.0}
    best_youden = {'thresh': 0.5, 'val': 0.0}
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        csi    = tp / (tp + fp + fn + 1e-10)
        youden = tp/(tp+fn+1e-10) + tn/(tn+fp+1e-10) - 1
        if csi    > best_csi['val']:    best_csi    = {'thresh': t, 'val': csi}
        if youden > best_youden['val']: best_youden = {'thresh': t, 'val': youden}
    return best_csi, best_youden

thresh_records = []
optimal_thresh = {}   # model → CSI-optimal threshold

print(f"\n  {'Model':<20} {'CSI thresh':>12} {'CSI val':>10} {'Youden thresh':>14} {'Youden val':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*14} {'-'*12}")
for model in MODELS:
    probs = prob_maps[model][all_valid]
    csi_r, youden_r = find_optimal_threshold(y_val, probs)
    optimal_thresh[model] = csi_r['thresh']
    thresh_records.append({
        'Model': model,
        'CSI_threshold': round(csi_r['thresh'], 4),
        'CSI_value':     round(csi_r['val'],    4),
        'Youden_threshold': round(youden_r['thresh'], 4),
        'Youden_value':     round(youden_r['val'],    4),
    })
    print(f"  {model:<20} {csi_r['thresh']:>12.4f} {csi_r['val']:>10.4f} "
          f"{youden_r['thresh']:>14.4f} {youden_r['val']:>12.4f}")

pd.DataFrame(thresh_records).to_csv(os.path.join(VALD_DIR, 'threshold_summary.csv'), index=False)
print("\nSaved: threshold_summary.csv")

# ── Step 3: Derive quantile breaks per model ──────────────────────────────────
print()
print("=" * 65)
print("STEP 3 — Computing quantile breaks for 5-class maps")
print("=" * 65)

quantile_breaks = {}    # model → [q20, q40, q60, q80]
for model in MODELS:
    path = os.path.join(PROB_DIR, f'FSM_{model}_probability.tif')
    # Read downsampled (1/10) to compute quantiles efficiently
    with rasterio.open(path) as ds:
        scale = 10
        arr_ds = ds.read(1, out_shape=(ds.height//scale, ds.width//scale),
                         resampling=rasterio.enums.Resampling.nearest)
    valid_pix = arr_ds[arr_ds != NODATA_IN].flatten()
    q = np.percentile(valid_pix, [20, 40, 60, 80])
    quantile_breaks[model] = q
    print(f"  {model:<20}  q20={q[0]:.4f}  q40={q[1]:.4f}  q60={q[2]:.4f}  q80={q[3]:.4f}")

# ── Step 4: Write classified rasters ─────────────────────────────────────────
print()
print("=" * 65)
print("STEP 4 — Writing binary & 5-class classified maps")
print("=" * 65)

def get_out_meta(prob_path, dtype, nodata):
    with rasterio.open(prob_path) as src:
        meta = src.meta.copy()
    meta.update(dtype=dtype, nodata=nodata,
                compress='lzw', tiled=True, blockxsize=512, blockysize=512)
    return meta

for model in MODELS:
    prob_path = os.path.join(PROB_DIR, f'FSM_{model}_probability.tif')
    bin_path  = os.path.join(CLSF_DIR, f'FSM_{model}_binary.tif')
    cls_path  = os.path.join(CLSF_DIR, f'FSM_{model}_5class.tif')
    thresh    = optimal_thresh[model]
    qb        = quantile_breaks[model]

    bin_meta  = get_out_meta(prob_path, 'int16', -9999)
    cls_meta  = get_out_meta(prob_path, 'int16', -9999)

    with rasterio.open(prob_path) as src, \
         rasterio.open(bin_path, 'w', **bin_meta) as bin_ds, \
         rasterio.open(cls_path, 'w', **cls_meta) as cls_ds:

        n_rows = src.height
        for row_start in range(0, n_rows, STRIP_ROWS):
            row_end = min(row_start + STRIP_ROWS, n_rows)
            win = Window(0, row_start, src.width, row_end - row_start)
            prob = src.read(1, window=win)
            nd_mask = prob == NODATA_IN

            # Binary
            binary = np.where(nd_mask, -9999, (prob >= thresh).astype(np.int16))
            bin_ds.write(binary.astype('int16'), 1, window=win)

            # 5-class
            cls = np.full_like(prob, -9999, dtype=np.int16)
            vld = ~nd_mask
            cls[vld & (prob < qb[0])]               = 1
            cls[vld & (prob >= qb[0]) & (prob < qb[1])] = 2
            cls[vld & (prob >= qb[1]) & (prob < qb[2])] = 3
            cls[vld & (prob >= qb[2]) & (prob < qb[3])] = 4
            cls[vld & (prob >= qb[3])]               = 5
            cls_ds.write(cls, 1, window=win)

    print(f"  {model:<20}  thresh={thresh:.4f}  → binary + 5-class saved")

# ── Step 5: Full validation metrics ──────────────────────────────────────────
print()
print("=" * 65)
print("STEP 5 — Validation metrics at optimal thresholds")
print("=" * 65)

def full_metrics(y_true, y_prob, thresh, model):
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Model':       model,
        'Threshold':   round(thresh, 4),
        'AUC':         round(roc_auc_score(y_true, y_prob), 4),
        'CSI':         round(tp/(tp+fp+fn+1e-10),           4),
        'F1':          round(f1_score(y_true, y_pred),      4),
        'Accuracy':    round(accuracy_score(y_true, y_pred),4),
        'Kappa':       round(cohen_kappa_score(y_true, y_pred), 4),
        'Sensitivity': round(tp/(tp+fn+1e-10),              4),
        'Specificity': round(tn/(tn+fp+1e-10),              4),
        'Precision':   round(tp/(tp+fp+1e-10),              4),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }

val_results = []
cm_data     = {}
print(f"\n  {'Model':<20} {'Thresh':>8} {'AUC':>7} {'F1':>7} {'CSI':>7} {'Sens':>7} {'Spec':>7} {'Kappa':>7}")
print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for model in MODELS:
    probs = prob_maps[model][all_valid]
    t     = optimal_thresh[model]
    m     = full_metrics(y_val, probs, t, model)
    val_results.append(m)
    cm_data[model] = confusion_matrix(y_val, (probs >= t).astype(int))
    print(f"  {model:<20} {t:>8.4f} {m['AUC']:>7.4f} {m['F1']:>7.4f} {m['CSI']:>7.4f} "
          f"{m['Sensitivity']:>7.4f} {m['Specificity']:>7.4f} {m['Kappa']:>7.4f}")

val_df = pd.DataFrame(val_results).sort_values('AUC', ascending=False)
val_df.to_csv(os.path.join(VALD_DIR, 'validation_metrics.csv'), index=False)
print("\nSaved: validation_metrics.csv")

for model, cm in cm_data.items():
    pd.DataFrame(cm, index=['True 0','True 1'], columns=['Pred 0','Pred 1'])\
      .to_csv(os.path.join(VALD_DIR, f'confusion_matrix_{model}.csv'))

# ── Step 6: Plots ─────────────────────────────────────────────────────────────
print()
print("STEP 6 — Generating plots")

# 6a. ROC curves
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1],'k--', lw=1.2, alpha=0.5, label='Random chance')
for model in MODELS:
    probs = prob_maps[model][all_valid]
    fpr, tpr, _ = roc_curve(y_val, probs)
    auc_val = auc(fpr, tpr)
    t = optimal_thresh[model]
    # mark optimal threshold point
    pred_t = (probs >= t).astype(int)
    tn,fp,fn,tp_ = confusion_matrix(y_val, pred_t).ravel()
    sens = tp_/(tp_+fn+1e-10); spec = tn/(tn+fp+1e-10)
    ax.plot(fpr, tpr, lw=2.2, color=COLORS[model],
            label=f'{model}  (AUC={auc_val:.4f}, t={t:.3f})')
    ax.plot(1-spec, sens, 'o', color=COLORS[model], ms=9, zorder=5)
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=15)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=15)
ax.set_title('ROC Curves — Validation with Optimal Threshold Points', fontsize=16, fontweight='bold')
ax.tick_params(labelsize=13)
ax.legend(loc='lower right', fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'roc_validation.png'), dpi=200, bbox_inches='tight')
plt.close()

# 6b. Metrics bar comparison
met_cols = ['AUC','F1','CSI','Sensitivity','Specificity','Kappa','Accuracy']
x = np.arange(len(met_cols)); w = 0.15
fig, ax = plt.subplots(figsize=(16, 6))
for i, (_, row) in enumerate(val_df.iterrows()):
    vals   = [row[m] for m in met_cols]
    offset = (i - 2) * w
    bars   = ax.bar(x + offset, vals, w, label=row['Model'],
                    color=COLORS[row['Model']], alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=50)
ax.set_xticks(x); ax.set_xticklabels(met_cols, fontsize=11)
ax.set_ylim(0, 1.18); ax.set_ylabel('Score', fontsize=12)
ax.set_title('Validation Performance at CSI-Optimal Thresholds', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'metrics_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

# 6c. Confusion matrices (2×3 grid, last panel empty)
# Canvas kept close to the manuscript's actual print width (~5.4in at
# width=\linewidth) so that print-size text does not require fonts far
# larger than the rest of the figure grammar.
fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
axes_flat = axes.flatten()
for i, model in enumerate(MODELS):
    ax = axes_flat[i]
    cm = cm_data[model]
    cm_pct = cm.astype(float) / cm.sum() * 100
    annot  = np.array([[f'{cm[r,c]}\n({cm_pct[r,c]:.1f}%)' for c in range(2)] for r in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=['No Flood','Flood'], yticklabels=['No Flood','Flood'],
                linewidths=1, linecolor='white', cbar=False,
                annot_kws={'fontsize': 13})
    t  = optimal_thresh[model]
    m_ = val_df[val_df['Model']==model].iloc[0]
    ax.set_title(f'{model}\nAUC={m_["AUC"]:.4f}  CSI={m_["CSI"]:.4f}  t={t:.3f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=13); ax.set_ylabel('Actual', fontsize=13)
    ax.tick_params(labelsize=12)
axes_flat[-1].set_visible(False)   # hide unused 6th panel
plt.suptitle('Confusion Matrices — CSI-Optimal Thresholds', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'confusion_matrices.png'), dpi=200, bbox_inches='tight')
plt.close()

# 6d. Threshold vs CSI curves (2×3 grid, last panel empty)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()
threshs = np.linspace(0.01, 0.99, 200)
for i, model in enumerate(MODELS):
    ax = axes_flat[i]
    probs = prob_maps[model][all_valid]
    csi_vals = []
    for t in threshs:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp_ = confusion_matrix(y_val, pred).ravel()
        csi_vals.append(tp_ / (tp_ + fp + fn + 1e-10))
    opt_t = optimal_thresh[model]
    opt_c = max(csi_vals)
    ax.plot(threshs, csi_vals, color=COLORS[model], lw=2)
    ax.axvline(opt_t, color='red', ls='--', lw=1.5, label=f'Optimal t={opt_t:.3f}')
    ax.plot(opt_t, opt_c, 'ro', ms=8)
    ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('CSI', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
axes_flat[-1].set_visible(False)
plt.suptitle('CSI vs Threshold Curves', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(VALD_DIR, 'csi_threshold_curves.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: roc_validation.png, metrics_comparison.png, confusion_matrices.png, csi_threshold_curves.png")

# ── Step 7: Word report ───────────────────────────────────────────────────────
print()
print("STEP 7 — Generating validation report (Word)")

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def shd_cell(cell, hex_fill, font_color=(255,255,255)):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),'clear'); shd.set(qn('w:color'),'auto')
    shd.set(qn('w:fill'), hex_fill); tcPr.append(shd)
    for run in cell.paragraphs[0].runs:
        run.font.color.rgb = RGBColor(*font_color)

def add_table_from_df(doc, df, header_color='1F497D'):
    tbl = doc.add_table(rows=1+len(df), cols=len(df.columns))
    tbl.style = 'Table Grid'
    hdr = tbl.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr[i].text = str(col)
        hdr[i].paragraphs[0].runs[0].bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(9)
        shd_cell(hdr[i], header_color)
    for r, (_, row) in enumerate(df.iterrows()):
        cells = tbl.rows[r+1].cells
        for c, val in enumerate(row):
            cells[c].text = str(val)
            cells[c].paragraphs[0].runs[0].font.size = Pt(9)
            if r % 2 == 0:
                shd_cell(cells[c], 'DEEAF1', (0,0,0))
    return tbl

doc = Document()
s = doc.styles['Normal']; s.font.name = 'Calibri'; s.font.size = Pt(11)

# Title
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('Flood Susceptibility Mapping — Threshold Optimisation & Validation Report')
r.bold = True; r.font.size = Pt(17); r.font.color.rgb = RGBColor(0x1F,0x49,0x7D)
p2 = doc.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.add_run('Slovenia | EPSG:3794 | 12.5 m | 10x sample size (500k/class) | RF · XGBoost · CatBoost · DNN · Ensemble').font.size = Pt(11)
doc.add_paragraph()

# Section 1 - Thresholds
h = doc.add_paragraph(); h.add_run('1. Optimal Thresholds').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)
doc.add_paragraph(
    'The CSI-optimal threshold minimises false alarms and misses simultaneously, '
    'making it the preferred choice for flood hazard mapping. Youden\'s J maximises '
    'the sum of sensitivity and specificity. Optimal threshold points are marked on '
    'the ROC curves.'
)
add_table_from_df(doc, pd.DataFrame(thresh_records).round(4))
doc.add_paragraph()

# Section 2 - Validation
h = doc.add_paragraph(); h.add_run('2. Validation Metrics (Test Set)').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)
doc.add_paragraph(
    f'Metrics computed on {all_valid.sum():,} test samples (10x split, held out from training) '
    f'at CSI-optimal thresholds. AUC is threshold-independent; all other metrics use the optimal threshold.'
)
add_table_from_df(doc,
    val_df[['Model','Threshold','AUC','CSI','F1','Accuracy','Kappa','Sensitivity','Specificity','Precision']].round(4))
doc.add_paragraph()

# Section 3 - Confusion matrices (text)
h = doc.add_paragraph(); h.add_run('3. Confusion Matrix Summary').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)
for _, row in val_df.iterrows():
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(f"{row['Model']}").bold = True
    p.add_run(f"  TP={row['TP']}  TN={row['TN']}  FP={row['FP']}  FN={row['FN']}  "
              f"(Threshold = {row['Threshold']})").font.size = Pt(10)
doc.add_paragraph()

# Section 4 - Class map description
h = doc.add_paragraph(); h.add_run('4. Susceptibility Classification (5-Class Maps)').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)
doc.add_paragraph(
    'Five flood susceptibility classes were derived by quantile classification (equal-frequency) '
    'of the probability values over all valid pixels within Slovenia. Each class contains ~20% '
    'of the valid area.'
)
cls_rows = []
for model in MODELS:
    qb = quantile_breaks[model]
    cls_rows.append({'Model': model,
        'Very Low (<q20)': f'<{qb[0]:.4f}',
        'Low (q20–q40)':   f'{qb[0]:.4f}–{qb[1]:.4f}',
        'Moderate (q40–q60)': f'{qb[1]:.4f}–{qb[2]:.4f}',
        'High (q60–q80)':  f'{qb[2]:.4f}–{qb[3]:.4f}',
        'Very High (>q80)': f'>{qb[3]:.4f}',
    })
add_table_from_df(doc, pd.DataFrame(cls_rows))
doc.add_paragraph()

# Section 5 - Output files
h = doc.add_paragraph(); h.add_run('5. Output Files').bold = True
h.runs[0].font.size = Pt(14); h.runs[0].font.color.rgb = RGBColor(0x1F,0x49,0x7D)
out_items = [
    ('classified_maps/', 'FSM_{model}_binary.tif',  'Binary flood map (0/1) at CSI-optimal threshold — Int16'),
    ('classified_maps/', 'FSM_{model}_5class.tif',  '5-class susceptibility map (1=Very Low … 5=Very High) — Int16'),
    ('validation/',      'validation_metrics.csv',  'Full metrics for all models'),
    ('validation/',      'threshold_summary.csv',   'CSI & Youden optimal thresholds per model'),
    ('validation/',      'roc_validation.png',      'ROC curves with optimal threshold points marked'),
    ('validation/',      'confusion_matrices.png',  'Confusion matrix grids for all models'),
    ('validation/',      'csi_threshold_curves.png','CSI vs threshold curves per model'),
    ('validation/',      'metrics_comparison.png',  'Bar chart comparing all metrics across models'),
]
tbl = doc.add_table(rows=1+len(out_items), cols=3)
tbl.style = 'Table Grid'
for i, h_txt in enumerate(['Folder','File','Description']):
    tbl.rows[0].cells[i].text = h_txt
    tbl.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    tbl.rows[0].cells[i].paragraphs[0].runs[0].font.size = Pt(9)
    shd_cell(tbl.rows[0].cells[i], '2E74B5')
for r, (folder, fname, desc) in enumerate(out_items):
    for c, txt in enumerate([folder, fname, desc]):
        tbl.rows[r+1].cells[c].text = txt
        tbl.rows[r+1].cells[c].paragraphs[0].runs[0].font.size = Pt(9)
        if r % 2 == 0:
            shd_cell(tbl.rows[r+1].cells[c], 'DEEAF1', (0,0,0))

doc.add_paragraph()
fp = doc.add_paragraph(); fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fp.add_run('Generated: 2026-05-25  |  Project: FSM Slovenia  |  Models: 10x (500k/class)').font.size = Pt(9)

report_path = os.path.join(VALD_DIR, 'validation_report.docx')
doc.save(report_path)
print(f"  Saved: validation_report.docx")

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\n{'Model':<20} {'Thresh':>8} {'AUC':>7} {'CSI':>7} {'F1':>7} {'Kappa':>7}")
print(f"{'-'*20} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for _, row in val_df.iterrows():
    print(f"{row['Model']:<20} {row['Threshold']:>8.4f} {row['AUC']:>7.4f} "
          f"{row['CSI']:>7.4f} {row['F1']:>7.4f} {row['Kappa']:>7.4f}")

print(f"\n5-class quantile breaks:")
for model in MODELS:
    q = quantile_breaks[model]
    print(f"  {model:<20}  {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {q[3]:.4f}")

print(f"\nClassified maps : {CLSF_DIR}")
print(f"Validation      : {VALD_DIR}")
