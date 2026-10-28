"""
Summary Report — Slovenia Flood Susceptibility Mapping
=======================================================
Generates a comprehensive Word (.docx) report covering the full
analysis pipeline: data, methods, multi-scale ML results, spatial CV,
flood area statistics, and SHAP explainability.

Output → ML_Results/FSM_Slovenia_Summary_Report.docx
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import date

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
OUT    = os.path.join(BASE, 'FSM_Slovenia_Summary_Report.docx')

VAL_DIR   = os.path.join(BASE, '08_Validation')
SCV_DIR   = os.path.join(BASE, '09_Analysis/spatial_cv')
SHAP_DIR  = os.path.join(BASE, '09_Analysis/shap')
ANA_DIR   = os.path.join(BASE, '09_Analysis')

SIZES  = ['1x', '2x', '3x', '4x', '10x']
MODELS = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']

# ── Load data ─────────────────────────────────────────────────────────────────
val_df  = pd.read_csv(os.path.join(VAL_DIR, 'multiscale_validation.csv'))
cv_df   = pd.read_csv(os.path.join(SCV_DIR, 'multiscale_cv_summary.csv'))
shap_df = pd.read_csv(os.path.join(SHAP_DIR, 'shap_importance_all_sizes.csv'))
bin_df  = pd.read_csv(os.path.join(ANA_DIR,  'binary_flood_statistics.csv'))
cls_df  = pd.read_csv(os.path.join(ANA_DIR,  'class_area_statistics.csv'))

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_col_width(table, col_idx, width_cm):
    for row in table.rows:
        row.cells[col_idx].width = Cm(width_cm)

def shade_cell(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.runs[0].font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)
    return h

def add_figure(doc, img_path, caption, width=6.0):
    if not os.path.exists(img_path):
        doc.add_paragraph(f'[Figure not found: {os.path.basename(img_path)}]')
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()

def header_row(table, values, bg='1A237E'):
    row = table.rows[0]
    for i, v in enumerate(values):
        cell = row.cells[i]
        cell.text = v
        shade_cell(cell, bg)
        run = cell.paragraphs[0].runs[0]
        run.font.bold  = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size  = Pt(9)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

def data_row(table, row_idx, values, bold_col=None, bg=None):
    row = table.rows[row_idx]
    for i, v in enumerate(values):
        cell = row.cells[i]
        cell.text = str(v)
        run  = cell.paragraphs[0].runs[0]
        run.font.size = Pt(9)
        if bold_col is not None and i == bold_col:
            run.font.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        if bg:
            shade_cell(cell, bg)

# ── Document ──────────────────────────────────────────────────────────────────
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── Title page ────────────────────────────────────────────────────────────────
doc.add_paragraph()
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
t_run = title.add_run('Flood Susceptibility Mapping of Slovenia')
t_run.bold = True
t_run.font.size = Pt(22)
t_run.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
s_run = sub.add_run('Multi-Scale Machine Learning Analysis — Summary Report')
s_run.font.size = Pt(14)
s_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_paragraph()
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run(f'Date: {date.today().strftime("%d %B %Y")}').font.size = Pt(11)
doc.add_paragraph()
meta2 = doc.add_paragraph()
meta2.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta2.add_run('Resolution: 12.5 m  |  CRS: EPSG:3794  |  Study area: ~20,254 km²').font.size = Pt(10)

doc.add_page_break()

# ── 1. Executive Summary ──────────────────────────────────────────────────────
add_heading(doc, '1. Executive Summary')
doc.add_paragraph(
    'This report presents a comprehensive flood susceptibility mapping (FSM) study for Slovenia '
    'using four machine learning (ML) models — Random Forest (RF), XGBoost, CatBoost, and a '
    'Deep Neural Network (DNN) — trained across five sample sizes (1x to 10x, corresponding to '
    '50,000–500,000 samples per class). The study evaluates model performance through both '
    'standard random-split validation and rigorous 5-fold spatial block cross-validation (50 km '
    'blocks) to assess true spatial generalisation. Explainability was addressed using SHAP '
    '(SHapley Additive exPlanations) for all models and sample sizes. Key findings include:'
)
bullets = [
    'XGBoost consistently achieved the highest standard validation performance across all sample sizes '
    '(AUC up to 0.994, CSI up to 0.940 at 10x).',
    'Spatial CV revealed that RF is the most spatially robust model (AUC ≈ 0.965, CSI ≈ 0.830 '
    'across all sizes), indicating superior generalisation to unseen spatial regions.',
    'Performance improved steadily from 1x to 10x training samples in standard validation, '
    'but spatial CV metrics plateaued — suggesting spatial autocorrelation is the dominant '
    'limiting factor rather than sample size.',
    'The ensemble mean model provides a reliable consensus flood susceptibility map, predicting '
    '~2,176 km² of flood-prone area (10.7% of Slovenia) at the CSI-optimal threshold.',
    'SHAP analysis consistently identified Soil Type, HAND (Height Above Nearest Drainage), '
    'Elevation, TRI (Terrain Ruggedness Index), and Lithology as the most influential '
    'conditioning factors across all models and sample sizes.',
]
for b in bullets:
    p = doc.add_paragraph(b, style='List Bullet')
    p.runs[0].font.size = Pt(10)

doc.add_page_break()

# ── 2. Study Area & Data ──────────────────────────────────────────────────────
add_heading(doc, '2. Study Area & Data')

add_heading(doc, '2.1 Study Area', level=2)
doc.add_paragraph(
    'The study covers the entire territory of Slovenia (approximately 20,254 km²) using a '
    '12.5 m spatial resolution raster grid aligned to the national coordinate reference system '
    'EPSG:3794 (Transverse Mercator, Bessel 1841 ellipsoid). The total analysis grid comprises '
    'approximately 129.6 million valid pixels.'
)

add_heading(doc, '2.2 Conditioning Factors', level=2)
doc.add_paragraph(
    'Nineteen flood conditioning factors were selected after VIF-based collinearity analysis '
    'from an initial set of terrain, hydrological, soil, and climatic variables:'
)

features = [
    ('Distance to River',    'Proximity to stream network (m)'),
    ('Distance to Road',     'Proximity to road network (m)'),
    ('Hydrogeological Soil', 'Grouped hydrogeological soil classes'),
    ('Soil Type',            'Pedological soil classification (grouped)'),
    ('River Density',        'Stream density per unit area'),
    ('Aspect',               'Slope facing direction (°)'),
    ('Drainage Density',     'Total channel length / catchment area'),
    ('Elevation',            'Digital elevation model (m a.s.l.)'),
    ('Flow Accumulation',    'Upstream contributing area (cells)'),
    ('HAND',                 'Height Above Nearest Drainage (m)'),
    ('Lithology',            'Grouped lithological classes'),
    ('LS Factor',            'Slope length and steepness factor'),
    ('Plan Curvature',       'Horizontal curvature of slope'),
    ('Profile Curvature',    'Vertical curvature of slope'),
    ('Rainfall P90',         'Annual max daily precipitation 90th percentile (mm)'),
    ('SPI',                  'Stream Power Index'),
    ('TRI',                  'Terrain Ruggedness Index'),
    ('TWI',                  'Topographic Wetness Index'),
    ('Valley Depth',         'Depth below surrounding terrain (m)'),
]

tbl = doc.add_table(rows=len(features)+1, cols=3)
tbl.style = 'Table Grid'
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl, ['#', 'Factor', 'Description'])
for i, (name, desc) in enumerate(features):
    bg = 'F5F5F5' if i % 2 == 0 else 'FFFFFF'
    data_row(tbl, i+1, [str(i+1), name, desc], bg=bg)
    tbl.rows[i+1].cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
doc.add_paragraph()

add_heading(doc, '2.3 Training Samples', level=2)
sample_info = [
    ('1x',  '50,000',  '100,000',  '9,957',   '19,957'),
    ('2x',  '100,000', '200,000',  '9,957',   '39,914'),
    ('3x',  '150,000', '300,000',  '9,957',   '59,871'),
    ('4x',  '200,000', '400,000',  '9,957',   '79,828'),
    ('10x', '500,000', '1,000,000','99,575',  '199,495'),
]
tbl2 = doc.add_table(rows=6, cols=5)
tbl2.style = 'Table Grid'
tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl2, ['Size', 'Train/class', 'Total train', 'Test/class', 'Total test'])
for i, row in enumerate(sample_info):
    bg = 'F5F5F5' if i % 2 == 0 else 'FFFFFF'
    data_row(tbl2, i+1, row, bg=bg)
doc.add_paragraph(
    'Flood and non-flood samples were drawn in equal proportions (balanced classes). '
    'Test sets are held-out spatially distributed samples not used during training.'
)

doc.add_page_break()

# ── 3. Methodology ────────────────────────────────────────────────────────────
add_heading(doc, '3. Methodology')

add_heading(doc, '3.1 Machine Learning Models', level=2)
model_desc = [
    ('Random Forest (RF)',
     'Ensemble of 500 decision trees (max depth 25, min samples leaf 3, balanced class weight). '
     'Feature importance derived from mean decrease in impurity.'),
    ('XGBoost',
     'Gradient-boosted trees (1,000 estimators, max depth 8, learning rate 0.05, GPU hist method). '
     'Early stopping on validation AUC (patience = 50).'),
    ('CatBoost',
     'Gradient boosting with native categorical handling (1,000 iterations, depth 8, '
     'learning rate 0.05, GPU). Early stopping on validation AUC (patience = 50).'),
    ('Deep Neural Network (DNN)',
     'Fully-connected MLP: 128 → 64 → 32 → 16 → 1 with BatchNorm, ReLU, Dropout(0.3). '
     'Trained with BCEWithLogitsLoss, Adam optimiser, CosineAnnealingLR, early stopping '
     'on validation AUC (patience = 20, max 200 epochs, batch size 4,096).'),
    ('Ensemble (mean)',
     'Simple unweighted average of the four individual model probability outputs.'),
]
for name, desc in model_desc:
    p = doc.add_paragraph()
    run = p.add_run(f'{name}: ')
    run.bold = True
    run.font.size = Pt(10)
    run2 = p.add_run(desc)
    run2.font.size = Pt(10)

add_heading(doc, '3.2 Validation Strategy', level=2)
doc.add_paragraph(
    'Two complementary validation approaches were applied:'
)
val_methods = [
    ('Standard random-split validation',
     'Models evaluated on a held-out test set (10% of total samples per size). '
     'Metrics: AUC-ROC, Average Precision, F1, Accuracy, Cohen\'s Kappa, CSI, Sensitivity, Specificity.'),
    ('5-fold spatial block cross-validation',
     'Samples partitioned into 5 folds by 50 km spatial blocks (checkerboard assignment). '
     'Each fold trains from scratch on 4 blocks, tests on the held-out spatial block. '
     'This evaluates true spatial generalisation beyond random-split optimism.'),
]
for name, desc in val_methods:
    p = doc.add_paragraph()
    p.add_run(f'{name}: ').bold = True
    p.runs[0].font.size = Pt(10)
    p.add_run(desc).font.size = Pt(10)

add_heading(doc, '3.3 Threshold Optimisation & Map Classification', level=2)
doc.add_paragraph(
    'Binary flood/no-flood maps were generated using the CSI-optimal threshold '
    '(maximising TP / (TP + FP + FN)) found by grid search over 200 threshold values. '
    'Five-class susceptibility maps (Very Low → Very High) were generated using equal-frequency '
    'quantile breaks (Q20, Q40, Q60, Q80) computed from the full probability map, ensuring '
    'approximately equal area per class (~20% each).'
)

add_heading(doc, '3.4 SHAP Explainability', level=2)
doc.add_paragraph(
    'SHAP (SHapley Additive exPlanations) was applied to all four models at each sample size. '
    'TreeExplainer (tree_path_dependent) was used for RF and CatBoost; native Booster '
    'pred_contribs for XGBoost (to avoid XGBoost 3.x / shap compatibility issues); and '
    'GradientExplainer for the DNN. Sample sizes: RF (1,000), XGBoost/CatBoost (3,000), '
    'DNN (1,000 test + 300 background). Outputs include beeswarm plots, bar plots, '
    'dependence plots for the top-3 features, and cross-size rank stability analysis.'
)

doc.add_page_break()

# ── 4. Standard Validation Results ───────────────────────────────────────────
add_heading(doc, '4. Standard Validation Results')

add_heading(doc, '4.1 Performance Across Sample Sizes', level=2)
doc.add_paragraph(
    'Table 1 presents the key validation metrics for all models across the five sample sizes. '
    'XGBoost consistently achieves the highest AUC and CSI. Performance improves with larger '
    'training sets, with the most significant gains between 1x and 2x.'
)

# Build table — best metric in each row bold
cols = ['Size', 'Model', 'AUC', 'Avg Prec', 'F1', 'Kappa', 'CSI', 'Sens.', 'Spec.']
n_rows = len(val_df) + 1
tbl3 = doc.add_table(rows=n_rows, cols=len(cols))
tbl3.style = 'Table Grid'
tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl3, cols)

val_sorted = val_df.sort_values(['SampleSize', 'AUC'], ascending=[True, False])
size_order  = {s: i for i, s in enumerate(SIZES)}
val_sorted['_ord'] = val_sorted['SampleSize'].map(size_order)
val_sorted = val_sorted.sort_values(['_ord', 'AUC'], ascending=[True, False])

prev_size = None
row_colors = {'1x':'EEF2FF','2x':'E8F5E9','3x':'FFF8E1','4x':'FCE4EC','10x':'E3F2FD'}
for i, (_, r) in enumerate(val_sorted.iterrows()):
    bg = row_colors.get(r['SampleSize'], 'FFFFFF')
    sz = r['SampleSize'] if r['SampleSize'] != prev_size else ''
    prev_size = r['SampleSize']
    vals = [sz, r['Model'],
            f"{r['AUC']:.4f}", f"{r['Avg_Prec']:.4f}", f"{r['F1']:.4f}",
            f"{r['Kappa']:.4f}", f"{r['CSI']:.4f}", f"{r['Sensitivity']:.4f}", f"{r['Specificity']:.4f}"]
    data_row(tbl3, i+1, vals, bg=bg)

p = doc.add_paragraph()
p.add_run('Table 1. ').bold = True
p.add_run('Standard validation metrics for all models and sample sizes. Rows sorted by AUC within each sample size group. Colour bands distinguish sample sizes.').italic = True
p.runs[0].font.size = Pt(9); p.runs[1].font.size = Pt(9)

doc.add_paragraph()
add_figure(doc,
    os.path.join(VAL_DIR, 'multiscale_comparison.png'),
    'Figure 1. Effect of sample size on validation performance (AUC, CSI, F1, Kappa) for all four models (1x–10x). '
    'Filled markers = standard validation; values annotated at each point.',
    width=6.2)

add_heading(doc, '4.2 Best Model Performance (10x)', level=2)
doc.add_paragraph(
    'At the largest sample size (10x, 500k/class), XGBoost achieved the best overall performance '
    '(AUC = 0.9938, CSI = 0.9399, F1 = 0.9690). The ensemble model provided the second-best '
    'results with strong agreement across individual models.'
)
add_figure(doc,
    os.path.join(VAL_DIR, '10x', 'roc_validation.png'),
    'Figure 2. ROC curves for all five models at 10x sample size. '
    'Filled circles indicate operating points at CSI-optimal thresholds.',
    width=5.5)

add_figure(doc,
    os.path.join(VAL_DIR, '10x', 'confusion_matrices.png'),
    'Figure 3. Confusion matrices for all five models at 10x sample size '
    '(CSI-optimal thresholds). Absolute counts and row-normalised percentages shown.',
    width=6.5)

add_figure(doc,
    os.path.join(VAL_DIR, '10x', 'metrics_comparison.png'),
    'Figure 4. Bar chart comparison of AUC, F1, CSI, Sensitivity, Specificity and Kappa '
    'for all five models at 10x sample size.',
    width=6.5)

doc.add_page_break()

# ── 5. Spatial Cross-Validation ───────────────────────────────────────────────
add_heading(doc, '5. Spatial Cross-Validation')
doc.add_paragraph(
    'Spatial block CV tests whether models generalise to geographic regions not seen during '
    'training — a critical evaluation for spatial prediction tasks where standard random-split '
    'validation is optimistic due to spatial autocorrelation.'
)

add_heading(doc, '5.1 Spatial CV Results — All Sizes', level=2)
cv_cols = ['Size', 'Model', 'AUC (mean±std)', 'CSI (mean±std)', 'F1 (mean±std)', 'Kappa (mean±std)']
cv_sorted = cv_df.copy()
cv_sorted['_ord'] = cv_sorted['SampleSize'].map(size_order)
cv_sorted = cv_sorted.sort_values(['_ord', 'AUC_mean'], ascending=[True, False])

tbl4 = doc.add_table(rows=len(cv_sorted)+1, cols=len(cv_cols))
tbl4.style = 'Table Grid'
tbl4.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl4, cv_cols)

prev_size = None
for i, (_, r) in enumerate(cv_sorted.iterrows()):
    bg = row_colors.get(r['SampleSize'], 'FFFFFF')
    sz = r['SampleSize'] if r['SampleSize'] != prev_size else ''
    prev_size = r['SampleSize']
    vals = [sz, r['Model'],
            f"{r['AUC_mean']:.4f}±{r['AUC_std']:.4f}",
            f"{r['CSI_mean']:.4f}±{r['CSI_std']:.4f}",
            f"{r['F1_mean']:.4f}±{r['F1_std']:.4f}",
            f"{r['Kappa_mean']:.4f}±{r['Kappa_std']:.4f}"]
    data_row(tbl4, i+1, vals, bg=bg)

p = doc.add_paragraph()
p.add_run('Table 2. ').bold = True
p.add_run('5-fold spatial block CV results (mean ± std) for all models and sample sizes.').italic = True
p.runs[0].font.size = Pt(9); p.runs[1].font.size = Pt(9)

doc.add_paragraph()
add_figure(doc,
    os.path.join(SCV_DIR, '10x', 'fold_map.png'),
    'Figure 5. Spatial fold assignment for 10x sample size. '
    'Five folds defined by 50 km × 50 km spatial blocks (checkerboard pattern).',
    width=5.5)

add_figure(doc,
    os.path.join(SCV_DIR, '10x', 'spatial_cv_comparison.png'),
    'Figure 6. Comparison of random-split validation (hatched bars) vs 5-fold spatial CV '
    '(solid bars ± std) for all four models at 10x sample size. '
    'Spatial CV metrics are consistently lower, revealing overfitting to spatial autocorrelation.',
    width=6.5)

add_figure(doc,
    os.path.join(SCV_DIR, '10x', 'per_fold_performance.png'),
    'Figure 7. Per-fold AUC, CSI, F1 and Kappa for all models at 10x. '
    'Fold 5 (low flood fraction, ~22%) shows reduced CSI across all models.',
    width=6.2)

add_heading(doc, '5.2 Key Findings from Spatial CV', level=2)
doc.add_paragraph(
    'Random Forest showed the most consistent spatial generalisation across all sample sizes '
    '(AUC ≈ 0.965, CSI ≈ 0.828–0.830), with the lowest standard deviation across folds. '
    'XGBoost and CatBoost demonstrated higher random-split performance but greater variance '
    'in spatial CV, suggesting some degree of spatial overfitting. The DNN showed stable '
    'spatial CV performance (AUC ≈ 0.963 at 10x), competitive with RF. '
    'Critically, spatial CV metrics did not improve beyond 4x training size, indicating '
    'that data quantity is not the bottleneck — spatial autocorrelation structure is.'
)

doc.add_page_break()

# ── 6. Flood Susceptibility Maps ──────────────────────────────────────────────
add_heading(doc, '6. Flood Susceptibility Maps & Area Statistics')

add_heading(doc, '6.1 Five-Class Susceptibility Maps', level=2)
doc.add_paragraph(
    'Five-class susceptibility maps (Very Low → Very High) were generated for all models at each '
    'sample size using equal-frequency quantile breaks (Q20/Q40/Q60/Q80) computed from the full '
    'probability raster. This ensures each class covers approximately 20% of the valid study area.'
)

# 5-class area table (10x)
cls_cols = ['Model', 'Very Low (km²)', 'Low (km²)', 'Moderate (km²)', 'High (km²)', 'Very High (km²)']
tbl5 = doc.add_table(rows=6, cols=len(cls_cols))
tbl5.style = 'Table Grid'
tbl5.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl5, cls_cols)
for i, model in enumerate(MODELS):
    sub = cls_df[cls_df.Model == model].sort_values('Class')
    bg  = row_colors.get('10x', 'E3F2FD')
    vals = [model] + [f"{r['Area_km2']:,.1f}\n({r['Pct']:.1f}%)" for _, r in sub.iterrows()]
    data_row(tbl5, i+1, vals, bg=bg)
p = doc.add_paragraph()
p.add_run('Table 3. ').bold = True
p.add_run('Five-class flood susceptibility area statistics (10x models). '
          'Each class covers approximately 20% of the valid study area (equal-frequency).').italic = True
p.runs[0].font.size = Pt(9); p.runs[1].font.size = Pt(9)

doc.add_paragraph()
add_figure(doc,
    os.path.join(ANA_DIR, 'class_area_comparison.png'),
    'Figure 8. Stacked bar chart of five-class susceptibility areas per model (10x). '
    'Near-equal class areas confirm well-calibrated equal-frequency quantile breaks.',
    width=6.2)

add_heading(doc, '6.2 Binary Flood Maps', level=2)
doc.add_paragraph(
    'Binary flood/no-flood maps were generated at the CSI-optimal threshold for each model. '
    'Table 4 summarises predicted flood areas across all five models.'
)

bin_cols = ['Model', 'CSI Threshold', 'Flood Area (km²)', 'Flood (%)', 'No-Flood (km²)']
tbl6 = doc.add_table(rows=6, cols=len(bin_cols))
tbl6.style = 'Table Grid'
tbl6.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl6, bin_cols)
for i, (_, r) in enumerate(bin_df.iterrows()):
    bg = row_colors.get('10x', 'E3F2FD')
    vals = [r['Model'], f"{r['CSI_Threshold']:.4f}",
            f"{r['Flood_km2']:,.1f}", f"{r['Flood_pct']:.2f}%", f"{r['NoFlood_km2']:,.1f}"]
    data_row(tbl6, i+1, vals, bg=bg)
p = doc.add_paragraph()
p.add_run('Table 4. ').bold = True
p.add_run('Binary flood area statistics for 10x models at CSI-optimal thresholds. '
          'Total valid study area: ~20,254 km².').italic = True
p.runs[0].font.size = Pt(9); p.runs[1].font.size = Pt(9)

doc.add_paragraph()
add_figure(doc,
    os.path.join(ANA_DIR, 'flood_area_comparison.png'),
    'Figure 9. Predicted flood area (km²) and percentage per model (10x) at CSI-optimal thresholds.',
    width=6.2)

doc.add_page_break()

# ── 7. SHAP Feature Importance ────────────────────────────────────────────────
add_heading(doc, '7. SHAP Feature Importance')

add_heading(doc, '7.1 Top Features per Model (10x)', level=2)
doc.add_paragraph(
    'SHAP mean absolute values were computed for all 19 conditioning factors. '
    'Table 5 lists the top five features per model at the 10x sample size.'
)

shap10 = shap_df[shap_df.SampleSize == '10x']
top_models = ['RF', 'XGBoost', 'CatBoost', 'DNN']
shap_cols = ['Rank'] + top_models
tbl7 = doc.add_table(rows=6, cols=len(shap_cols))
tbl7.style = 'Table Grid'
tbl7.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl7, shap_cols)

tops = {}
for m in top_models:
    sub = shap10[shap10.Model == m].sort_values('MeanAbsSHAP', ascending=False).head(5)
    tops[m] = [(r['Feature'], r['MeanAbsSHAP']) for _, r in sub.iterrows()]

for rank in range(5):
    bg = 'F5F5F5' if rank % 2 == 0 else 'FFFFFF'
    vals = [str(rank+1)] + [f"{tops[m][rank][0]}\n({tops[m][rank][1]:.4f})" for m in top_models]
    data_row(tbl7, rank+1, vals, bg=bg)

p = doc.add_paragraph()
p.add_run('Table 5. ').bold = True
p.add_run('Top 5 SHAP features per model at 10x sample size. Values in parentheses are mean |SHAP|.').italic = True
p.runs[0].font.size = Pt(9); p.runs[1].font.size = Pt(9)

doc.add_paragraph()
add_figure(doc,
    os.path.join(SHAP_DIR, '10x', 'shap_importance_comparison.png'),
    'Figure 10. Multi-model SHAP feature importance comparison (10x). '
    'Bar length = mean |SHAP| value; features sorted by RF importance.',
    width=6.5)

add_heading(doc, '7.2 Beeswarm Plots', level=2)
doc.add_paragraph(
    'Beeswarm plots show both the magnitude and direction of SHAP effects. '
    'High HAND values consistently reduce flood susceptibility (negative SHAP), '
    'while specific soil types and low elevations increase susceptibility.'
)
add_figure(doc,
    os.path.join(SHAP_DIR, '10x', 'shap_beeswarm_rf.png'),
    'Figure 11. SHAP beeswarm plot for Random Forest (10x). Each dot is one test sample; '
    'colour indicates feature value (red = high, blue = low).',
    width=6.2)

add_figure(doc,
    os.path.join(SHAP_DIR, '10x', 'shap_beeswarm_xgboost.png'),
    'Figure 12. SHAP beeswarm plot for XGBoost (10x).',
    width=6.2)

add_heading(doc, '7.3 Cross-Size Feature Rank Stability', level=2)
doc.add_paragraph(
    'Feature importance ranks were compared across all five sample sizes to assess stability. '
    'HAND, Elevation, Soil Type, and TRI maintained consistent top rankings regardless of '
    'sample size, confirming their role as primary flood conditioning factors in Slovenia.'
)
add_figure(doc,
    os.path.join(SHAP_DIR, 'shap_rank_stability.png'),
    'Figure 13. Feature rank stability heatmap across all sample sizes (1x–10x) and models. '
    'Lower rank number = higher importance. Consistent colours indicate stable rankings.',
    width=6.5)

doc.add_page_break()

# ── 8. Discussion ─────────────────────────────────────────────────────────────
add_heading(doc, '8. Discussion')

add_heading(doc, '8.1 Sample Size Effect', level=2)
doc.add_paragraph(
    'Increasing training sample size from 1x (50k/class) to 10x (500k/class) yielded consistent '
    'improvements in standard validation metrics, with the largest gains between 1x and 2x '
    '(e.g., XGBoost AUC: 0.987 → 0.991). However, spatial CV metrics showed negligible change '
    'beyond 2x–3x. This divergence suggests that for spatially distributed environmental data, '
    'the marginal value of additional samples diminishes quickly once the feature space is '
    'adequately covered. The dominant source of uncertainty is spatial extrapolation, not '
    'insufficient training data quantity.'
)

add_heading(doc, '8.2 Model Comparison', level=2)
doc.add_paragraph(
    'XGBoost demonstrated the highest discriminative power in standard validation across all '
    'sample sizes, benefiting from gradient boosting\'s ability to capture complex non-linear '
    'interactions with relatively lower overfitting risk. CatBoost performed similarly, with '
    'particularly strong handling of categorical features (Soil Type, Lithology). '
    'RF showed the most consistent spatial generalisation, likely due to its built-in '
    'variance-reducing ensemble mechanism and lower susceptibility to spatial autocorrelation '
    'compared to boosting algorithms. The DNN was competitive in spatial CV (AUC ≈ 0.963 at 10x) '
    'despite being the most complex model, benefiting from regularisation via BatchNorm and Dropout. '
    'The ensemble mean provided a robust consensus that slightly outperformed individual models '
    'in several metrics, particularly precision and specificity.'
)

add_heading(doc, '8.3 Physical Interpretation', level=2)
doc.add_paragraph(
    'The SHAP analysis provides physically interpretable results consistent with flood dynamics. '
    'HAND (Height Above Nearest Drainage) emerges as the dominant topographic driver — areas '
    'close to stream level (low HAND) are inherently more prone to flooding. Elevation acts '
    'as a broader topographic control, with lower-elevation floodplains showing higher susceptibility. '
    'Soil Type is the leading non-topographic factor, reflecting infiltration capacity and '
    'runoff generation potential — poorly draining soils increase susceptibility. '
    'TRI captures terrain roughness that controls lateral flow pathways. '
    'The 90th percentile of annual maximum daily rainfall (Rain_P90) emerged prominently in '
    'boosting models, correctly reflecting the role of extreme precipitation events in driving '
    'flood occurrence.'
)

add_heading(doc, '8.4 Limitations', level=2)
lims = [
    'Flood inventory completeness: the reference flood layer may not capture all historical '
    'events, potentially underrepresenting flash floods in steep terrain.',
    'Static conditioning factors: time-varying factors (soil moisture, snowmelt, land use '
    'change) are not included. The maps represent long-term susceptibility, not event-specific forecast.',
    'Spatial autocorrelation: even with 50 km block CV, some spatial autocorrelation may remain '
    'within blocks, making CV metrics an upper bound of true spatial generalisation.',
    'Non-flood sampling: non-flood samples were drawn randomly from non-flooded areas and '
    'may include some flood-prone pixels not captured in the reference dataset.',
]
for l in lims:
    p = doc.add_paragraph(l, style='List Bullet')
    p.runs[0].font.size = Pt(10)

doc.add_page_break()

# ── 9. Conclusions ────────────────────────────────────────────────────────────
add_heading(doc, '9. Conclusions')
doc.add_paragraph(
    'This study produced high-resolution (12.5 m) flood susceptibility maps for Slovenia using '
    'four ML models trained at five sample sizes, with comprehensive validation including '
    '5-fold spatial block cross-validation and SHAP explainability analysis. Key conclusions:'
)
conclusions = [
    'XGBoost is the best-performing model for standard prediction tasks (AUC = 0.994, CSI = 0.940 at 10x).',
    'Random Forest shows the most robust spatial generalisation (spatial CV AUC ≈ 0.965 across all sizes) '
    'and is recommended for applications requiring high transferability to new regions.',
    'The ensemble mean model offers a reliable, model-agnostic flood susceptibility estimate '
    'predicting ~2,176 km² of flood-prone land (~10.7% of Slovenia).',
    'HAND, Elevation, Soil Type, TRI, and Lithology are the dominant conditioning factors, '
    'consistent across all models and sample sizes — confirming the physical validity of the approach.',
    'Spatial CV reveals that increasing training size beyond 3x–4x provides diminishing returns '
    'on spatial generalisation; future work should focus on reducing spatial autocorrelation bias.',
    'The 10x (500k/class) models represent the recommended operational configuration, '
    'balancing computational cost against marginal performance gains.',
]
for c in conclusions:
    p = doc.add_paragraph(c, style='List Bullet')
    p.runs[0].font.size = Pt(10)

doc.add_page_break()

# ── 10. Output File Index ──────────────────────────────────────────────────────
add_heading(doc, '10. Output File Index')
doc.add_paragraph(
    'All outputs are saved under ML_Results/ in the project directory. '
    'The following directory structure summarises the key files:'
)

file_index = [
    ('models_{size}/',           'Trained model files (model_rf.pkl, model_xgb.json, model_cat.cbm, model_dnn.pth, scaler.pkl)'),
    ('probability_maps/{size}/', 'Five probability rasters (FSM_{model}_probability.tif) per sample size'),
    ('classified_maps/{size}/',  'Binary (flood/no-flood) and 5-class susceptibility TIFs per model per size'),
    ('validation/{size}/',       'Threshold summary CSV, validation metrics CSV, ROC / confusion / CSI / bar plots'),
    ('validation/',              'multiscale_validation.csv, multiscale_comparison.png (cross-size)'),
    ('spatial_cv/{size}/',       'Fold assignments CSV, per-fold metrics, mean metrics, fold map, CV comparison plots'),
    ('spatial_cv/',              'multiscale_cv_summary.csv (cross-size spatial CV)'),
    ('shap/{size}/',             'Beeswarm, bar, dependence plots per model; shap_importance_all_models.csv'),
    ('shap/',                    'shap_importance_all_sizes.csv, shap_rank_stability.png (cross-size)'),
    ('analysis/',                'class_area_statistics.csv, binary_flood_statistics.csv, area comparison plots'),
]

tbl8 = doc.add_table(rows=len(file_index)+1, cols=2)
tbl8.style = 'Table Grid'
tbl8.alignment = WD_TABLE_ALIGNMENT.CENTER
header_row(tbl8, ['Directory', 'Contents'])
for i, (d, c) in enumerate(file_index):
    bg = 'F5F5F5' if i % 2 == 0 else 'FFFFFF'
    row = tbl8.rows[i+1]
    row.cells[0].text = d
    row.cells[0].paragraphs[0].runs[0].font.name = 'Courier New'
    row.cells[0].paragraphs[0].runs[0].font.size = Pt(8)
    row.cells[1].text = c
    row.cells[1].paragraphs[0].runs[0].font.size = Pt(9)
    shade_cell(row.cells[0], bg)
    shade_cell(row.cells[1], bg)

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(OUT)
print(f'Report saved → {OUT}')
size_mb = os.path.getsize(OUT) / 1e6
print(f'File size: {size_mb:.1f} MB')
