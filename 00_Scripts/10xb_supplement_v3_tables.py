"""
10xb — Supporting-information tables added in the v3 revision (co-author review 2026-10-22), and the machine-readable
audit tables moved out of the PDF supplement.
    python 00_Scripts/10xb_supplement_v3_tables.py
Writes Manuscript/$PACKAGE/supp_v2/{foldcomp,resolution,lulc_v2,shift,fr_categorical}.tex (default JFRM_v3) from
    08g  09_Analysis/spatial_cv_v2/fold_composition/          fold composition, buffer, polygons spanning folds
    09o  09_Analysis/resolution_v2/                           aggregation to 25/50/100 m
    05v2 05_Results_v2/v2_1x_abl_base, v2_1x_noLULC           land-use ablation with the corrected predictors
    08e  09_Analysis/spatial_cv_v2/10x_v2[_shift25000]/        block grid shifted by 25 km
    09b  09_Analysis/frequency_ratio.csv                      categorical rows only (complete table deposited as CSV)
and deposits complete tables as CSV in Manuscript/$PACKAGE/supp_tables/. Tables whose inputs are missing are skipped.
"""
import os, shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, '09_Analysis')
PKG = os.path.join(ROOT, 'Manuscript', os.environ.get('PACKAGE', 'JFRM_v3'))
OUT, CSV = os.path.join(PKG, 'supp_v2'), os.path.join(PKG, 'supp_tables')
os.makedirs(OUT, exist_ok=True); os.makedirs(CSV, exist_ok=True)
ORDER = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
NAME = {'RF': 'Random Forest', 'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost', 'DNN': 'DNN', 'Ensemble_mean': 'Ensemble mean'}
n = lambda v: f'{int(round(v)):,}'.replace(',', '\\,')
f3 = lambda v: f'{v:.3f}'
f4 = lambda v: f'{v:.4f}'


def table(caption, label, cols, rows, align=None, size='\\footnotesize', sep='4pt'):
    align = align or 'l' + 'r' * (len(cols) - 1)
    body = '\n'.join(' & '.join(r) + ' \\\\' for r in rows)
    return (f"\\begin{{table}}[!htbp]\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n\\setlength{{\\tabcolsep}}{{{sep}}}\n\\centering\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(cols)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def write(name, text):
    open(os.path.join(OUT, name), 'w').write(text); print('wrote', name)


# ---- fold composition (08g) ------------------------------------------------------------------------------------
fc = os.path.join(A, 'spatial_cv_v2', 'fold_composition')
C = pd.read_csv(os.path.join(fc, 'fold_composition.csv'))
rows = [[str(int(r.fold)), str(int(r.blocks)), n(r.n_test), n(r.test_flood), n(r.test_nonflood), f'{r.test_flood_pct:.1f}', n(r.n_train),
         n(r.n_train_buffered), f'{r.buffer_removed_pct:.1f}', f'{r.nn_dist_median_m / 1000:.1f}', f'{r.nn_dist_buffered_median_m / 1000:.1f}']
        for _, r in C.iterrows()]
rows.append(['All', str(C.blocks.sum()), n(C.n_test.sum()), n(C.test_flood.sum()), n(C.test_nonflood.sum()),
             f'{100 * C.test_flood.sum() / C.n_test.sum():.1f}', '', '', '', '', ''])
t1 = table('Composition of the five spatial cross-validation folds at the operational baseline ($10\\times$; train and test samples pooled). '
           'Test: the withheld fold. Training: the other four folds, before and after removing every training observation within 5\\,km of '
           'the nearest test observation. Distance: median distance from a test observation to the nearest training observation, before '
           'and after buffering (after buffering the minimum is 5.0\\,km by construction).', 'tab:s-foldcomp',
           ['Fold', 'Blocks', 'Test', 'Flood', 'Non-flood', 'Flood \\%', 'Training', 'Buffered', 'Removed \\%', 'Dist. (km)', 'Buffered (km)'],
           rows, align='lrrrrrrrrrr', size='\\scriptsize', sep='2.6pt')
P = pd.read_csv(os.path.join(fc, 'polygon_crossing.csv'))
lab = {'Frequent': 'Frequent flood hazard', 'Rare': 'Rare flood hazard', 'Extreme_Rare': 'Extremely rare flood hazard',
       'Union_components': 'Connected areas of union', 'Any source polygon': 'Any source polygon'}
rows = [[lab[r.layer], '' if pd.isna(r.polygons_with_samples) else n(r.polygons_with_samples),
         '' if pd.isna(r.polygons_multi_fold) else f'{n(r.polygons_multi_fold)} ({r.pct_polygons_multi_fold:.1f}\\,\\%)',
         n(r.flood_samples_in_layer), f'{n(r.flood_samples_in_multi_fold)} ({r.pct_flood_samples_in_multi_fold:.1f}\\,\\%)']
        for _, r in P.iterrows()]
t2 = table('Source flood polygons (Slovenian Water Agency hazard classes) whose flood samples fall in more than one spatial fold, $10\\times$ '
           'configuration. Folds are assigned by block, so a polygon crossing a block boundary contributes samples to several folds; '
           'the 5\\,km buffered design (Table~\\ref{tab:s-buffer5000}) tests the sensitivity of the results to this dependence.',
           'tab:s-polycross', ['Layer', 'Sampled polygons', 'In $>1$ fold', 'Flood samples', 'In those polygons'],
           rows, align='lrrrr')
write('foldcomp.tex', t1 + '\n' + t2)
shutil.copy(os.path.join(fc, 'fold_composition.csv'), os.path.join(CSV, 'fold_composition.csv'))
shutil.copy(os.path.join(fc, 'polygon_crossing.csv'), os.path.join(CSV, 'polygon_crossing.csv'))

# ---- grid-origin shift (08e) -----------------------------------------------------------------------------------
sh = os.path.join(A, 'spatial_cv_v2', '10x_v2_shift25000', 'mean_metrics.csv')
if os.path.exists(sh):
    B = pd.read_csv(os.path.join(A, 'spatial_cv_v2', '10x_v2', 'mean_metrics.csv'))
    S = pd.read_csv(sh)
    rows = []
    for m in ORDER:
        b = B[(B.Model == m) & (B.threshold_rule == 'national')].iloc[0]; s = S[(S.Model == m) & (S.threshold_rule == 'national')].iloc[0]
        rows.append([NAME[m], f'{b.AUC_mean:.3f} $\\pm$ {b.AUC_std:.3f}', f'{s.AUC_mean:.3f} $\\pm$ {s.AUC_std:.3f}',
                     f'{b.CSI_mean:.3f} $\\pm$ {b.CSI_std:.3f}', f'{s.CSI_mean:.3f} $\\pm$ {s.CSI_std:.3f}'])
    write('shift.tex', table('Sensitivity of spatial block cross-validation to the origin of the 50\\,km block grid ($10\\times$, national threshold, '
                             'mean $\\pm$ s.d. over five folds). Shifted: origin moved by 25\\,km in both directions, which changes '
                             'every block boundary and the fold membership of every sample.', 'tab:s-shift',
                             ['Model', '$AUC$, published', '$AUC$, shifted', '$CSI$, published', '$CSI$, shifted'], rows, align='lrrrr', sep='3pt'))
    pd.concat([B.assign(grid='published'), S.assign(grid='shifted 25 km')]).to_csv(os.path.join(CSV, 'spatial_cv_grid_shift.csv'), index=False)

# ---- land-use ablation with the corrected predictors (05v2) -----------------------------------------------------
a, b = (os.path.join(ROOT, '05_Results_v2', t, 'metrics_holdout.csv') for t in ('v2_1x_abl_base', 'v2_1x_noLULC'))
if os.path.exists(a) and os.path.exists(b):
    X, Y = pd.read_csv(a).set_index('Model'), pd.read_csv(b).set_index('Model')
    rows = []
    for m in ORDER:
        r = [NAME[m]]
        for k in ('AUC', 'CSI', 'F1', 'Kappa'):
            d = X.loc[m, k] - Y.loc[m, k]
            r += [f4(Y.loc[m, k]), f4(X.loc[m, k]), f'${d:+.4f}$']
        rows.append(r)
    head = ['Model'] + [f'{h} {k}' for k in ('$AUC$', '$CSI$', '$F1$', '$\\kappa$') for h in ('19', '20', '$\\Delta$')]
    write('lulc_v2.tex', table('Paired ablation of the categorical land-use factor with the corrected predictor set: 19 factors without land use '
                               'against the 20 used in the main text, same samples, split, seed and settings, $1\\times$ configuration; '
                               'hold-out scores at each model\'s own validation-split threshold.', 'tab:s-lulc-v2', head, rows,
                               align='l' + 'r' * 12, size='\\scriptsize', sep='2pt'))
    pd.concat([Y.assign(features='19, without land use'), X.assign(features='20')]).to_csv(os.path.join(CSV, 'lulc_ablation_v2_1x.csv'))

# ---- resolution sensitivity (09o) -------------------------------------------------------------------------------
R = pd.read_csv(os.path.join(A, 'resolution_v2', 'metrics.csv'))
cor = pd.read_csv(os.path.join(A, 'resolution_v2', 'corridors.csv'))
nar = cor[cor.zone == 'narrow corridors'].set_index(['res_m', 'model', 'threshold_rule']).POD
rows = []
for agg in ('mean', 'max'):
    for m in ORDER:
        for rule in (('screening', 'prevalence') if m == 'Ensemble_mean' else ('screening',)):
            g = R[(R.model == m) & (R.threshold_rule == rule) & ((R['agg'] == agg) | (R.res_m == 12.5))].sort_values('res_m')
            for _, r in g.iterrows():
                if agg == 'max' and r.res_m == 12.5:
                    continue
                pod_n = nar.get((r.res_m, m, rule), float('nan')) if agg == 'mean' else float('nan')
                rows.append([agg if r.res_m > 12.5 else '--', NAME[m] + (' (prev.)' if rule == 'prevalence' else ''), f'{r.res_m:g}',
                             f4(r.AUC_frac), f3(r.AP_frac), f3(r.CSI_frac), f3(r.POD_frac), f3(r.FAR_frac), f'{r.FBI_frac:.2f}',
                             f3(r.CSI_any), f3(r.CSI_half), n(r.flagged_km2), f'{r.area_change_pct:+.1f}', f3(r.Jaccard),
                             '' if pd.isna(pod_n) else f'{pod_n:.2f}', f3(r.reports_flagged), f3(r.report_AUC)])
D = pd.read_csv(os.path.join(A, 'resolution_v2', 'disagreement.csv'))
t1 = table('Sensitivity to output resolution: every model, mean and maximum aggregation of the delivered 12.5\\,m probabilities, '
           'screening threshold (and the ensemble prevalence threshold, 0.885). $AUC$, $AP$, $CSI$, $POD$, $FAR$, $FBI$: against the '
           'inventory with each coarse cell carrying its flooded and non-flooded 12.5\\,m area (fractional reference). $CSI_{any}$, '
           '$CSI_{50}$: against a coarse binary reference, flooded if any or at least 50\\,\\% of the cell is flooded. Area: flagged area '
           'and change from 12.5\\,m. Jaccard: agreement with the 12.5\\,m binary map on the 12.5\\,m support. Narrow: $POD$ for mapped '
           'flood extents no more than 50\\,m wide. Reports: river-flood sites outside the inventory flagged, and their $AUC$ against '
           'wind, snow and ice sites.', 'tab:s-resolution',
           ['Agg.', 'Model', 'm', '$AUC$', '$AP$', '$CSI$', '$POD$', '$FAR$', '$FBI$', '$CSI_{any}$', '$CSI_{50}$', 'Area', '$\\Delta$\\%',
            'Jacc.', 'Narrow', 'Rep.', 'Rep. $AUC$'], rows, align='llr' + 'r' * 14, size='\\tiny', sep='1.5pt')
rows = [[f'{r.res_m:g}', r['agg'], f'{r.none_pct:.1f}', f'{r.disputed_pct:.2f}', f'{r.all4_pct:.2f}'] for _, r in D.iterrows()]
t2 = table('Four-learner agreement after aggregation (screening thresholds): share of the national area flagged by none, by one to three '
           '(disputed) and by all four learners.', 'tab:s-resolution-agree', ['Cell (m)', 'Aggregation', 'None (\\%)', 'Disputed (\\%)',
                                                                              'All four (\\%)'], rows, align='rlrrr')
L = pd.read_csv(os.path.join(A, 'resolution_v2', 'change_location.csv'))
L = L[L.threshold_rule == 'screening']
rows = [[f'{r.res_m:g}', NAME[r.model], n(r.changed_cells), f'{r.in_margin_pct:.1f}', f'{r.in_narrow_corridor_pct:.1f}',
         f'{r.in_karst_units_pct:.1f}'] for _, r in L.iterrows()]
t3 = table(f'Where the 12.5\\,m cells that change class after mean aggregation lie (screening threshold): share of changed cells within '
           f'50\\,m of the inventory edge (which covers {L.margin_share_of_area_pct.iloc[0]:.1f}\\,\\% of the country), in narrow mapped '
           f'corridors, and in the Dinaric karst warning units ({L.karst_share_of_area_pct.iloc[0]:.1f}\\,\\% of the country).',
           'tab:s-resolution-where', ['Cell (m)', 'Model', 'Changed cells', 'Margin (\\%)', 'Narrow (\\%)', 'Karst units (\\%)'], rows,
           align='rlrrrr')
write('resolution.tex', '\n'.join([t1, t2, t3]))
for f in ('metrics.csv', 'change_location.csv', 'corridors.csv', 'disagreement.csv'):
    shutil.copy(os.path.join(A, 'resolution_v2', f), os.path.join(CSV, f'resolution_{f}'))

# ---- frequency ratios: categorical factors in the PDF, complete table as CSV ------------------------------------
F = pd.read_csv(os.path.join(A, 'frequency_ratio.csv'))
F.to_csv(os.path.join(CSV, 'frequency_ratio_full_first_predictor_set.csv'), index=False)
shutil.copy(os.path.join(A, 'fr_hand_v2.csv'), os.path.join(CSV, 'frequency_ratio_corrected_layers.csv'))
cat = {'LULC': 'Land use', 'Pedoloska_grouped': 'Soil type', 'lito_poly_250_d96_grouped': 'Lithology'}
rows = [[cat[r.Feature], str(r.Bin_Label), n(r.Total_Px), n(r.Flood_Px), f3(r.FR)] for _, r in F[F.Feature.isin(cat)].iterrows()]
write('fr_categorical.tex', table(
    'Frequency ratios of the grouped categorical factors (land use, soil type, lithology), computed on every tenth row and column of the '
    'national grid. $FR>1$ indicates over-representation of mapped flood extent relative to the class\'s share of national area. '
    'This model-free measure is the consistency check against which the model-conditional SHAP hierarchy is compared. Hydrogeology, '
    'HAND and flow accumulation were corrected in the input audit and are in Table~\\ref{tab:s-fr-changed}; the complete table for '
    'every factor and bin is deposited as \\texttt{supp\\_tables/frequency\\_ratio\\_full\\_first\\_predictor\\_set.csv}.',
    'tab:s-fr-full', ['Factor', 'Class', 'Total cells', 'Flood cells', '$FR$'], rows, align='llrrr'))

# ---- per-intensity tables moved out of the PDF ------------------------------------------------------------------
H = []
for t in ('v2_1x', 'v2_2x', 'v2_3x', 'v2_4x', 'v2'):
    H.append(pd.read_csv(os.path.join(ROOT, '05_Results_v2', t, 'metrics_holdout.csv')).assign(config=t.replace('v2_', '').replace('v2', '10x')))
pd.concat(H).to_csv(os.path.join(CSV, 'holdout_all_configurations.csv'), index=False)
Fo = []
for t, d in (('1x', '1x_v2_1x'), ('2x', '2x_v2_2x'), ('3x', '3x_v2_3x'), ('4x', '4x_v2_4x'), ('10x', '10x_v2')):
    Fo.append(pd.read_csv(os.path.join(A, 'spatial_cv_v2', d, 'per_fold_metrics.csv')).assign(config=t))
pd.concat(Fo).to_csv(os.path.join(CSV, 'spatial_cv_per_fold_all_configurations.csv'), index=False)
for mode in ('buffer5000', 'regime'):
    shutil.copy(os.path.join(A, 'transfer_v2', mode, 'per_fold_metrics.csv'), os.path.join(CSV, f'transfer_{mode}_per_fold.csv'))
print('->', OUT, '|', CSV, sorted(os.listdir(CSV)))
