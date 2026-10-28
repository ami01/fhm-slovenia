"""
10x — LaTeX tables for the JFRM v2 supporting information, generated from the v2 result files (review 2026-10-11).
    PYTHONPATH=<dir with openpyxl> python 00_Scripts/10x_supplement_v2_tables.py
Writes Manuscript/$PACKAGE/supp_v2/*.tex (default JFRM_v3), each \\input by the package's supplementary .tex:
  holdout.tex      hold-out validation at the five sampling intensities (05v2_train)
  folds.tex        per-fold spatial block CV at each intensity, national threshold (08e, SAMPLE_SET=v2[_Nx])
  sizetrend.tex    spatial CV mean by intensity
  shap.tex         mean |SHAP| (10x) and the 20-attribution rank table (08v2_shap)
  fr_changed.tex   frequency ratios of the three corrected layers (HAND, flow accumulation, hydrogeology), 1/10 scale as 09b
  prevalence.tex   prevalence threshold chosen on the validation split (07w)
  units.tex        flood-prone share of all 26 warning units
  incidents.tex    incident AUC with CI, baselines, elevation bands, flood types
  transfer.tex     per-fold buffered CV and per-regime results, all learners (08f), AOA summary (09n)
  operating.tex    operating points of the threshold curve (09m)
"""
import os, json, numpy as np, pandas as pd, rasterio
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, '09_Analysis'); OUT = os.path.join(ROOT, 'Manuscript', os.environ.get('PACKAGE', 'JFRM_v3'), 'supp_v2'); os.makedirs(OUT, exist_ok=True)
SIZES = {'1x': ('v2_1x', '50\\,000'), '2x': ('v2_2x', '100\\,000'), '3x': ('v2_3x', '150\\,000'), '4x': ('v2_4x', '200\\,000'),
         '10x': ('v2', '500\\,000')}
ORDER = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
NAME = {'RF': 'Random Forest', 'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost', 'DNN': 'DNN', 'Ensemble_mean': 'Ensemble mean'}
SHORT = {'Distance_to_River': 'Distance to river', 'Distance_to_Road': 'Distance to road', 'Hgk_grouped': 'Hydrogeology',
         'Pedoloska_grouped': 'Soil type', 'River_Density': 'River density', 'aspect': 'Aspect', 'drainage_density': 'Drainage density',
         'elevation': 'Elevation', 'flow_accumulation': 'Flow accumulation', 'hand': 'HAND', 'lito_poly_250_d96_grouped': 'Lithology',
         'ls_factor': 'LS factor', 'plan_curvature': 'Plan curvature', 'profile_curvature': 'Profile curvature',
         'slovenia_annual_max_daily_p90_3794_12m5': 'Rainfall P90', 'spi': 'SPI', 'tri': 'TRI', 'twi': 'TWI', 'valley_depth': 'Valley depth',
         'LULC': 'Land use'}


def table(caption, label, cols, rows, align=None, size='\\footnotesize'):
    align = align or 'l' + 'r' * (len(cols) - 1)
    esc = lambda c: c if '\\%' in c else c.replace('%', '\\%').replace('_', '\\_') if '$' not in c else c.replace('%', '\\%')
    body = '\n'.join(' & '.join(esc(c) for c in r) + ' \\\\' for r in rows)
    return (f"\\begin{{table}}[!htbp]\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n\\centering\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{' & '.join(cols)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


f4 = lambda v: f'{v:.4f}'
f3 = lambda v: f'{v:.3f}'
X = lambda sz: sz.replace('x', '$\\times$')

# ---- hold-out ------------------------------------------------------------------------------------------------------
out = []
for size, (tag, n) in SIZES.items():
    P = pd.read_parquet(os.path.join(ROOT, '05_Results_v2', tag, 'holdout_predictions.parquet'))
    thr = json.load(open(os.path.join(ROOT, '04_Models_v2', tag, 'thresholds.json')))['thresholds']
    y = P.label.values.astype(int); rows = []
    for m in ORDER:
        p = P[m].values; q = (p >= thr[m]).astype(int)
        tp, tn, fp, fn = ((q == 1) & (y == 1)).sum(), ((q == 0) & (y == 0)).sum(), ((q == 1) & (y == 0)).sum(), ((q == 0) & (y == 1)).sum()
        po = (tp + tn) / len(y); pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / len(y) ** 2
        rows.append([NAME[m], f'{thr[m]:.3f}', f4(roc_auc_score(y, p)), f4(average_precision_score(y, p)), f4(tp / (tp + fp + fn)),
                     f4(2 * tp / (2 * tp + fp + fn)), f4((po - pe) / (1 - pe)), f4(po), f4(tp / (tp + fn)), f4(tn / (tn + fp))])
    out.append(table(f'Random hold-out validation at the {X(size)} sampling configuration ({n} points per class), '
                     f'at each model\'s $CSI$-optimal threshold chosen on the validation split.', f'tab:s-holdout-{size}',
                     ['Model', 'Thr.', '$AUC$', 'AP', '$CSI$', '$F1$', '$\\kappa$', 'Acc.', 'Sens.', 'Spec.'], rows))
open(os.path.join(OUT, 'holdout.tex'), 'w').write('\n'.join(out))

# ---- spatial CV per fold and by size ----------------------------------------------------------------------------------
out, trend = [], []
for size, (tag, n) in SIZES.items():
    f = os.path.join(A, 'spatial_cv_v2', f'{size}_{tag}' if size != '10x' else '10x_v2', 'per_fold_metrics.csv')
    if not os.path.exists(f):
        continue
    F = pd.read_csv(f); F = F[F.threshold_rule == 'national']
    rows = [[NAME[r.Model], str(int(r.Fold)), f4(r.AUC), f4(r.CSI), f4(r.F1), f4(r.Kappa), f4(r.Sensitivity), f4(r.Specificity),
             f'{int(r.N_test):,}'.replace(',', '\\,')] for _, r in F.sort_values(['Fold', 'Model'], key=lambda c: c.map({m: i for i, m in enumerate(ORDER)}) if c.name == 'Model' else c).iterrows()]
    out.append(table(f'Per-fold results of 5-fold spatial block cross-validation (50\\,km blocks) at the {X(size)} '
                     f'configuration ({n} points per class), at each model\'s national threshold; every model, including the ensemble, refitted per fold.',
                     f'tab:s-fold-{size}', ['Model', 'Fold', '$AUC$', '$CSI$', '$F1$', '$\\kappa$', 'Sens.', 'Spec.', '$N_\\mathrm{test}$'], rows))
    g = F.groupby('Model')[['AUC', 'CSI']].agg(['mean', 'std'])
    for m in ORDER:
        trend.append(dict(size=size, model=m, AUC=g.loc[m, ('AUC', 'mean')], AUCsd=g.loc[m, ('AUC', 'std')], CSI=g.loc[m, ('CSI', 'mean')],
                          CSIsd=g.loc[m, ('CSI', 'std')]))
open(os.path.join(OUT, 'folds.tex'), 'w').write('\n'.join(out))
T = pd.DataFrame(trend); T.to_csv(os.path.join(A, 'spatial_cv_v2', 'size_trend.csv'), index=False)
sizes = list(dict.fromkeys(T['size']))
rows = [[NAME[m]] + [f"{T[(T['size'] == s) & (T.model == m)].AUC.iloc[0]:.4f} / {T[(T['size'] == s) & (T.model == m)].CSI.iloc[0]:.3f}" for s in sizes]
        for m in ORDER]
open(os.path.join(OUT, 'sizetrend.tex'), 'w').write(table(
    'Spatial block cross-validation by sampling intensity: mean $AUC$ / mean $CSI$ over five folds at the national threshold.',
    'tab:s-sizetrend-v2', ['Model'] + [s.replace('x', '$\\times$') for s in sizes], rows))

# ---- SHAP -------------------------------------------------------------------------------------------------------------
S = pd.read_csv(os.path.join(A, 'shap_v2', 'shap_importance_all_sizes.csv'))
S['r'] = S.groupby(['SampleSize', 'Model']).MeanAbsSHAP.rank(ascending=False).astype(int)
s10 = S[S.SampleSize == '10x'].pivot(index='Feature', columns='Model', values='MeanAbsSHAP')
order = S.groupby('Feature').r.mean().sort_values().index
rows = [[SHORT.get(f, f)] + [f'{s10.loc[f, m]:.4f}' for m in ['RF', 'XGBoost', 'CatBoost', 'DNN']] for f in order]
t1 = table('Mean $|$SHAP$|$ of every conditioning factor for the four learners at the operational baseline (10$\\times$). Values are on '
           'each learner\'s own output scale (probability for RF, log-odds for the others), so compare within a column.',
           'tab:s-shap-full', ['Conditioning factor', 'RF', 'XGBoost', 'CatBoost', 'DNN'], rows)
AB = {'RF': 'RF', 'XGBoost': 'XGB', 'CatBoost': 'Cat', 'DNN': 'DNN'}
cols = [(s, m) for s in ['1x', '2x', '3x', '4x', '10x'] for m in ['RF', 'XGBoost', 'CatBoost', 'DNN']]
R = S.pivot_table(index='Feature', columns=['SampleSize', 'Model'], values='r')
rows = [[SHORT.get(f, f)] + [str(int(R.loc[f, c])) for c in cols] for f in order]
hdr = ['Factor'] + [AB[m] + '\\,' + s for s, m in cols]
t2 = table('SHAP importance rank (1 = largest) of every factor for the four learners at all five sampling intensities (twenty '
           'attributions; the samples are nested, so these are not independent replicates). Rows ordered by mean rank.',
           'tab:s-shap-rank', hdr, rows, size='\\tiny').replace('\\begin{tabular}', '\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}').replace(
           '\\end{tabular}', '\\end{tabular}\n}')
open(os.path.join(OUT, 'shap.tex'), 'w').write(t1 + '\n' + t2)

# ---- frequency ratios of the corrected layers (1/10 scale, as 09b) -----------------------------------------------------
with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
    h, w = d.height // 10, d.width // 10
    inv = d.read(1, out_shape=(h, w), resampling=rasterio.enums.Resampling.nearest); ok0 = inv != d.nodata
rows = []
for f, cat in (('hand', False), ('flow_accumulation', False), ('Hgk_grouped', True)):
    with rasterio.open(os.path.join(ROOT, '01_Input_Features_v2', f + '.tif')) as d:
        a = d.read(1, out_shape=(h, w), resampling=rasterio.enums.Resampling.nearest).astype('float64')
        if d.nodata is not None: a[a == d.nodata] = np.nan
    ok = ok0 & np.isfinite(a); v = a[ok]; fl = inv[ok] == 1
    if cat:
        vr = np.round(v); bins = [(int(c), vr == c, f'class {int(c)}') for c in np.unique(vr)]
    else:
        q = np.unique(np.nanpercentile(v, np.linspace(0, 100, 11))); idx = np.digitize(v, q[1:-1])
        bins = [(i, idx == i, f'{q[i]:.3g}--{q[i + 1]:.3g}') for i in range(len(q) - 1)]
    for i, m, lab in bins:
        fr = (fl[m].sum() / fl.sum()) / (m.sum() / len(v)) if m.sum() else np.nan
        rows.append([SHORT[f] + (' (flow path)' if f == 'hand' else ''), lab, f'{m.sum():,}'.replace(',', '\\,'), f'{fl[m].sum():,}'.replace(',', '\\,'), f'{fr:.3f}'])
open(os.path.join(OUT, 'fr_changed.tex'), 'w').write(table(
    'Frequency ratios of the three layers corrected in the input audit (Section~S16), computed as in Table~\\ref{tab:s-fr-full} '
    '(1/10 scale, nearest neighbour, quantile bins for continuous factors). They replace the rows of these factors in that table.',
    'tab:s-fr-changed', ['Factor', 'Bin', 'Total cells', 'Flood cells', '$FR$'], rows, align='llrrr'))

# ---- prevalence threshold -----------------------------------------------------------------------------------------------
P = pd.read_csv(os.path.join(A, 'threshold_prevalence', 'v2.csv'))
rows = [[NAME[r.model], f'{r.threshold:.3f}', f3(r.POD), f3(r.FAR), f'{r.FBI:.2f}', f3(r.CSI), f3(r.ETS), f'{r.flagged_km2:,.0f}'.replace(',', '\\,')]
        for _, r in P.set_index('model').loc[ORDER].reset_index().iterrows()]
open(os.path.join(OUT, 'prevalence.tex'), 'w').write(table(
    'Prevalence threshold: chosen on the training validation split with non-flood samples reweighted to the national prevalence '
    '(5.63\\,\\%), then applied to the national maps and verified over all 129\\,016\\,006 valid cells. The test set and the maps '
    'were not used to choose it; most flood cells are nevertheless training cells.', 'tab:s-prevalence',
    ['Model', 'Thr.', '$POD$', '$FAR$', '$FBI$', '$CSI$', '$ETS$', 'Area (km$^2$)'], rows))

# ---- warning units ------------------------------------------------------------------------------------------------------
U = pd.read_csv(os.path.join(A, 'unit_share_v2', 'warning_units_flood_prone.csv'))
rows = [[r.unit.replace('Č', '\\v{C}').replace('č', '\\v{c}').replace('Š', '\\v{S}').replace('š', '\\v{s}').replace('Ž', '\\v{Z}').replace('ž', '\\v{z}'),
         f'{r.valid_km2:,.0f}'.replace(',', '\\,'), f'{r.flood_prone_km2:,.1f}', f'{r.share_pct:.1f}', f'{r.inventory_pct:.1f}'] for _, r in U.iterrows()]
open(os.path.join(OUT, 'units.tex'), 'w').write(table(
    'Flood-prone share of all 26 ARSO hydrological warning units (ensemble, screening threshold) and the share covered by the inventory.',
    'tab:s-units', ['Warning unit', 'Area (km$^2$)', 'Flood-prone (km$^2$)', 'Flood-prone (\\%)', 'Inventory (\\%)'], rows))

# ---- incidents ----------------------------------------------------------------------------------------------------------
I = os.path.join(A, 'incident_reports_v2_v2')
C = pd.read_csv(os.path.join(I, 'auc_ci.csv')).set_index('model').loc[ORDER]
rows = [[NAME[m], f'{r.AUC:.3f} ({r.CI_lo:.3f}--{r.CI_hi:.3f})', f'{r.AUC_aug2023:.3f} ({r.CI_lo_aug:.3f}--{r.CI_hi_aug:.3f})'] for m, r in C.iterrows()]
t1 = table('$AUC$ separating river-flood sites outside the inventory from wind, snow and ice sites, with bootstrap 95\\,\\% intervals.',
           'tab:s-inc-auc', ['Model', 'All dates ($n = 5\\,019$)', 'August 2023 ($n = 3\\,076$)'], rows)
Bz = pd.read_csv(os.path.join(I, 'baselines.csv'))
rows = [[r.predictor.replace('−', '$-$'), f3(r.AUC)] for _, r in Bz.iterrows()]
t2 = table('What the map adds to terrain: $AUC$ of single terrain variables (sign reversed where lower means more flood-prone) and of '
           'logistic models fitted to the test sites themselves by five-fold cross-validation.', 'tab:s-inc-base', ['Predictor', '$AUC$'], rows, align='lr')
E = pd.read_csv(os.path.join(I, 'elevation_bands.csv'))
eb = E[E.band.astype(str).str.fullmatch(r'\d+')]
rows = [[f'{r.elev_min:.0f}--{r.elev_max:.0f}', f'{int(r.n_flood)}', f'{int(r.n_comp)}', f3(r.AUC_ensemble), f3(r.AUC_HAND)] for _, r in eb.iterrows()]
for _, r in E[~E.band.astype(str).str.fullmatch(r'\d+')].iterrows():
    rows.append([r.band.replace('Močan veter', 'wind').replace('Visok sneg', 'snow').replace('Žled', 'ice'), '', '' if pd.isna(r.n_comp) else f'{int(r.n_comp)}', f3(r.AUC_ensemble), ''])
t3 = table('Incident $AUC$ (ensemble and flow-path HAND alone) by elevation quintile of the sites, for the elevation-matched comparison, '
           'and against each comparison hazard separately.', 'tab:s-inc-elev', ['Elevation (m) / variant', 'Flood', 'Comparison', 'Ensemble', 'HAND'], rows)
Sm = pd.read_csv(os.path.join(ROOT, '_v2_view', '09_Analysis', 'incident_reports', 'summary.csv'))
e = Sm[(Sm.model == 'Ensemble_mean') & Sm.auc_vs_control.notna()]
rows = [[r.hazard.replace('riverine', 'river').replace('pluvial', 'surface water'), r.subset, f'{int(r.n_sites):,}'.replace(',', '\\,'), f3(r.flagged), f3(r.auc_vs_control)] for _, r in e.iterrows()]
t4 = table('Ensemble map against each type of flood report: share of sites flagged at the screening threshold and $AUC$ against wind, snow and ice sites.',
           'tab:s-inc-types', ['Flood type', 'Subset', 'Sites', 'Flagged', '$AUC$'], rows, align='llrrr')
open(os.path.join(OUT, 'incidents.tex'), 'w').write('\n'.join([t1, t2, t3, t4]))

# ---- transferability ---------------------------------------------------------------------------------------------------
out = []
for mode, cap, unit in (('buffer5000', 'Per-fold results of 50\\,km spatial block cross-validation with every training observation '
                                       'within 5\\,km of the nearest test observation removed', 'Fold'),
                        ('regime', 'Per-regime results of leave-one-regime-out cross-validation', 'Regime')):
    F = pd.read_csv(os.path.join(A, 'transfer_v2', mode, 'per_fold_metrics.csv')); F = F[F.rule == 'national']
    rows = [[str(r.fold), NAME[r.model], f'{r.n_train:,}'.replace(',', '\\,'), f'{r.n_test:,}'.replace(',', '\\,'), f3(r.flood_share), f3(r.AUC), f3(r.CSI), f3(r.POD), f3(r.Specificity)]
            for _, r in F.sort_values(['fold', 'model'], key=lambda c: c.map({m: i for i, m in enumerate(ORDER)}) if c.name == 'model' else c).iterrows()]
    out.append(table(f'{cap}, at each model\'s national threshold.', f'tab:s-{mode}',
                     [unit, 'Model', '$N_\\mathrm{train}$', '$N_\\mathrm{test}$', 'Flood share', '$AUC$', '$CSI$', '$POD$', 'Spec.'], rows, align='llrrrrrrr'))
J = json.load(open(os.path.join(A, 'aoa_v2', 'summary.json')))
K = pd.read_csv(os.path.join(A, 'aoa_v2', 'skill_by_di.csv'))
rows = [[r.cv, r.DI_bin, f'{int(r.n):,}'.replace(',', '\\,'), f3(r.flood_share), f3(r.AUC)] for _, r in K.iterrows()]
out.append(table(f'Ensemble skill by dissimilarity index (share of the area-of-applicability threshold, {J["threshold"]:.3f}). '
                 f'{100 * J["inside"]:.1f}\\,\\% of Slovenia lies inside the area of applicability; '
                 f'{J["outside_km2"]:.0f}\\,km$^2$ lies outside. Buffered-design rows reuse the unbuffered index.',
                 'tab:s-aoa', ['Design', 'DI bin', 'Samples', 'Flood share', '$AUC$'], rows, align='llrrr'))
open(os.path.join(OUT, 'transfer.tex'), 'w').write('\n'.join(out))

# ---- operating curve --------------------------------------------------------------------------------------------------
O = pd.read_csv(os.path.join(A, 'operating_curve', 'curve.csv'))
O = O[O.threshold.round(3).isin([0.3, 0.5, 0.57, 0.6, 0.7, 0.8, 0.885, 0.9])]
rows = [[r.set.split(' (')[0], f'{r.threshold:.3f}', f3(r.CSI), f3(r.POD), f3(r.FAR), f'{r.FBI:.2f}', f'{r.area_km2:,.0f}'.replace(',', '\\,'),
         f3(r.reports_flagged), f3(r.comparison_flagged)] for _, r in O.iterrows()]
open(os.path.join(OUT, 'operating.tex'), 'w').write(table(
    'Operating points of the ensemble map: agreement with the inventory at the true base rate against the share of independent '
    'river-flood reports outside the inventory, and of wind, snow and ice sites, that the map flags.', 'tab:s-operating',
    ['Sampling', 'Thr.', '$CSI$', '$POD$', '$FAR$', '$FBI$', 'Area (km$^2$)', 'Reports', 'Comparison'], rows, align='lrrrrrrrr'))
print('->', OUT, sorted(os.listdir(OUT)))
