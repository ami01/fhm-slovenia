"""
09j — Independent incident validation, v2 (review 2026-10-09): baselines, confidence intervals, robustness.
==========================================================================================================
Same sites as 09i (river-flood sites outside the inventory vs wind/snow/ice sites never flood-reported;
placeholder coordinates reused >= 20 times removed). Adds what a referee will ask:
  * bootstrap 95 % CIs (1000 resamples, stratified) for every model's AUC;
  * terrain baselines on the same sites: HAND, distance to river, elevation (no fitting), and a 5-fold
    cross-validated logistic model on their ranks (fitted on the test sites, so favourable to the baseline);
  * the 50 m tolerance applied to the comparison sites too (09i applied it to flood sites only);
  * AUC within elevation quintiles, and with comparison sites reweighted to the flood sites' elevation;
  * AUC against each comparison group (wind, snow, ice).
Needs openpyxl (set PYTHONPATH to a directory that has it if the env lacks it).
Outputs → 09_Analysis/incident_reports_v2/: auc_ci.csv, baselines.csv, elevation_bands.csv, tolerance50m.csv, notes.txt
"""
import os, numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = os.environ.get('MAP_SIZE', '10x')
MAPSET = os.environ.get('MAP_SET')                 # None = production maps; 'v1'/'v2' = 06v2_predict outputs
F = os.path.join(ROOT, '01_Input_Features' if MAPSET != 'v2' else '01_Input_Features_v2')
PM = os.path.join(ROOT, '06_Probability_Maps_v2', MAPSET) if MAPSET else os.path.join(ROOT, '06_Probability_Maps', SIZE)
CM = os.path.join(ROOT, '07_Classified_Maps_v2', MAPSET) if MAPSET else os.path.join(ROOT, '07_Classified_Maps', SIZE)
OUT = os.path.join(ROOT, '09_Analysis', 'incident_reports_v2' + (f'_{MAPSET}' if MAPSET else '')); os.makedirs(OUT, exist_ok=True)
MODELS = ['XGBoost', 'Ensemble_mean', 'CatBoost', 'RF', 'DNN']
RIV, CONTROL, TOL, B, SEED = 'poplave ob vodotokih', ['Močan veter', 'Visok sneg', 'Žled'], 4, 1000, 42
notes = []
def note(s): print(s, flush=True); notes.append(s)

ev = pd.read_excel(os.path.join(ROOT, 'Events_natural-disasters_2005-2026_all-atts.xlsx'),
                   usecols=['NastanekDatumCas', 'WGSLat', 'WGSLon', 'SkupinaDogodka', 'Dogodek']).dropna(subset=['WGSLat', 'WGSLon'])
ev['key'] = ev.WGSLat.round(5).astype(str) + ',' + ev.WGSLon.round(5).astype(str)
fl = ev[ev.Dogodek.isin([RIV, 'poplave meteorne vode', 'poplave zaradi dviga podtalnice, kanalizacijskih vod'])].copy()
fl = fl[fl.key.map(fl.key.value_counts()) < 20]
fl['aug23'] = fl.NastanekDatumCas.between('2023-08-03', '2023-08-20', inclusive='left')
s = fl[fl.Dogodek == RIV].groupby('key').agg(lat=('WGSLat', 'first'), lon=('WGSLon', 'first'), aug23=('aug23', 'any')).reset_index()
ct = ev[ev.SkupinaDogodka.isin(CONTROL)]; ct = ct[ct.key.map(ct.key.value_counts()) < 20].drop_duplicates('key')
ct = ct[~ct.key.isin(set(ev.loc[ev.SkupinaDogodka == 'Poplave', 'key']))]
tr = Transformer.from_crs(4326, 3794, always_xy=True)
sx, sy = tr.transform(s.lon.values, s.lat.values); cx, cy = tr.transform(ct.WGSLon.values, ct.WGSLat.values)


def sample(path, xs, ys, k=0):
    """Value at the cell (k = 0) or max over a (2k+1)² window (k = TOL: 50 m tolerance)."""
    with rasterio.open(path) as d:
        a = d.read(1); nd = d.nodata
        r, c = rasterio.transform.rowcol(d.transform, xs, ys); r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1]); v = np.full(len(r), np.nan)
    for i in np.flatnonzero(ok):
        w = a[max(r[i] - k, 0):r[i] + k + 1, max(c[i] - k, 0):c[i] + k + 1].astype('float64')
        if nd is not None: w[w == nd] = np.nan
        v[i] = np.nanmax(w) if np.isfinite(w).any() else np.nan
    return v


lay = {'inv': os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif'), 'elev': os.path.join(F, 'elevation.tif'),
       'hand': os.path.join(F, 'hand.tif'), 'd2r': os.path.join(F, 'Distance_to_River.tif')}
lay.update({f'p_{m}': os.path.join(PM, f'FSM_{m}_probability.tif') for m in MODELS})
S = pd.DataFrame({k: sample(v, np.asarray(sx), np.asarray(sy)) for k, v in lay.items()}); S['aug23'] = s.aug23.values; S['y'] = 1
C = pd.DataFrame({k: sample(v, np.asarray(cx), np.asarray(cy)) for k, v in lay.items()}); C['aug23'] = False; C['y'] = 0
C['grp'] = ct.SkupinaDogodka.values
S = S[(S.inv == 0) & S.p_Ensemble_mean.notna()]; C = C[C.p_Ensemble_mean.notna()]
S_idx, C_idx = S.index.values, C.index.values
for m in MODELS:
    b = os.path.join(CM, f'FSM_{m}_binary.tif')
    S[f'b_{m}'] = sample(b, np.asarray(sx), np.asarray(sy))[S_idx]; C[f'b_{m}'] = sample(b, np.asarray(cx), np.asarray(cy))[C_idx]
    S[f'b50_{m}'] = sample(b, np.asarray(sx), np.asarray(sy), TOL)[S_idx]; C[f'b50_{m}'] = sample(b, np.asarray(cx), np.asarray(cy), TOL)[C_idx]
D = pd.concat([S, C], ignore_index=True)
note(f'flood sites outside inventory {len(S)}, of which Aug 2023 {int(S.aug23.sum())}; comparison sites {len(C)}')

rng = np.random.default_rng(SEED)
i1, i0 = np.flatnonzero(D.y.values == 1), np.flatnonzero(D.y.values == 0)
def boot_ci(score, sub=None):
    a1, a0 = (i1, i0) if sub is None else (np.intersect1d(i1, sub), i0)
    vals = []
    for _ in range(B):
        idx = np.r_[rng.choice(a1, a1.size), rng.choice(a0, a0.size)]
        vals.append(roc_auc_score(D.y.values[idx], score[idx]))
    return np.percentile(vals, [2.5, 97.5])
rows = []
aug = np.flatnonzero(D.aug23.values)
for m in MODELS:
    p = D[f'p_{m}'].values
    lo, hi = boot_ci(p); la, ha = boot_ci(p, aug)
    sub = np.r_[np.intersect1d(i1, aug), i0]
    rows.append(dict(model=m, AUC=roc_auc_score(D.y, p), CI_lo=lo, CI_hi=hi,
                     AUC_aug2023=roc_auc_score(D.y.values[sub], p[sub]), CI_lo_aug=la, CI_hi_aug=ha))
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, 'auc_ci.csv'), index=False)
note(pd.DataFrame(rows).round(3).to_string(index=False))

# terrain baselines
d = D.dropna(subset=['hand', 'd2r', 'elev']).copy()
base = [dict(predictor='ensemble map', AUC=roc_auc_score(d.y, d.p_Ensemble_mean)),
        dict(predictor='distance to river (−)', AUC=roc_auc_score(d.y, -d.d2r)),
        dict(predictor='HAND (−)', AUC=roc_auc_score(d.y, -d.hand)),
        dict(predictor='elevation (−)', AUC=roc_auc_score(d.y, -d.elev))]
for name, cols in (('HAND + distance to river + elevation, 5-fold logistic on the test sites', ['hand', 'd2r', 'elev']),
                   ('ensemble + the three terrain variables, 5-fold logistic', ['p_Ensemble_mean', 'hand', 'd2r', 'elev'])):
    pr = cross_val_predict(LogisticRegression(max_iter=1000), d[cols].rank(pct=True).values, d.y, cv=5, method='predict_proba')[:, 1]
    base.append(dict(predictor=name, AUC=roc_auc_score(d.y, pr)))
pd.DataFrame(base).round(4).to_csv(os.path.join(OUT, 'baselines.csv'), index=False)
note(pd.DataFrame(base).round(3).to_string(index=False))

# elevation robustness
D['eq'] = pd.qcut(D.elev, 5, labels=False)
eb = [dict(band=int(b), elev_min=g.elev.min(), elev_max=g.elev.max(), n_flood=int(g.y.sum()), n_comp=int((g.y == 0).sum()),
           AUC_ensemble=roc_auc_score(g.y, g.p_Ensemble_mean), AUC_HAND=roc_auc_score(g.dropna(subset=['hand']).y, -g.dropna(subset=['hand']).hand))
      for b, g in D.groupby('eq')]
bins = np.quantile(D.elev, np.linspace(0, 1, 21)); D['b20'] = np.clip(np.digitize(D.elev, bins[1:-1]), 0, 19)
pf, pc = D[D.y == 1].b20.value_counts(normalize=True), D[D.y == 0].b20.value_counts(normalize=True)
w = np.where(D.y == 1, 1.0, D.b20.map(pf / pc).fillna(0).values)
eb.append(dict(band='elevation-matched (comparison reweighted)', AUC_ensemble=roc_auc_score(D.y, D.p_Ensemble_mean, sample_weight=w)))
for g in CONTROL:
    dd = D[(D.y == 1) | (D.grp == g)]; eb.append(dict(band=f'vs {g} only', n_comp=int((dd.y == 0).sum()), AUC_ensemble=roc_auc_score(dd.y, dd.p_Ensemble_mean)))
pd.DataFrame(eb).round(4).to_csv(os.path.join(OUT, 'elevation_bands.csv'), index=False)
note(pd.DataFrame(eb).round(3).to_string(index=False))
note(f'median elevation: flood {S.elev.median():.0f} m, comparison {C.elev.median():.0f} m')

# 50 m tolerance for both classes
tol = [dict(model=m, flagged_flood=S[f'b_{m}'].mean(), flagged_comp=C[f'b_{m}'].mean(), ratio=S[f'b_{m}'].mean() / C[f'b_{m}'].mean(),
            flagged50_flood=S[f'b50_{m}'].mean(), flagged50_comp=C[f'b50_{m}'].mean(), ratio50=S[f'b50_{m}'].mean() / C[f'b50_{m}'].mean())
       for m in MODELS]
pd.DataFrame(tol).round(4).to_csv(os.path.join(OUT, 'tolerance50m.csv'), index=False)
note(pd.DataFrame(tol).round(3).to_string(index=False))
open(os.path.join(OUT, 'notes.txt'), 'w').write('\n'.join(notes) + '\n')
