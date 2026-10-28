"""
09n — Area of applicability (Meyer & Pebesma 2021) of the v2 models (review 2026-10-09).
=======================================================================================
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<dir with openpyxl> python 00_Scripts/09n_aoa.py
Dissimilarity index DI = distance to the nearest training sample in predictor space / mean training-to-training distance.
  * Predictors: the 16 continuous inputs (log1p for flow accumulation, SPI, LS factor, as in training), standardised with
    the training mean/sd and weighted by mean |SHAP| (10x, mean of the four learners, 09_Analysis/shap_v2).
    The four categorical inputs (LULC, soil, lithology, hydrogeology) are label codes, for which a Euclidean distance has
    no meaning; they are checked separately: a cell is "novel" if any of its classes never occurs in the training data.
  * Reference: a random 150,000 of the 798k training samples (seeded), the same for the map and the threshold.
  * Threshold: DI of each sample to the training samples of the OTHER spatial-CV folds (08e blocks), outlier-trimmed
    maximum (largest value <= Q3 + 1.5 IQR), as in CAST::aoa.
Outputs → 09_Analysis/aoa_v2/: DI_100m.tif, AOA_100m.tif (1 inside, 0 outside, 2 categorical novelty), summary.json,
skill_by_di.csv (spatial-CV and regime-CV fold predictions binned by DI), incidents_by_aoa.csv
"""
import json, os, numpy as np, pandas as pd, rasterio, torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '09_Analysis', 'aoa_v2'); os.makedirs(OUT, exist_ok=True)
FEATURES = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
CAT = ['LULC', 'Pedoloska_grouped', 'lito_poly_250_d96_grouped', 'Hgk_grouped']
CONT = [f for f in FEATURES if f not in CAT]
LOG = ['flow_accumulation', 'spi', 'ls_factor']
NREF, F, SEED = 150_000, 8, 42                        # F: 12.5 m -> 100 m decimation
DEV = torch.device('cuda')
rng = np.random.default_rng(SEED)

S = pd.read_parquet(os.path.join(ROOT, '03_Samples_v2', 'v2', 'samples_train.parquet'))
mu, sd = S[CONT].mean(), S[CONT].std()
sh = pd.read_csv(os.path.join(ROOT, '09_Analysis', 'shap_v2', 'shap_importance_all_sizes.csv'))
w = sh[(sh.SampleSize == '10x') & (sh.Model != 'Ensemble_mean')].groupby('Feature').MeanAbsSHAP.mean().reindex(CONT)
assert w.notna().all(), w[w.isna()]
w = w / w.sum(); print('weights:', w.round(3).to_dict(), flush=True)
z = lambda D: torch.tensor((((D[CONT] - mu) / sd) * w).values.astype(np.float32), device=DEV)
seen = {c: set(np.unique(S[c].values)) for c in CAT}


def nn_dist(Q, R, chunk=8192):
    out = torch.empty(len(Q), device=DEV)
    for i in range(0, len(Q), chunk):
        out[i:i + chunk] = torch.cdist(Q[i:i + chunk], R).min(1).values
    return out.cpu().numpy()


ref = S.iloc[rng.choice(len(S), NREF, replace=False)]
R = z(ref)
k = rng.choice(NREF, 20_000, replace=False)
dbar = float(torch.cdist(R[k[:10000]], R[k[10000:]]).mean())     # mean training-to-training distance
print(f'mean training distance {dbar:.4f}', flush=True)

# --- threshold from cross-fold DI (08e folds) and DI of every CV sample -----------------------------------------
A = pd.concat([pd.read_parquet(os.path.join(ROOT, '03_Samples_v2', 'v2', f'samples_{s}.parquet')) for s in ('train', 'test')],
              ignore_index=True)
bx = ((A.x_3794 - A.x_3794.min()) // 50_000).astype(int); by = ((A.y_3794 - A.y_3794.min()) // 50_000).astype(int)
A['fold'] = ((bx * 2 + by) % 5) + 1
A['DI_cv'] = np.nan
for f in range(1, 6):
    o = A[A.fold != f]; rf = o.iloc[rng.choice(len(o), min(NREF, len(o)), replace=False)]
    m = (A.fold == f).values
    A.loc[m, 'DI_cv'] = nn_dist(z(A[m]), z(rf)) / dbar
q1, q3 = np.percentile(A.DI_cv, [25, 75]); thr = float(A.DI_cv[A.DI_cv <= q3 + 1.5 * (q3 - q1)].max())
print(f'AOA threshold DI = {thr:.4f}; CV samples outside: {(A.DI_cv > thr).mean():.3%}', flush=True)

# --- national DI map at 100 m -----------------------------------------------------------------------------------
FD = os.path.join(ROOT, '01_Input_Features_v2')
with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
    h, wd = d.height // F, d.width // F
    inv = d.read(1, out_shape=(h, wd), resampling=rasterio.enums.Resampling.nearest); ok = inv != d.nodata
    T = d.transform * d.transform.scale(d.width / wd, d.height / h); crs = d.crs
G = {}
for f in FEATURES:
    with rasterio.open(os.path.join(FD, f + '.tif')) as d:
        a = d.read(1, out_shape=(h, wd), resampling=rasterio.enums.Resampling.nearest).astype('float64')
        if d.nodata is not None and not np.isnan(d.nodata): a[a == d.nodata] = np.nan
    G[f] = np.log1p(np.maximum(a, 0)) if f in LOG else a
    ok &= np.isfinite(G[f])
M = pd.DataFrame({f: G[f][ok] for f in FEATURES})
di = np.full((h, wd), -9999, np.float32); di[ok] = nn_dist(z(M), R) / dbar
nov = np.zeros(ok.sum(), bool)
for c in CAT:
    nov |= ~np.isin(M[c].values, list(seen[c]))
aoa = np.full((h, wd), 255, np.uint8); aoa[ok] = np.where(nov, 2, (di[ok] <= thr).astype(np.uint8))
prof = dict(driver='GTiff', height=h, width=wd, count=1, crs=crs, transform=T, compress='deflate')
with rasterio.open(os.path.join(OUT, 'DI_100m.tif'), 'w', dtype='float32', nodata=-9999, **prof) as d: d.write(di, 1)
with rasterio.open(os.path.join(OUT, 'AOA_100m.tif'), 'w', dtype='uint8', nodata=255, **prof) as d: d.write(aoa, 1)
v = aoa[ok]
summ = dict(threshold=thr, dbar=dbar, n_ref=NREF, weights=w.round(4).to_dict(), cells_100m=int(ok.sum()),
            inside=float((v == 1).mean()), outside_DI=float((v == 0).mean()), categorical_novelty=float((v == 2).mean()),
            outside_km2=float((v != 1).sum() * 0.01), cv_samples_outside=float((A.DI_cv > thr).mean()),
            DI_quantiles={str(p): float(np.percentile(di[ok], p)) for p in (50, 90, 95, 99, 99.9)})
for c in CAT:
    summ[f'novel_{c}'] = float((~np.isin(M[c].values, list(seen[c]))).mean())
json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2); print(json.dumps(summ, indent=1), flush=True)


# --- skill by DI: spatial-CV fold predictions (and regime CV if present) -----------------------------------------
def skill(P, tag, di_col):
    rows = []
    bins = [0, 0.25 * thr, 0.5 * thr, 0.75 * thr, thr, np.inf]
    P['bin'] = pd.cut(P[di_col], bins, labels=['0-25%', '25-50%', '50-75%', '75-100%', 'outside AOA'], include_lowest=True)
    for b, g in list(P.groupby('bin', observed=True)) + [('inside AOA', P[P[di_col] <= thr]), ('all', P)]:
        y = g.label.values; p = g.Ensemble_mean.values
        if len(np.unique(y)) < 2: continue
        q = p >= 0.5; tp = (q & (y == 1)).sum()
        rows.append(dict(cv=tag, DI_bin=str(b), n=len(g), flood_share=y.mean(), AUC=roc_auc_score(y, p),
                         CSI_at_0_5=tp / max(tp + (q & (y == 0)).sum() + (~q & (y == 1)).sum(), 1)))
    return rows


K = ['x_3794', 'y_3794']
rows = skill(pd.read_parquet(os.path.join(ROOT, '09_Analysis', 'spatial_cv_v2', '10x_v2', 'fold_predictions.parquet'))
             .merge(A[K + ['DI_cv']], on=K, how='left'), 'block CV (08e)', 'DI_cv')
for mode in ('regime', 'buffer5000'):
    fp = os.path.join(ROOT, '09_Analysis', 'transfer_v2', mode, 'fold_predictions.parquet')
    if not os.path.exists(fp): continue
    P = pd.read_parquet(fp)
    if mode == 'regime':                                  # DI to the training data of the other regimes
        P = P.merge(A[K + FEATURES], on=K, how='left'); P['DI'] = np.nan
        for g in P.fold.unique():
            m = (P.fold == g).values; o = P[~m]
            P.loc[m, 'DI'] = nn_dist(z(P[m]), z(o.iloc[rng.choice(len(o), min(NREF, len(o)), replace=False)])) / dbar
        rows += skill(P, 'leave-one-regime-out', 'DI')
    else:
        rows += skill(P.merge(A[K + ['DI_cv']], on=K, how='left'), 'block CV, 5 km buffer', 'DI_cv')
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, 'skill_by_di.csv'), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

# --- incident sites inside vs outside the AOA ---------------------------------------------------------------------
from pyproj import Transformer
ev = pd.read_excel(os.path.join(ROOT, 'Events_natural-disasters_2005-2026_all-atts.xlsx'),
                   usecols=['WGSLat', 'WGSLon', 'SkupinaDogodka', 'Dogodek']).dropna(subset=['WGSLat', 'WGSLon'])
ev['key'] = ev.WGSLat.round(5).astype(str) + ',' + ev.WGSLon.round(5).astype(str)
fl = ev[ev.Dogodek == 'poplave ob vodotokih']; fl = fl[fl.key.map(ev[ev.Dogodek.isin(
    ['poplave ob vodotokih', 'poplave meteorne vode', 'poplave zaradi dviga podtalnice, kanalizacijskih vod'])].key.value_counts()) < 20]
s = fl.drop_duplicates('key'); ct = ev[ev.SkupinaDogodka.isin(['Močan veter', 'Visok sneg', 'Žled'])]
ct = ct[ct.key.map(ct.key.value_counts()) < 20].drop_duplicates('key'); ct = ct[~ct.key.isin(set(ev.loc[ev.SkupinaDogodka == 'Poplave', 'key']))]
tr = Transformer.from_crs(4326, 3794, always_xy=True)
with rasterio.open(os.path.join(ROOT, '06_Probability_Maps_v2', 'v2', 'FSM_Ensemble_mean_probability.tif')) as dp, \
        rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as di_:
    pmap = dp.read(1); invf = di_.read(1); Tf = dp.transform


def sample(df_, lab):
    x, y = tr.transform(df_.WGSLon.values, df_.WGSLat.values)
    r, c = map(np.asarray, rasterio.transform.rowcol(Tf, x, y)); r2, c2 = map(np.asarray, rasterio.transform.rowcol(T, x, y))
    k = (r >= 0) & (r < pmap.shape[0]) & (c >= 0) & (c < pmap.shape[1]) & (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < wd)
    d = pd.DataFrame(dict(label=lab, p=pmap[r[k], c[k]], inv=invf[r[k], c[k]], aoa=aoa[r2[k], c2[k]], DI=di[r2[k], c2[k]]))
    return d[(d.p != -9999) & (d.aoa != 255)]


I = pd.concat([sample(s, 1), sample(ct, 0)]); I = I[~((I.label == 1) & (I.inv == 1))]   # reports outside the inventory
rows = []
for name, g in (('all', I), ('inside AOA', I[I.aoa == 1]), ('outside AOA (DI)', I[I.aoa == 0]), ('categorical novelty', I[I.aoa == 2])):
    rows.append(dict(sites=name, n_flood=int((g.label == 1).sum()), n_comparison=int((g.label == 0).sum()),
                     AUC=roc_auc_score(g.label, g.p) if g.label.nunique() == 2 and len(g) > 20 else np.nan))
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, 'incidents_by_aoa.csv'), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
