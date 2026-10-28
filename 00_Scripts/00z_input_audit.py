"""
00z — Audit of every conditioning factor and the flood inventory before (re)training (review 2026-10-09).
========================================================================================================
For each raster in 01_Input_Features: grid (shape, transform, CRS) vs the inventory raster; dtype / nodata;
valid-cell coverage vs the national mask (inventory valid cells); value range and percentiles; sentinel values
(-9999, -3.4e38, 65535) present as DATA; physically impossible values (negative distances, densities,
accumulation, slope, TRI); non-integer values in categorical layers (a sign of bilinear regridding); exact-zero
share; and, for the training samples, sentinel or NaN values that reached the feature table.
Writes 09_Analysis/input_audit/{raster_audit.csv, sample_audit.csv, notes.txt}.
"""
import os, numpy as np, pandas as pd, rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, '01_Input_Features')
OUT = os.path.join(ROOT, '09_Analysis', 'input_audit'); os.makedirs(OUT, exist_ok=True)
CATEGORICAL = {'LULC', 'Pedoloska_grouped', 'lito_poly_250_d96_grouped', 'Hgk_grouped', 'Flooded_NoFlood'}
NONNEG = {'Distance_to_River', 'Distance_to_Road', 'River_Density', 'drainage_density', 'flow_accumulation', 'slope',
          'tri', 'ls_factor', 'spi', 'roughness', 'slovenia_annual_max_daily_mean_3794_12m5',
          'slovenia_annual_max_daily_p90_3794_12m5', 'slovenia_mean_annual_rainfall_3794_12m5'}
SENT = [-9999.0, -3.4028234663852886e+38, 65535.0, -32768.0]
with rasterio.open(os.path.join(F, 'Flooded_NoFlood.tif')) as d:
    ref_t, ref_s, ref_c = d.transform, d.shape, d.crs
    mask = d.read(1) != d.nodata                            # national valid mask (inside Slovenia)
rows = []
for f in sorted(x for x in os.listdir(F) if x.endswith('.tif')):
    name = f[:-4]
    with rasterio.open(os.path.join(F, f)) as d:
        a = d.read(1); nd = d.nodata
        r = dict(layer=name, dtype=str(a.dtype), nodata=nd, same_grid=(d.transform == ref_t and d.shape == ref_s),
                 crs=d.crs.to_epsg() if d.crs else None)
    isnd = (a == nd) if nd is not None and not np.isnan(nd) else np.zeros(a.shape, bool)
    fin = np.isfinite(a) & ~isnd
    r['valid_in_mask'] = (fin & mask).sum() / mask.sum()
    r['valid_outside_mask'] = int((fin & ~mask).sum())
    v = a[fin & mask].astype('float64')
    sentinel_hits = {s: int(np.isclose(v, s).sum()) for s in SENT}
    r['sentinel_as_data'] = sum(sentinel_hits.values())
    if v.size:
        p = np.percentile(v[::max(1, v.size // 5_000_000)], [0, 1, 50, 99, 100])
        r.update(min=p[0], p1=p[1], median=p[2], p99=p[3], max=p[4], zero_share=float((v == 0).mean()))
        if name in NONNEG:
            r['negative_share'] = float((v < 0).mean())
        if name in CATEGORICAL:
            r['non_integer_share'] = float((np.abs(v - np.round(v)) > 1e-6).mean())
            r['n_classes'] = int(np.unique(np.round(v)).size)
        if name == 'hand':
            r['negative_share'] = float((v < 0).mean())
    rows.append(r); print(name, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'raster_audit.csv'), index=False)
pd.set_option('display.width', 250); print(R.round(4).to_string(index=False))

# samples
feats = [l.strip() for l in open(os.path.join(ROOT, '02_Collinearity', 'selected_features.txt')) if l.strip()]
srows = []
for fn in ('samples_train_10x.csv', 'samples_test_10x.csv'):
    s = pd.read_csv(os.path.join(ROOT, '03_Samples', fn))
    for c in feats:
        x = s[c].astype('float64')
        srows.append(dict(file=fn, feature=c, n=len(x), nan=int(x.isna().sum()),
                          sentinel=int(sum(np.isclose(x, v).sum() for v in SENT)), min=x.min(), max=x.max(),
                          flood_mean=x[s.label == 1].mean(), noflood_mean=x[s.label == 0].mean()))
S = pd.DataFrame(srows); S.to_csv(os.path.join(OUT, 'sample_audit.csv'), index=False)
print(S[(S.nan > 0) | (S.sentinel > 0)].to_string(index=False) or 'samples: no NaN or sentinel values')
