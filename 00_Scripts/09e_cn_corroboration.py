#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09e -- Corroborate the susceptibility map against SCS Curve Number, per catchment.

The paper already cross-checks its SHAP hierarchy against a model-free statistic
(bivariate frequency ratio) computed from the same inventory.  That is a check
against the labels; it cannot detect an error shared by the model and the
inventory.  This is a check against an *external, physically-based* quantity:
the SCS Curve Number, which encodes soil hydrologic group and land cover as
runoff potential and is derived independently of both the flood inventory and
the ML models.

If the mapped susceptibility carries hydrological signal rather than merely
reproducing the inventory's footprint, catchments with higher CN -- greater
runoff potential, less infiltration -- should carry a larger flood-prone
fraction.  A null or inverted relationship would be a warning.

Data: CN_zonal_stats_227catchments.gpkg, 227 gauge-connected catchments with CN
already summarised for the three antecedent moisture conditions
(dobri / povprecni / slabi = AMC I / II / III).

Two properties of that file are handled explicitly rather than ignored:
  * it totals 22,168 km2 against Slovenia's 20,271 -- catchments are cut to
    gauges, not to the national border, so they overhang it;
  * it carries `coverage_ok`, false where the CN raster covers only part of the
    catchment (Mura Tromeja is 34 % covered).  Those catchments are excluded
    from the correlation and reported separately, because a CN mean over a third
    of a catchment is not a catchment mean.

Susceptibility per catchment is the flood-prone fraction of the *binary*
ensemble map, which is the product a user acts on, restricted to valid cells so
the overhang outside Slovenia cannot dilute it.

Outputs -> 09_Analysis/cn_corroboration/
    catchment_cn_vs_susceptibility.csv
    scatter_cn_vs_susceptibility.png
    correlations.csv
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import geometry_mask
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = os.environ.get('MAP_SIZE', '10x')
GPKG = os.path.join(ROOT, 'CN_zonal_stats_227catchments.gpkg')
BIN  = os.path.join(ROOT, '07_Classified_Maps', SIZE, 'FSM_Ensemble_mean_binary.tif')
REF  = os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')
OUT  = os.path.join(ROOT, '09_Analysis', 'cn_corroboration')
os.makedirs(OUT, exist_ok=True)
FIG_W, DPI = 6.5, 300
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'savefig.dpi': DPI})

c = gpd.read_file(GPKG).to_crs(3794)
print('catchments: %d   coverage_ok: %d' % (len(c), int(c.coverage_ok.astype(bool).sum())), flush=True)

with rasterio.open(BIN) as ds:
    binar = ds.read(1); tf, H, W = ds.transform, ds.height, ds.width; nod = ds.nodata
with rasterio.open(REF) as ds:
    ref = ds.read(1); rnod = ds.nodata
inside = (ref != rnod) if rnod is not None else np.isfinite(ref)
print('binary map %dx%d  valid cells %d' % (H, W, int(inside.sum())), flush=True)

rows = []
for _, r in c.iterrows():
    m = geometry_mask([r.geometry], out_shape=(H, W), transform=tf, invert=True)
    m &= inside
    n = int(m.sum())
    if n == 0:
        rows.append(dict(CATCH_ID=r.CATCH_ID, name=r.CATCH_NAME, n_cells=0)); continue
    v = binar[m]
    v = v[v != nod] if nod is not None else v
    rows.append(dict(CATCH_ID=r.CATCH_ID, name=r.CATCH_NAME,
                     n_cells=n, flood_frac=float((v == 1).sum()) / max(len(v), 1),
                     inv_frac=float((ref[m] == 1).mean()),
                     area_km2=r.geometry.area / 1e6,
                     cn_dry=r.CN_dobri_mean, cn_avg=r.CN_povprec_mean,
                     cn_wet=r.CN_slabi_mean, cn_range=r.CN_range_mean,
                     coverage_pct=r.coverage_pct, coverage_ok=bool(r.coverage_ok),
                     region=r.REGION))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, 'catchment_cn_vs_susceptibility.csv'), index=False)

use = d[(d.coverage_ok) & (d.n_cells > 0) & d.cn_avg.notna()].copy()
print('\nused in correlation: %d of %d (coverage_ok and inside Slovenia)' % (len(use), len(d)), flush=True)

cors = []
for col, lab in [('cn_dry', 'CN AMC I (dry)'), ('cn_avg', 'CN AMC II (average)'),
                 ('cn_wet', 'CN AMC III (wet)'), ('cn_range', 'CN range')]:
    pr, pp = pearsonr(use[col], use.flood_frac)
    sr, sp = spearmanr(use[col], use.flood_frac)
    cors.append(dict(cn_variable=lab, column=col, n=len(use),
                     pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp))
    print('  %-22s  r=%+.3f (p=%.2g)   rho=%+.3f (p=%.2g)' % (lab, pr, pp, sr, sp), flush=True)
# The same correlation for the training inventory itself.  If the labels already
# carry most of the CN association, the map's correlation is a consistency check,
# not independent evidence -- CN is also built from soil group and land cover,
# which are close to two of the model inputs.
for col, lab in [('cn_dry', 'CN AMC I (dry)'), ('cn_avg', 'CN AMC II (average)'),
                 ('cn_wet', 'CN AMC III (wet)'), ('cn_range', 'CN range')]:
    pr, pp = pearsonr(use[col], use.inv_frac)
    sr, sp = spearmanr(use[col], use.inv_frac)
    cors.append(dict(cn_variable=lab + ' vs inventory', column=col, n=len(use),
                     pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp))
    print('  %-22s  inventory: r=%+.3f   rho=%+.3f' % (lab, pr, sr), flush=True)
rk = lambda x: rankdata(x)
def _resid(y, x):
    X = np.c_[np.ones(len(x)), x]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
part = np.corrcoef(_resid(rk(use.flood_frac), rk(use.inv_frac)),
                   _resid(rk(use.cn_avg), rk(use.inv_frac)))[0, 1]
sr, sp = spearmanr(use.flood_frac, use.inv_frac)
cors.append(dict(cn_variable='map vs inventory', column='inv_frac', n=len(use),
                 spearman_rho=sr, spearman_p=sp))
cors.append(dict(cn_variable='CN AMC II vs map, partial on inventory (rank)',
                 column='cn_avg', n=len(use), spearman_rho=part))
print('  map vs inventory rho=%+.3f   partial rho(map, CN II | inventory)=%+.3f' % (sr, part), flush=True)
pd.DataFrame(cors).to_csv(os.path.join(OUT, 'correlations.csv'), index=False)

fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.5), layout='constrained')
for ax, (col, lab) in zip(axes, [('cn_dry', 'CN AMC I (dry)'), ('cn_avg', 'CN AMC II (average)'),
                                 ('cn_wet', 'CN AMC III (wet)')]):
    ax.scatter(use[col], 100 * use.flood_frac, s=11, alpha=0.7,
               color='#2196F3', edgecolor='none')
    b = np.polyfit(use[col], 100 * use.flood_frac, 1)
    xg = np.linspace(use[col].min(), use[col].max(), 50)
    ax.plot(xg, np.polyval(b, xg), color='#D32F2F', lw=1.2)
    sr, sp = spearmanr(use[col], use.flood_frac)
    ax.set_title('%s\n$\\rho$=%+.3f (p=%.1e)' % (lab, sr, sp), fontsize=8)
    ax.set_xlabel(lab, fontsize=7.5); ax.grid(alpha=0.3)
axes[0].set_ylabel('flood-prone area (%)', fontsize=7.5)
fig.suptitle('Ensemble susceptibility vs SCS Curve Number, %d gauge catchments' % len(use),
             fontsize=9.5, fontweight='bold')
fig.savefig(os.path.join(OUT, 'scatter_cn_vs_susceptibility.png'), dpi=DPI); plt.close(fig)
print('\nSaved -> 09_Analysis/cn_corroboration/', flush=True)
