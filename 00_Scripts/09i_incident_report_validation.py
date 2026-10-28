#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09i -- Independent validation against civil-protection flood incident reports.

Every other check in the paper is scored against the flood inventory the models
were trained on, or against layers derived from it (the frequent / rare /
extremely-rare zones ARE the inventory).  This scores the delivered MAP_SIZE
(default 10x) probability and binary maps against a different kind of record:
geolocated intervention reports, 2005-2026, from the national natural-disaster
incident log (Events_natural-disasters_2005-2026_all-atts.xlsx).

What a report is, and is not.  A report marks where a crew responded -- a house,
a road, a culvert -- not the extent of inundation, so a report one or two cells
off the floodplain is not a model error.  The binary hit rate is therefore also
given with a 50 m tolerance (max over a 9 x 9 cell window).

Three design choices, each argued because the naive version is badly biased:

  * Placeholder coordinates.  About a fifth of flood rows sit on coordinates
    reused >= 20 times (one point carries 2,880 reports); they are default
    locations, not flood sites, and are excluded.  Sites are deduplicated so a
    street reported 40 times in one event counts once.
  * Independence.  Sites inside the training inventory are reported, but the
    headline is the subset OUTSIDE it, which the models never saw as positive.
  * Reporting bias.  Reports come from where people live, and people live in
    valleys, so random background cells (which include empty Alpine terrain)
    inflate AUC.  The comparison class is therefore wind, snow and ice incident
    sites that never carried a flood report: same reporting process, a hazard
    unrelated to flooding.  AUC against random cells is written too, labelled.

The August 2023 flood (3-19 Aug) is scored separately.  The inventory's version
dates (DATP_ZAC) are 2014-12-01 and 2017-12-01 except two extremely-rare polygons
(0.48 km2) revised in 2023-11 and 2025-09; the number of August 2023 sites inside
those two is written to the summary so the "post-dates the inventory" claim is
checked, not assumed.

Hazard types (Dogodek): riverine "poplave ob vodotokih" is the process the
inventory maps; pluvial "poplave meteorne vode" and groundwater/sewer
"poplave zaradi dviga podtalnice, kanalizacijskih vod" are reported as contrasts.

Needs openpyxl to read the workbook.

Outputs -> 09_Analysis/incident_reports/
    summary.csv                 per model x hazard x subset: sites, hit rates, AUCs
    fr_by_decile.csv            report frequency ratio per national probability decile (ensemble)
    catchment_density.csv       report sites per 100 km2 vs flood-prone share, 227 catchments
    notes.txt                   data screening counts, inventory-date check, catchment correlation
    Manuscript figure           Manuscript/figures/jem/JEM_Fig08b_incident_validation.png
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '00_Scripts'))
from _figstyle import FIG_W, RCPARAMS, MODELS, model as model_label

SIZE = os.environ.get('MAP_SIZE', '10x')
XLSX = os.environ.get('EVENTS_XLSX',
                      os.path.join(ROOT, 'Events_natural-disasters_2005-2026_all-atts.xlsx'))
CLS_DIR = os.environ.get('FLOOD_CLASS_DIR',
    '/home/arora/FSM_Slovenia/FSM_Pipeline_LULC_archive/Vectors/Flooded_Areas')
PROB = os.path.join(ROOT, '06_Probability_Maps', SIZE, 'FSM_%s_probability.tif')
REF = os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')
THR = os.path.join(ROOT, '08_Validation', SIZE, 'threshold_summary.csv')
GPKG = os.path.join(ROOT, 'CN_zonal_stats_227catchments.gpkg')
OUT = os.path.join(ROOT, '09_Analysis', 'incident_reports')
FIG = os.path.join(ROOT, 'Manuscript', 'figures', 'jem', 'JEM_Fig08b_incident_validation.png')
os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.dirname(FIG), exist_ok=True)

RIVER = 'poplave ob vodotokih'
HAZARDS = {RIVER: 'riverine',
           'poplave meteorne vode': 'pluvial',
           'poplave zaradi dviga podtalnice, kanalizacijskih vod': 'groundwater/sewer'}
CONTROL = ['Močan veter', 'Visok sneg', 'Žled']       # wind, snow, ice
AUG23 = ('2023-08-03', '2023-08-20')
PLACEHOLDER_N, TOL, SEED = 20, 4, 42                   # TOL cells each side = 50 m
t0 = time.time()
notes = []
def note(s):
    print(s, flush=True); notes.append(s)

# ── incident log ──────────────────────────────────────────────────────────────
ev = pd.read_excel(XLSX, usecols=['NastanekDatumCas', 'WGSLat', 'WGSLon',
                                  'SkupinaDogodka', 'Dogodek'])
note('incident log: %d rows, %s to %s' % (len(ev), ev.NastanekDatumCas.min().date(),
                                          ev.NastanekDatumCas.max().date()))
ev = ev.dropna(subset=['WGSLat', 'WGSLon'])
ev['key'] = ev.WGSLat.round(5).astype(str) + ',' + ev.WGSLon.round(5).astype(str)

fl = ev[ev.Dogodek.isin(HAZARDS)].copy()
rep = fl.key.map(fl.key.value_counts())
note('flood rows by type: ' + ', '.join('%s %d' % (HAZARDS[k], v)
                                       for k, v in fl.Dogodek.value_counts().items()))
note('flood rows on placeholder coordinates (reused >= %d x): %d (%.1f %%)'
     % (PLACEHOLDER_N, (rep >= PLACEHOLDER_N).sum(), 100 * (rep >= PLACEHOLDER_N).mean()))
fl = fl[rep < PLACEHOLDER_N]
fl['aug23'] = fl.NastanekDatumCas.between(*AUG23, inclusive='left')
# one row per (site, hazard); a site counts for Aug 2023 if any report there fell in the event
sites = (fl.groupby(['key', 'Dogodek'])
           .agg(lat=('WGSLat', 'first'), lon=('WGSLon', 'first'), aug23=('aug23', 'any'))
           .reset_index())
note('unique flood sites: ' + ', '.join('%s %d' % (HAZARDS[k], v)
                                       for k, v in sites.Dogodek.value_counts().items()))

ct = ev[ev.SkupinaDogodka.isin(CONTROL)]
ct = ct[ct.key.map(ct.key.value_counts()) < PLACEHOLDER_N].drop_duplicates('key')
ct = ct[~ct.key.isin(set(ev.loc[ev.SkupinaDogodka == 'Poplave', 'key']))]
note('control sites (wind/snow/ice, never flood-reported): %d' % len(ct))

tr = Transformer.from_crs(4326, 3794, always_xy=True)
sites['x'], sites['y'] = tr.transform(sites.lon.values, sites.lat.values)
cx, cy = tr.transform(ct.WGSLon.values, ct.WGSLat.values)


def rc(ds, xs, ys):
    r, c = rasterio.transform.rowcol(ds.transform, xs, ys)
    r, c = np.asarray(r), np.asarray(c)
    return r, c, (r >= 0) & (r < ds.height) & (c >= 0) & (c < ds.width)


def at(a, r, c, ok, nd):
    v = np.full(len(r), np.nan)
    v[ok] = a[r[ok], c[ok]]
    if nd is not None:
        v[v == nd] = np.nan
    return v


def win_max(a, r, c, ok, nd, k):
    v = np.full(len(r), np.nan)
    for i in np.flatnonzero(ok):
        w = a[max(r[i] - k, 0):r[i] + k + 1, max(c[i] - k, 0):c[i] + k + 1].astype('float32')
        if nd is not None:
            w[w == nd] = np.nan
        if np.isfinite(w).any():
            v[i] = np.nanmax(w)
    return v


# ── training inventory at each site ───────────────────────────────────────────
with rasterio.open(REF) as ds:
    inv = ds.read(1); ind = ds.nodata; tf = ds.transform; shape = (ds.height, ds.width)
    r, c, ok = rc(ds, sites.x.values, sites.y.values)
sites['inv'] = at(inv, r, c, ok, ind)
sites = sites[sites.inv.notna()].reset_index(drop=True)
note('sites inside Slovenia valid grid: %d' % len(sites))

# inventory-date check for the Aug 2023 subset
try:
    xr = gpd.read_file(os.path.join(CLS_DIR, 'Extreme_Rare_Flood_Slovenia.shp')).to_crs(3794)
    late = xr[xr.DATP_ZAC.astype(str).str[:10] >= '2023/08/01']
    pts = gpd.GeoSeries(gpd.points_from_xy(sites.x, sites.y), crs=3794)
    inside_late = pts.within(late.geometry.union_all()) if len(late) else pd.Series(False, index=sites.index)
    note('extremely-rare polygons revised after 2023-08-01: %d (%.2f km2); '
         'August 2023 riverine sites inside them: %d'
         % (len(late), late.area.sum() / 1e6,
            int((inside_late & sites.aug23 & (sites.Dogodek == RIVER)).sum())))
except Exception as e:                                     # layer not on this machine
    note('inventory-date check skipped: %s' % e)

# ── score every model ─────────────────────────────────────────────────────────
thr = pd.read_csv(THR).set_index('Model').CSI_threshold
rng = np.random.default_rng(SEED)
valid_idx = np.flatnonzero((inv != ind).ravel())
bg_idx = rng.choice(valid_idx, 2_000_000, replace=False)
del valid_idx

subsets = {
    'all sites':                  lambda s: np.ones(len(s), bool),
    'outside inventory':          lambda s: s.inv.values == 0,
    'Aug 2023, outside inventory': lambda s: (s.inv.values == 0) & s.aug23.values,
    'other dates, outside inventory': lambda s: (s.inv.values == 0) & ~s.aug23.values,
}
rows, ens_p, ens_bg, ens_ct = [], None, None, None
for m in MODELS:
    with rasterio.open(PROB % m) as ds:
        P = ds.read(1); pnd = ds.nodata
        sr_, sc_, sok = rc(ds, sites.x.values, sites.y.values)
        kr, kc, kok = rc(ds, cx, cy)
    p0 = at(P, sr_, sc_, sok, pnd)
    riv = sites.Dogodek.values == RIVER
    p50 = np.full(len(sites), np.nan)
    p50[riv] = win_max(P, sr_[riv], sc_[riv], sok[riv], pnd, TOL)
    pc = at(P, kr, kc, kok, pnd); pc = pc[np.isfinite(pc)]
    pb = P.ravel()[bg_idx].astype(float); pb = pb[pb != pnd] if pnd is not None else pb
    del P
    t = thr[m]
    rows.append(dict(model=m, hazard='control (wind/snow/ice)', subset='control sites',
                     n_sites=len(pc), flagged=np.mean(pc >= t), threshold=t))
    rows.append(dict(model=m, hazard='random valid cells', subset='background',
                     n_sites=len(pb), flagged=np.mean(pb >= t), threshold=t))
    for hz, hname in HAZARDS.items():
        for sname, f in subsets.items():
            sel = (sites.Dogodek.values == hz) & f(sites) & np.isfinite(p0)
            if sel.sum() < 30:
                continue
            p = p0[sel]
            rows.append(dict(
                model=m, hazard=hname, subset=sname, n_sites=int(sel.sum()), threshold=t,
                inside_inventory=float(np.mean(sites.inv.values[sel] == 1)),
                flagged=float(np.mean(p >= t)),
                flagged_50m=float(np.mean(p50[sel] >= t)) if hz == RIVER else np.nan,
                auc_vs_control=roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(pc))], np.r_[p, pc]),
                auc_vs_random=roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(pb))], np.r_[p, pb])))
    if m == 'Ensemble_mean':
        ens_p, ens_bg, ens_ct = p0, pb, pc
    print('  %-14s done  (%.0f s)' % (m, time.time() - t0), flush=True)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
print(summ[summ.hazard == 'riverine'][['model', 'subset', 'n_sites', 'flagged', 'flagged_50m',
                                       'auc_vs_control', 'auc_vs_random']].round(3).to_string(index=False))

# ── frequency ratio by national probability decile (ensemble) ─────────────────
edges = np.quantile(ens_bg, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
area = np.histogram(ens_bg, edges)[0] / len(ens_bg)
def fr(p):
    p = p[np.isfinite(p)]
    return (np.histogram(p, edges)[0] / len(p)) / area
riv_out = (sites.Dogodek.values == RIVER) & (sites.inv.values == 0)
fr_tab = pd.DataFrame({
    'decile': np.arange(1, 11), 'p_low': edges[:-1], 'p_high': edges[1:],
    'riverine_outside_inventory': fr(ens_p[riv_out]),
    'riverine_aug2023_outside_inventory': fr(ens_p[riv_out & sites.aug23.values]),
    'pluvial_outside_inventory': fr(ens_p[(sites.Dogodek.values == 'poplave meteorne vode')
                                          & (sites.inv.values == 0)]),
    'control_wind_snow_ice': fr(ens_ct)})
fr_tab.to_csv(os.path.join(OUT, 'fr_by_decile.csv'), index=False)
print(fr_tab.round(3).to_string(index=False))

# ── catchment level (reported for completeness; weak by design of the data) ──
cat = gpd.read_file(GPKG).to_crs(3794)
lab = rasterize(((g, i) for g, i in zip(cat.geometry, cat.CATCH_ID)), out_shape=shape,
                transform=tf, fill=0, dtype='int32')
with rasterio.open(os.path.join(ROOT, '07_Classified_Maps', SIZE, 'FSM_Ensemble_mean_binary.tif')) as ds:
    B = ds.read(1)
m_ = inv != ind
n_cells = np.bincount(lab[m_], minlength=lab.max() + 1)
map_frac = np.bincount(lab[m_], weights=(B[m_] == 1), minlength=lab.max() + 1) / np.maximum(n_cells, 1)
inv_frac = np.bincount(lab[m_], weights=(inv[m_] == 1), minlength=lab.max() + 1) / np.maximum(n_cells, 1)
del B, lab
pts = gpd.GeoDataFrame(sites[sites.Dogodek == RIVER], crs=3794,
                       geometry=gpd.points_from_xy(sites.x[sites.Dogodek == RIVER],
                                                   sites.y[sites.Dogodek == RIVER]))
j = gpd.sjoin(pts, cat[['CATCH_ID', 'geometry']], predicate='within')
cd = pd.DataFrame({'CATCH_ID': cat.CATCH_ID})
cd['slo_km2'] = n_cells[cd.CATCH_ID] * 156.25 / 1e6
cd['map_frac'] = map_frac[cd.CATCH_ID]; cd['inv_frac'] = inv_frac[cd.CATCH_ID]
cd['sites_per_100km2'] = cd.CATCH_ID.map(j.groupby('CATCH_ID').size()).fillna(0) / cd.slo_km2 * 100
cd['sites_outside_inv_per_100km2'] = (cd.CATCH_ID.map(j[j.inv == 0].groupby('CATCH_ID').size())
                                      .fillna(0) / cd.slo_km2 * 100)
cd = cd[cd.slo_km2 >= 10]
cd.to_csv(os.path.join(OUT, 'catchment_density.csv'), index=False)
for a_, b_ in [('map_frac', 'sites_per_100km2'), ('map_frac', 'sites_outside_inv_per_100km2'),
               ('inv_frac', 'sites_per_100km2')]:
    s_, p_ = spearmanr(cd[a_], cd[b_])
    note('catchments (n=%d, >=10 km2 in Slovenia): rho(%s, %s) = %.3f (p=%.1e)'
         % (len(cd), a_, b_, s_, p_))
open(os.path.join(OUT, 'notes.txt'), 'w').write('\n'.join(notes) + '\n')

# ── manuscript figure ─────────────────────────────────────────────────────────
plt.rcParams.update(RCPARAMS)
BLUE, ORANGE, GREY, MUTED = '#2a78d6', '#eb6834', '#8a8a86', '#6b6b67'
fig, (a, b) = plt.subplots(1, 2, figsize=(FIG_W, 2.6), gridspec_kw=dict(wspace=0.62))
x = fr_tab.decile.values
a.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
a.text(0.8, 1.08, 'FR = 1', color=MUTED, fontsize=6, va='bottom')
series = [('riverine_outside_inventory', 'River-flood sites outside inventory', BLUE, 'o'),
          ('riverine_aug2023_outside_inventory', 'of which August 2023', ORANGE, 's'),
          ('control_wind_snow_ice', 'Wind/snow/ice sites (comparison)', GREY, '^')]
for col, lab_, colr, mk in series:
    a.plot(x, fr_tab[col], color=colr, lw=1.6, marker=mk, ms=4, mec='white', mew=0.6,
           label=lab_, zorder=3)
a.set_yscale('log'); a.set_xticks(x); a.minorticks_off()
a.set_xlabel('Ensemble probability decile (national)')
a.set_ylabel('Frequency ratio (sites / area)')

rv = summ[summ.hazard == 'riverine'].set_index(['subset', 'model'])
yy = np.arange(len(MODELS))[::-1]
for sname, colr, mk, dy in [('outside inventory', BLUE, 'o', 0.12),
                            ('Aug 2023, outside inventory', ORANGE, 's', -0.12)]:
    b.scatter([rv.loc[(sname, m_), 'auc_vs_control'] for m_ in MODELS], yy + dy, s=22,
              color=colr, marker=mk, edgecolor='white', linewidth=0.6, zorder=3)
b.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
b.set_yticks(yy); b.set_yticklabels([model_label(m_) for m_ in MODELS])
b.set_xlim(0.45, 0.85); b.set_ylim(yy[-1] - 0.6, yy[0] + 0.75)
b.set_xlabel('$AUC$, flood sites vs comparison sites')
b.text(0.505, yy[0] + 0.45, 'no skill', color=MUTED, fontsize=6, va='bottom')
for ax_, l in [(a, 'a'), (b, 'b')]:
    ax_.text(-0.22, 1.02, '(%s)' % l, transform=ax_.transAxes, fontweight='bold', fontsize=8.5)
    ax_.grid(True, which='major', color='#e6e6e3', lw=0.5, zorder=0)
    for sp in ('top', 'right'):
        ax_.spines[sp].set_visible(False)
fig.legend(*a.get_legend_handles_labels(), frameon=False, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 0.0), handlelength=1.8, fontsize=6.5)
fig.savefig(FIG, bbox_inches='tight'); plt.close(fig)
print('\nSaved -> %s and %s  (%.0f s)' % (os.path.relpath(OUT, ROOT), os.path.relpath(FIG, ROOT),
                                           time.time() - t0), flush=True)
