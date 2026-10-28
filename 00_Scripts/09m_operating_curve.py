"""
09m — Operating curve: agreement with the inventory vs detection of independent flood reports (review 2026-10-09).
==================================================================================================================
For the ensemble probability map of each model set and thresholds 0.05-0.95: base-rate CSI / POD / FAR / FBI against the
inventory (all valid cells, via probability histograms), and the share of river-flood incident sites outside the
inventory and of wind/snow/ice comparison sites that the binary map flags (exact cell; and within 50 m).
Shows the trade-off between matching an incomplete inventory and finding flood terrain it misses.
    PYTHONPATH=<dir with openpyxl> python 00_Scripts/09m_operating_curve.py
Outputs → 09_Analysis/operating_curve/curve.csv, fig_operating_curve.{png,pdf}
"""
import os, sys, numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from scipy.ndimage import maximum_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '09_Analysis', 'operating_curve'); os.makedirs(OUT, exist_ok=True)
SETS = {'balanced sampling (adopted)': 'v2', 'hard-negative sampling (sensitivity test)': 'v2_hardneg'}
TS = np.round(np.r_[np.arange(0.05, 0.951, 0.01), 0.885], 3); TS.sort()

ev = pd.read_excel(os.path.join(ROOT, 'Events_natural-disasters_2005-2026_all-atts.xlsx'),
                   usecols=['WGSLat', 'WGSLon', 'SkupinaDogodka', 'Dogodek']).dropna(subset=['WGSLat', 'WGSLon'])
ev['key'] = ev.WGSLat.round(5).astype(str) + ',' + ev.WGSLon.round(5).astype(str)
fl = ev[ev.Dogodek == 'poplave ob vodotokih']; fl = fl[fl.key.map(ev[ev.Dogodek.isin(
    ['poplave ob vodotokih', 'poplave meteorne vode', 'poplave zaradi dviga podtalnice, kanalizacijskih vod'])].key.value_counts()) < 20]
s = fl.drop_duplicates('key'); ct = ev[ev.SkupinaDogodka.isin(['Močan veter', 'Visok sneg', 'Žled'])]
ct = ct[ct.key.map(ct.key.value_counts()) < 20].drop_duplicates('key'); ct = ct[~ct.key.isin(set(ev.loc[ev.SkupinaDogodka == 'Poplave', 'key']))]
tr = Transformer.from_crs(4326, 3794, always_xy=True)
with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
    inv = d.read(1); ok0 = inv != d.nodata; T = d.transform; px = abs(T.a) ** 2 / 1e6


def rc(lon, lat):
    x, y = tr.transform(lon, lat); r, c = rasterio.transform.rowcol(T, x, y); r, c = np.asarray(r), np.asarray(c)
    k = (r >= 0) & (r < inv.shape[0]) & (c >= 0) & (c < inv.shape[1]); return r[k], c[k]


fr, fc = rc(s.WGSLon.values, s.WGSLat.values); out = inv[fr, fc] == 0; fr, fc = fr[out], fc[out]
cr, cc = rc(ct.WGSLon.values, ct.WGSLat.values)
rows = []
for label, tag in SETS.items():
    with rasterio.open(os.path.join(ROOT, '06_Probability_Maps_v2', tag, 'FSM_Ensemble_mean_probability.tif')) as d:
        p = d.read(1)
    ok = ok0 & (p != -9999)
    o = inv[ok] == 1; pv = p[ok]
    hf, _ = np.histogram(pv[o], bins=np.r_[TS, 1.01]); hn, _ = np.histogram(pv[~o], bins=np.r_[TS, 1.01])
    tp_c, fp_c = np.cumsum(hf[::-1])[::-1], np.cumsum(hn[::-1])[::-1]          # cells with p >= t
    nflood = o.sum()
    p50 = maximum_filter(np.where(ok, p, -1), size=9)                            # 50 m tolerance (4 cells each side)
    sf, sc_, sf50, sc50 = p[fr, fc], p[cr, cc], p50[fr, fc], p50[cr, cc]
    vf, vc = sf != -9999, sc_ != -9999
    for i, t in enumerate(TS):
        tp, fp = tp_c[i], fp_c[i]; fn = nflood - tp
        rows.append(dict(set=label, threshold=t, CSI=tp / (tp + fp + fn), POD=tp / nflood, FAR=fp / max(tp + fp, 1),
                         FBI=(tp + fp) / nflood, area_km2=(tp + fp) * px,
                         reports_flagged=(sf[vf] >= t).mean(), comparison_flagged=(sc_[vc] >= t).mean(),
                         reports_flagged50=(sf50[vf] >= t).mean(), comparison_flagged50=(sc50[vc] >= t).mean()))
C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, 'curve.csv'), index=False)
print(C[C.threshold.isin([0.3, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9])].round(3).to_string(index=False))

import matplotlib.pyplot as plt
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
cols = {'balanced sampling (adopted)': '#2a78d6', 'hard-negative sampling (sensitivity test)': '#eb6834'}
for label, g in C.groupby('set', sort=False):
    axs[0].plot(g.threshold, g.CSI, color=cols[label], lw=1.8, label=f'{label}: CSI vs inventory')
    axs[0].plot(g.threshold, g.reports_flagged, color=cols[label], lw=1.2, ls='--', label=f'{label}: independent reports flagged')
    axs[1].plot(g.reports_flagged, g.CSI, color=cols[label], lw=1.8, label=label)
    if label.startswith('balanced'):
        for t, name, off in ((0.57, 'screening product\n(t = 0.57)', (6, 4)), (0.885, 'area estimate\n(t = 0.885)', (6, -48))):
            r = g[np.isclose(g.threshold, t)].iloc[0]
            axs[1].plot(r.reports_flagged, r.CSI, 'o', ms=5, mfc=cols[label], mec=SURF, zorder=5)
            axs[1].annotate(name, (r.reports_flagged, r.CSI), xytext=off, textcoords='offset points', fontsize=6, color=INK)
            for a in (axs[0],):
                a.axvline(t, color=INK2, lw=0.6, ls=':')
axs[0].set_xlabel('threshold on ensemble probability'); axs[0].set_ylabel('score'); axs[0].legend(fontsize=5.4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
axs[1].set_xlabel('independent flood reports flagged (outside inventory)')
axs[1].set_ylabel('CSI against the inventory (true base rate)'); axs[1].legend(fontsize=5.8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
for a in axs:
    a.grid(color='#e6e5e0', lw=0.5); [a.spines[k].set_visible(False) for k in ('top', 'right')]
axs[0].set_title('(a) Inventory agreement and report detection by threshold', loc='left', fontsize=7)
axs[1].set_title('(b) The trade-off: no setting maximises both', loc='left', fontsize=7)
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig_operating_curve.{e}'), dpi=300, bbox_inches='tight')
print('->', OUT)
