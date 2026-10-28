"""
10z — Graphical abstract for the JFRM v2 package (review 2026-10-11): v2 ensemble probability cut at its national
quintiles (equal-area classes, as the five-class product) and the validation tiers of v2.
    python 00_Scripts/10z_graphical_abstract_v2.py  → Manuscript/JFRM_v2/graphical_abstract.png
"""
import importlib.util, os, sys
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location('f10s', os.path.join(HERE, '10s_jem_final_figures.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
ROOT = os.path.dirname(HERE)
p = f._map(os.path.join(ROOT, '06_Probability_Maps_v2', 'v2', 'FSM_Ensemble_mean_probability.tif'))
ok = np.isfinite(p) & (p >= 0)
q = np.quantile(p[ok], [0.2, 0.4, 0.6, 0.8])
a = np.where(ok, np.digitize(p, q) + 1, np.nan)
cols = ['#2C7BB6', '#ABD9E9', '#FFFFBF', '#FDAE61', '#D7191C']
fig = plt.figure(figsize=(7.0, 3.2))
ax = fig.add_axes([0.01, 0.12, 0.56, 0.86])
ax.imshow(np.ma.masked_invalid(a), cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(0.5, 6), 5), interpolation='nearest'); ax.axis('off')
ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4, label=l) for c, l in
                   zip(cols, ['very low', 'low', 'moderate', 'high', 'very high'])],
          loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=7, frameon=False, handlelength=1.0,
          columnspacing=0.8, title='Ensemble flood susceptibility, 12.5 m (national quintiles)', title_fontsize=7.5)
tiers = [('Random hold-out', 'AUC 0.99', 'where the inventory already is', '#9e9e9e'),
         ('Unseen regions', 'AUC 0.97', '50 km blocks, 5 km buffer, regimes withheld;\nweakest: sub-Pannonian plains (0.93)', '#3f7cac'),
         ('Civil-protection reports', 'AUC 0.73', 'flood sites vs wind/snow/ice sites;\nAugust 2023: 0.76', '#2d6a3e')]
for k, (t, v, sub, c) in enumerate(tiers):
    b = fig.add_axes([0.60, 0.70 - k * 0.30, 0.38, 0.25]); b.axis('off')
    b.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0,rounding_size=0.06', facecolor='white',
                                        edgecolor=c, lw=1.4, transform=b.transAxes))
    b.text(0.05, 0.72, t, fontsize=8, fontweight='bold', color=c, va='center', transform=b.transAxes)
    b.text(0.95, 0.72, v, fontsize=10, fontweight='bold', ha='right', va='center', transform=b.transAxes)
    b.text(0.05, 0.30, sub, fontsize=6.8, color='#333', va='center', transform=b.transAxes)
out = os.path.join(ROOT, 'Manuscript', 'JFRM_v2', 'graphical_abstract.png')
fig.savefig(out, dpi=300, bbox_inches='tight'); print('->', out)
