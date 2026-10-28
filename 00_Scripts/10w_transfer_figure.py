"""
10w — Transferability figure (review 2026-10-09): (a) dissimilarity index and area of applicability (09n),
(b) ensemble AUC per held-out unit under block CV, 5 km buffered block CV and leave-one-regime-out (08e, 08f).
    python 00_Scripts/10w_transfer_figure.py  → 09_Analysis/aoa_v2/fig_transfer.{png,pdf}
"""
import json, os, numpy as np, pandas as pd, rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, '09_Analysis', 'aoa_v2'); TV = os.path.join(ROOT, '09_Analysis', 'transfer_v2')
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
thr = json.load(open(os.path.join(A, 'summary.json')))['threshold']
with rasterio.open(os.path.join(A, 'DI_100m.tif')) as d:
    di = d.read(1, masked=True); b = d.bounds
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]
bins = [0, 0.25 * thr, 0.5 * thr, 0.75 * thr, thr, 10]
cols = ['#dbe7f6', '#a9c6ec', '#6f9fdc', '#2a78d6', '#eb6834']
cls = np.ma.masked_array(np.digitize(di.filled(0), bins[1:-1]), di.mask)
im = ax.imshow(cls, cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(-0.5, 5), 5), extent=(b.left, b.right, b.bottom, b.top),
               interpolation='nearest')
ax.set_axis_off()
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, ticks=range(5))
cb.ax.set_yticklabels(['DI 0–25 %', '25–50 %', '50–75 %', '75–100 %', 'outside AOA'], fontsize=5.8); cb.outline.set_visible(False)
ax.set_title('(a) Dissimilarity index (share of AOA threshold)', loc='left', fontsize=7)

ax = axs[1]
F = []
p = os.path.join(ROOT, '09_Analysis', 'spatial_cv_v2', '10x_v2', 'per_fold_metrics.csv')
e = pd.read_csv(p); e = e[(e.Model == 'Ensemble_mean') & (e.threshold_rule == 'national')]
F.append(('50 km block CV', e.Fold.astype(str).values, e.AUC.values))
for mode, lab in (('buffer5000', 'block CV, 5 km buffer'), ('regime', 'leave-one-regime-out')):
    q = os.path.join(TV, mode, 'per_fold_metrics.csv')
    if os.path.exists(q):
        r = pd.read_csv(q); r = r[(r.model == 'Ensemble_mean') & (r.rule == 'national')]
        F.append((lab, r.fold.astype(str).values, r.AUC.values))
for i, (lab, names, auc) in enumerate(F):
    o = np.argsort(auc); names, auc = names[o], auc[o]
    yy_ = np.full(len(auc), i) + (np.linspace(-0.3, 0.3, len(auc)) if 'regime' in lab else np.linspace(-0.12, 0.12, len(auc)))
    ax.plot(auc, yy_, 'o', ms=5, mfc='#2a78d6', mec=SURF, mew=1)
    ax.plot([auc.mean()] * 2, [i - 0.35, i + 0.35], color=INK, lw=1.5)
    ax.annotate(f'mean {auc.mean():.3f}', (auc.mean(), i - 0.47), ha='center', fontsize=5.8, color=INK2)
    if 'regime' in lab:
        for n, a, yy in zip(names, auc, yy_):
            ax.annotate(n, (a, yy), xytext=(4, -2), textcoords='offset points', fontsize=5.2, color=INK2)
ax.set_yticks(range(len(F))); ax.set_yticklabels([f[0] for f in F]); ax.set_ylim(-0.6, len(F) - 0.55)
ax.set_xlabel('ensemble AUC on the held-out unit'); ax.grid(axis='x', color='#e6e5e0', lw=0.5)
[ax.spines[k].set_visible(False) for k in ('top', 'right')]
ax.set_title('(b) Skill under increasingly strict spatial separation', loc='left', fontsize=7)
fig.tight_layout()
for x in ('png', 'pdf'):
    fig.savefig(os.path.join(A, f'fig_transfer.{x}'), dpi=300, bbox_inches='tight')
print('->', A)
