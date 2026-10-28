"""10u — SHAP rank-stability heatmap for the v2 models (same layout and scale as the v1 Fig. 8).
    python 00_Scripts/10u_shap_rank_v2.py <shap_importance_all_sizes.csv> <out.png>"""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
NAMES = {'HAND': 'HAND', 'Soil_Type': 'Soil type', 'Elevation': 'Elevation', 'TRI': 'TRI', 'Lithology': 'Lithology',
         'Valley_Dep': 'Valley depth', 'Rain_P90': 'Rainfall P90', 'Hydro_Soil': 'Hydrogeological soil',
         'Dist_River': 'Distance to river', 'LS_Factor': 'LS factor', 'River_Den': 'River density', 'LULC': 'Land use',
         'TWI': 'TWI', 'Dist_Road': 'Distance to road', 'Aspect': 'Aspect', 'Flow_Acc': 'Flow accumulation',
         'Drain_Den': 'Drainage density', 'Prof_Curv': 'Profile curvature', 'SPI': 'SPI', 'Plan_Curv': 'Plan curvature'}
D = pd.read_csv(sys.argv[1]); D['Feature'] = D.Feature.replace({'LULC': 'LULC'})
D['rank'] = D.groupby(['SampleSize', 'Model']).MeanAbsSHAP.rank(ascending=False, method='first').astype(int)
order = D.groupby('Feature')['rank'].mean().sort_values().index.tolist()
sizes, models = ['1x', '2x', '3x', '4x', '10x'], ['RF', 'XGBoost', 'CatBoost', 'DNN']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, axs = plt.subplots(1, 4, figsize=(9.6, 8.6), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, m in zip(axs, models):
    M = D[D.Model == m].pivot(index='Feature', columns='SampleSize', values='rank').loc[order, sizes].values
    im = ax.imshow(M, cmap='RdYlGn_r', vmin=1, vmax=20, aspect='auto')
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, str(v), ha='center', va='center', fontsize=8, color='white' if v <= 3 or v >= 18 else 'black')
    ax.set_title(m, fontweight='bold'); ax.set_xticks(range(5), sizes)
    ax.set_yticks(range(len(order)), [NAMES.get(f, f) for f in order])
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, ticks=[1, 5, 10, 15, 20]); cb.set_label('SHAP rank (1 = most important)')
fig.savefig(sys.argv[2], dpi=300, bbox_inches='tight'); print('->', sys.argv[2])
