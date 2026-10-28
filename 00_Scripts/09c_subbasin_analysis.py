"""
Sub-Basin Level Flood Susceptibility Analysis — Slovenia
=========================================================
Zonal statistics for all ML outputs extracted per HGOII sub-basin.

Outputs → Manuscript/figures/subbasin/
  Tables (CSV):
    subbasin_prob_stats.csv        — mean/std/median probability per model
    subbasin_binary_stats.csv      — % flood-susceptible area per model
    subbasin_class_dist.csv        — % area in each susceptibility class (ensemble)
    subbasin_agreement_stats.csv   — mean agreement score per basin

  Maps (choropleth, no scale bar / north arrow / grid):
    map_prob_ensemble.png          — mean ensemble probability per basin
    map_susceptible_pct.png        — % Very High susceptibility (ensemble)
    map_agreement.png              — mean model agreement per basin

  Charts:
    chart_prob_heatmap.png         — basins × models, mean probability
    chart_susceptible_bar.png      — % susceptible area per basin (all models)
    chart_class_dist_stacked.png   — 5-class distribution per basin (ensemble)
    chart_parent_basin_summary.png — susceptibility summary by parent basin
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from rasterstats import zonal_stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
# Per-size subfolder: the flat directory holds only a stale 1x DNN map (see 09a).
PROB_DIR  = f'{BASE}/06_Probability_Maps/' + __import__('os').environ.get('MAP_SIZE', '10x')
CLSF_DIR  = f'{BASE}/07_Classified_Maps'
AGREE_TIF = f'{BASE}/09_Analysis/spatial_agreement_map.tif'
BASIN_SHP = ('/home/arora/Documents/FSM_Slovenia/Raw_Data/'
             'Catchment Boundaries/DRSV_HGO_II/DRSV_HGO_II.shp')
OUT_DIR   = f'{BASE}/Manuscript/figures/subbasin'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load sub-basins ────────────────────────────────────────────────────────────
gdf = gpd.read_file(BASIN_SHP)
gdf['area_km2'] = gdf.geometry.area / 1e6

# Clean up: use sub-basin name as index label
gdf = gdf.rename(columns={'HGOII_IME': 'basin_name', 'HGOI_IME': 'parent_basin'})
gdf = gdf.sort_values('parent_basin').reset_index(drop=True)

print(f"Sub-basins loaded: {len(gdf)}")
print(gdf[['basin_name', 'parent_basin', 'area_km2']].to_string(index=False))

MODELS      = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
MODEL_LABEL = {
    'RF': 'Random Forest', 'XGBoost': 'XGBoost',
    'CatBoost': 'CatBoost', 'DNN': 'DNN',
    'Ensemble_mean': 'Ensemble',
}
CLASS_COLORS = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
CLASS_LABELS  = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

# Parent-basin colour palette (6 parents)
PARENT_COLORS = {
    'Porečje Save':  '#2166ac',
    'Porečje Kolpe': '#74add1',
    'Porečje Drave': '#4dac26',
    'Porečje Mure':  '#b8e186',
    'Porečje Obale': '#f46d43',
    'Porečje Soče':  '#d73027',
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Probability maps — zonal stats (mean, std, median)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Probability zonal stats ──')
prob_rows = []
for model in MODELS:
    tif = f'{PROB_DIR}/FSM_{model}_probability.tif'
    stats = zonal_stats(gdf, tif, stats=['mean', 'std', 'median'],
                        nodata=-9999.0, all_touched=False)
    for i, s in enumerate(stats):
        prob_rows.append({
            'basin_name':   gdf.loc[i, 'basin_name'],
            'parent_basin': gdf.loc[i, 'parent_basin'],
            'area_km2':     gdf.loc[i, 'area_km2'],
            'model':        model,
            'prob_mean':    round(s['mean'] or 0, 4),
            'prob_std':     round(s['std']  or 0, 4),
            'prob_median':  round(s['median'] or 0, 4),
        })
    print(f'  {model}: done')

df_prob = pd.DataFrame(prob_rows)
df_prob.to_csv(f'{OUT_DIR}/subbasin_prob_stats.csv', index=False)
print(f'  Saved: subbasin_prob_stats.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Binary maps — % susceptible pixels per basin
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Binary zonal stats ──')
bin_rows = []
for model in MODELS:
    tif = f'{CLSF_DIR}/FSM_{model}_binary.tif'
    stats = zonal_stats(gdf, tif, stats=['mean', 'count', 'sum'],
                        nodata=-9999.0, all_touched=False)
    for i, s in enumerate(stats):
        total = s['count'] or 1
        susc  = s['sum']   or 0
        bin_rows.append({
            'basin_name':    gdf.loc[i, 'basin_name'],
            'parent_basin':  gdf.loc[i, 'parent_basin'],
            'area_km2':      gdf.loc[i, 'area_km2'],
            'model':         model,
            'pct_susceptible': round(100 * susc / total, 2),
        })
    print(f'  {model}: done')

df_bin = pd.DataFrame(bin_rows)
df_bin.to_csv(f'{OUT_DIR}/subbasin_binary_stats.csv', index=False)
print(f'  Saved: subbasin_binary_stats.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 3. 5-class distribution per basin (Ensemble)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 5-class distribution (Ensemble) ──')
tif_5cl = f'{CLSF_DIR}/FSM_Ensemble_mean_5class.tif'
stats_5 = zonal_stats(gdf, tif_5cl,
                      stats=['count'], categorical=True,
                      nodata=-9999.0, all_touched=False)
cls_rows = []
for i, s in enumerate(stats_5):
    total = sum(v for k, v in s.items() if k not in ['count', 'min','max','mean'])
    if total == 0:
        total = 1
    row = {
        'basin_name':   gdf.loc[i, 'basin_name'],
        'parent_basin': gdf.loc[i, 'parent_basin'],
        'area_km2':     gdf.loc[i, 'area_km2'],
    }
    for cls in range(1, 6):
        pct = round(100 * s.get(cls, 0) / total, 2)
        row[f'pct_class{cls}'] = pct
    cls_rows.append(row)

df_cls = pd.DataFrame(cls_rows)
df_cls.to_csv(f'{OUT_DIR}/subbasin_class_dist.csv', index=False)
print(f'  Saved: subbasin_class_dist.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 4. Spatial agreement per basin
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Spatial agreement zonal stats ──')
ag_stats = zonal_stats(gdf, AGREE_TIF,
                       stats=['mean', 'std'], categorical=True,
                       nodata=-9999.0, all_touched=False)
ag_rows = []
for i, s in enumerate(ag_stats):
    total = sum(v for k, v in s.items() if isinstance(k, (int, float)))
    pct_all5 = round(100 * s.get(5, 0) / max(total, 1), 2)
    pct_4or5 = round(100 * (s.get(4, 0) + s.get(5, 0)) / max(total, 1), 2)
    ag_rows.append({
        'basin_name':    gdf.loc[i, 'basin_name'],
        'parent_basin':  gdf.loc[i, 'parent_basin'],
        'area_km2':      gdf.loc[i, 'area_km2'],
        'mean_agreement': round(s['mean'] or 0, 4),
        'std_agreement':  round(s['std']  or 0, 4),
        'pct_all5_agree': pct_all5,
        'pct_4or5_agree': pct_4or5,
    })

df_ag = pd.DataFrame(ag_rows)
df_ag.to_csv(f'{OUT_DIR}/subbasin_agreement_stats.csv', index=False)
print(f'  Saved: subbasin_agreement_stats.csv')


# ══════════════════════════════════════════════════════════════════════════════
# Merge all stats into the geodataframe for mapping
# ══════════════════════════════════════════════════════════════════════════════
ens_prob = df_prob[df_prob['model'] == 'Ensemble_mean'][['basin_name','prob_mean']].rename(
    columns={'prob_mean': 'ens_prob_mean'})
ens_bin  = df_bin[df_bin['model'] == 'Ensemble_mean'][['basin_name','pct_susceptible']].rename(
    columns={'pct_susceptible': 'ens_pct_susc'})
ens_vh   = df_cls[['basin_name','pct_class5']].rename(columns={'pct_class5': 'pct_very_high'})

gdf = (gdf
       .merge(ens_prob, on='basin_name', how='left')
       .merge(ens_bin,  on='basin_name', how='left')
       .merge(ens_vh,   on='basin_name', how='left')
       .merge(df_ag[['basin_name','mean_agreement','pct_all5_agree']], on='basin_name', how='left'))

print('\n── Summary (Ensemble) ──')
print(gdf[['basin_name','ens_prob_mean','ens_pct_susc','pct_very_high','pct_all5_agree']]
      .sort_values('ens_prob_mean', ascending=False)
      .to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════
def clean_map_ax(ax):
    ax.set_aspect('equal')
    ax.axis('off')

def basin_labels(ax, gdf_sub, col='basin_name', fontsize=5.5):
    for _, row in gdf_sub.iterrows():
        c = row.geometry.centroid
        ax.annotate(row[col], xy=(c.x, c.y), fontsize=fontsize,
                    ha='center', va='center',
                    color='#222222', fontweight='normal',
                    path_effects=[
                        mpe.withStroke(linewidth=1.5, foreground='white')
                    ])


# ══════════════════════════════════════════════════════════════════════════════
# Map 1 — Mean ensemble probability per basin
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Map: ensemble probability ──')
cmap_prob = LinearSegmentedColormap.from_list(
    'prob', ['#f7f7f7','#fee08b','#fc8d59','#d73027','#4d0000'])

fig, ax = plt.subplots(figsize=(11, 7))
gdf.plot(column='ens_prob_mean', cmap=cmap_prob, vmin=0, vmax=1,
         linewidth=0.5, edgecolor='white', ax=ax, legend=False)
# Parent basin boundaries
gdf.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333333', linewidth=1.0)
basin_labels(ax, gdf)
clean_map_ax(ax)

sm = plt.cm.ScalarMappable(cmap=cmap_prob,
                            norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
cbar.set_label('Mean Flood Susceptibility Probability', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title('Mean Flood Susceptibility Probability by Sub-Basin\n(Ensemble Model)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/map_prob_ensemble.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: map_prob_ensemble.png')


# ══════════════════════════════════════════════════════════════════════════════
# Map 2 — % Very High susceptibility (Ensemble 5-class)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Map: % Very High susceptibility ──')
cmap_vh = LinearSegmentedColormap.from_list(
    'vh', ['#ffffd4','#fed98e','#fe9929','#d95f0e','#993404'])

fig, ax = plt.subplots(figsize=(11, 7))
gdf.plot(column='pct_very_high', cmap=cmap_vh, vmin=0, vmax=50,
         linewidth=0.5, edgecolor='white', ax=ax, legend=False)
gdf.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333333', linewidth=1.0)
basin_labels(ax, gdf)
clean_map_ax(ax)

sm = plt.cm.ScalarMappable(cmap=cmap_vh, norm=plt.Normalize(vmin=0, vmax=50))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
cbar.set_label('% Area in Very High Susceptibility Class', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title('Percentage Area in Very High Flood Susceptibility Class\n(Ensemble Model)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/map_susceptible_pct.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: map_susceptible_pct.png')


# ══════════════════════════════════════════════════════════════════════════════
# Map 3 — Mean model agreement
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Map: spatial agreement ──')
cmap_ag = LinearSegmentedColormap.from_list(
    'ag', ['#f7f7f7','#c6dbef','#6baed6','#2171b5','#08306b'])

fig, ax = plt.subplots(figsize=(11, 7))
gdf.plot(column='mean_agreement', cmap=cmap_ag, vmin=0, vmax=5,
         linewidth=0.5, edgecolor='white', ax=ax, legend=False)
gdf.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333333', linewidth=1.0)
basin_labels(ax, gdf)
clean_map_ax(ax)

sm = plt.cm.ScalarMappable(cmap=cmap_ag, norm=plt.Normalize(vmin=0, vmax=5))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30,
                    ticks=[0,1,2,3,4,5])
cbar.set_label('Mean No. of Models Agreeing on Very High Susceptibility', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title('Spatial Agreement on Very High Susceptibility by Sub-Basin\n(0 = none, 5 = all models)',
             fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/map_agreement.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: map_agreement.png')


# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 — Heatmap: sub-basins × models, mean probability
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Chart: probability heatmap ──')
pivot = df_prob.pivot(index='basin_name', columns='model', values='prob_mean')
pivot = pivot[[m for m in MODELS if m in pivot.columns]]
pivot.columns = [MODEL_LABEL[m] for m in pivot.columns]
pivot = pivot.loc[gdf['basin_name'].tolist()]   # keep parent-sorted order

parent_map = gdf.set_index('basin_name')['parent_basin']
parent_colors_list = [PARENT_COLORS[parent_map[b]] for b in pivot.index]

fig, ax = plt.subplots(figsize=(10, 9))
cmap_hm = LinearSegmentedColormap.from_list(
    'hm', ['#f7f7f7','#fee08b','#fc8d59','#d73027','#4d0000'])
im = ax.imshow(pivot.values, cmap=cmap_hm, vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9, rotation=30, ha='right')
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=8)

# Colour tick labels by parent basin
for tick, basin in zip(ax.get_yticklabels(), pivot.index):
    tick.set_color(PARENT_COLORS[parent_map[basin]])

# Annotate cell values
for r in range(len(pivot)):
    for c in range(len(pivot.columns)):
        val = pivot.values[r, c]
        txt_col = 'white' if val > 0.6 else 'black'
        ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color=txt_col)

ax.set_title('Mean Flood Susceptibility Probability by Sub-Basin and Model',
             fontsize=11, fontweight='bold', pad=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('Mean Probability', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Parent basin legend
leg_patches = [mpatches.Patch(facecolor=c, label=p)
               for p, c in PARENT_COLORS.items()]
ax.legend(handles=leg_patches, title='Parent Basin', title_fontsize=8,
          fontsize=7.5, loc='lower right', bbox_to_anchor=(1.28, 0.0),
          framealpha=0.9, edgecolor='grey')

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/chart_prob_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: chart_prob_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 — % susceptible area per basin, grouped by model
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Chart: % susceptible bar ──')
pivot_bin = df_bin.pivot(index='basin_name', columns='model', values='pct_susceptible')
pivot_bin = pivot_bin[[m for m in MODELS if m in pivot_bin.columns]]
pivot_bin.columns = [MODEL_LABEL[m] for m in pivot_bin.columns]
pivot_bin = pivot_bin.loc[gdf['basin_name'].tolist()]
pivot_bin = pivot_bin.sort_values('Ensemble', ascending=True)

MODEL_COLORS = ['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0']
n_models = len(pivot_bin.columns)
n_basins = len(pivot_bin)
x = np.arange(n_basins)
bar_w = 0.14

fig, ax = plt.subplots(figsize=(14, 7))
for j, (col, col_color) in enumerate(zip(pivot_bin.columns, MODEL_COLORS)):
    offset = (j - n_models/2 + 0.5) * bar_w
    bars = ax.barh(x + offset, pivot_bin[col], height=bar_w,
                   color=col_color, alpha=0.85, label=col)

ax.set_yticks(x)
ax.set_yticklabels(pivot_bin.index, fontsize=8)

# Colour y-tick labels by parent basin
sorted_basins = pivot_bin.index.tolist()
for tick, basin in zip(ax.get_yticklabels(), sorted_basins):
    tick.set_color(PARENT_COLORS[parent_map[basin]])

ax.set_xlabel('% Area Classified as Flood Susceptible (Binary Map)', fontsize=9)
ax.set_title('Flood Susceptible Area (%) by Sub-Basin and Model', fontsize=11, fontweight='bold')
ax.legend(title='Model', fontsize=8, title_fontsize=9, framealpha=0.9)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)

# Parent basin legend on right
leg_patches = [mpatches.Patch(facecolor=c, label=p)
               for p, c in PARENT_COLORS.items()]
ax2 = ax.twinx()
ax2.axis('off')
ax2.legend(handles=leg_patches, title='Parent Basin', title_fontsize=8,
           fontsize=7.5, loc='lower right', framealpha=0.9, edgecolor='grey')

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/chart_susceptible_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: chart_susceptible_bar.png')


# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 — Stacked bar: 5-class distribution per basin (Ensemble)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Chart: 5-class stacked bar ──')
cls_cols = [f'pct_class{c}' for c in range(1, 6)]
df_cls_plot = df_cls.set_index('basin_name')[cls_cols].loc[gdf['basin_name'].tolist()]
df_cls_plot.columns = CLASS_LABELS
df_cls_plot = df_cls_plot.sort_values('Very High', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
left = np.zeros(len(df_cls_plot))
for cls_label, color in zip(CLASS_LABELS, CLASS_COLORS):
    vals = df_cls_plot[cls_label].values
    ax.barh(range(len(df_cls_plot)), vals, left=left,
            color=color, label=cls_label, height=0.7)
    left += vals

ax.set_yticks(range(len(df_cls_plot)))
ax.set_yticklabels(df_cls_plot.index, fontsize=8)

for tick, basin in zip(ax.get_yticklabels(), df_cls_plot.index):
    tick.set_color(PARENT_COLORS[parent_map[basin]])

ax.set_xlabel('% Area', fontsize=9)
ax.set_xlim(0, 100)
ax.set_title('Flood Susceptibility Class Distribution by Sub-Basin\n(Ensemble Model)',
             fontsize=11, fontweight='bold')
ax.legend(title='Susceptibility Class', fontsize=8, title_fontsize=9,
          loc='lower right', framealpha=0.9)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/chart_class_dist_stacked.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: chart_class_dist_stacked.png')


# ══════════════════════════════════════════════════════════════════════════════
# Chart 4 — Parent basin summary (box/dot plot of prob by parent basin)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── Chart: parent basin summary ──')
ens_df = df_prob[df_prob['model'] == 'Ensemble_mean'].copy()
parent_order = (ens_df.groupby('parent_basin')['prob_mean']
                .mean().sort_values(ascending=False).index.tolist())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: mean probability per parent basin
ax = axes[0]
parent_means = ens_df.groupby('parent_basin')['prob_mean'].agg(['mean','std']).reindex(parent_order)
colors = [PARENT_COLORS[p] for p in parent_order]
bars = ax.barh(parent_order, parent_means['mean'], xerr=parent_means['std'],
               color=colors, alpha=0.85, height=0.5,
               error_kw=dict(ecolor='#333333', lw=1.2, capsize=4))
ax.set_xlabel('Mean Flood Susceptibility Probability', fontsize=9)
ax.set_title('Mean Probability by Parent Basin\n(Ensemble ± std)', fontsize=10, fontweight='bold')
ax.spines[['top','right']].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
ax.set_axisbelow(True)
for i, (p, row) in enumerate(parent_means.iterrows()):
    ax.text(row['mean'] + row['std'] + 0.005, i,
            f"{row['mean']:.3f}", va='center', fontsize=8)

# Right: % Very High per parent basin
ax = axes[1]
ens_cls = df_cls.copy()   # parent_basin already present in df_cls
parent_vh = (ens_cls.groupby('parent_basin')['pct_class5']
             .agg(['mean','std']).reindex(parent_order))
ax.barh(parent_order, parent_vh['mean'], xerr=parent_vh['std'],
        color=colors, alpha=0.85, height=0.5,
        error_kw=dict(ecolor='#333333', lw=1.2, capsize=4))
ax.set_xlabel('Mean % Area in Very High Susceptibility', fontsize=9)
ax.set_title('% Very High Area by Parent Basin\n(Ensemble ± std)', fontsize=10, fontweight='bold')
ax.spines[['top','right']].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
ax.set_axisbelow(True)
for i, (p, row) in enumerate(parent_vh.iterrows()):
    ax.text(row['mean'] + row['std'] + 0.3, i,
            f"{row['mean']:.1f}%", va='center', fontsize=8)

fig.suptitle('Flood Susceptibility Summary by Parent River Basin', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/chart_parent_basin_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved: chart_parent_basin_summary.png')

print(f'\nAll outputs saved to:\n  {OUT_DIR}/')
