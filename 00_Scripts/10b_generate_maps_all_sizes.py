"""
Generate All Maps & Figures — All Sample Sizes (1x → 10x)
==========================================================
For each size produces:
  figures/{size}/
    Fig1_Probability_Maps.png         5-panel probability
    Fig2_Susceptibility_5class.png    5-panel 5-class
    Fig3_Binary_Maps.png              5-panel binary
    Fig4_Ensemble_Main.png            standalone ensemble + flood zones
    Fig5_Spatial_Agreement.png        model consensus map
    individual/
      Prob_{model}.png  (×5)
      5class_{model}.png (×5)
      Binary_{model}.png (×5)
    subbasin/
      subbasin_prob_stats.csv
      subbasin_binary_stats.csv
      subbasin_class_dist.csv
      subbasin_agreement_stats.csv
      map_prob_ensemble.png
      map_susceptible_pct.png
      map_agreement.png
      chart_prob_heatmap.png
      chart_susceptible_bar.png
      chart_class_dist_stacked.png
      chart_parent_basin_summary.png

Cross-size comparison figures saved to figures/comparison/
"""

import warnings
warnings.filterwarnings('ignore')

import os, gc, time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from rasterstats import zonal_stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
ML_BASE     = BASE
VEC_DIR   = f'{BASE}/Vectors'
FIG_BASE  = f'{BASE}/Manuscript/figures'

BASIN_SHP = ('/home/arora/Documents/FSM_Slovenia/Raw_Data/'
             'Catchment Boundaries/DRSV_HGO_II/DRSV_HGO_II.shp')

SIZES  = ['1x', '2x', '3x', '4x', '10x']
MODELS = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
LABELS = {
    'RF':            'Random Forest',
    'XGBoost':       'XGBoost',
    'CatBoost':      'CatBoost',
    'DNN':           'Deep Neural Network',
    'Ensemble_mean': 'Ensemble (Mean)',
}
PANEL_LETTERS = ['(a)', '(b)', '(c)', '(d)', '(e)']
MODEL_COLORS  = {
    'RF': '#2196F3', 'XGBoost': '#FF5722', 'CatBoost': '#4CAF50',
    'DNN': '#FF9800', 'Ensemble_mean': '#9C27B0',
}

# ── Colour palettes ────────────────────────────────────────────────────────────
CLASS_COLORS = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
CLASS_LABELS  = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
CMAP_5CLASS   = ListedColormap(CLASS_COLORS)
NORM_5CLASS   = BoundaryNorm([0.5,1.5,2.5,3.5,4.5,5.5], CMAP_5CLASS.N)
CMAP_PROB     = LinearSegmentedColormap.from_list('fsm_prob',
    [(0.,'#f7f7f7'),(0.25,'#fee08b'),(0.50,'#fc8d59'),(0.75,'#d73027'),(1.,'#4d0000')])
CMAP_BIN      = ListedColormap(['#d1e5f0', '#b2182b'])
NORM_BIN      = BoundaryNorm([-0.5,0.5,1.5], CMAP_BIN.N)
AGREE_COLORS  = ['#f7f7f7','#c6dbef','#6baed6','#2171b5','#08306b','#08001a']
CMAP_AGREE    = ListedColormap(AGREE_COLORS)
NORM_AGREE    = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5,5.5], 6)
PARENT_COLORS = {
    'Porečje Save':  '#2166ac', 'Porečje Kolpe': '#74add1',
    'Porečje Drave': '#4dac26', 'Porečje Mure':  '#b8e186',
    'Porečje Obale': '#f46d43', 'Porečje Soče':  '#d73027',
}
NODATA = -9999.0
DSAMP  = 8

# ── Load static vector layers ─────────────────────────────────────────────────
print('Loading vector layers ...')
boundary = gpd.read_file(f'{VEC_DIR}/Slovenia_Boundary.shp')
admn2    = gpd.read_file(f'{VEC_DIR}/Admn_02_EPSG_3794.shp')
freq_fl  = gpd.read_file(f'{VEC_DIR}/Flooded_Areas/Frequent_Flood_Slovenia.shp')
rare_fl  = gpd.read_file(f'{VEC_DIR}/Flooded_Areas/Rare_Flood_Slovenia.shp')
extr_fl  = gpd.read_file(f'{VEC_DIR}/Flooded_Areas/Extreme_Rare_Flood_Slovenia.shp')

print('Loading sub-basin layer ...')
gdf_basins = gpd.read_file(BASIN_SHP)
gdf_basins['area_km2'] = gdf_basins.geometry.area / 1e6
gdf_basins = gdf_basins.rename(columns={'HGOII_IME':'basin_name','HGOI_IME':'parent_basin'})
gdf_basins = gdf_basins.sort_values('parent_basin').reset_index(drop=True)
print(f'  {len(gdf_basins)} sub-basins loaded')

MODEL_LABEL = {'RF':'Random Forest','XGBoost':'XGBoost',
               'CatBoost':'CatBoost','DNN':'DNN','Ensemble_mean':'Ensemble'}

# ── Helpers ────────────────────────────────────────────────────────────────────
def read_raster(path, factor=DSAMP, resampling=Resampling.average):
    with rasterio.open(path) as src:
        h = max(1, src.height // factor)
        w = max(1, src.width  // factor)
        data = src.read(1, out_shape=(h,w), resampling=resampling).astype('float32')
        nd   = src.nodata if src.nodata is not None else NODATA
        ext  = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    data[data == nd] = np.nan
    return data, ext

def read_classified(path, factor=DSAMP):
    return read_raster(path, factor=factor, resampling=Resampling.nearest)

def panel_label(ax, letter):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8, ec='none'), zorder=20)

def five_panel_fig(title=None):
    fig = plt.figure(figsize=(16, 9.5))
    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)
    gs  = fig.add_gridspec(2, 3, hspace=0.06, wspace=0.04,
                           left=0.02, right=0.91, top=0.95, bottom=0.04)
    axes = [fig.add_subplot(gs[r,c]) for r,c in [(0,0),(0,1),(0,2),(1,0),(1,1)]]
    cb_ax = fig.add_axes([0.93, 0.12, 0.018, 0.72])
    return fig, axes, cb_ax

def add_boundary(ax, lw=1.2):
    boundary.boundary.plot(ax=ax, color='black', linewidth=lw, zorder=7)

def clean_ax(ax):
    ax.set_aspect('equal'); ax.axis('off')

def savefig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved: {os.path.basename(path)}')

def make_individual(data, ext, cmap, norm, title, fname,
                    cb_label=None, legend_handles=None, legend_title=None,
                    vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(left=0.01, right=0.82, top=0.93, bottom=0.04)
    kw = dict(extent=ext, origin='upper')
    if vmin is not None:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear', **kw)
    else:
        im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest', **kw)
    add_boundary(ax)
    clean_ax(ax)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    if legend_handles:
        ax.legend(handles=legend_handles, title=legend_title, title_fontsize=9, fontsize=8.5,
                  bbox_to_anchor=(1.01, 0.5), loc='center left', framealpha=0.9, edgecolor='grey',
                  handlelength=1.3, handleheight=1.3)
    else:
        cb_ax = fig.add_axes([0.84, 0.10, 0.025, 0.78])
        cbar  = fig.colorbar(im, cax=cb_ax)
        if cb_label:
            cbar.set_label(cb_label, fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8)
    savefig(fig, fname)

def basin_labels(ax, gdf_sub, fontsize=5.5):
    for _, row in gdf_sub.iterrows():
        c = row.geometry.centroid
        ax.annotate(row['basin_name'], xy=(c.x, c.y), fontsize=fontsize,
                    ha='center', va='center', color='#222222',
                    path_effects=[mpe.withStroke(linewidth=1.5, foreground='white')])

# ── Spatial agreement TIF generator ──────────────────────────────────────────
def build_agreement_tif(clsf_dir, out_tif, strip_rows=500):
    """Sum binary maps across 4 tree models (exclude Ensemble) → 0–4 agreement."""
    tree_models = ['RF', 'XGBoost', 'CatBoost', 'DNN']
    paths = [os.path.join(clsf_dir, f'FSM_{m}_binary.tif') for m in tree_models]
    with rasterio.open(paths[0]) as src:
        meta = src.meta.copy()
        H, W = src.height, src.width
    meta.update(dtype='int16', nodata=-9999,
                compress='lzw', tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(out_tif, 'w', **meta) as dst:
        srcs = [rasterio.open(p) for p in paths]
        for r0 in range(0, H, strip_rows):
            r1  = min(r0+strip_rows, H)
            win = Window(0, r0, W, r1-r0)
            arr = np.zeros((r1-r0, W), dtype='int16')
            nd_mask = np.zeros_like(arr, dtype=bool)
            for s in srcs:
                b = s.read(1, window=win)
                nd_mask |= (b == -9999)
                arr += np.where(b == -9999, 0, b.astype('int16'))
            arr[nd_mask] = -9999
            dst.write(arr, 1, window=win)
        for s in srcs:
            s.close()
    print(f'    Agreement TIF saved: {os.path.basename(out_tif)}')

# =============================================================================
# MAIN LOOP
# =============================================================================
total_t0 = time.time()

# Accumulate cross-size summary data
all_size_prob_stats = []
all_size_bin_stats  = []

for size in SIZES:
    t0 = time.time()
    print()
    print('=' * 65)
    print(f'  SIZE: {size}')
    print('=' * 65)

    prob_dir  = os.path.join(ML_BASE, '06_Probability_Maps', size)
    clsf_dir  = os.path.join(ML_BASE, '07_Classified_Maps',  size)
    agree_tif = os.path.join(ML_BASE, '07_Classified_Maps',  size, 'spatial_agreement_map.tif')
    out_dir   = os.path.join(FIG_BASE, size)
    ind_dir   = os.path.join(out_dir, 'individual')
    sub_dir   = os.path.join(out_dir, 'subbasin')
    for d in [out_dir, ind_dir, sub_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Build agreement TIF if missing ────────────────────────────────────────
    if not os.path.exists(agree_tif):
        print(f'  Building spatial agreement map [{size}] ...')
        build_agreement_tif(clsf_dir, agree_tif)

    # ── Fig 1 — Probability Maps ──────────────────────────────────────────────
    print(f'  Fig 1: Probability Maps [{size}] ...')
    fig, axes, cb_ax = five_panel_fig(f'Flood Susceptibility Probability — {size} ({size.replace("x","")}×50k samples/class)')
    im_last = None
    for i, (model, ax) in enumerate(zip(MODELS, axes)):
        data, ext = read_raster(os.path.join(prob_dir, f'FSM_{model}_probability.tif'))
        im_last = ax.imshow(data, extent=ext, cmap=CMAP_PROB, vmin=0, vmax=1,
                            origin='upper', interpolation='bilinear')
        ax.set_title(LABELS[model], fontsize=9, fontweight='bold', pad=3)
        panel_label(ax, PANEL_LETTERS[i])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im_last, cax=cb_ax)
    cbar.set_label('Flood Susceptibility Probability', fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8)
    savefig(fig, os.path.join(out_dir, 'Fig1_Probability_Maps.png'))

    # ── Fig 2 — 5-Class Maps ──────────────────────────────────────────────────
    print(f'  Fig 2: 5-Class Susceptibility Maps [{size}] ...')
    fig, axes, cb_ax = five_panel_fig(f'Flood Susceptibility Class — {size}')
    cb_ax.set_visible(False)
    for i, (model, ax) in enumerate(zip(MODELS, axes)):
        data, ext = read_classified(os.path.join(clsf_dir, f'FSM_{model}_5class.tif'))
        ax.imshow(data, extent=ext, cmap=CMAP_5CLASS, norm=NORM_5CLASS,
                  origin='upper', interpolation='nearest')
        ax.set_title(LABELS[model], fontsize=9, fontweight='bold', pad=3)
        panel_label(ax, PANEL_LETTERS[i])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    patches_cls = [mpatches.Patch(facecolor=c, edgecolor='grey', lw=0.3, label=l)
                   for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    fig.legend(handles=patches_cls, title='Susceptibility Class', title_fontsize=8, fontsize=8,
               loc='center right', bbox_to_anchor=(0.999,0.5), framealpha=0.9,
               edgecolor='grey', handlelength=1.2, handleheight=1.2)
    savefig(fig, os.path.join(out_dir, 'Fig2_Susceptibility_5class.png'))

    # ── Fig 3 — Binary Maps ───────────────────────────────────────────────────
    print(f'  Fig 3: Binary Maps [{size}] ...')
    fig, axes, cb_ax = five_panel_fig(f'Binary Flood Susceptibility — {size}')
    cb_ax.set_visible(False)
    for i, (model, ax) in enumerate(zip(MODELS, axes)):
        data, ext = read_classified(os.path.join(clsf_dir, f'FSM_{model}_binary.tif'))
        ax.imshow(data, extent=ext, cmap=CMAP_BIN, norm=NORM_BIN,
                  origin='upper', interpolation='nearest')
        ax.set_title(LABELS[model], fontsize=9, fontweight='bold', pad=3)
        panel_label(ax, PANEL_LETTERS[i])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
    patches_bin = [
        mpatches.Patch(facecolor='#d1e5f0', edgecolor='grey', lw=0.3, label='Non-susceptible'),
        mpatches.Patch(facecolor='#b2182b', edgecolor='grey', lw=0.3, label='Flood Susceptible'),
    ]
    fig.legend(handles=patches_bin, title='Classification', title_fontsize=8, fontsize=8,
               loc='center right', bbox_to_anchor=(0.999,0.5), framealpha=0.9,
               edgecolor='grey', handlelength=1.2, handleheight=1.2)
    savefig(fig, os.path.join(out_dir, 'Fig3_Binary_Maps.png'))

    # ── Fig 4 — Ensemble Main ─────────────────────────────────────────────────
    print(f'  Fig 4: Ensemble Main [{size}] ...')
    data_ens, ext = read_classified(
        os.path.join(clsf_dir, 'FSM_Ensemble_mean_5class.tif'), factor=5)
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.imshow(data_ens, extent=ext, cmap=CMAP_5CLASS, norm=NORM_5CLASS,
              origin='upper', interpolation='nearest')
    admn2.boundary.plot(ax=ax, color='#333333', linewidth=0.35, linestyle='--',
                        alpha=0.6, zorder=4)
    freq_fl.boundary.plot(ax=ax, color='#1a1aff', linewidth=0.7, zorder=6)
    rare_fl.boundary.plot(ax=ax, color='#00bfff', linewidth=0.5, linestyle='--', zorder=5)
    extr_fl.boundary.plot(ax=ax, color='#00e5ff', linewidth=0.4, linestyle=':', zorder=5)
    add_boundary(ax)
    clean_ax(ax)
    patches_c = [mpatches.Patch(facecolor=c, edgecolor='#555', lw=0.4, label=l)
                 for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    flood_h = [
        Line2D([0],[0], color='#1a1aff', lw=1.2, label='Frequent flood zone'),
        Line2D([0],[0], color='#00bfff', lw=0.9, linestyle='--', label='Rare flood zone'),
        Line2D([0],[0], color='#00e5ff', lw=0.7, linestyle=':', label='Extreme/Rare flood zone'),
        Line2D([0],[0], color='#333', lw=0.6, linestyle='--', label='Municipality boundary'),
    ]
    leg1 = ax.legend(handles=patches_c, title='Flood Susceptibility Class',
                     title_fontsize=9, fontsize=8.5, loc='upper left',
                     framealpha=0.88, edgecolor='grey', handlelength=1.2, handleheight=1.2)
    ax.add_artist(leg1)
    ax.legend(handles=flood_h, title='Reference Layers', title_fontsize=9, fontsize=8.5,
              loc='lower right', framealpha=0.88, edgecolor='grey')
    ax.set_title(f'Ensemble Flood Susceptibility Map — Slovenia [{size}]',
                 fontsize=12, fontweight='bold', pad=8)
    savefig(fig, os.path.join(out_dir, 'Fig4_Ensemble_Main.png'))

    # ── Fig 5 — Spatial Agreement ─────────────────────────────────────────────
    print(f'  Fig 5: Spatial Agreement [{size}] ...')
    data_ag, ext = read_classified(agree_tif, factor=5)
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.imshow(data_ag, extent=ext, cmap=CMAP_AGREE, norm=NORM_AGREE,
              origin='upper', interpolation='nearest')
    admn2.boundary.plot(ax=ax, color='#555555', linewidth=0.3, linestyle='--', alpha=0.5, zorder=4)
    add_boundary(ax)
    clean_ax(ax)
    agree_labels = ['0 models','1 model','2 models','3 models','4 models (all)']
    patches_ag = [mpatches.Patch(facecolor=c, edgecolor='#555', lw=0.4, label=l)
                  for c, l in zip(AGREE_COLORS[:5], agree_labels)]
    ax.legend(handles=patches_ag, title='Models Classifying\nas Flood Susceptible',
              title_fontsize=9, fontsize=8.5, loc='upper left',
              framealpha=0.88, edgecolor='grey', handlelength=1.2, handleheight=1.2)
    ax.set_title(f'Spatial Agreement Map — Flood Susceptibility Consensus [{size}]',
                 fontsize=12, fontweight='bold', pad=8)
    savefig(fig, os.path.join(out_dir, 'Fig5_Spatial_Agreement.png'))

    # ── Individual maps ───────────────────────────────────────────────────────
    print(f'  Individual maps [{size}] ...')
    patches_cls = [mpatches.Patch(facecolor=c, edgecolor='#555', lw=0.4, label=l)
                   for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
    patches_bin_ind = [
        mpatches.Patch(facecolor='#d1e5f0', edgecolor='grey', lw=0.4, label='Non-susceptible'),
        mpatches.Patch(facecolor='#b2182b', edgecolor='grey', lw=0.4, label='Flood Susceptible'),
    ]
    for model in MODELS:
        # Probability
        data, ext = read_raster(os.path.join(prob_dir, f'FSM_{model}_probability.tif'))
        make_individual(data, ext, CMAP_PROB, None,
                        f'Flood Susceptibility Probability — {LABELS[model]} [{size}]',
                        os.path.join(ind_dir, f'Prob_{model}.png'),
                        cb_label='Flood Susceptibility Probability', vmin=0, vmax=1)
        # 5-class
        data, ext = read_classified(os.path.join(clsf_dir, f'FSM_{model}_5class.tif'))
        make_individual(data, ext, CMAP_5CLASS, NORM_5CLASS,
                        f'Flood Susceptibility (5-Class) — {LABELS[model]} [{size}]',
                        os.path.join(ind_dir, f'5class_{model}.png'),
                        legend_handles=patches_cls, legend_title='Susceptibility Class')
        # Binary
        data, ext = read_classified(os.path.join(clsf_dir, f'FSM_{model}_binary.tif'))
        make_individual(data, ext, CMAP_BIN, NORM_BIN,
                        f'Binary Flood Susceptibility — {LABELS[model]} [{size}]',
                        os.path.join(ind_dir, f'Binary_{model}.png'),
                        legend_handles=patches_bin_ind, legend_title='Classification')

    # ── Subbasin Analysis ─────────────────────────────────────────────────────
    print(f'  Subbasin analysis [{size}] ...')
    gdf = gdf_basins.copy()

    # Probability zonal stats
    prob_rows = []
    for model in MODELS:
        tif = os.path.join(prob_dir, f'FSM_{model}_probability.tif')
        stats = zonal_stats(gdf, tif, stats=['mean','std','median'],
                            nodata=NODATA, all_touched=False)
        for i, s in enumerate(stats):
            prob_rows.append({
                'basin_name':   gdf.loc[i,'basin_name'],
                'parent_basin': gdf.loc[i,'parent_basin'],
                'area_km2':     round(gdf.loc[i,'area_km2'], 3),
                'model': model,
                'prob_mean':   round(s['mean']   or 0, 4),
                'prob_std':    round(s['std']    or 0, 4),
                'prob_median': round(s['median'] or 0, 4),
                'size': size,
            })
    df_prob = pd.DataFrame(prob_rows)
    df_prob.to_csv(os.path.join(sub_dir, 'subbasin_prob_stats.csv'), index=False)

    # Binary zonal stats
    bin_rows = []
    for model in MODELS:
        tif = os.path.join(clsf_dir, f'FSM_{model}_binary.tif')
        stats = zonal_stats(gdf, tif, stats=['mean','count','sum'],
                            nodata=-9999, all_touched=False)
        for i, s in enumerate(stats):
            total = s['count'] or 1
            susc  = s['sum']   or 0
            bin_rows.append({
                'basin_name':      gdf.loc[i,'basin_name'],
                'parent_basin':    gdf.loc[i,'parent_basin'],
                'area_km2':        round(gdf.loc[i,'area_km2'], 3),
                'model': model,
                'pct_susceptible': round(100*susc/total, 2),
                'size': size,
            })
    df_bin = pd.DataFrame(bin_rows)
    df_bin.to_csv(os.path.join(sub_dir, 'subbasin_binary_stats.csv'), index=False)

    # 5-class distribution (Ensemble)
    tif_5cl = os.path.join(clsf_dir, 'FSM_Ensemble_mean_5class.tif')
    stats_5 = zonal_stats(gdf, tif_5cl, categorical=True, nodata=-9999, all_touched=False)
    cls_rows = []
    for i, s in enumerate(stats_5):
        total = sum(v for k,v in s.items() if isinstance(k, (int,float)))
        total = max(total, 1)
        row = {'basin_name': gdf.loc[i,'basin_name'],
               'parent_basin': gdf.loc[i,'parent_basin'],
               'area_km2': round(gdf.loc[i,'area_km2'],3), 'size': size}
        for cls in range(1,6):
            row[f'pct_class{cls}'] = round(100*s.get(cls,0)/total, 2)
        cls_rows.append(row)
    df_cls = pd.DataFrame(cls_rows)
    df_cls.to_csv(os.path.join(sub_dir, 'subbasin_class_dist.csv'), index=False)

    # Agreement zonal stats
    ag_stats = zonal_stats(gdf, agree_tif, stats=['mean','std'], categorical=True,
                           nodata=-9999, all_touched=False)
    ag_rows = []
    for i, s in enumerate(ag_stats):
        total = sum(v for k,v in s.items() if isinstance(k,(int,float)))
        total = max(total, 1)
        ag_rows.append({
            'basin_name':    gdf.loc[i,'basin_name'],
            'parent_basin':  gdf.loc[i,'parent_basin'],
            'area_km2':      round(gdf.loc[i,'area_km2'],3),
            'mean_agreement': round(s['mean'] or 0, 4),
            'std_agreement':  round(s['std']  or 0, 4),
            'pct_all4_agree': round(100*s.get(4,0)/total, 2),
            'pct_3or4_agree': round(100*(s.get(3,0)+s.get(4,0))/total, 2),
            'size': size,
        })
    df_ag = pd.DataFrame(ag_rows)
    df_ag.to_csv(os.path.join(sub_dir, 'subbasin_agreement_stats.csv'), index=False)

    # Merge into gdf for mapping
    ens_prob = df_prob[df_prob.model=='Ensemble_mean'][['basin_name','prob_mean']].rename(
        columns={'prob_mean':'ens_prob_mean'})
    ens_bin  = df_bin[df_bin.model=='Ensemble_mean'][['basin_name','pct_susceptible']].rename(
        columns={'pct_susceptible':'ens_pct_susc'})
    ens_vh   = df_cls[['basin_name','pct_class5']].rename(columns={'pct_class5':'pct_very_high'})
    gdf = (gdf.merge(ens_prob, on='basin_name', how='left')
               .merge(ens_bin,  on='basin_name', how='left')
               .merge(ens_vh,   on='basin_name', how='left')
               .merge(df_ag[['basin_name','mean_agreement','pct_all4_agree']],
                      on='basin_name', how='left'))

    parent_map = gdf.set_index('basin_name')['parent_basin']

    # ── Subbasin Map 1: ensemble probability ──────────────────────────────────
    cmap_p = LinearSegmentedColormap.from_list('p',
        ['#f7f7f7','#fee08b','#fc8d59','#d73027','#4d0000'])
    fig, ax = plt.subplots(figsize=(11,7))
    gdf.plot(column='ens_prob_mean', cmap=cmap_p, vmin=0, vmax=1,
             linewidth=0.5, edgecolor='white', ax=ax, legend=False)
    gdf.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333333', linewidth=1.0)
    basin_labels(ax, gdf)
    clean_ax(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap_p, norm=plt.Normalize(0,1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cb.set_label('Mean Flood Susceptibility Probability', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f'Mean Flood Susceptibility Probability by Sub-Basin [{size}]\n(Ensemble)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(sub_dir, 'map_prob_ensemble.png'))

    # ── Subbasin Map 2: % Very High ────────────────────────────────────────────
    cmap_vh = LinearSegmentedColormap.from_list('vh',
        ['#ffffd4','#fed98e','#fe9929','#d95f0e','#993404'])
    fig, ax = plt.subplots(figsize=(11,7))
    gdf.plot(column='pct_very_high', cmap=cmap_vh, vmin=0, vmax=50,
             linewidth=0.5, edgecolor='white', ax=ax, legend=False)
    gdf.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333333', linewidth=1.0)
    basin_labels(ax, gdf)
    clean_ax(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap_vh, norm=plt.Normalize(0,50))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cb.set_label('% Area in Very High Susceptibility', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f'% Very High Susceptibility Area by Sub-Basin [{size}]\n(Ensemble)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(sub_dir, 'map_susceptible_pct.png'))

    # ── Subbasin Map 3: agreement ──────────────────────────────────────────────
    cmap_ag = LinearSegmentedColormap.from_list('ag',
        ['#f7f7f7','#c6dbef','#6baed6','#2171b5','#08306b'])
    fig, ax = plt.subplots(figsize=(11,7))
    gdf.plot(column='mean_agreement', cmap=cmap_ag, vmin=0, vmax=4,
             linewidth=0.5, edgecolor='white', ax=ax, legend=False)
    gdf.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333333', linewidth=1.0)
    basin_labels(ax, gdf)
    clean_ax(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap_ag, norm=plt.Normalize(0,4))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=30, ticks=[0,1,2,3,4])
    cb.set_label('Mean No. of Models Agreeing', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f'Spatial Agreement on Flood Susceptibility by Sub-Basin [{size}]\n(0–4 models)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(sub_dir, 'map_agreement.png'))

    # ── Chart 1: probability heatmap ───────────────────────────────────────────
    pivot = df_prob.pivot(index='basin_name', columns='model', values='prob_mean')
    pivot = pivot[[m for m in MODELS if m in pivot.columns]]
    pivot.columns = [MODEL_LABEL[m] for m in pivot.columns]
    pivot = pivot.loc[gdf['basin_name'].tolist()]
    fig, ax = plt.subplots(figsize=(10,9))
    cmap_hm = LinearSegmentedColormap.from_list('hm',
        ['#f7f7f7','#fee08b','#fc8d59','#d73027','#4d0000'])
    im = ax.imshow(pivot.values, cmap=cmap_hm, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for tick, basin in zip(ax.get_yticklabels(), pivot.index):
        if basin in parent_map.index:
            tick.set_color(PARENT_COLORS.get(parent_map[basin], '#333'))
    for r in range(len(pivot)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if val > 0.6 else 'black')
    ax.set_title(f'Mean Flood Susceptibility Probability by Sub-Basin and Model [{size}]',
                 fontsize=11, fontweight='bold', pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Mean Probability', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    leg_patches = [mpatches.Patch(facecolor=c, label=p)
                   for p, c in PARENT_COLORS.items()]
    ax.legend(handles=leg_patches, title='Parent Basin', title_fontsize=8, fontsize=7.5,
              loc='lower right', bbox_to_anchor=(1.28,0.0), framealpha=0.9, edgecolor='grey')
    fig.tight_layout()
    savefig(fig, os.path.join(sub_dir, 'chart_prob_heatmap.png'))

    # ── Chart 2: % susceptible bar ─────────────────────────────────────────────
    pivot_bin = df_bin.pivot(index='basin_name', columns='model', values='pct_susceptible')
    pivot_bin = pivot_bin[[m for m in MODELS if m in pivot_bin.columns]]
    pivot_bin.columns = [MODEL_LABEL[m] for m in pivot_bin.columns]
    pivot_bin = pivot_bin.loc[gdf['basin_name'].tolist()]
    pivot_bin = pivot_bin.sort_values('Ensemble', ascending=True)
    m_colors = ['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0']
    x = np.arange(len(pivot_bin)); bar_w = 0.14
    fig, ax = plt.subplots(figsize=(14,7))
    for j, (col, cc) in enumerate(zip(pivot_bin.columns, m_colors)):
        offset = (j - len(pivot_bin.columns)/2 + 0.5) * bar_w
        ax.barh(x+offset, pivot_bin[col], height=bar_w, color=cc, alpha=0.85, label=col)
    ax.set_yticks(x)
    ax.set_yticklabels(pivot_bin.index, fontsize=8)
    for tick, basin in zip(ax.get_yticklabels(), pivot_bin.index):
        if basin in parent_map.index:
            tick.set_color(PARENT_COLORS.get(parent_map[basin], '#333'))
    ax.set_xlabel('% Area Classified as Flood Susceptible', fontsize=9)
    ax.set_title(f'Flood Susceptible Area (%) by Sub-Basin and Model [{size}]',
                 fontsize=11, fontweight='bold')
    ax.legend(title='Model', fontsize=8, title_fontsize=9, framealpha=0.9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax2 = ax.twinx(); ax2.axis('off')
    ax2.legend(handles=[mpatches.Patch(facecolor=c, label=p) for p,c in PARENT_COLORS.items()],
               title='Parent Basin', title_fontsize=8, fontsize=7.5,
               loc='lower right', framealpha=0.9, edgecolor='grey')
    fig.tight_layout()
    savefig(fig, os.path.join(sub_dir, 'chart_susceptible_bar.png'))

    # ── Chart 3: 5-class stacked bar ───────────────────────────────────────────
    cls_cols_plot = [f'pct_class{c}' for c in range(1,6)]
    df_cls_p = df_cls.set_index('basin_name')[cls_cols_plot].loc[gdf['basin_name'].tolist()]
    df_cls_p.columns = CLASS_LABELS
    df_cls_p = df_cls_p.sort_values('Very High', ascending=True)
    fig, ax = plt.subplots(figsize=(12,8))
    left = np.zeros(len(df_cls_p))
    for cl, color in zip(CLASS_LABELS, CLASS_COLORS):
        vals = df_cls_p[cl].values
        ax.barh(range(len(df_cls_p)), vals, left=left, color=color, label=cl, height=0.7)
        left += vals
    ax.set_yticks(range(len(df_cls_p)))
    ax.set_yticklabels(df_cls_p.index, fontsize=8)
    for tick, basin in zip(ax.get_yticklabels(), df_cls_p.index):
        if basin in parent_map.index:
            tick.set_color(PARENT_COLORS.get(parent_map[basin], '#333'))
    ax.set_xlabel('% Area', fontsize=9)
    ax.set_xlim(0,100)
    ax.set_title(f'Susceptibility Class Distribution by Sub-Basin [{size}] — Ensemble',
                 fontsize=11, fontweight='bold')
    ax.legend(title='Susceptibility Class', fontsize=8, title_fontsize=9,
              loc='lower right', framealpha=0.9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    savefig(fig, os.path.join(sub_dir, 'chart_class_dist_stacked.png'))

    # ── Chart 4: parent basin summary ─────────────────────────────────────────
    ens_df    = df_prob[df_prob.model=='Ensemble_mean'].copy()
    par_order = (ens_df.groupby('parent_basin')['prob_mean']
                 .mean().sort_values(ascending=False).index.tolist())
    fig, axes2 = plt.subplots(1,2,figsize=(14,6))
    parent_means = ens_df.groupby('parent_basin')['prob_mean'].agg(['mean','std']).reindex(par_order)
    colors_par   = [PARENT_COLORS.get(p,'#999') for p in par_order]
    axes2[0].barh(par_order, parent_means['mean'], xerr=parent_means['std'],
                  color=colors_par, alpha=0.85, height=0.5,
                  error_kw=dict(ecolor='#333', lw=1.2, capsize=4))
    axes2[0].set_xlabel('Mean Flood Susceptibility Probability', fontsize=9)
    axes2[0].set_title(f'Mean Probability by Parent Basin [{size}]\n(Ensemble ± std)',
                       fontsize=10, fontweight='bold')
    axes2[0].spines[['top','right']].set_visible(False)
    axes2[0].xaxis.grid(True, linestyle='--', alpha=0.4, lw=0.6)
    axes2[0].set_axisbelow(True)
    for i, (p, row) in enumerate(parent_means.iterrows()):
        axes2[0].text(row['mean']+row['std']+0.005, i, f"{row['mean']:.3f}", va='center', fontsize=8)
    parent_vh = df_cls.groupby('parent_basin')['pct_class5'].agg(['mean','std']).reindex(par_order)
    axes2[1].barh(par_order, parent_vh['mean'], xerr=parent_vh['std'],
                  color=colors_par, alpha=0.85, height=0.5,
                  error_kw=dict(ecolor='#333', lw=1.2, capsize=4))
    axes2[1].set_xlabel('Mean % Area in Very High Susceptibility', fontsize=9)
    axes2[1].set_title(f'% Very High by Parent Basin [{size}]\n(Ensemble ± std)',
                       fontsize=10, fontweight='bold')
    axes2[1].spines[['top','right']].set_visible(False)
    axes2[1].xaxis.grid(True, linestyle='--', alpha=0.4, lw=0.6)
    axes2[1].set_axisbelow(True)
    for i, (p, row) in enumerate(parent_vh.iterrows()):
        axes2[1].text(row['mean']+row['std']+0.3, i, f"{row['mean']:.1f}%", va='center', fontsize=8)
    fig.suptitle(f'Flood Susceptibility Summary by Parent River Basin [{size}]',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    savefig(fig, os.path.join(sub_dir, 'chart_parent_basin_summary.png'))

    # Accumulate for cross-size
    all_size_prob_stats.append(df_prob)
    all_size_bin_stats.append(df_bin)

    elapsed = time.time()-t0
    print(f'  [{size}] done in {elapsed/60:.1f} min')
    gc.collect()

# =============================================================================
# CROSS-SIZE COMPARISON FIGURES
# =============================================================================
print()
print('=' * 65)
print('  CROSS-SIZE COMPARISON FIGURES')
print('=' * 65)

comp_dir = os.path.join(FIG_BASE, 'comparison')
os.makedirs(comp_dir, exist_ok=True)

df_all_prob = pd.concat(all_size_prob_stats, ignore_index=True)
df_all_bin  = pd.concat(all_size_bin_stats,  ignore_index=True)
df_all_prob.to_csv(os.path.join(comp_dir, 'all_sizes_subbasin_prob_stats.csv'), index=False)
df_all_bin.to_csv(os.path.join(comp_dir,  'all_sizes_subbasin_binary_stats.csv'), index=False)

# ── Cross-size: ensemble probability per size (choropleth multi-panel) ─────────
print('  Cross-size ensemble probability map panel ...')
cmap_cp = LinearSegmentedColormap.from_list('cp',
    ['#f7f7f7','#fee08b','#fc8d59','#d73027','#4d0000'])
fig = plt.figure(figsize=(17, 10))
gs  = fig.add_gridspec(2, 6, hspace=0.08, wspace=0.04,
                        left=0.02, right=0.98, top=0.88, bottom=0.13)
ax_specs = [gs[0,0:2], gs[0,2:4], gs[0,4:6], gs[1,1:3], gs[1,3:5]]
axes_cs  = [fig.add_subplot(sp) for sp in ax_specs]
for ax, size in zip(axes_cs, SIZES):
    df_sz  = df_all_prob[(df_all_prob.model=='Ensemble_mean') & (df_all_prob['size']==size)]
    gdf_sz = gdf_basins.merge(df_sz[['basin_name','prob_mean']], on='basin_name', how='left')
    gdf_sz.plot(column='prob_mean', cmap=cmap_cp, vmin=0, vmax=1,
                linewidth=0.4, edgecolor='white', ax=ax, legend=False)
    gdf_sz.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333', linewidth=0.8)
    clean_ax(ax)
    ax.set_title(size, fontsize=13, fontweight='bold', pad=4)
sm = plt.cm.ScalarMappable(cmap=cmap_cp, norm=plt.Normalize(0,1))
sm.set_array([])
cbar_ax = fig.add_axes([0.22, 0.055, 0.56, 0.022])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.set_label('Mean Flood Susceptibility Probability (Ensemble)', fontsize=10, labelpad=4)
cb.ax.tick_params(labelsize=9)
fig.suptitle('Mean Ensemble Flood Susceptibility Probability by Sub-Basin — All Sample Sizes',
             fontsize=12, fontweight='bold', y=0.95)
fig.savefig(os.path.join(comp_dir, 'comparison_subbasin_prob_ensemble.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ── Cross-size: % susceptible per size ────────────────────────────────────────
print('  Cross-size % susceptible map panel ...')
cmap_sp = LinearSegmentedColormap.from_list('sp',
    ['#ffffd4','#fed98e','#fe9929','#d95f0e','#993404'])
fig = plt.figure(figsize=(17, 10))
gs  = fig.add_gridspec(2, 6, hspace=0.08, wspace=0.04,
                        left=0.02, right=0.98, top=0.88, bottom=0.13)
ax_specs = [gs[0,0:2], gs[0,2:4], gs[0,4:6], gs[1,1:3], gs[1,3:5]]
axes_cs  = [fig.add_subplot(sp) for sp in ax_specs]
for ax, size in zip(axes_cs, SIZES):
    df_sz  = df_all_bin[(df_all_bin.model=='Ensemble_mean') & (df_all_bin['size']==size)]
    gdf_sz = gdf_basins.merge(df_sz[['basin_name','pct_susceptible']], on='basin_name', how='left')
    gdf_sz.plot(column='pct_susceptible', cmap=cmap_sp, vmin=0, vmax=30,
                linewidth=0.4, edgecolor='white', ax=ax, legend=False)
    gdf_sz.dissolve(by='parent_basin').boundary.plot(ax=ax, color='#333', linewidth=0.8)
    clean_ax(ax)
    ax.set_title(size, fontsize=13, fontweight='bold', pad=4)
sm = plt.cm.ScalarMappable(cmap=cmap_sp, norm=plt.Normalize(0,30))
sm.set_array([])
cbar_ax = fig.add_axes([0.22, 0.055, 0.56, 0.022])
cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cb.set_label('% Flood Susceptible Area (Ensemble, CSI-Optimal Threshold)', fontsize=10, labelpad=4)
cb.ax.tick_params(labelsize=9)
fig.suptitle('% Flood Susceptible Area by Sub-Basin (Ensemble, CSI-Optimal Threshold) — All Sample Sizes',
             fontsize=12, fontweight='bold', y=0.95)
fig.savefig(os.path.join(comp_dir, 'comparison_subbasin_susceptible_pct.png'),
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ── Cross-size: line chart — ensemble prob per basin across sizes ──────────────
print('  Cross-size basin probability trend lines ...')
ens_by_size = df_all_prob[df_all_prob.model=='Ensemble_mean'].copy()
pivot_cs    = ens_by_size.pivot_table(
    index='basin_name', columns='size', values='prob_mean')[SIZES]
pivot_cs    = pivot_cs.sort_values('10x', ascending=False)

fig, ax = plt.subplots(figsize=(14,7))
cmap_lines = plt.cm.get_cmap('tab20', len(pivot_cs))
for i, basin in enumerate(pivot_cs.index):
    vals = pivot_cs.loc[basin].values
    ax.plot(SIZES, vals, '-o', lw=1.2, ms=5,
            color=cmap_lines(i), alpha=0.7, label=basin)
ax.set_xlabel('Sample Size', fontsize=10)
ax.set_ylabel('Mean Flood Susceptibility Probability', fontsize=10)
ax.set_title('Ensemble Mean Probability per Sub-Basin Across Sample Sizes',
             fontsize=12, fontweight='bold')
ax.legend(title='Sub-Basin', fontsize=6.5, title_fontsize=8,
          bbox_to_anchor=(1.01,0.5), loc='center left', ncol=2,
          framealpha=0.9, edgecolor='grey')
ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
savefig(fig, os.path.join(comp_dir, 'comparison_basin_prob_trends.png'))

# ── Cross-size: % susceptible trend lines per model ───────────────────────────
print('  Cross-size % susceptible per model across sizes ...')
tree_models_plot = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
ens_pct = (df_all_bin.groupby(['size','model'])['pct_susceptible']
           .mean().reset_index())
ens_pct['_ord'] = ens_pct['size'].map({s:i for i,s in enumerate(SIZES)})
ens_pct = ens_pct.sort_values('_ord')

fig, ax = plt.subplots(figsize=(9,6))
for model in tree_models_plot:
    sub = ens_pct[ens_pct.model==model]
    ax.plot(sub['size'], sub['pct_susceptible'], '-o', lw=2, ms=8,
            color=MODEL_COLORS.get(model,'#999'), label=LABELS.get(model,model))
    for _, r in sub.iterrows():
        ax.annotate(f"{r['pct_susceptible']:.1f}%",
                    (r['size'], r['pct_susceptible']),
                    textcoords='offset points', xytext=(0,8),
                    ha='center', fontsize=7.5)
ax.set_xlabel('Sample Size', fontsize=10)
ax.set_ylabel('Mean % Susceptible Area across Sub-Basins', fontsize=10)
ax.set_title('Effect of Sample Size on Predicted Flood Susceptible Area\n(Mean across all sub-basins)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
savefig(fig, os.path.join(comp_dir, 'comparison_model_pct_susceptible_by_size.png'))

# ── Copy key comparison figures into Manuscript/figures/comparison ─────────────
print()
total_elapsed = time.time()-total_t0
print('=' * 65)
print(f'ALL DONE  ({total_elapsed/60:.1f} min total)')
print('=' * 65)
for size in SIZES:
    n = len([f for f in os.listdir(os.path.join(FIG_BASE,size)) if f.endswith('.png')])
    ns = len([f for f in os.listdir(os.path.join(FIG_BASE,size,'individual')) if f.endswith('.png')])
    nb = len([f for f in os.listdir(os.path.join(FIG_BASE,size,'subbasin')) if f.endswith('.png')])
    print(f'  {size}: {n} panel + {ns} individual + {nb} subbasin figures')
print(f'  comparison/: {len([f for f in os.listdir(comp_dir) if f.endswith(".png")])} figures')
print(f'\nAll outputs under: {FIG_BASE}/')
