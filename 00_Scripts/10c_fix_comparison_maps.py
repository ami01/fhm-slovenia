"""
Fix comparison_subbasin_prob_ensemble.png and comparison_subbasin_susceptible_pct.png
— removes overlapping legend/colorbar
— changes layout from 1×5 to 3-top + 2-bottom
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE     = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
COMP_DIR = f'{BASE}/Manuscript/figures/comparison'
BASIN_SHP = (f'{BASE}/../Raw_Data/Catchment Boundaries/DRSV_HGO_II/DRSV_HGO_II.shp')
SIZES    = ['1x', '2x', '3x', '4x', '10x']

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading sub-basins ...')
gdf_basins = gpd.read_file(BASIN_SHP)
gdf_basins = gdf_basins.rename(columns={'HGOII_IME': 'basin_name', 'HGOI_IME': 'parent_basin'})
gdf_basins = gdf_basins.sort_values('parent_basin').reset_index(drop=True)
print(f'  {len(gdf_basins)} sub-basins')

print('Loading CSVs ...')
df_all_prob = pd.read_csv(os.path.join(COMP_DIR, 'all_sizes_subbasin_prob_stats.csv'))
df_all_bin  = pd.read_csv(os.path.join(COMP_DIR, 'all_sizes_subbasin_binary_stats.csv'))


def clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_3top2bottom_map(df_all, col, col_label, cmap, vmin, vmax,
                         merge_col, title, out_path):
    """
    Build a 3-top + 2-bottom choropleth panel.
    df_all  — full multi-size dataframe (already filtered to Ensemble_mean if needed)
    col     — column to plot (e.g. 'prob_mean')
    cmap    — colormap
    """
    fig = plt.figure(figsize=(17, 10))
    # GridSpec: 2 rows × 6 cols
    #   top row uses columns 0:2, 2:4, 4:6  → 3 maps
    #   bottom row uses columns 1:3, 3:5   → 2 maps centred
    gs = fig.add_gridspec(
        2, 6,
        hspace=0.08, wspace=0.04,
        left=0.02, right=0.98,
        top=0.88, bottom=0.13,
    )

    ax_specs = [
        gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],   # top: 1x 2x 3x
        gs[1, 1:3], gs[1, 3:5],                 # bottom: 4x 10x (centred)
    ]
    axes_cs = [fig.add_subplot(sp) for sp in ax_specs]

    for ax, size in zip(axes_cs, SIZES):
        df_sz  = df_all[df_all['size'] == size]
        gdf_sz = gdf_basins.merge(df_sz[['basin_name', col]], on='basin_name', how='left')
        gdf_sz.plot(column=col, cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidth=0.4, edgecolor='white', ax=ax, legend=False)
        gdf_sz.dissolve(by='parent_basin').boundary.plot(
            ax=ax, color='#333333', linewidth=0.8)
        clean_ax(ax)
        ax.set_title(size, fontsize=13, fontweight='bold', pad=4)

    # Horizontal colorbar centred at bottom — placed BEFORE tight operations
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.22, 0.055, 0.56, 0.022])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label(col_label, fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.95)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {os.path.basename(out_path)}')


# ── Figure 1: ensemble probability ───────────────────────────────────────────
print('\nGenerating comparison_subbasin_prob_ensemble.png ...')
cmap_cp = LinearSegmentedColormap.from_list('cp',
    ['#f7f7f7', '#fee08b', '#fc8d59', '#d73027', '#4d0000'])
df_ens_prob = df_all_prob[df_all_prob.model == 'Ensemble_mean'].copy()
make_3top2bottom_map(
    df_all     = df_ens_prob,
    col        = 'prob_mean',
    col_label  = 'Mean Flood Susceptibility Probability (Ensemble)',
    cmap       = cmap_cp,
    vmin       = 0, vmax=1,
    merge_col  = 'basin_name',
    title      = ('Mean Ensemble Flood Susceptibility Probability by Sub-Basin'
                  ' — All Sample Sizes'),
    out_path   = os.path.join(COMP_DIR, 'comparison_subbasin_prob_ensemble.png'),
)

# ── Figure 2: % susceptible ───────────────────────────────────────────────────
print('\nGenerating comparison_subbasin_susceptible_pct.png ...')
cmap_sp = LinearSegmentedColormap.from_list('sp',
    ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'])
df_ens_bin = df_all_bin[df_all_bin.model == 'Ensemble_mean'].copy()
make_3top2bottom_map(
    df_all     = df_ens_bin,
    col        = 'pct_susceptible',
    col_label  = '% Flood Susceptible Area (Ensemble, CSI-Optimal Threshold)',
    cmap       = cmap_sp,
    vmin       = 0, vmax=30,
    merge_col  = 'basin_name',
    title      = ('% Flood Susceptible Area by Sub-Basin (Ensemble, CSI-Optimal Threshold)'
                  ' — All Sample Sizes'),
    out_path   = os.path.join(COMP_DIR, 'comparison_subbasin_susceptible_pct.png'),
)

print('\nDONE.')
