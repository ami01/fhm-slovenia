"""
Flood Susceptibility Maps — Slovenia
=====================================
Publication-quality figures from ML_Results.

Outputs → Manuscript/figures/
  Fig1_Probability_Maps.png        — 5-panel probability maps (all models)
  Fig2_Susceptibility_5class.png   — 5-panel 5-class susceptibility maps
  Fig3_Binary_Maps.png             — 5-panel binary maps
  Fig4_Ensemble_Main.png           — Standalone ensemble result (5-class)
                                     with national flood zones overlay
  Fig5_Spatial_Agreement.png       — Model consensus map (0–5 models agree)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from rasterio.enums import Resampling

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC'
# Per-size subfolder (see 09a). The flat directory is empty of model maps.
PROB_DIR   = f'{BASE}/06_Probability_Maps/' + __import__('os').environ.get('MAP_SIZE', '10x')
CLSF_DIR   = f'{BASE}/07_Classified_Maps'
AGREE_TIF  = f'{BASE}/09_Analysis/spatial_agreement_map.tif'
VEC_DIR    = f'{BASE}/Vectors'
OUT_DIR    = f'{BASE}/Manuscript/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Vector layers ──────────────────────────────────────────────────────────────
boundary = gpd.read_file(f'{VEC_DIR}/Slovenia_Boundary.shp')
admn2    = gpd.read_file(f'{VEC_DIR}/Admn_02_EPSG_3794.shp')
freq_fl  = gpd.read_file(f'{VEC_DIR}/Flooded_Areas/Frequent_Flood_Slovenia.shp')
rare_fl  = gpd.read_file(f'{VEC_DIR}/Flooded_Areas/Rare_Flood_Slovenia.shp')
extr_fl  = gpd.read_file(f'{VEC_DIR}/Flooded_Areas/Extreme_Rare_Flood_Slovenia.shp')

# ── Model config ───────────────────────────────────────────────────────────────
MODELS = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
LABELS = {
    'RF':            'Random Forest',
    'XGBoost':       'XGBoost',
    'CatBoost':      'CatBoost',
    'DNN':           'Deep Neural Network',
    'Ensemble_mean': 'Ensemble (Mean)',
}
PANEL_LETTERS = ['(a)', '(b)', '(c)', '(d)', '(e)']

# ── Colour palettes ────────────────────────────────────────────────────────────
# 5-class discrete: Very Low → Very High
CLASS_COLORS = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
CLASS_LABELS  = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
CMAP_5CLASS   = ListedColormap(CLASS_COLORS)
NORM_5CLASS   = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP_5CLASS.N)

# Probability: white → yellow → orange → dark red
CMAP_PROB = LinearSegmentedColormap.from_list(
    'fsm_prob',
    [(0.00, '#f7f7f7'),
     (0.25, '#fee08b'),
     (0.50, '#fc8d59'),
     (0.75, '#d73027'),
     (1.00, '#4d0000')],
)

# Binary: non-flood grey, flood red
CMAP_BIN  = ListedColormap(['#d1e5f0', '#b2182b'])
NORM_BIN  = BoundaryNorm([-0.5, 0.5, 1.5], CMAP_BIN.N)

# Spatial agreement: 0–5
CMAP_AGREE = LinearSegmentedColormap.from_list(
    'agree',
    [(0.0, '#f7f7f7'),
     (0.2, '#c6dbef'),
     (0.4, '#6baed6'),
     (0.6, '#2171b5'),
     (0.8, '#08306b'),
     (1.0, '#08001a')],
)
NORM_AGREE = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 6)

# ── Helpers ────────────────────────────────────────────────────────────────────
NODATA = -9999.0
DSAMP  = 8      # downsample factor — gives ~1625×2489 display arrays

def read_raster(path, factor=DSAMP, resampling=Resampling.average, dtype='float32'):
    """Read raster at reduced resolution; mask nodata → NaN."""
    with rasterio.open(path) as src:
        h = max(1, src.height // factor)
        w = max(1, src.width  // factor)
        data = src.read(1, out_shape=(h, w), resampling=resampling).astype(dtype)
        nd   = src.nodata if src.nodata is not None else NODATA
        ext  = [src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top]
    data = data.astype('float32')
    data[data == nd] = np.nan
    return data, ext

def read_classified(path, factor=DSAMP):
    return read_raster(path, factor=factor, resampling=Resampling.nearest, dtype='float32')

def panel_label(ax, letter, fontsize=9):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8, ec='none'),
            zorder=20)

def five_panel_fig(title=None):
    """2×3 grid: maps in [0,0–2] and [1,0–1], colorbar slot at [1,2]."""
    fig = plt.figure(figsize=(16, 9.5))
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.06, wspace=0.04,
                          left=0.02, right=0.91, top=0.95, bottom=0.04)
    axes = [fig.add_subplot(gs[r, c])
            for r, c in [(0,0),(0,1),(0,2),(1,0),(1,1)]]
    cb_ax = fig.add_axes([0.93, 0.12, 0.018, 0.72])   # colorbar
    return fig, axes, cb_ax


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Probability Maps
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 1: Probability Maps ...')
fig, axes, cb_ax = five_panel_fig()

for i, (model, ax) in enumerate(zip(MODELS, axes)):
    path = f'{PROB_DIR}/FSM_{model}_probability.tif'
    data, ext = read_raster(path)
    im = ax.imshow(data, extent=ext, cmap=CMAP_PROB, vmin=0, vmax=1,
                   origin='upper', interpolation='bilinear')
    ax.set_title(LABELS[model], fontsize=9, fontweight='bold', pad=3)
    panel_label(ax, PANEL_LETTERS[i])

cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_label('Flood Susceptibility Probability', fontsize=9, labelpad=6)
cbar.ax.tick_params(labelsize=8)

fig.savefig(f'{OUT_DIR}/Fig1_Probability_Maps.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('  Saved: Fig1_Probability_Maps.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — 5-Class Susceptibility Maps
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 2: 5-Class Susceptibility Maps ...')
fig, axes, cb_ax = five_panel_fig()

for i, (model, ax) in enumerate(zip(MODELS, axes)):
    path = f'{CLSF_DIR}/FSM_{model}_5class.tif'
    data, ext = read_classified(path)
    ax.imshow(data, extent=ext, cmap=CMAP_5CLASS, norm=NORM_5CLASS,
              origin='upper', interpolation='nearest')
    ax.set_title(LABELS[model], fontsize=9, fontweight='bold', pad=3)
    panel_label(ax, PANEL_LETTERS[i])

# Discrete legend
patches = [mpatches.Patch(facecolor=c, edgecolor='grey', linewidth=0.3, label=l)
           for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
fig.legend(handles=patches, title='Susceptibility Class', title_fontsize=8,
           fontsize=8, loc='center right',
           bbox_to_anchor=(0.999, 0.5), framealpha=0.9,
           edgecolor='grey', handlelength=1.2, handleheight=1.2)

fig.savefig(f'{OUT_DIR}/Fig2_Susceptibility_5class.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('  Saved: Fig2_Susceptibility_5class.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Binary Maps
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 3: Binary Maps ...')
fig, axes, cb_ax = five_panel_fig()
cb_ax.set_visible(False)

for i, (model, ax) in enumerate(zip(MODELS, axes)):
    path = f'{CLSF_DIR}/FSM_{model}_binary.tif'
    data, ext = read_classified(path)
    ax.imshow(data, extent=ext, cmap=CMAP_BIN, norm=NORM_BIN,
              origin='upper', interpolation='nearest')
    ax.set_title(LABELS[model], fontsize=9, fontweight='bold', pad=3)
    panel_label(ax, PANEL_LETTERS[i])

patches_bin = [
    mpatches.Patch(facecolor='#d1e5f0', edgecolor='grey', linewidth=0.3, label='Non-susceptible'),
    mpatches.Patch(facecolor='#b2182b', edgecolor='grey', linewidth=0.3, label='Flood Susceptible'),
]
fig.legend(handles=patches_bin, title='Classification', title_fontsize=8,
           fontsize=8, loc='center right',
           bbox_to_anchor=(0.999, 0.5), framealpha=0.9,
           edgecolor='grey', handlelength=1.2, handleheight=1.2)

fig.savefig(f'{OUT_DIR}/Fig3_Binary_Maps.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('  Saved: Fig3_Binary_Maps.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Ensemble Main Result (standalone, large)
# with national flood zone overlay
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 4: Ensemble Main Result ...')

data_ens, ext = read_classified(f'{CLSF_DIR}/FSM_Ensemble_mean_5class.tif', factor=5)

fig, ax = plt.subplots(figsize=(14, 9.5))
im = ax.imshow(data_ens, extent=ext, cmap=CMAP_5CLASS, norm=NORM_5CLASS,
               origin='upper', interpolation='nearest')

# Administrative boundaries (municipalities)
admn2.boundary.plot(ax=ax, color='#333333', linewidth=0.35, linestyle='--',
                    alpha=0.6, zorder=4, label='Municipalities')
# National flood zones
freq_fl.boundary.plot(ax=ax, color='#1a1aff', linewidth=0.7, zorder=6,
                      label='Frequent flood zone (≤10 yr)')
rare_fl.boundary.plot(ax=ax, color='#00bfff', linewidth=0.5, linestyle='--',
                      zorder=5, label='Rare flood zone (100 yr)')
extr_fl.boundary.plot(ax=ax, color='#00e5ff', linewidth=0.4, linestyle=':',
                      zorder=5, label='Extreme/Rare flood zone (500 yr)')
# Country boundary
boundary.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=7)

ax.set_aspect('equal')
ax.axis('off')

# Susceptibility class legend
patches_cls = [mpatches.Patch(facecolor=c, edgecolor='#555', linewidth=0.4, label=l)
               for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
# Flood zone legend
flood_handles = [
    Line2D([0],[0], color='#1a1aff', lw=1.2, label='Frequent flood zone'),
    Line2D([0],[0], color='#00bfff', lw=0.9, linestyle='--', label='Rare flood zone'),
    Line2D([0],[0], color='#00e5ff', lw=0.7, linestyle=':', label='Extreme/Rare flood zone'),
    Line2D([0],[0], color='#333', lw=0.6, linestyle='--', label='Municipality boundary'),
]

leg1 = ax.legend(handles=patches_cls, title='Flood Susceptibility Class',
                 title_fontsize=9, fontsize=8.5,
                 loc='upper left', framealpha=0.88, edgecolor='grey',
                 handlelength=1.2, handleheight=1.2)
ax.add_artist(leg1)
ax.legend(handles=flood_handles, title='Reference Layers',
          title_fontsize=9, fontsize=8.5,
          loc='lower right', framealpha=0.88, edgecolor='grey')

ax.set_title('Ensemble Flood Susceptibility Map — Slovenia', fontsize=12,
             fontweight='bold', pad=8)

fig.savefig(f'{OUT_DIR}/Fig4_Ensemble_Main.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('  Saved: Fig4_Ensemble_Main.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Spatial Agreement Map
# ══════════════════════════════════════════════════════════════════════════════
print('Figure 5: Spatial Agreement Map ...')

data_ag, ext = read_classified(AGREE_TIF, factor=5)

fig, ax = plt.subplots(figsize=(14, 9.5))

AGREE_COLORS = ['#f7f7f7', '#c6dbef', '#6baed6', '#2171b5', '#08306b', '#08001a']
AGREE_CMAP   = ListedColormap(AGREE_COLORS)
AGREE_NORM   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 6)

im = ax.imshow(data_ag, extent=ext, cmap=AGREE_CMAP, norm=AGREE_NORM,
               origin='upper', interpolation='nearest')
admn2.boundary.plot(ax=ax, color='#555555', linewidth=0.3, linestyle='--', alpha=0.5, zorder=4)
boundary.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=5)

ax.set_aspect('equal')
ax.axis('off')

agree_labels = ['0 models', '1 model', '2 models',
                '3 models', '4 models', '5 models (all)']
patches_ag = [mpatches.Patch(facecolor=c, edgecolor='#555', linewidth=0.4, label=l)
              for c, l in zip(AGREE_COLORS, agree_labels)]
ax.legend(handles=patches_ag, title='Models Classifying\nas Very High Susceptibility',
          title_fontsize=9, fontsize=8.5,
          loc='upper left', framealpha=0.88, edgecolor='grey',
          handlelength=1.2, handleheight=1.2)

ax.set_title('Spatial Agreement Map — Very High Susceptibility Consensus', fontsize=12,
             fontweight='bold', pad=8)

fig.savefig(f'{OUT_DIR}/Fig5_Spatial_Agreement.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.close(fig)
print('  Saved: Fig5_Spatial_Agreement.png')

print(f'\nAll panel figures saved to:\n  {OUT_DIR}/')


# ══════════════════════════════════════════════════════════════════════════════
# Individual figures — one per model per map type
# ══════════════════════════════════════════════════════════════════════════════
IND_DIR = f'{OUT_DIR}/individual'
os.makedirs(IND_DIR, exist_ok=True)

def make_individual(data, ext, cmap, norm, title, fname,
                    cb_label=None, legend_handles=None, legend_title=None,
                    vmin=None, vmax=None):
    """Single-map figure with colorbar or legend on the right."""
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(left=0.01, right=0.82, top=0.93, bottom=0.04)

    kw = dict(extent=ext, origin='upper')
    if vmin is not None:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='bilinear', **kw)
    else:
        im = ax.imshow(data, cmap=cmap, norm=norm,
                       interpolation='nearest', **kw)

    # Boundary
    boundary.boundary.plot(ax=ax, color='black', linewidth=1.2, zorder=5)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)

    # Colorbar or legend
    if legend_handles:
        ax.legend(handles=legend_handles, title=legend_title,
                  title_fontsize=9, fontsize=8.5,
                  bbox_to_anchor=(1.01, 0.5), loc='center left',
                  framealpha=0.9, edgecolor='grey',
                  handlelength=1.3, handleheight=1.3)
    else:
        cb_ax = fig.add_axes([0.84, 0.10, 0.025, 0.78])
        cbar  = fig.colorbar(im, cax=cb_ax)
        if cb_label:
            cbar.set_label(cb_label, fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Individual probability maps ───────────────────────────────────────────────
print('\nIndividual probability maps ...')
for model in MODELS:
    path = f'{PROB_DIR}/FSM_{model}_probability.tif'
    data, ext = read_raster(path)
    fname = f'{IND_DIR}/Prob_{model}.png'
    make_individual(
        data, ext, CMAP_PROB, None,
        title=f'Flood Susceptibility Probability — {LABELS[model]}',
        fname=fname,
        cb_label='Flood Susceptibility Probability',
        vmin=0, vmax=1,
    )
    print(f'  Saved: Prob_{model}.png')

# ── Individual 5-class maps ───────────────────────────────────────────────────
print('\nIndividual 5-class susceptibility maps ...')
patches_cls = [mpatches.Patch(facecolor=c, edgecolor='#555', linewidth=0.4, label=l)
               for c, l in zip(CLASS_COLORS, CLASS_LABELS)]
for model in MODELS:
    path = f'{CLSF_DIR}/FSM_{model}_5class.tif'
    data, ext = read_classified(path)
    fname = f'{IND_DIR}/5class_{model}.png'
    make_individual(
        data, ext, CMAP_5CLASS, NORM_5CLASS,
        title=f'Flood Susceptibility Map (5-Class) — {LABELS[model]}',
        fname=fname,
        legend_handles=patches_cls,
        legend_title='Susceptibility Class',
    )
    print(f'  Saved: 5class_{model}.png')

# ── Individual binary maps ────────────────────────────────────────────────────
print('\nIndividual binary maps ...')
patches_bin = [
    mpatches.Patch(facecolor='#d1e5f0', edgecolor='grey', linewidth=0.4, label='Non-susceptible'),
    mpatches.Patch(facecolor='#b2182b', edgecolor='grey', linewidth=0.4, label='Flood Susceptible'),
]
for model in MODELS:
    path = f'{CLSF_DIR}/FSM_{model}_binary.tif'
    data, ext = read_classified(path)
    fname = f'{IND_DIR}/Binary_{model}.png'
    make_individual(
        data, ext, CMAP_BIN, NORM_BIN,
        title=f'Binary Flood Susceptibility Map — {LABELS[model]}',
        fname=fname,
        legend_handles=patches_bin,
        legend_title='Classification',
    )
    print(f'  Saved: Binary_{model}.png')

print(f'\nAll individual figures saved to:\n  {IND_DIR}/')