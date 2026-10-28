#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10s -- Final figure set for the JEM submission package.

Builds every figure used by Manuscript/JEM_final/manuscript_jem.tex and
supplementary_jem.tex into Manuscript/JEM_final/figures/{main,supplementary}.
Nothing in Manuscript/figures/ is overwritten.

Conventions applied throughout (author's instruction, 2026-09-29):
  * no figure carries a heading: what a figure shows is said once, in the
    LaTeX caption; panels carry only a letter and, where needed, a short label;
  * legends, colour bars and map furniture sit outside the mapped territory
    and never overlap each other or the data;
  * panel lettering is flat -- (a), (b), (c) ... -- never nested.

Fixed relative to the previous JEM set:
  * the cross-model agreement panel read 07_Classified_Maps/<size>/
    spatial_agreement_map.tif, a stale layer whose maximum is 4 models; the
    counts quoted in the text (76.1 % none, 16.4 % all five) come from
    09_Analysis/spatial_agreement_map.tif, which is what is drawn now;
  * the old Fig. 3(b) plotted skill against the CSI threshold (a column-name
    bug fixed in 10l but never propagated to figures/jem); panel (b)-(e) of the
    validation-design figure now plot hold-out skill against sampling
    configuration, as the text describes;
  * the national probability grid no longer displaces the CatBoost panel.

Usage:
    python3 10s_jem_final_figures.py            # everything
    python3 10s_jem_final_figures.py fig01 s_roc   # selected builders
"""
import os, sys, shutil, warnings, textwrap
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import (LightSource, LinearSegmentedColormap, ListedColormap,
                               BoundaryNorm, Normalize)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _figstyle as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, '01_Input_Features')
V = os.path.join(ROOT, '08_Validation')
A = os.path.join(ROOT, '09_Analysis')
RES = os.path.join(ROOT, '05_Results')
COLL = os.path.join(ROOT, '02_Collinearity')
SAMP = os.path.join(ROOT, '03_Samples')
PROB = os.path.join(ROOT, '06_Probability_Maps')
CLSF = os.path.join(ROOT, '07_Classified_Maps')
SRC_MAIN = os.path.join(ROOT, 'Manuscript', 'figures', 'main')
SRC_SUPP = os.path.join(ROOT, 'Manuscript', 'figures', 'supplementary')
SRC_JEM = os.path.join(ROOT, 'Manuscript', 'figures', 'jem')
PKG = os.path.join(ROOT, 'Manuscript', 'JEM_final')
OUT_M = os.path.join(PKG, 'figures', 'main')
OUT_S = os.path.join(PKG, 'figures', 'supplementary')
NE = '/usr/share/magics/50m'
SIZE = '10x'
SIZES = ['1x', '2x', '3x', '4x', '10x']
FIG_W, DPI = S.FIG_W, S.DPI           # main text: 5.40 in (elsarticle preprint)
SUP_W, SUP_DPI = 6.5, 300             # supplement: A4, 2.2 cm margins
MODELS, CV_MODELS, COLORS = S.MODELS, S.CV_MODELS, S.COLORS
plt.rcParams.update(S.RCPARAMS)
for d in (OUT_M, OUT_S):
    os.makedirs(d, exist_ok=True)


def save(fig, path, dpi=DPI, **kw):
    fig.savefig(path, dpi=dpi, facecolor='white', **kw)
    plt.close(fig)
    print('  wrote %s' % os.path.relpath(path, PKG), flush=True)


def copy(src, dst):
    shutil.copy2(src, dst)
    print('  copied %s -> %s' % (os.path.relpath(src, ROOT), os.path.relpath(dst, PKG)),
          flush=True)


def plabel(ax, letter, text=None, size=7.5):
    ax.set_title('(%s)%s' % (letter, (' ' + text) if text else ''), fontsize=size,
                 fontweight='bold', loc='left', pad=3)


def read_dec(path, dec, how='average'):
    """Decimated read.  out_shape must be (h, w) with a scalar band index --
    `read(1, out_shape=(1,h,w))[0]` slices off a row (see CLAUDE.md)."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as ds:
        h, w = ds.height // dec, ds.width // dec
        a = ds.read(1, out_shape=(h, w), resampling=getattr(Resampling, how)).astype(np.float32)
        nd, b, tf = ds.nodata, ds.bounds, ds.transform
    if nd is not None:
        a[a == nd] = np.nan
    return a, b, tf * tf.scale(ds.width / w, ds.height / h)


def read_block_max(path, dec, rows=40):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as ds:
        h, w = ds.height // dec, ds.width // dec
        out = np.zeros((h, w), np.float32)
        for r0 in range(0, h, rows):
            r1 = min(r0 + rows, h)
            blk = ds.read(1, window=Window(0, r0 * dec, w * dec, (r1 - r0) * dec)).astype(np.float32)
            out[r0:r1] = blk.reshape(r1 - r0, dec, w, dec).max(axis=(1, 3))
    return out


def dilate(m, n=1):
    for _ in range(n):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def crop(a):
    m = np.isfinite(a)
    r = np.flatnonzero(m.any(1)); c = np.flatnonzero(m.any(0))
    return a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]


# =============================================================================
# Fig. 1  Study area, with Europe locator inset
# =============================================================================
def slovenia_polygon(land, tf):
    """Vectorise the national land mask into one EPSG:3794 polygon."""
    import rasterio.features, shapely.geometry as sg
    from shapely.ops import unary_union
    geoms = [sg.shape(g) for g, v in rasterio.features.shapes(
        land.astype(np.uint8), mask=land, transform=tf) if v == 1]
    return unary_union(geoms).buffer(0)


def fig01():
    import geopandas as gpd
    from shapely.geometry import box
    dec = 8
    dem, b, tf = read_dec(os.path.join(FEAT, 'elevation.tif'), dec)
    land = np.isfinite(dem) & (dem > -1000)
    flood = (read_block_max(os.path.join(FEAT, 'Flooded_NoFlood.tif'), dec) == 1) & land
    flood = dilate(flood, 1) & land
    facc = read_block_max(os.path.join(FEAT, 'flow_accumulation.tif'), dec)
    facc = np.where(land, facc, np.nan)
    rivers = dilate(np.nan_to_num(facc) > np.nanpercentile(facc, 99.7), 1) & land

    # coarse mask for the vector outline (fast, and ample at inset scale)
    dem32, _, tf32 = read_dec(os.path.join(FEAT, 'elevation.tif'), 32)
    slo = slovenia_polygon(np.isfinite(dem32) & (dem32 > -1000), tf32)
    slo_g = gpd.GeoSeries([slo], crs=3794)

    # map frame in km, padded so the neighbours are legible
    X0, X1, Y0, Y1 = 338e3, 662e3, 14e3, 204e3
    ne_land = gpd.read_file(os.path.join(NE, 'ne_50m_land.shp'))
    ne_bnd = gpd.read_file(os.path.join(NE, 'ne_50m_admin_0_boundary_lines_land.shp'))
    bb4326 = gpd.GeoSeries([box(X0, Y0, X1, Y1)], crs=3794).to_crs(4326).total_bounds
    clipbox = box(bb4326[0] - 1, bb4326[1] - 1, bb4326[2] + 1, bb4326[3] + 1)
    land_loc = gpd.clip(ne_land, clipbox).to_crs(3794)
    bnd_loc = gpd.clip(ne_bnd, clipbox).to_crs(3794)

    SEA, NBR, NBR_EDGE = '#dde8f0', '#eceae4', '#b5b0a6'
    terrain = LinearSegmentedColormap.from_list('hyps', [
        (0.00, '#3f7f3f'), (0.12, '#88b96a'), (0.30, '#d6cf90'), (0.50, '#c39a68'),
        (0.72, '#8d6b4e'), (0.88, '#b9a99c'), (1.00, '#ffffff')])
    dem_m = np.where(land, dem, np.nan)
    vmax = float(np.nanpercentile(dem_m, 99.9))
    shade = LightSource(315, 45).hillshade(np.where(land, dem, np.nanmin(dem_m)),
                                           vert_exag=6, dx=12.5 * dec, dy=12.5 * dec)
    rgb = terrain(Normalize(0, vmax)(dem_m))
    rgb[..., :3] *= (0.42 + 0.58 * shade)[..., None]
    rgb[..., 3] = np.where(land, 1.0, 0.0)
    ext = (b.left, b.left + tf.a * dem.shape[1], b.top + tf.e * dem.shape[0], b.top)

    fig = plt.figure(figsize=(FIG_W, 2.52))
    ax = fig.add_axes([0.035, 0.065, 0.69, 0.93])
    ax.set_facecolor(SEA)
    land_loc.plot(ax=ax, color=NBR, edgecolor='none', zorder=1)
    bnd_loc.plot(ax=ax, color=NBR_EDGE, lw=0.5, zorder=2)
    ax.imshow(rgb, extent=ext, origin='upper', interpolation='bilinear', zorder=3)
    for m, c in ((rivers, '#1f6fb4'), (flood, '#d7191c')):
        ax.imshow(np.ma.masked_where(~m, np.ones_like(dem)), extent=ext, origin='upper',
                  cmap=ListedColormap([c]), interpolation='nearest', zorder=4)
    slo_g.boundary.plot(ax=ax, color='#1a1a1a', lw=0.6, zorder=5)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect('equal')
    # geographic graticule (WGS84) drawn in the projected frame, labelled at the edges
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 3794, always_xy=True)
    xt, xl, yt, yl = [], [], [], []
    for lon in np.arange(13.5, 16.51, 0.5):
        lx, ly = tr.transform(np.full(200, lon), np.linspace(44.8, 47.3, 200))
        ax.plot(lx, ly, color='0.45', lw=0.35, ls=(0, (4, 3)), zorder=5.5)
        xt.append(np.interp(Y0, ly, lx)); xl.append('%g°E' % lon)
    for lat in np.arange(45.5, 46.76, 0.5):
        lx, ly = tr.transform(np.linspace(12.5, 17.5, 200), np.full(200, lat))
        ax.plot(lx, ly, color='0.45', lw=0.35, ls=(0, (4, 3)), zorder=5.5)
        yt.append(np.interp(X0, lx, ly)); yl.append('%g°N' % lat)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_yticks(yt); ax.set_yticklabels(yl, rotation=90, va='center')
    ax.tick_params(labelsize=5.8, length=2, pad=1.5)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color('0.35')

    # neighbour labels placed in national-grid metres, inside the frame
    for name, x, y in (('AUSTRIA', 480e3, 190e3), ('ITALY', 352e3, 135e3),
                       ('HUNGARY', 638e3, 186e3), ('CROATIA', 600e3, 72e3),
                       ('Adriatic\nSea', 364e3, 27e3)):
        ax.text(x, y, name, fontsize=6.0 if name.isupper() else 5.6,
                color='#6b6760' if name.isupper() else '#4f7390', style='italic',
                ha='center', va='center', zorder=6, clip_on=True)

    # scale bar and north arrow in the empty south-east corner (Croatia)
    sx, sy = 575e3, 26e3
    for j in range(4):
        ax.add_patch(mpatches.Rectangle((sx + j * 12.5e3, sy), 12.5e3, 2.6e3,
                     facecolor='k' if j % 2 == 0 else 'w', edgecolor='k', lw=0.4, zorder=7))
    for f, t in ((0, '0'), (25e3, '25'), (50e3, '50 km')):
        ax.text(sx + f, sy - 2.5e3, t, fontsize=5.6, ha='center', va='top', zorder=7)
    ax.annotate('', xy=(648e3, 50e3), xytext=(648e3, 28e3),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=0.9), zorder=7)
    ax.text(648e3, 51.5e3, 'N', fontsize=7, fontweight='bold', ha='center', va='bottom',
            zorder=7)

    # -------------------------------------------------------- Europe inset
    ai = fig.add_axes([0.73, 0.47, 0.265, 0.52])
    eur = box(-12, 33, 42, 72)
    land_e = gpd.clip(ne_land, eur).to_crs(3035)
    bnd_e = gpd.clip(ne_bnd, eur).to_crs(3035)
    ai.set_facecolor(SEA)
    land_e.plot(ax=ai, color=NBR, edgecolor='none')
    bnd_e.plot(ax=ai, color=NBR_EDGE, lw=0.3)
    slo_e = slo_g.to_crs(3035)
    slo_e.plot(ax=ai, color='#c0392b', edgecolor='#7b1f16', lw=0.3, zorder=4)
    ai.set_xlim(2.55e6, 6.35e6); ai.set_ylim(1.35e6, 5.35e6)
    ai.set_xticks([]); ai.set_yticks([])
    for sp in ai.spines.values():
        sp.set_linewidth(0.6); sp.set_color('0.35')
    c = slo_e.iloc[0].centroid
    ai.annotate('Slovenia', xy=(c.x, c.y), xytext=(c.x + 0.55e6, c.y - 0.95e6),
                fontsize=6.0, ha='center', va='top',
                arrowprops=dict(arrowstyle='-', color='k', lw=0.5), zorder=5)

    # -------------------------------------------------------- legend block
    al = fig.add_axes([0.73, 0.0, 0.265, 0.44]); al.axis('off')
    al.set_xlim(0, 1); al.set_ylim(0, 1)
    rows = [('patch', '#d7191c', 'Flood inventory'), ('line', '#1f6fb4', 'Major rivers'),
            ('line', '#1a1a1a', 'National border'), ('line', NBR_EDGE, 'Other borders')]
    for i, (k, col, lab) in enumerate(rows):
        y = 0.94 - i * 0.12
        if k == 'patch':
            al.add_patch(mpatches.Rectangle((0.02, y - 0.035), 0.13, 0.07, color=col))
        else:
            al.plot([0.02, 0.15], [y, y], color=col, lw=1.4)
        al.text(0.20, y, lab, fontsize=6.2, va='center')
    grad = np.linspace(0, 1, 256)[None, :]
    al.imshow(grad, cmap=terrain, aspect='auto', extent=(0.02, 0.96, 0.18, 0.27))
    al.add_patch(mpatches.Rectangle((0.02, 0.18), 0.94, 0.09, fill=False, lw=0.4,
                                    edgecolor='0.35'))
    for t in np.arange(0, vmax, 500):
        al.text(0.02 + 0.94 * t / vmax, 0.155, '%d' % t, fontsize=5.4, ha='center', va='top')
    al.text(0.49, 0.30, 'Elevation (m a.s.l.)', fontsize=6.0, ha='center', va='bottom')
    save(fig, os.path.join(OUT_M, 'Fig01_study_area.png'))


# =============================================================================
# Fig. 3  Methodological framework
# =============================================================================
WF = dict(W=8.2, M=0.10, HDR=1.62, GAP_N=0.24, PADV=0.09, GAP=0.10,
          FS_L=14, FS_BT=10.8, FS_REF=8.4, FS_NT=10.0, FS_NS=8.7)
PAL = {'blue': ('#1f4e79', '#3f7cac'), 'orange': ('#b1560f', '#d97b2e'),
       'green': ('#2d6a3e', '#4d9a61'), 'purple': ('#5a3d7a', '#8766ad')}
BANDS = [
    ('A', 'Data and common grid', '§2.1–2.4', 'blue', 'layers', [
        ('Conditioning data', 'DEM, hydrography, roads, soils, geology, rainfall, land use'),
        ('Flood inventory', 'frequent, rare and extremely rare zones; 5.6 % of cells'),
        ('Common grid', '12.5 m, EPSG:3794; 129.7 M valid cells')]),
    ('B', 'Factor screening', '§2.5', 'blue', 'funnel', [
        ('24 candidates', 'topographic, hydrological, climatic, substrate, cover'),
        ('Iterative VIF cut', 'remove highest, recompute, stop below 10'),
        ('20 factors retained', 'maximum VIF 3.98')]),
    ('C', 'Sampling and training', '§3.1–3.2', 'orange', 'models', [
        ('5 nested samples', '50 000–500 000 per class, balanced, 20 % hold-out'),
        ('4 learners + ensemble', 'RF, XGBoost, CatBoost, DNN, unweighted mean'),
        ('DNN seed control', 'ten seeds, reported as mean ± s.d.')]),
    ('D', 'Validation against the inventory', '§3.3–3.4', 'orange', 'split', [
        ('Random hold-out', 'AUC, CSI, F1, κ on the balanced test set'),
        ('Spatial block CV', 'five folds of 50 km blocks, retrained per fold'),
        ('Decision thresholds', 'CSI-optimal, Youden’s J and prevalence-corrected')]),
    ('E', 'Verification and independent evidence', '§3.5–3.6', 'green', 'target', [
        ('Delivered maps', 'cell by cell at the 5.6 % base rate; 26 warning units'),
        ('Incident reports', '7 621 river-flood sites vs wind, snow and ice sites; '
                             'August 2023'),
        ('Curve Number', '216 catchments, AMC I–III; consistency check')]),
    ('F', 'Interpretation and sensitivity', '§3.7; Suppl.', 'green', 'chart', [
        ('SHAP, 20 attributions', 'four learners × five samples; rank stability'),
        ('Frequency ratio', 'model-free, every bin of every factor'),
        ('Sensitivity tests', 'land-use ablation, encodings, class ratio, Q500 target')]),
    ('G', 'National products', '§3.8', 'purple', 'map', [
        ('Susceptibility maps', 'probability, binary and five-class, all models'),
        ('Agreement layer', 'number of models calling each cell Very High'),
        ('Sub-basin statistics', '31 water-balance sub-basins; hotspots')]),
]


def fig03(refs=None, out_dir=None, stem='Fig03_methodological_framework'):
    bands = [b if not refs else (b[0], b[1], refs[b[0]], b[3], b[4], b[5]) for b in BANDS]
    out_dir = out_dir or OUT_M
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
    W, M, HDR = WF['W'], WF['M'], WF['HDR']
    IN = 1 / 72.

    def wl(t, w, fs):
        return textwrap.wrap(t, max(8, int(w * 72 / (fs * 0.56))),
                             break_long_words=False, break_on_hyphens=False)

    def th(lines, fs, lead=1.28):
        return len(lines) * fs * lead * IN

    area = (W - 2 * M) - HDR - 0.12
    nw = (area - WF['GAP_N'] * 2) / 3

    def band_h(bd):
        node = max(0.11 + th(wl(t, nw - 0.16, WF['FS_NT']), WF['FS_NT']) + 0.06
                   + th(wl(s, nw - 0.16, WF['FS_NS']), WF['FS_NS']) + 0.08
                   for t, s in bd[5])
        head = 0.47 + th(wl(bd[1], HDR - 0.22, WF['FS_BT']), WF['FS_BT']) + 0.30
        return max(node + 2 * WF['PADV'], head)

    hs = [band_h(b) for b in bands]
    H = sum(hs) + WF['GAP'] * (len(bands) - 1) + 0.04
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    def icon(kind, cx, cy, c, r=0.14):
        lw = 1.3
        if kind == 'layers':
            for i, dy in enumerate([-0.085, 0, 0.085]):
                ax.add_patch(Rectangle((cx - r, cy + dy - 0.03), 2 * r, 0.052,
                             facecolor=c if i < 2 else 'none', edgecolor=c, lw=lw, alpha=0.6 + 0.2 * i))
        elif kind == 'funnel':
            ax.add_patch(Polygon([(cx - r, cy + .75 * r), (cx + r, cy + .75 * r),
                                  (cx + .25 * r, cy), (cx + .25 * r, cy - .85 * r),
                                  (cx - .25 * r, cy - .85 * r), (cx - .25 * r, cy)],
                                 fill=False, edgecolor=c, lw=lw))
        elif kind == 'models':
            for dx, dy in [(-.075, -.04), (.075, -.04), (0, .085)]:
                ax.add_patch(Circle((cx + dx, cy + dy), r * .5, fill=False, edgecolor=c, lw=lw))
        elif kind == 'split':
            for sgn in (-1, 1):
                ax.add_patch(FancyArrowPatch((cx, cy + .9 * r), (cx + sgn * .85 * r, cy - .8 * r),
                             arrowstyle='-|>', mutation_scale=8, lw=lw, color=c))
        elif kind == 'target':
            for rr in (r, r * .6, r * .2):
                ax.add_patch(Circle((cx, cy), rr, fill=rr < r * .3, facecolor=c, edgecolor=c, lw=lw))
        elif kind == 'chart':
            for dx, h in zip([-.085, 0, .085], [.1, .2, .15]):
                ax.add_patch(Rectangle((cx + dx - .027, cy - .7 * r), .054, h, color=c, alpha=.85))
        elif kind == 'map':
            ax.add_patch(Polygon([(cx - r, cy - .6 * r), (cx - .3 * r, cy + .7 * r),
                                  (cx + .4 * r, cy + .3 * r), (cx + r, cy + .8 * r),
                                  (cx + .8 * r, cy - .7 * r)], fill=False, edgecolor=c, lw=lw))

    y = H - 0.02
    for k, bd in enumerate(bands):
        letter, title, ref, pal, ic, nodes = bd
        dark, mid = PAL[pal]
        h = hs[k]; yb = y - h
        ax.add_patch(FancyBboxPatch((M, yb), HDR, h, boxstyle='round,pad=0,rounding_size=0.06',
                                    facecolor=dark, edgecolor=dark))
        ax.text(M + 0.28, y - 0.23, letter, fontsize=WF['FS_L'], fontweight='bold',
                color='w', ha='center', va='center')
        icon(ic, M + HDR - 0.34, y - 0.23, 'w')
        ax.text(M + HDR / 2, y - 0.44, '\n'.join(wl(title, HDR - 0.22, WF['FS_BT'])),
                fontsize=WF['FS_BT'], fontweight='bold', color='w', ha='center', va='top',
                linespacing=1.2)
        ax.text(M + HDR / 2, yb + 0.07, ref, fontsize=WF['FS_REF'], color='w',
                style='italic', ha='center', va='bottom', alpha=0.9)
        x0 = M + HDR + 0.12
        ny0, ny1 = yb + WF['PADV'], y - WF['PADV']
        for i, (t, st) in enumerate(nodes):
            a0 = x0 + i * (nw + WF['GAP_N']); a1 = a0 + nw
            ax.add_patch(FancyBboxPatch((a0, ny0), nw, ny1 - ny0,
                         boxstyle='round,pad=0,rounding_size=0.05',
                         facecolor='w', edgecolor='#c9ced6', lw=1.0))
            ax.add_patch(Rectangle((a0, ny1 - 0.05), nw, 0.05, color=mid))
            tl = wl(t, nw - 0.16, WF['FS_NT'])
            ax.text((a0 + a1) / 2, ny1 - 0.10, '\n'.join(tl), fontsize=WF['FS_NT'],
                    fontweight='bold', color='#1a202c', ha='center', va='top', linespacing=1.28)
            ax.text((a0 + a1) / 2, ny1 - 0.10 - th(tl, WF['FS_NT']) - 0.06,
                    '\n'.join(wl(st, nw - 0.16, WF['FS_NS'])), fontsize=WF['FS_NS'],
                    color=dark, ha='center', va='top', linespacing=1.25)
            if i < 2:
                ax.add_patch(FancyArrowPatch((a1, (ny0 + ny1) / 2), (a1 + WF['GAP_N'], (ny0 + ny1) / 2),
                             arrowstyle='-|>', mutation_scale=10, lw=1.3, color=mid))
        if k < len(bands) - 1:
            ax.add_patch(FancyArrowPatch((M + HDR / 2, yb - 0.01), (M + HDR / 2, yb - WF['GAP'] + 0.01),
                         arrowstyle='-|>', mutation_scale=13, lw=1.6, color=dark))
        y = yb - WF['GAP']
    fig.savefig(os.path.join(out_dir, stem + '.pdf'), facecolor='white')
    save(fig, os.path.join(out_dir, stem + '.png'), dpi=300)


# =============================================================================
# Fig. 4  Validation design: spatial folds + hold-out skill vs sampling size
# =============================================================================
FOLD_C = {1: '#2196F3', 2: '#FF5722', 3: '#4CAF50', 4: '#9C27B0', 5: '#FF9800'}


def fig04():
    d = pd.read_csv(os.path.join(A, 'spatial_cv', SIZE, 'fold_assignments.csv'),
                    usecols=['x_3794', 'y_3794', 'fold'])
    ms = pd.read_csv(os.path.join(V, 'multiscale_validation.csv'))
    ms = ms[ms.SampleSize.isin(SIZES)].copy()
    ms['_o'] = ms.SampleSize.map({s: i for i, s in enumerate(SIZES)})
    fig = plt.figure(figsize=(FIG_W, 6.1), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    for f, c in FOLD_C.items():
        s = d[d.fold == f]
        ax.scatter(s.x_3794, s.y_3794, s=0.04, c=c, linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.axis('off'); plabel(ax, 'a')
    ax.legend(handles=[mlines.Line2D([], [], ls='', marker='o', ms=4.5, color=c,
                                     label='Fold %d' % f) for f, c in FOLD_C.items()],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5,
              frameon=False, columnspacing=1.4, handletextpad=0.2)
    for k, met in enumerate(['AUC', 'CSI', 'F1', 'Kappa']):
        a = fig.add_subplot(gs[1 + k // 2, k % 2])
        for m in MODELS:
            s = ms[ms.Model == m].sort_values('_o')
            a.plot(s._o, s[met], 'o-', ms=2.8, lw=1.1, color=COLORS[m], label=S.model(m))
        a.set_xticks(range(len(SIZES))); a.set_xticklabels(SIZES)
        a.set_xlabel('sampling configuration', fontsize=7); a.set_ylabel(S.metric(met), fontsize=7)
        a.grid(alpha=0.3); a.tick_params(labelsize=6.3)
        plabel(a, 'bcde'[k])
        if k == 0:
            h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=5, fontsize=6.4, frameon=False)
    save(fig, os.path.join(OUT_M, 'Fig04_validation_design.png'))


# =============================================================================
# Fig. 8  Spatial breakdown of skill (ARSO units) and Curve Number consistency
# =============================================================================
def fig08():
    import geopandas as gpd
    from scipy.stats import spearmanr
    units = gpd.read_file(os.path.join(ROOT, 'geo_static.json')).to_crs(3794)
    sk = pd.read_csv(os.path.join(A, 'arso_units', 'arso_unit_skill.csv'))
    sp = pd.read_csv(os.path.join(A, 'arso_units', 'arso_unit_support.csv'))
    e = sk[sk.model == 'Ensemble_mean'].merge(sp[['unit', 'prevalence']], on='unit')
    u = units.merge(e[['unit', 'AUC', 'CSI_nat', 'reported']], left_on='ime',
                    right_on='unit', how='left')
    u.loc[u.reported != True, ['AUC', 'CSI_nat']] = np.nan
    cn = pd.read_csv(os.path.join(A, 'cn_corroboration', 'catchment_cn_vs_susceptibility.csv'))
    cn = cn[(cn.coverage_ok) & (cn.n_cells > 0) & cn.cn_avg.notna()]

    fig = plt.figure(figsize=(FIG_W, 6.0), layout='constrained')
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.95, 0.95])
    for k, (col, cmap, lab) in enumerate([('AUC', 'viridis', '$AUC$'),
                                          ('CSI_nat', 'RdYlGn', '$CSI$')]):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        u.plot(column=col, ax=ax, cmap=cmap, edgecolor='white', linewidth=0.35, legend=True,
               missing_kwds={'color': '#d9d9d9', 'edgecolor': 'white', 'hatch': '///'},
               legend_kwds={'orientation': 'horizontal', 'shrink': 0.72, 'pad': 0.01,
                            'aspect': 26, 'label': lab})
        ax.axis('off'); plabel(ax, 'ab'[k])
    ax = fig.add_subplot(gs[1, :])
    r = e[e.reported == True]
    ax.scatter(100 * r.prevalence, r.AUC, s=16, color='#2196F3', zorder=3)
    ax.set_xlabel("flood fraction of the unit's held-out sample (%)", fontsize=7.5)
    ax.set_ylabel('$AUC$ (circles)', fontsize=7.5, color='#2196F3')
    ax.tick_params(axis='y', labelcolor='#2196F3', labelsize=6.5); ax.tick_params(axis='x', labelsize=6.5)
    ax.grid(alpha=0.3)
    a2 = ax.twinx()
    a2.scatter(100 * r.prevalence, r.CSI_nat, s=16, color='#D32F2F', marker='s', zorder=3)
    a2.set_ylabel('$CSI$ (squares)', fontsize=7.5, color='#D32F2F')
    a2.tick_params(axis='y', labelcolor='#D32F2F', labelsize=6.5)
    plabel(ax, 'c')
    for k, (col, lab) in enumerate([('cn_dry', 'AMC I (dry)'), ('cn_avg', 'AMC II (average)'),
                                    ('cn_wet', 'AMC III (wet)')]):
        a = fig.add_subplot(gs[2, 2 * k:2 * k + 2])
        a.scatter(cn[col], 100 * cn.flood_frac, s=7, alpha=0.65, color='#2196F3', edgecolor='none')
        bb = np.polyfit(cn[col], 100 * cn.flood_frac, 1)
        xg = np.linspace(cn[col].min(), cn[col].max(), 50)
        a.plot(xg, np.polyval(bb, xg), color='#D32F2F', lw=1.1)
        rho, _ = spearmanr(cn[col], cn.flood_frac)
        a.text(0.04, 0.95, r'$\rho = %+.3f$' % rho, transform=a.transAxes, fontsize=6.3,
               va='top', ha='left', bbox=dict(facecolor='w', edgecolor='none', pad=1))
        a.set_xlabel('Curve Number', fontsize=7); a.grid(alpha=0.3); a.tick_params(labelsize=6.3)
        if k == 0:
            a.set_ylabel('flood-prone area (%)', fontsize=7)
        plabel(a, 'def'[k], lab, size=7)
    save(fig, os.path.join(OUT_M, 'Fig08_spatial_skill_and_cn.png'))


# =============================================================================
# Fig. 11  National probability and binary maps, all five products
# =============================================================================
LABEL = {'RF': 'Random Forest', 'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost',
         'DNN': 'DNN', 'Ensemble_mean': 'Ensemble mean'}


def _map(path, dec=16):
    a, _, _ = read_dec(path, dec, how='nearest')
    return crop(a)


def fig11():
    probs = [_map(os.path.join(PROB, SIZE, 'FSM_%s_probability.tif' % m)) for m in MODELS]
    bins = [_map(os.path.join(CLSF, SIZE, 'FSM_%s_binary.tif' % m)) for m in MODELS]
    h, w = probs[0].shape
    W, cw, gap = FIG_W, FIG_W / 3, 0.03
    mh = (cw - gap) * h / w                     # map height, in
    th, lab = 0.18, 0.20                        # title strip, block label
    blk = lab + 2 * (mh + th) + 0.04
    H = 2 * blk + 0.05
    fig = plt.figure(figsize=(W, H))
    cols = ['#E8EAF0', '#1A5FB4']

    def place(r, c, top):
        x = c * cw + gap / 2
        y = top - lab - (r + 1) * (mh + th)
        return fig.add_axes([x / W, y / H, (cw - gap) / W, mh / H])

    for b, (arrs, letter) in enumerate(((probs, 'a'), (bins, 'b'))):
        top = H - b * blk
        fig.text(0.004, (top - 0.03) / H, '(%s)' % letter, fontsize=8.5, fontweight='bold',
                 va='top')
        for i, a in enumerate(arrs):
            ax = place(i // 3, i % 3, top)
            if b == 0:
                im = ax.imshow(a, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            else:
                ax.imshow(a, cmap=ListedColormap(cols), norm=BoundaryNorm([-.5, .5, 1.5], 2),
                          interpolation='nearest')
            ax.set_title(LABEL[MODELS[i]], fontsize=6.4, fontweight='bold', pad=1.5)
            ax.axis('off')
        # the sixth cell of each block holds its key, well clear of the maps
        x6 = 2 * cw + gap / 2
        y6 = top - lab - 2 * (mh + th)
        if b == 0:
            cax = fig.add_axes([(x6 + 0.25 * cw) / W, (y6 + 0.45 * mh) / H,
                                (0.5 * cw) / W, 0.07 / H])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cb.set_label('susceptibility probability', fontsize=6.3)
            cb.ax.tick_params(labelsize=5.8)
        else:
            lax = fig.add_axes([x6 / W, y6 / H, (cw - gap) / W, mh / H]); lax.axis('off')
            lax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4, label=l)
                                for c, l in zip(cols, ['not flood-prone', 'flood-prone'])],
                       loc='center', fontsize=6.4, frameon=False)
    save(fig, os.path.join(OUT_M, 'Fig11_national_maps.png'))


# =============================================================================
# Fig. 12  Ensemble five-class map and cross-model agreement
# =============================================================================
def fig12():
    fig = plt.figure(figsize=(FIG_W, 5.1), layout='constrained')
    gs = fig.add_gridspec(2, 1, hspace=0.06)
    ax = fig.add_subplot(gs[0])
    a = _map(os.path.join(CLSF, SIZE, 'FSM_Ensemble_mean_5class.tif'))
    cols = ['#2C7BB6', '#ABD9E9', '#FFFFBF', '#FDAE61', '#D7191C']
    ax.imshow(a, cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(0.5, 6), 5),
              interpolation='nearest')
    ax.axis('off'); plabel(ax, 'a', size=8.5)
    ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4, label=l) for c, l in
                       zip(cols, ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])],
              loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=6.4,
              frameon=False, columnspacing=1.0, handlelength=1.1, handletextpad=0.4)
    ax2 = fig.add_subplot(gs[1])
    # 09_Analysis layer: the one behind the 76.1 % / 16.4 % figures in the text
    b = _map(os.path.join(A, 'spatial_agreement_map.tif'))
    cols2 = ['#F2F2F2', '#FEE5D9', '#FCAE91', '#FB6A4A', '#DE2D26', '#A50F15']
    ax2.imshow(b, cmap=ListedColormap(cols2), norm=BoundaryNorm(np.arange(-0.5, 6), 6),
               interpolation='nearest')
    ax2.axis('off'); plabel(ax2, 'b', size=8.5)
    ax2.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4, label=l) for c, l in
                        zip(cols2, ['0 models', '1', '2', '3', '4', 'all 5'])],
               loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=6.4,
               frameon=False, columnspacing=1.0, handlelength=1.1, handletextpad=0.4,
               title='models classifying the cell Very High', title_fontsize=6.4)
    save(fig, os.path.join(OUT_M, 'Fig12_ensemble_and_agreement.png'))


# =============================================================================
# Main-text figures carried over unchanged (already heading-free)
# =============================================================================
def copies_main():
    for src, dst in (
            (os.path.join(SRC_MAIN, 'Fig02_conditioning_factors.png'), 'Fig02_conditioning_factors.png'),
            (os.path.join(SRC_MAIN, 'Fig06_roc_validation.png'), 'Fig05_roc_validation.png'),
            (os.path.join(SRC_MAIN, 'Fig07_confusion_matrices.png'), 'Fig06_confusion_matrices.png'),
            (os.path.join(SRC_MAIN, 'Fig08_spatial_cv_vs_random.png'), 'Fig07_spatial_cv_vs_random.png'),
            (os.path.join(SRC_JEM, 'JEM_Fig08b_incident_validation.png'), 'Fig09_incident_validation.png'),
            (os.path.join(SRC_MAIN, 'Fig12_shap_rank_stability.png'), 'Fig10_shap_rank_stability.png')):
        copy(src, os.path.join(OUT_M, dst))


# =============================================================================
# Supplementary S1-S28, regenerated without headings (after 10i)
# =============================================================================
def sfig(i, off):
    return 1 + 5 * i + off


def _sample(size_points, models, size):
    """Probability at points, by direct array indexing (as stage 07 does)."""
    import rasterio
    xs, ys = size_points
    out = {}
    for m in models:
        with rasterio.open(os.path.join(PROB, size, 'FSM_%s_probability.tif' % m)) as ds:
            t = ds.transform
            c = np.clip(np.floor((xs - t.c) / t.a).astype(int), 0, ds.width - 1)
            r = np.clip(np.floor((ys - t.f) / t.e).astype(int), 0, ds.height - 1)
            arr = ds.read(1); v = arr[r, c].astype(np.float64); del arr
            nd = ds.nodata
        if nd is not None:
            v[v == nd] = np.nan
        out[m] = v
    return out


def s_roc():
    from sklearn.metrics import roc_curve, roc_auc_score
    for i, size in enumerate(SIZES):
        suf = '' if size == '1x' else '_%s' % size
        df = pd.read_csv(os.path.join(SAMP, 'samples_test%s.csv' % suf))
        y = df['label'].values.astype(int)
        P = _sample((df.x_3794.values, df.y_3794.values), MODELS, size)
        ok = np.all([np.isfinite(P[m]) for m in MODELS], axis=0)
        thr = pd.read_csv(os.path.join(V, size, 'threshold_summary.csv'))
        fig, ax = plt.subplots(figsize=(SUP_W, 4.6), layout='constrained')
        ax.plot([0, 1], [0, 1], ls='--', lw=0.9, color='gray', label='random chance')
        for m in MODELS:
            fpr, tpr, _ = roc_curve(y[ok], P[m][ok])
            t = thr[thr.Model == m]['CSI_threshold']
            ax.plot(fpr, tpr, lw=1.4, color=COLORS[m],
                    label='%s ($AUC$ = %.4f, $t$ = %.3f)' % (S.model(m), roc_auc_score(y[ok], P[m][ok]),
                                                          t.values[0]))
        ax.set_xlabel('false positive rate (1 $-$ specificity)'); ax.set_ylabel('true positive rate (sensitivity)')
        ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01); ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=7)
        save(fig, os.path.join(OUT_S, 'FigS%02d_roc_%s.png' % (sfig(i, 0), size)), dpi=SUP_DPI)


def s_confusion():
    import seaborn as sns
    for i, size in enumerate(SIZES):
        df = pd.read_csv(os.path.join(V, size, 'validation_metrics.csv'))
        fig, axes = plt.subplots(2, 3, figsize=(SUP_W, 4.5), layout='constrained')
        fl = axes.flatten()
        for j, m in enumerate(MODELS):
            r = df[df.Model == m].iloc[0]
            cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], float); pct = cm / cm.sum() * 100
            ann = np.array([['%d\n(%.1f%%)' % (cm[a, b], pct[a, b]) for b in range(2)] for a in range(2)])
            sns.heatmap(cm, annot=ann, fmt='', cmap='Blues', ax=fl[j], cbar=False,
                        xticklabels=['No flood', 'Flood'], yticklabels=['No flood', 'Flood'],
                        linewidths=0.6, linecolor='white', annot_kws={'fontsize': 6.5})
            fl[j].set_title('(%s) %s\n$AUC$ = %.4f, $CSI$ = %.4f\n$t$ = %.3f'
                            % ('abcde'[j], S.model(m), r.AUC, r.CSI, r.Threshold), fontsize=6.4)
            fl[j].set_xlabel('predicted', fontsize=7); fl[j].set_ylabel('actual', fontsize=7)
            fl[j].tick_params(labelsize=6.5)
        fl[-1].set_visible(False)
        save(fig, os.path.join(OUT_S, 'FigS%02d_confusion_%s.png' % (sfig(i, 1), size)), dpi=SUP_DPI)


def s_csi():
    from sklearn.metrics import confusion_matrix
    dft = pd.read_csv(os.path.join(SAMP, 'samples_test_10x.csv'))
    y = dft['label'].values.astype(int)
    ts = np.linspace(0.01, 0.99, 200)
    for i, size in enumerate(SIZES):
        P = _sample((dft.x_3794.values, dft.y_3794.values), MODELS, size)
        ok = np.all([np.isfinite(P[m]) for m in MODELS], axis=0)
        thr = pd.read_csv(os.path.join(V, size, 'threshold_summary.csv'))
        fig, axes = plt.subplots(2, 3, figsize=(SUP_W, 4.3), layout='constrained')
        fl = axes.flatten()
        for j, m in enumerate(MODELS):
            p, yy = P[m][ok], y[ok]
            order = np.argsort(-p); ps, ys_ = p[order], yy[order]
            tp_c = np.cumsum(ys_); n_pos = yy.sum()
            csi = []
            for t in ts:
                k = np.searchsorted(-ps, -t, side='right')   # predicted positive: p >= t
                tp = tp_c[k - 1] if k else 0; fp = k - tp; fn = n_pos - tp
                csi.append(tp / (tp + fp + fn))
            t0 = float(thr[thr.Model == m]['CSI_threshold'].values[0])
            a = fl[j]
            a.plot(ts, csi, color=COLORS[m], lw=1.3)
            a.axvline(t0, color='k', ls='--', lw=0.8, label='$t$ = %.3f' % t0)
            a.set_xlabel('threshold', fontsize=7); a.set_ylabel('$CSI$', fontsize=7)
            a.legend(fontsize=6, loc='lower left'); a.grid(alpha=0.3); a.tick_params(labelsize=6.3)
            plabel(a, 'abcde'[j], S.model(m), size=7)
        fl[-1].set_visible(False)
        save(fig, os.path.join(OUT_S, 'FigS%02d_csi_curves_%s.png' % (sfig(i, 2), size)), dpi=SUP_DPI)


def s_spatial_cv():
    for i, size in enumerate(SIZES):
        mn = pd.read_csv(os.path.join(A, 'spatial_cv', size, 'mean_metrics.csv'))
        rd = pd.read_csv(os.path.join(V, size, 'validation_metrics.csv'))
        fig, axes = plt.subplots(2, 2, figsize=(SUP_W, 4.8), layout='constrained')
        x = np.arange(len(CV_MODELS)); w = 0.35
        for k, (ax, met) in enumerate(zip(axes.flat, ['AUC', 'CSI', 'F1', 'Kappa'])):
            rv = [rd[rd.Model == m][met].values[0] for m in CV_MODELS]
            b1 = ax.bar(x - w / 2, rv, w, color=[COLORS[m] for m in CV_MODELS], alpha=0.45,
                        edgecolor='gray', lw=0.5, hatch='//')
            cm_ = [mn[mn.Model == m]['%s_mean' % met].values[0] for m in CV_MODELS]
            cs = [mn[mn.Model == m]['%s_std' % met].values[0] for m in CV_MODELS]
            ax.bar(x + w / 2, cm_, w, yerr=cs, capsize=2, color=[COLORS[m] for m in CV_MODELS],
                   alpha=0.9, edgecolor='k', lw=0.5)
            for bar, v in zip(b1, rv):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02, '%.3f' % v,
                        ha='center', va='top', fontsize=6, rotation=90, color='#37474F')
            ax.set_xticks(x); ax.set_xticklabels([S.model(m) for m in CV_MODELS], fontsize=7)
            ax.set_ylim(0, 1.14); ax.grid(axis='y', alpha=0.3); ax.set_ylabel(S.metric(met))
            plabel(ax, 'abcd'[k])
        fig.legend(handles=[mpatches.Patch(facecolor='gray', alpha=0.45, hatch='//', label='random split (80/20)'),
                            mpatches.Patch(facecolor='dimgray', alpha=0.9, label='spatial CV (5-fold, 50 km)')],
                   loc='outside lower center', ncol=2, fontsize=7, frameon=False)
        save(fig, os.path.join(OUT_S, 'FigS%02d_spatial_cv_%s.png' % (sfig(i, 3), size)), dpi=SUP_DPI)


def s_shap():
    for i, size in enumerate(SIZES):
        d = pd.read_csv(os.path.join(A, 'shap', size, 'shap_importance_all_models.csv'), index_col=0)
        cols = [m for m in CV_MODELS if m in d.columns]
        sh = d[cols].div(d[cols].sum(axis=0), axis=1)
        sh = sh.assign(_m=sh.mean(axis=1)).sort_values('_m').drop(columns='_m')
        n, nm, bh = len(sh), len(cols), 0.19
        offs = np.linspace(-(nm - 1) / 2, (nm - 1) / 2, nm) * bh; y = np.arange(n)
        fig, ax = plt.subplots(figsize=(SUP_W, 5.6), layout='constrained')
        for k, m in enumerate(cols):
            ax.barh(y + offs[k], 100 * sh[m].values, bh, label=S.model(m), color=COLORS[m],
                    alpha=0.85, edgecolor='white', lw=0.3)
        ax.set_yticks(y); ax.set_yticklabels(S.features(sh.index))
        ax.set_xlabel("share of the model's total mean $|$SHAP$|$ attribution (%)")
        ax.legend(loc='lower right'); ax.grid(axis='x', alpha=0.3)
        save(fig, os.path.join(OUT_S, 'FigS%02d_shap_%s.png' % (sfig(i, 4), size)), dpi=SUP_DPI)


SHORT = {'Distance_to_River': 'Dist_River', 'Distance_to_Road': 'Dist_Road',
         'Hgk_grouped': 'Hydro_Soil', 'Pedoloska_grouped': 'Soil_Type',
         'River_Density': 'River_Den', 'aspect': 'Aspect', 'drainage_density': 'Drain_Den',
         'elevation': 'Elevation', 'flow_accumulation': 'Flow_Acc', 'hand': 'HAND',
         'lito_poly_250_d96_grouped': 'Lithology', 'ls_factor': 'LS_Factor',
         'plan_curvature': 'Plan_Curv', 'profile_curvature': 'Prof_Curv',
         'slovenia_annual_max_daily_p90_3794_12m5': 'Rain_P90', 'spi': 'SPI', 'tri': 'TRI',
         'twi': 'TWI', 'valley_depth': 'Valley_Dep', 'LULC': 'LULC'}


def s_heatmap():
    import seaborn as sns
    c = pd.read_csv(os.path.join(COLL, 'correlation_final.csv'), index_col=0)
    lab = [S.feature(SHORT.get(k, k), short=True) for k in c.index]
    c.index = lab; c.columns = lab
    fig, ax = plt.subplots(figsize=(SUP_W, SUP_W * 0.9), layout='constrained')
    sns.heatmap(c, mask=np.triu(np.ones_like(c, bool), 1), annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.4, linecolor='white', annot_kws={'size': 5.6},
                ax=ax, square=True, cbar_kws={'shrink': 0.6, 'label': 'Pearson $r$'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6.8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.8)
    save(fig, os.path.join(OUT_S, 'FigS26_correlation_heatmap.png'), dpi=SUP_DPI)


def s_seed():
    df = pd.read_csv(os.path.join(RES, 'dnn_seed_variation.csv'))
    mets = ['AUC', 'CSI', 'F1', 'Kappa', 'Sensitivity', 'Specificity']
    fig, axes = plt.subplots(2, 3, figsize=(SUP_W, 4.3), layout='constrained')
    for k, (ax, m) in enumerate(zip(axes.ravel(), mets)):
        ax.plot(df['seed'], df[m], 'o-', color='#FF9800', lw=1.1, ms=3)
        mu, sd = df[m].mean(), df[m].std()
        ax.axhline(mu, color='#2196F3', ls='--', lw=0.9)
        ax.fill_between(df['seed'], mu - sd, mu + sd, color='#2196F3', alpha=0.15)
        ax.set_xlabel('seed', fontsize=7); ax.grid(alpha=0.3)
        ax.tick_params(labelsize=6.3)
        nm = m.lower() if m in ('Sensitivity', 'Specificity') else S.metric(m)
        plabel(ax, 'abcdef'[k], '%s: %.4f $\\pm$ %.4f' % (nm, mu, sd), size=6.8)
    fig.legend(handles=[mlines.Line2D([], [], color='#FF9800', marker='o', ms=3, lw=1.1, label='single seed'),
                        mlines.Line2D([], [], color='#2196F3', ls='--', lw=0.9, label='ten-seed mean'),
                        mpatches.Patch(color='#2196F3', alpha=0.15, label='$\\pm$1 s.d.')],
               loc='outside lower center', ncol=3, fontsize=6.8, frameon=False)
    save(fig, os.path.join(OUT_S, 'FigS27_dnn_seed_variation.png'), dpi=SUP_DPI)


def s_lulc():
    df = pd.read_csv(os.path.join(RES, 'lulc_encoding_study.csv'))
    order = ['baseline19', 'ordinal20', 'onehot', 'fractions']
    colors = ['#9E9E9E', '#D32F2F', '#2196F3', '#4CAF50']
    base = df[df.variant == 'baseline19']
    fig, axes = plt.subplots(2, 2, figsize=(SUP_W, 4.6), layout='constrained')
    for k, (ax, m) in enumerate(zip(axes.flat, ['AUC', 'CSI', 'F1', 'Kappa'])):
        bp = ax.boxplot([df[df.variant == v][m].values for v in order], tick_labels=order,
                        patch_artist=True, widths=0.55)
        for p_, c in zip(bp['boxes'], colors):
            p_.set_facecolor(c); p_.set_alpha(0.55)
        ax.axhline(base[m].mean(), color='#9E9E9E', ls='--', lw=0.9)
        ax.grid(alpha=0.3, axis='y'); ax.tick_params(labelsize=6.5)
        ax.set_ylabel(S.metric(m))
        plabel(ax, 'abcd'[k])
    fig.legend(handles=[mlines.Line2D([], [], color='#9E9E9E', ls='--', lw=0.9,
                                      label='baseline19 ten-seed mean')],
               loc='outside lower center', fontsize=6.8, frameon=False)
    save(fig, os.path.join(OUT_S, 'FigS28_lulc_encoding_study.png'), dpi=SUP_DPI)


def copies_supp():
    for src, dst in (
            (os.path.join(SRC_MAIN, 'Fig09_per_fold_performance.png'), 'FigS29_per_fold_performance.png'),
            (os.path.join(SRC_MAIN, 'Fig11_shap_importance.png'), 'FigS30_shap_importance.png'),
            (os.path.join(SRC_SUPP, 'FigS29_shap_beeswarm_models.png'), 'FigS31_shap_beeswarm.png')):
        copy(src, os.path.join(OUT_S, dst))


BUILDERS = {'fig01': fig01, 'fig03': fig03, 'fig04': fig04, 'fig08': fig08,
            'fig11': fig11, 'fig12': fig12, 'copies_main': copies_main,
            's_confusion': s_confusion, 's_spatial_cv': s_spatial_cv, 's_shap': s_shap,
            's_heatmap': s_heatmap, 's_seed': s_seed, 's_lulc': s_lulc,
            'copies_supp': copies_supp, 's_roc': s_roc, 's_csi': s_csi}


# =============================================================================
# JFRM version (Manuscript/JFRM/): 8 main figures + graphical abstract
# =============================================================================
JPKG = os.path.join(ROOT, 'Manuscript', 'JFRM')
J_M = os.path.join(JPKG, 'figures', 'main')
J_S = os.path.join(JPKG, 'figures', 'supplementary')
JFRM_REFS = {'A': '§2.1–2.3', 'B': '§2.3', 'C': '§3.1', 'D': '§3.2', 'E': '§3.3–3.4',
             'F': '§3.5; Suppl.', 'G': '§3.5'}


def j_fig03():
    os.makedirs(J_M, exist_ok=True)
    fig03(refs=JFRM_REFS, out_dir=J_M, stem='Fig03_methodological_framework')


def j_fig04():
    """Validation design (a) and ROC on training (b) and hold-out (c) samples."""
    from sklearn.metrics import roc_curve, roc_auc_score
    os.makedirs(J_M, exist_ok=True)
    d = pd.read_csv(os.path.join(A, 'spatial_cv', SIZE, 'fold_assignments.csv'),
                    usecols=['x_3794', 'y_3794', 'fold'])
    sets = {}
    for key, f in (('train', 'samples_train_10x.csv'), ('test', 'samples_test_10x.csv')):
        df = pd.read_csv(os.path.join(SAMP, f), usecols=['label', 'x_3794', 'y_3794'])
        P = _sample((df.x_3794.values, df.y_3794.values), MODELS, SIZE)
        ok = np.all([np.isfinite(P[m]) for m in MODELS], axis=0)
        sets[key] = (df.label.values.astype(int)[ok], {m: P[m][ok] for m in MODELS}, ok.sum())
    fig = plt.figure(figsize=(FIG_W, 5.3), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, :])
    for f, c in FOLD_C.items():
        q = d[d.fold == f]
        ax.scatter(q.x_3794, q.y_3794, s=0.04, c=c, linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.axis('off'); plabel(ax, 'a')
    ax.legend(handles=[mlines.Line2D([], [], ls='', marker='o', ms=4.5, color=c,
                                     label='Fold %d' % f) for f, c in FOLD_C.items()],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5,
              frameon=False, columnspacing=1.4, handletextpad=0.2)
    for k, (key, name) in enumerate((('train', 'training'), ('test', 'validation (hold-out)'))):
        y, P, n = sets[key]
        a = fig.add_subplot(gs[1, k])
        a.plot([0, 1], [0, 1], ls='--', lw=0.7, color='gray')
        for m in MODELS:
            fpr, tpr, _ = roc_curve(y, P[m])
            a.plot(fpr, tpr, lw=1.1, color=COLORS[m],
                   label='%s (%.4f)' % (S.model(m), roc_auc_score(y, P[m])))
            print('   %s %s AUC %.4f n=%d' % (key, m, roc_auc_score(y, P[m]), n), flush=True)
        a.set_xlim(-0.01, 1); a.set_ylim(0, 1.01); a.grid(alpha=0.3)
        a.set_xlabel('false positive rate', fontsize=7); a.set_ylabel('true positive rate', fontsize=7)
        a.tick_params(labelsize=6.3)
        a.legend(loc='lower right', fontsize=5.8, title='$AUC$', title_fontsize=6, frameon=True)
        plabel(a, 'bc'[k], name, size=7)
    save(fig, os.path.join(J_M, 'Fig04_validation_design_roc.png'))


def j_fig05():
    """Random hold-out vs spatial block CV skill, (a)-(d)."""
    os.makedirs(J_M, exist_ok=True)
    mn = pd.read_csv(os.path.join(A, 'spatial_cv', SIZE, 'mean_metrics.csv'))
    rd = pd.read_csv(os.path.join(V, SIZE, 'validation_metrics.csv'))
    fig = plt.figure(figsize=(FIG_W, 4.0), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    x = np.arange(len(CV_MODELS)); w = 0.35
    for k, met in enumerate(['AUC', 'CSI', 'F1', 'Kappa']):
        a = fig.add_subplot(gs[k // 2, k % 2])
        rv = [rd[rd.Model == m][met].values[0] for m in CV_MODELS]
        b1 = a.bar(x - w / 2, rv, w, color=[COLORS[m] for m in CV_MODELS], alpha=0.45,
                   edgecolor='gray', lw=0.5, hatch='//')
        cm_ = [mn[mn.Model == m]['%s_mean' % met].values[0] for m in CV_MODELS]
        cs = [mn[mn.Model == m]['%s_std' % met].values[0] for m in CV_MODELS]
        a.bar(x + w / 2, cm_, w, yerr=cs, capsize=2, color=[COLORS[m] for m in CV_MODELS],
              alpha=0.9, edgecolor='k', lw=0.5)
        for bar, v in zip(b1, rv):
            a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02, '%.3f' % v,
                   ha='center', va='top', fontsize=5.6, rotation=90, color='#37474F')
        a.set_xticks(x); a.set_xticklabels([S.model(m) for m in CV_MODELS], fontsize=6.3)
        a.set_ylim(0, 1.12); a.grid(axis='y', alpha=0.3); a.set_ylabel(S.metric(met), fontsize=7)
        a.tick_params(labelsize=6.3); plabel(a, 'abcd'[k])
    fig.legend(handles=[mpatches.Patch(facecolor='gray', alpha=0.45, hatch='//', label='random hold-out (20 %)'),
                        mpatches.Patch(facecolor='dimgray', alpha=0.9, label='spatial block CV (5 folds, 50 km)')],
               loc='outside lower center', ncol=2, fontsize=6.5, frameon=False)
    save(fig, os.path.join(J_M, 'Fig05_random_vs_spatial.png'))


def j_graphical():
    """Graphical abstract: consensus map + the three validation tiers."""
    os.makedirs(JPKG, exist_ok=True)
    a = _map(os.path.join(CLSF, SIZE, 'FSM_Ensemble_mean_5class.tif'))
    cols = ['#2C7BB6', '#ABD9E9', '#FFFFBF', '#FDAE61', '#D7191C']
    fig = plt.figure(figsize=(7.0, 3.2))
    ax = fig.add_axes([0.01, 0.12, 0.56, 0.86])
    ax.imshow(a, cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(0.5, 6), 5),
              interpolation='nearest'); ax.axis('off')
    ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4, label=l) for c, l in
                       zip(cols, ['very low', 'low', 'moderate', 'high', 'very high'])],
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=7, frameon=False,
              handlelength=1.0, columnspacing=0.8, title='Ensemble flood susceptibility, 12.5 m',
              title_fontsize=7.5)
    tiers = [('Random hold-out', 'AUC 0.99', 'where the inventory already is', '#9e9e9e'),
             ('Spatial block CV (50 km)', 'AUC 0.96', 'in regions left out of training', '#3f7cac'),
             ('Civil-protection reports', 'AUC 0.73', 'flood sites vs wind/snow/ice sites;\nAugust 2023: 0.76', '#2d6a3e')]
    for k, (t, v, sub, c) in enumerate(tiers):
        y0 = 0.70 - k * 0.30
        b = fig.add_axes([0.60, y0, 0.38, 0.25]); b.axis('off')
        b.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0,rounding_size=0.06',
                    facecolor='white', edgecolor=c, lw=1.4, transform=b.transAxes))
        b.text(0.05, 0.72, t, fontsize=8, fontweight='bold', color=c, va='center', transform=b.transAxes)
        b.text(0.95, 0.72, v, fontsize=10, fontweight='bold', ha='right', va='center', transform=b.transAxes)
        b.text(0.05, 0.30, sub, fontsize=6.8, color='#333', va='center', transform=b.transAxes)
    save(fig, os.path.join(JPKG, 'graphical_abstract.png'), dpi=300)


def j_copies():
    os.makedirs(J_M, exist_ok=True); os.makedirs(J_S, exist_ok=True)
    for src, dst in (('Fig01_study_area.png', 'Fig01_study_area.png'),
                     ('Fig02_conditioning_factors.png', 'Fig02_conditioning_factors.png'),
                     ('Fig08_spatial_skill_and_cn.png', 'Fig06_spatial_skill_and_cn.png'),
                     ('Fig09_incident_validation.png', 'Fig07_incident_validation.png'),
                     ('Fig10_shap_rank_stability.png', 'Fig08_shap_rank_stability.png'),
):
        copy(os.path.join(OUT_M, src), os.path.join(J_M, dst))
    for f in sorted(os.listdir(OUT_S)):
        copy(os.path.join(OUT_S, f), os.path.join(J_S, f))
    for src, dst in ((os.path.join(OUT_M, 'Fig06_confusion_matrices.png'), 'FigS32_confusion_matrices.png'),
                     (os.path.join(SRC_MAIN, 'Fig05_multiscale_performance.png'), 'FigS33_sampling_configuration.png')):
        copy(src, os.path.join(J_S, dst))


BUILDERS.update({'j_fig03': j_fig03, 'j_fig04': j_fig04, 'j_fig05': j_fig05, 'j_graphical': j_graphical,
                 'j_copies': j_copies})


def j_fig_maps():
    """Probability (a) and binary (b) maps: four learners 2x2, ensemble enlarged."""
    os.makedirs(J_M, exist_ok=True)
    learners = ['RF', 'XGBoost', 'CatBoost', 'DNN']
    probs = {m: _map(os.path.join(PROB, SIZE, 'FSM_%s_probability.tif' % m)) for m in MODELS}
    bins = {m: _map(os.path.join(CLSF, SIZE, 'FSM_%s_binary.tif' % m)) for m in MODELS}
    h, w = probs['RF'].shape
    asp = w / h
    W, gap, th, lab, key = FIG_W, 0.05, 0.17, 0.20, 0.42
    sw = (W - th * asp - 4 * gap) / 4            # small map width
    sh = sw / asp
    ew = 2 * sw + th * asp                       # ensemble width: spans two small rows
    eh = ew / asp
    blk = lab + 2 * (sh + th) + key
    hw = 0.66 * W                                # agreement map, row (c)
    hh = hw / asp
    row3 = lab + th + hh + 0.54
    H = 2 * blk + row3 + 0.05
    fig = plt.figure(figsize=(W, H))
    cols = ['#E8EAF0', '#1A5FB4']

    def ax_at(x, y, ww, hh):
        return fig.add_axes([x / W, y / H, ww / W, hh / H])

    for b, (arrs, letter) in enumerate(((probs, 'a'), (bins, 'b'))):
        top = H - b * blk
        fig.text(0.004, (top - 0.03) / H, '(%s)' % letter, fontsize=8.5, fontweight='bold', va='top')
        y_row = [top - lab - th - sh, top - lab - 2 * (th + sh)]
        for i, m in enumerate(learners):
            ax = ax_at(gap / 2 + (i % 2) * (sw + gap), y_row[i // 2], sw, sh)
            if b == 0:
                im = ax.imshow(arrs[m], cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            else:
                ax.imshow(arrs[m], cmap=ListedColormap(cols), norm=BoundaryNorm([-.5, .5, 1.5], 2),
                          interpolation='nearest')
            ax.set_title(LABEL[m], fontsize=6.4, fontweight='bold', pad=1.5); ax.axis('off')
        ex = W - ew - gap / 2
        ey = y_row[1]
        ax = ax_at(ex, ey, ew, eh)
        if b == 0:
            ax.imshow(arrs['Ensemble_mean'], cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
        else:
            ax.imshow(arrs['Ensemble_mean'], cmap=ListedColormap(cols),
                      norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation='nearest')
        ax.set_title('Ensemble mean', fontsize=7.4, fontweight='bold', pad=2); ax.axis('off')
        ky = ey - key + 0.05
        if b == 0:
            cax = ax_at(W / 2 - 1.1, ky + 0.16, 2.2, 0.07)
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cb.set_label('susceptibility probability', fontsize=6.3, labelpad=1)
            cb.ax.tick_params(labelsize=5.8, length=2, pad=1)
        else:
            lax = ax_at(0, ky, W, 0.3); lax.axis('off')
            lax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4, label=l)
                                for c, l in zip(cols, ['not flood-prone', 'flood-prone'])],
                       loc='center', ncol=2, fontsize=6.4, frameon=False)
    # row (c): cross-model agreement on the Very High class
    top = H - 2 * blk
    y3 = top - lab - th - hh
    fig.text(0.004, (top - 0.03) / H, '(c)', fontsize=8.5, fontweight='bold', va='top')
    # distinct hue and lightness per count (viridis steps); 0 stays a neutral grey
    cl = ['#E6E6E6', '#FDE725', '#5EC962', '#21918C', '#3B528B', '#440154']
    # national share of each agreement count: the table behind the 76.1 % / 16.4 % in the text
    st = pd.read_csv(os.path.join(A, 'spatial_agreement_stats.csv')).sort_values('Models_Agree_VeryHigh')
    pct = st.Pct_Slovenia.values
    names = ['0 of 5', '1', '2', '3', '4', 'all 5']
    x = gap
    ax = ax_at(x, y3, hw, hh)
    ax.imshow(_map(os.path.join(A, 'spatial_agreement_map.tif')), cmap=ListedColormap(cl),
              norm=BoundaryNorm(np.arange(-0.5, 6), 6), interpolation='nearest')
    ax.set_title('Products classifying the cell Very High', fontsize=7.4, fontweight='bold', pad=2)
    ax.axis('off')
    # share of national area per count, beside the map
    bx0 = x + hw + 0.42
    # the 0-of-5 class is the grey background; it is described in the caption, not keyed
    bax = ax_at(bx0, y3 + 0.18 * hh, W - bx0 - 0.40, 0.64 * hh)
    yy = np.arange(5)[::-1]
    bax.barh(yy, pct[1:], color=cl[1:], edgecolor='0.35', lw=0.4, height=0.72)
    for yv, v in zip(yy, pct[1:]):
        bax.text(v + 0.4, yv, '%.1f %%' % v, va='center', ha='left', fontsize=5.9)
    bax.set_yticks(yy); bax.set_yticklabels(names[1:], fontsize=6.0)
    bax.set_xlim(0, 20); bax.set_xticks([0, 10, 20])
    bax.tick_params(axis='x', labelsize=5.8, length=2, pad=1); bax.tick_params(axis='y', length=0)
    bax.set_xlabel('share of Slovenia (%)', fontsize=6.2, labelpad=1)
    for sp in ('top', 'right'):
        bax.spines[sp].set_visible(False)
    lax = ax_at(0, y3 - 0.50, W, 0.42); lax.axis('off')
    lax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4,
                                       label='%s (%.1f %%)' % (n, v))
                        for c, n, v in zip(cl[1:], names[1:], pct[1:])],
               loc='center', ncol=5, fontsize=6.3, frameon=False, handlelength=1.1,
               columnspacing=1.4, handletextpad=0.4)
    save(fig, os.path.join(J_M, 'Fig09_national_maps.png'))


BUILDERS['j_fig_maps'] = j_fig_maps

if __name__ == '__main__':
    want = sys.argv[1:] or list(BUILDERS)
    for k in want:
        print('[%s]' % k, flush=True)
        BUILDERS[k]()
