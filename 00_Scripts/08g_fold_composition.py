"""
08g — Composition of the spatial cross-validation folds (co-author review 2026-10-22).
=====================================================================================
Reconstructs the 08e / 08f fold design from the v2 samples (03_Samples_v2/v2, train + test pooled, as in 08e) and
reports, per fold: number of 50 km blocks, test and training sizes, flood counts and prevalence, the training samples
removed by the 5 km buffer of 08f (count and %), and the distance from each test sample to the nearest training sample
before and after buffering. Also records the origin of the block grid and, for the source flood-hazard polygons
(frequent, rare, extremely rare; DRSV) and for the connected components of their union, how many contribute flood
samples to more than one fold.

Block assignment is identical to 08e/08f: bx = floor((x - min x) / 50 km), by = floor((y - min y) / 50 km),
fold = (2 bx + by) mod 5 + 1, with min x / min y taken over the pooled samples.

Outputs → 09_Analysis/spatial_cv_v2/fold_composition/{fold_composition.csv, polygon_crossing.csv, grid.json}
"""
import json, os
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY_DIR = os.environ.get('POLY_DIR', '/home/arora/FSM_Slovenia/FSM_Pipeline_LULC_archive/Vectors/Flooded_Areas')
OUT = os.path.join(ROOT, '09_Analysis', 'spatial_cv_v2', 'fold_composition'); os.makedirs(OUT, exist_ok=True)
BLOCK, BUF = 50_000, 5_000

df = pd.concat([pd.read_parquet(os.path.join(ROOT, '03_Samples_v2', 'v2', f'samples_{k}.parquet'),
                                columns=['label', 'x_3794', 'y_3794']) for k in ('train', 'test')], ignore_index=True)
x0, y0 = float(df.x_3794.min()), float(df.y_3794.min())
df['bx'] = ((df.x_3794 - x0) // BLOCK).astype(int); df['by'] = ((df.y_3794 - y0) // BLOCK).astype(int)
df['fold'] = (df.bx * 2 + df.by) % 5 + 1
json.dump(dict(origin_x=x0, origin_y=y0, block_m=BLOCK, crs='EPSG:3794', n_samples=len(df),
               n_blocks=int(df.groupby(['bx', 'by']).ngroups), rule='fold = (2*bx + by) mod 5 + 1'),
          open(os.path.join(OUT, 'grid.json'), 'w'), indent=2)

rows = []
for f in range(1, 6):
    te, tr = df[df.fold == f], df[df.fold != f]
    xy_te, xy_tr = te[['x_3794', 'y_3794']].values, tr[['x_3794', 'y_3794']].values
    d_tr, _ = cKDTree(xy_te).query(xy_tr, k=1)                 # training sample → nearest test sample (08f rule)
    keep = d_tr >= BUF
    d0, _ = cKDTree(xy_tr).query(xy_te, k=1)                   # test sample → nearest training sample
    d1, _ = cKDTree(xy_tr[keep]).query(xy_te, k=1)
    rows.append(dict(fold=f, blocks=te.groupby(['bx', 'by']).ngroups, n_test=len(te), test_flood=int(te.label.sum()),
                     test_nonflood=int((te.label == 0).sum()), test_flood_pct=100 * te.label.mean(),
                     n_train=len(tr), train_flood_pct=100 * tr.label.mean(),
                     n_train_buffered=int(keep.sum()), buffer_removed=int((~keep).sum()),
                     buffer_removed_pct=100 * (~keep).mean(),
                     buffer_removed_flood_pct=100 * (~keep & (tr.label.values == 1)).sum() / (tr.label == 1).sum(),
                     train_flood_pct_buffered=100 * tr.label.values[keep].mean(),
                     nn_dist_median_m=float(np.median(d0)), nn_dist_buffered_min_m=float(d1.min()),
                     nn_dist_buffered_median_m=float(np.median(d1))))
C = pd.DataFrame(rows); C.to_csv(os.path.join(OUT, 'fold_composition.csv'), index=False)
print(C.round(3).to_string())

# --- source flood polygons whose flood samples fall in more than one fold -------------------------------------
fl = df[df.label == 1]
pts = gpd.GeoDataFrame(fl[['fold']], geometry=gpd.points_from_xy(fl.x_3794, fl.y_3794), crs=3794)
layers = {k: gpd.read_file(os.path.join(POLY_DIR, f'{k}_Flood_Slovenia.shp')).to_crs(3794)[['geometry']]
          for k in ('Frequent', 'Rare', 'Extreme_Rare')}
allg = gpd.GeoSeries(pd.concat(layers.values()).geometry.buffer(0), crs=3794)
u = allg.union_all() if hasattr(allg, 'union_all') else allg.unary_union
layers['Union_components'] = gpd.GeoDataFrame(geometry=list(getattr(u, 'geoms', [u])), crs=3794)
P = []
spanning_any = np.zeros(len(pts), bool)
for name, g in layers.items():
    g = g.reset_index(drop=True); g['pid'] = np.arange(len(g))
    j = gpd.sjoin(pts, g[['pid', 'geometry']], how='inner', predicate='within')
    nf = j.groupby('pid').fold.nunique()
    span = nf[nf > 1].index
    in_span = j[j.pid.isin(span)]
    if name != 'Union_components':
        spanning_any[np.isin(np.arange(len(pts)), pts.index.get_indexer(in_span.index.unique()))] = True
    P.append(dict(layer=name, polygons=len(g), polygons_with_samples=int(nf.size), polygons_multi_fold=int(span.size),
                  pct_polygons_multi_fold=100 * span.size / max(nf.size, 1),
                  flood_samples_in_layer=int(j.index.nunique()),
                  flood_samples_in_multi_fold=int(in_span.index.nunique()),
                  pct_flood_samples_in_multi_fold=100 * in_span.index.nunique() / max(j.index.nunique(), 1),
                  area_km2_multi_fold=float(g[g.pid.isin(span)].area.sum() / 1e6)))
P.append(dict(layer='Any source polygon', flood_samples_in_layer=len(pts),
              flood_samples_in_multi_fold=int(spanning_any.sum()),
              pct_flood_samples_in_multi_fold=100 * spanning_any.mean()))
P = pd.DataFrame(P); P.to_csv(os.path.join(OUT, 'polygon_crossing.csv'), index=False)
print(P.round(2).to_string())
print('->', OUT)
