"""
03h — Hard-negative training sample (review 2026-10-09). Design fixed a priori, one configuration:
same flood training points and same test set as 10x; HALF of the 10x non-flood training points are replaced by
non-flood cells drawn uniformly from the 'hard' zone = valid, outside the inventory, within 1 km of an inventory cell
and HAND (v2) <= 10 m. Rationale: 70 % of the over-flagged area lies within 500 m of mapped zones, where random
non-flood points (mostly mountains) give the learners almost no examples.
    python 00_Scripts/03h_hard_negative_samples.py -> 03_Samples/samples_train_10x_hardneg.csv (x, y, label)
"""
import os, numpy as np, pandas as pd, rasterio
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(42)
tr = pd.read_csv(os.path.join(ROOT, '03_Samples', 'samples_train_10x.csv'), usecols=['label', 'x_3794', 'y_3794'])
te = pd.read_csv(os.path.join(ROOT, '03_Samples', 'samples_test_10x.csv'), usecols=['x_3794', 'y_3794'])
with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
    inv = d.read(1); nd = d.nodata; T = d.transform
with rasterio.open(os.path.join(ROOT, '01_Input_Features_v2', 'hand.tif')) as d:
    hand = d.read(1)
dist = np.load(os.path.join(ROOT, '09_Analysis', 'overflag_v2', '_dist_to_inventory_m.npy'))
hard = (inv == 0) & (dist <= 1000) & (hand <= 10) & (hand != -9999)
print(f'hard zone: {hard.sum():,} cells ({hard.sum() * 156.25 / 1e6:.0f} km2)')
tr_r, tr_c = rasterio.transform.rowcol(T, te.x_3794.values, te.y_3794.values)
hard[np.asarray(tr_r), np.asarray(tr_c)] = False                       # never draw a test cell
neg = tr[tr.label == 0]; n_new = len(neg) // 2
keep = neg.sample(len(neg) - n_new, random_state=42)
idx = rng.choice(np.flatnonzero(hard.ravel()), n_new, replace=False)
r, c = np.unravel_index(idx, hard.shape)
x, y = rasterio.transform.xy(T, r, c)
new = pd.DataFrame({'label': 0, 'x_3794': np.asarray(x), 'y_3794': np.asarray(y)})
out = pd.concat([tr[tr.label == 1], keep, new], ignore_index=True).sample(frac=1, random_state=42)
out.to_csv(os.path.join(ROOT, '03_Samples', 'samples_train_10x_hardneg.csv'), index=False)
print(f'flood {int((out.label == 1).sum()):,}, non-flood kept {len(keep):,} + hard {n_new:,} -> {len(out):,}')
