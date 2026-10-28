"""09l — Frequency ratio by HAND decile, v1 vs v2 HAND (review 2026-10-09). Same rule as Table S-HAND: bins are
quantiles of the national HAND distribution (deciles), FR = share of flood cells in the bin / share of all cells.
    python 00_Scripts/09l_fr_hand_v2.py -> 09_Analysis/fr_hand_v2.csv"""
import os, numpy as np, pandas as pd, rasterio
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
    inv = d.read(1); ok = inv != d.nodata
rows = []
for tag, path in (('v1', '01_Input_Features/hand.tif'), ('v2', '01_Input_Features_v2/hand.tif')):
    with rasterio.open(os.path.join(ROOT, path)) as d:
        h = d.read(1).astype('float64'); m = ok & (h != d.nodata)
    hv, fl = h[m], inv[m] == 1
    edges = np.unique(np.quantile(hv, np.linspace(0, 1, 11)))
    b = np.clip(np.digitize(hv, edges[1:-1]), 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        s = b == k
        rows.append(dict(hand=tag, bin=k, lo=edges[k], hi=edges[k + 1], cells=int(s.sum()), flood=int((s & fl).sum()),
                         FR=(s & fl).sum() / fl.sum() / (s.sum() / len(hv))))
R = pd.DataFrame(rows).round(3); R.to_csv(os.path.join(ROOT, '09_Analysis', 'fr_hand_v2.csv'), index=False); print(R.to_string(index=False))
