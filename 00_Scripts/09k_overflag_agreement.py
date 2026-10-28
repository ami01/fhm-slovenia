"""
09k — Where does the over-flagged area lie, and how much do the four learners agree? (review 2026-10-09)
=====================================================================================================
For each binary map: area flagged outside the inventory, split by distance to the nearest inventory cell
(0-100 m, 100-500 m, 0.5-2 km, > 2 km) and by HAND (< 2 m). Agreement layer v2: number of the FOUR learners'
binary maps flagging a cell (the ensemble is their mean, not an independent fifth vote; the binary maps
avoid the quintile-class construction of 09_Analysis/spatial_agreement_map.tif). Also: of the river-flood
incident sites outside the inventory that the ensemble flags, how far are they from the inventory?
Outputs → 09_Analysis/overflag_v2/: overflag_by_distance.csv, agreement4.csv, agreement4_binary.tif,
          incident_hits_by_distance.csv
"""
import os, numpy as np, pandas as pd, rasterio
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = os.environ.get('MAP_SIZE', '10x')
MAPSET = os.environ.get('MAP_SET')
F = os.path.join(ROOT, '01_Input_Features')
CM = os.path.join(ROOT, '07_Classified_Maps_v2', MAPSET) if MAPSET else os.path.join(ROOT, '07_Classified_Maps', SIZE)
OUT = os.path.join(ROOT, '09_Analysis', 'overflag_v2' + (f'_{MAPSET}' if MAPSET else '')); os.makedirs(OUT, exist_ok=True)
HAND = os.path.join(ROOT, '01_Input_Features_v2' if MAPSET == 'v2' else '01_Input_Features', 'hand.tif')
MODELS = ['XGBoost', 'Ensemble_mean', 'CatBoost', 'RF', 'DNN']
with rasterio.open(os.path.join(F, 'Flooded_NoFlood.tif')) as d:
    inv = d.read(1); nd = d.nodata; prof = d.profile; px = abs(d.transform.a) ** 2 / 1e6
with rasterio.open(HAND) as d:
    hand = d.read(1).astype('float32'); hand[hand == d.nodata] = np.nan
valid = inv != nd; fl = (inv == 1) & valid
dist = distance_transform_edt(~fl) * abs(prof['transform'].a)
bands = [(0, 100), (100, 500), (500, 2000), (2000, np.inf)]
rows, B = [], {}
for m in MODELS:
    with rasterio.open(os.path.join(CM, f'FSM_{m}_binary.tif')) as d:
        b = d.read(1); B[m] = (b == 1) & valid
    fp = B[m] & ~fl
    r = dict(model=m, flagged_km2=B[m].sum() * px, inventory_km2=fl.sum() * px, outside_km2=fp.sum() * px,
             outside_HAND_lt2_share=np.nanmean(hand[fp] < 2))
    for a, z in bands:
        sel = fp & (dist >= a) & (dist < z)
        r[f'km2_{a}_{z}'] = sel.sum() * px; r[f'share_{a}_{z}'] = sel.sum() / max(fp.sum(), 1)
    rows.append(r)
pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, 'overflag_by_distance.csv'), index=False)
print(pd.DataFrame(rows).round(3).T.to_string())
n4 = sum(B[m].astype(np.uint8) for m in ('RF', 'XGBoost', 'CatBoost', 'DNN'))
ag = [dict(n_models=k, km2=((n4 == k) & valid).sum() * px, share=((n4 == k) & valid).sum() / valid.sum(),
           inventory_share=((n4 == k) & fl).sum() / max(((n4 == k) & valid).sum(), 1)) for k in range(5)]
pd.DataFrame(ag).round(4).to_csv(os.path.join(OUT, 'agreement4.csv'), index=False)
print(pd.DataFrame(ag).round(3).to_string(index=False))
with rasterio.open(os.path.join(OUT, 'agreement4_binary.tif'), 'w', **(prof | dict(dtype='uint8', nodata=255, compress='deflate'))) as d:
    d.write(np.where(valid, n4, 255).astype('uint8'), 1)
np.save(os.path.join(OUT, '_dist_to_inventory_m.npy'), dist.astype('float32'))   # reused below / by the manuscript checks
print('->', OUT)
