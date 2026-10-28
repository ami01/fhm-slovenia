"""
07v2 — Verification of delivered binary maps at the true base rate, v1 production maps vs v2 maps (review 2026-10-09).
    python 00_Scripts/07v2_verify.py   -> 09_Analysis/verification_v2/verification.csv
Scores every cell valid in the inventory AND in all compared maps (one common cell set): POD, FAR, POFD, FBI, CSI, ETS,
flagged area. 'v1' = production maps (07_Classified_Maps/10x), 'v2' = 07_Classified_Maps_v2/v2.
"""
import os, numpy as np, pandas as pd, rasterio
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '09_Analysis', 'verification_v2'); os.makedirs(OUT, exist_ok=True)
MODELS = ['XGBoost', 'Ensemble_mean', 'CatBoost', 'RF', 'DNN']
SETS = {'v1': os.path.join(ROOT, '07_Classified_Maps', '10x'), 'v2': os.path.join(ROOT, '07_Classified_Maps_v2', 'v2')}
with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
    inv = d.read(1); ok = inv != d.nodata; px = abs(d.transform.a) ** 2 / 1e6
maps = {}
for s, dd in SETS.items():
    for m in MODELS:
        with rasterio.open(os.path.join(dd, f'FSM_{m}_binary.tif')) as d:
            b = d.read(1); maps[(s, m)] = b; ok &= (b != d.nodata)
o = (inv == 1)[ok]; n = ok.sum()
rows = []
for (s, m), b in maps.items():
    f = (b == 1)[ok]; tp = (f & o).sum(); fp = (f & ~o).sum(); fn = (~f & o).sum(); tn = n - tp - fp - fn
    hr = (tp + fp) * (tp + fn) / n
    rows.append(dict(set=s, model=m, POD=tp / (tp + fn), FAR=fp / (tp + fp), POFD=fp / (fp + tn), FBI=(tp + fp) / (tp + fn),
                     CSI=tp / (tp + fp + fn), ETS=(tp - hr) / (tp + fp + fn - hr), flagged_km2=(tp + fp) * px))
R = pd.DataFrame(rows).round(4); R.to_csv(os.path.join(OUT, 'verification.csv'), index=False)
print(f'common valid cells {n:,}, base rate {o.mean():.4f}'); print(R.to_string(index=False))
