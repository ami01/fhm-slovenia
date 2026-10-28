#!/usr/bin/env python3
"""Sanity-check the RABA fraction raster against its reference grid."""
import numpy as np
import rasterio

OUT = "/home/arora/Documents/FSM_Slovenia/For_Analysis/raba_frac_202606.tif"
REF = "/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs_Aligned/Flooded_NoFlood.tif"

with rasterio.open(OUT) as ds, rasterio.open(REF) as ref:
    print(f"bands: {ds.count}  size: {ds.width} x {ds.height}  dtype: {ds.dtypes[0]}")
    print(f"descriptions: {ds.descriptions}")
    print(f"tags: {ds.tags()}")

    # 1. exact grid match with the feature stack
    same = (ds.transform == ref.transform and ds.width == ref.width
            and ds.height == ref.height and ds.crs == ref.crs)
    print(f"\ngrid identical to Flooded_NoFlood.tif: {same}")
    if not same:
        print(f"  out: {ds.transform}\n  ref: {ref.transform}")

    names = [d or f"band{i}" for i, d in enumerate(ds.descriptions, 1)]
    n_frac = sum(1 for n in names if n.startswith("frac_"))
    cov_idx = names.index("coverage") + 1

    # 2/3. streamed over blocks: fractions must sum to coverage, and ranges sane
    max_resid = 0.0
    tot = np.zeros(n_frac)
    cov_hist = np.zeros(5, dtype=np.int64)   # <0.5, <0.9, <0.999, ~1, nan
    n_valid = 0
    rng_min = np.full(ds.count, np.inf)
    rng_max = np.full(ds.count, -np.inf)

    for _, win in ds.block_windows(1):
        arr = ds.read(window=win).astype(np.float64)
        valid = ~np.isnan(arr[0])
        if not valid.any():
            cov_hist[4] += (~valid).sum()
            continue
        for b in range(ds.count):
            v = arr[b][valid]
            if v.size:
                rng_min[b] = min(rng_min[b], v.min())
                rng_max[b] = max(rng_max[b], v.max())

        fr = arr[:n_frac][:, valid]
        cov = arr[cov_idx - 1][valid]
        max_resid = max(max_resid, np.abs(fr.sum(axis=0) - cov).max())
        tot += fr.sum(axis=1)
        n_valid += valid.sum()
        cov_hist[0] += (cov < 0.5).sum()
        cov_hist[1] += ((cov >= 0.5) & (cov < 0.9)).sum()
        cov_hist[2] += ((cov >= 0.9) & (cov < 0.999)).sum()
        cov_hist[3] += (cov >= 0.999).sum()
        cov_hist[4] += (~valid).sum()

    print(f"\nvalid cells: {n_valid:,}   nodata cells: {cov_hist[4]:,}")
    print(f"max |sum(fractions) - coverage|: {max_resid:.3e}   "
          f"({'OK' if max_resid < 1e-5 else 'FAIL'})")

    print("\nper-band range over valid cells:")
    for i, n in enumerate(names):
        print(f"  {n:<22} min={rng_min[i]:.4f}  max={rng_max[i]:.4f}")

    print("\ncoverage distribution (valid cells):")
    for lbl, c in zip(["<0.5", "0.5-0.9", "0.9-0.999", ">=0.999"], cov_hist[:4]):
        print(f"  {lbl:<10} {c:>14,}  ({100 * c / max(n_valid, 1):5.2f}%)")

    print("\nareal share of Slovenia by group (mean fraction over valid cells):")
    for n, t in sorted(zip(names[:n_frac], tot), key=lambda x: -x[1]):
        print(f"  {n:<22} {100 * t / max(n_valid, 1):6.2f}%   {t * 12.5 * 12.5 / 1e6:9.1f} km2")
