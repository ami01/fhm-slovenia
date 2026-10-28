"""
00y — Rebuild the faulty conditioning factors (v2, review 2026-10-09). Writes 01_Input_Features_v2/.
====================================================================================================
Faults found by 00z_input_audit.py and fixed here, on the reference 12.5 m grid (19,909 × 12,996, EPSG:3794):

1. hand: the v1 layer is elevation above the EUCLIDEAN-nearest drainage cell (01a compute_hand), then regridded
   bilinearly; 8.7 % of cells are negative (min −559 m), where the nearest channel lies across a divide.
   v2 = flow-path HAND (Rennó et al. 2008; Nobre et al. 2011): pyflwdir D8 from a priority-flood on the DEM
   (14 interior nodata holes filled first); drainage = upstream area ≥ DRAIN_KM2 (1 km², fixed a priori, not tuned).
   Needs pyflwdir (not in rapids_env; set PYTHONPATH to a directory that has it).
2. flow_accumulation: v1 was regridded bilinearly (non-integer cell counts). v2 = cells from the same D8 routing.
3. Hgk_grouped: 1.4 % of cells carry non-integer class codes (a categorical layer resampled bilinearly).
   v2 = each non-integer cell takes the most frequent integer class in its 5 × 5 neighbourhood.
Heavy tails (flow accumulation, SPI, LS factor: max 5e7-1e8) are log-transformed at sampling time
(00x_resample_v2.py), not in these rasters.
All other layers are symlinked unchanged. Valley depth is a 51-cell TPI (01a) and is valid as such (renamed in
the text, not changed).
"""
import os, sys, time, numpy as np, rasterio
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1, F2 = os.path.join(ROOT, '01_Input_Features'), os.path.join(ROOT, '01_Input_Features_v2')
os.makedirs(F2, exist_ok=True)
DRAIN_KM2 = 1.0
t0 = time.time()

FALLBACK_ONLY = '--fallback-only' in sys.argv
cell_km2 = 12.5 * 12.5 / 1e6


def rebuild():
    """Steps 1-3 and the symlinks (the first run of 2026-10-09)."""
    global cell_km2
    with rasterio.open(os.path.join(F1, 'elevation.tif')) as d:
        z = d.read(1).astype('float64'); prof = d.profile; nd = d.nodata
    hole = (z == nd) | ~np.isfinite(z)
    # interior nodata holes (14 components, 22 cells): in D8 routing a nodata cell is an outlet, so one hole on a
    # river swallows everything upstream (first run: max accumulation 1,404 km² instead of > 8,000 km²). Fill them
    # from the nearest valid cell; only holes not connected to the raster edge are touched.
    lab, _ = ndimage.label(hole)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    inner = hole & ~np.isin(lab, edge[edge > 0])
    _, (ri, ci) = ndimage.distance_transform_edt(hole, return_indices=True)     # nearest valid cell
    z[inner] = z[ri[inner], ci[inner]]
    print(f'filled {inner.sum()} interior nodata cells', flush=True)
    valid = (z != nd) & np.isfinite(z)
    cell_km2 = abs(prof['transform'].a) ** 2 / 1e6
    print(f'DEM {z.shape}, valid {valid.mean():.3f}', flush=True)

    # ── 1-2. flow-path HAND and flow accumulation (pyflwdir) ──────────────────────────────────────────────
    # pyflwdir derives D8 directions with a priority-flood (Wang & Liu 2006), so there are no unresolved flats.
    # (pysheds left 26,250 flat/pit cells unresolved, which cut the rivers at reservoirs: max accumulation
    # 1,404 km²; patching its directions created loops. Both attempts are logged in notes.)
    import pyflwdir
    zf = np.where(valid, z, -9999.0).astype('float32')
    flw = pyflwdir.from_dem(data=zf, nodata=-9999.0, transform=prof['transform'], latlon=False)
    print(f'  flow directions (priority-flood) {time.time() - t0:.0f} s', flush=True)
    acc_np = flw.upstream_area(unit='cell').astype('float64'); print(f'  accumulation {time.time() - t0:.0f} s', flush=True)
    drain = (acc_np >= DRAIN_KM2 / cell_km2) & valid
    hand = flw.hand(drain=drain, elevtn=zf).astype('float32'); print(f'  HAND {time.time() - t0:.0f} s', flush=True)
    hand = np.where(valid & np.isfinite(hand) & (hand > -9999), np.maximum(hand, 0), -9999.0).astype('float32')
    meta = prof | dict(dtype='float32', nodata=-9999.0, compress='deflate', tiled=True, blockxsize=512, blockysize=512, BIGTIFF='YES')
    with rasterio.open(os.path.join(F2, 'hand.tif'), 'w', **meta) as d:
        d.write(hand, 1)
    with rasterio.open(os.path.join(F2, 'flow_accumulation.tif'), 'w', **meta) as d:
        d.write(np.where(valid & (acc_np > 0), acc_np, -9999.0).astype('float32'), 1)
    hv = hand[valid & (hand != -9999)]
    print(f'HAND v2: min {hv.min():.2f}, median {np.median(hv):.2f}, p99 {np.percentile(hv, 99):.1f}, zero share '
          f'{(hv == 0).mean():.4f} (drainage share {drain[valid].mean():.4f}), unreachable cells '
          f'{((hand == -9999) & valid).sum():,}; max upstream area {acc_np.max() * cell_km2:.0f} km²', flush=True)
    del flw, acc_np, drain, hand

    # ── 3. Hgk_grouped: non-integer boundary cells → 5 × 5 integer mode ─────────────────────────────────────
    with rasterio.open(os.path.join(F1, 'Hgk_grouped.tif')) as d:
        h = d.read(1); hp = d.profile; hnd = d.nodata
    bad = (h != hnd) & (np.abs(h - np.round(h)) > 1e-6)
    hi = np.where((h != hnd) & ~bad, np.round(h), 0).astype('int16')             # 0 = unknown for the vote
    classes = [c for c in np.unique(hi) if c != 0]
    votes = np.stack([ndimage.uniform_filter((hi == c).astype('float32'), 5) for c in classes])
    mode = np.asarray(classes, 'int16')[votes.argmax(0)]
    fixed = np.where(bad, mode, h).astype('float32')
    with rasterio.open(os.path.join(F2, 'Hgk_grouped.tif'), 'w', **(hp | dict(compress='deflate'))) as d:
        d.write(fixed, 1)
    print(f'Hgk: {bad.sum():,} non-integer cells ({bad[h != hnd].mean():.2%}) reassigned', flush=True)

    # ── symlink the unchanged layers ────────────────────────────────────────────────────────────────────────
    for f in os.listdir(F1):
        if f.endswith('.tif') and not os.path.exists(os.path.join(F2, f)):
            os.symlink(os.path.realpath(os.path.join(F1, f)), os.path.join(F2, f))
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {F2}')


# ── 4. cells whose flow path leaves Slovenia before reaching a drainage cell (added after the first run) ───
# The routing leaves them without a drain (about 4 % of cells, mostly along the border). Dropping them would bias samples away
# from the border, so they take the height above the Euclidean-nearest v2 drainage cell, clipped at 0, and are
# flagged in hand_fallback_mask.tif. Run on its own with: python 00y_rebuild_features_v2.py --fallback-only
def hand_fallback():
    with rasterio.open(os.path.join(F1, 'elevation.tif')) as d:
        zz = d.read(1).astype('float32'); vv = (zz != d.nodata) & np.isfinite(zz)
    with rasterio.open(os.path.join(F2, 'flow_accumulation.tif')) as d:
        ac = d.read(1)
    with rasterio.open(os.path.join(F2, 'hand.tif')) as d:
        hh = d.read(1); hp_ = d.profile
    drain_ = vv & (ac >= DRAIN_KM2 / cell_km2)
    miss = vv & (hh == -9999)
    _, (ri, ci) = ndimage.distance_transform_edt(~drain_, return_indices=True)
    hh[miss] = np.maximum(zz[miss] - zz[ri[miss], ci[miss]], 0)
    with rasterio.open(os.path.join(F2, 'hand.tif'), 'w', **hp_) as d:
        d.write(hh, 1)
    with rasterio.open(os.path.join(F2, 'hand_fallback_mask.tif'), 'w',
                       **(hp_ | dict(dtype='uint8', nodata=255))) as d:
        d.write(np.where(vv, miss, 255).astype('uint8'), 1)
    print(f'HAND fallback: {miss.sum():,} cells ({miss[vv].mean():.2%}); median fallback HAND {np.median(hh[miss]):.1f} m')


if __name__ == '__main__':
    if not FALLBACK_ONLY:
        rebuild()
    hand_fallback()
    print(f'total {(time.time() - t0) / 60:.1f} min')
