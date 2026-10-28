"""Extract the modal band of raba_frac_202606.tif as a categorical LULC factor,
matching the CgFs_Aligned convention (Float32, nodata=0, LZW, 512 blocks)."""
import numpy as np, rasterio, collections

SRC = "/home/arora/Documents/FSM_Slovenia/For_Analysis/raba_frac_202606.tif"
DST = "/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs_Aligned/LULC.tif"
REF = "/home/arora/Documents/FSM_Slovenia/For_Analysis/CgFs_Aligned/Hgk_grouped.tif"

CLASSES = ["arable", "permanent_crop", "grassland", "woody",
           "built_up", "water_wetland", "bare_open"]

with rasterio.open(SRC) as src:
    modal_idx = list(src.descriptions).index("modal_gid") + 1
    prof = src.profile.copy()
    prof.update(count=1, dtype="float32", nodata=0.0, compress="lzw",
                predictor=1, tiled=True, blockxsize=512, blockysize=512)
    prof.pop("zlevel", None)

    counts = collections.Counter()
    with rasterio.open(DST, "w", **prof) as dst:
        for _, win in src.block_windows(1):
            a = src.read(modal_idx, window=win).astype(np.float32)
            a[np.isnan(a)] = 0.0          # outside mask -> nodata
            dst.write(a, 1, window=win)   # 0 already = RABA gap -> nodata
            v, c = np.unique(a.astype(np.int16), return_counts=True)
            counts.update(dict(zip(v.tolist(), c.tolist())))
        dst.set_band_description(1, "LULC")
        dst.update_tags(
            source="RABA_20260630 via raba_fractions.py (modal of 1.25 m burn)",
            classes=",".join(f"{i}={n}" for i, n in enumerate(CLASSES, 1)),
            nodata_meaning="0 = outside Slovenia mask or no RABA polygon coverage",
        )

# verify against the reference grid
with rasterio.open(DST) as d, rasterio.open(REF) as r:
    print(f"grid matches CgFs_Aligned: "
          f"{d.transform == r.transform and (d.width, d.height) == (r.width, r.height) and d.crs == r.crs}")
    print(f"size {d.width} x {d.height}  dtype {d.dtypes[0]}  nodata {d.nodata}")

tot = sum(c for v, c in counts.items() if v != 0)
print(f"\nclass distribution ({tot:,} valid cells):")
for i, n in enumerate(CLASSES, 1):
    c = counts.get(i, 0)
    print(f"  {i} {n:<16} {c:>13,}  {100*c/tot:5.2f}%")
print(f"  0 nodata           {counts.get(0,0):>13,}")
