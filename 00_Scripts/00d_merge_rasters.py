"""
merge_rasters.py
────────────────
Mosaics multiple GeoTIFF tiles from a download folder into a single merged GeoTIFF.

Input:  All *.tif files in the specified folder (e.g. Terrascope downloads)
Output: merged_output.tif in the same folder

Usage:
    Update 'folder' path and run:
    python merge_rasters.py

Dependencies: rasterio
"""

import glob
from rasterio.merge import merge
import rasterio

folder = "/home/arora/Downloads/terrascope_download_20260416_151715/"
tifs = sorted(glob.glob(folder + "*.tif"))

print(f"Found {len(tifs)} rasters: {[f.split('/')[-1] for f in tifs]}")

srcs = [rasterio.open(f) for f in tifs]
mosaic, transform = merge(srcs)

out_meta = srcs[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform,
    "compress": "lzw"
})

out_path = folder + "merged_output.tif"
with rasterio.open(out_path, "w", **out_meta) as dest:
    dest.write(mosaic)

for s in srcs:
    s.close()

print(f"Merged raster saved to: {out_path}")