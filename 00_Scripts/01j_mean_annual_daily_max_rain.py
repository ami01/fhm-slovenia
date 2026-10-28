"""
Slovenia Rainfall Interpolation Maps
=====================================
Source data: E-OBS ECMWF ensembles (rr_ens_mean_0.1deg, v31.0e), 1950-2024 (75 annual maxima)
Inputs:
  - slovenia_mean_annual_rainfall.tif  (1 band, mm/year)
  - slovenia_annual_max_daily.tif      (75 bands, mm/day, one per year)

Workflow:
  1. Read rasters → extract cell-center lon/lat arrays
  2. Fill NaN border cells with nearest-neighbor
  3. Cubic spline interpolation (RectBivariateSpline) → 0.01° fine grid
  4. Light Gaussian smoothing (sigma=1.5 cells)
  5. Slovenia polygon mask
  6. ARSO-style contourf visualization

Dependencies:
  pip install rasterio geopandas scipy matplotlib shapely
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import geopandas as gpd
from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import mapping
import math, warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD RASTERS ──────────────────────────────────────────────────────────
f_mean = 'slovenia_mean_annual_rainfall.tif'
f_maxd = 'slovenia_annual_max_daily.tif'

with rasterio.open(f_mean) as src:
    mean_rain = src.read(1)
    T = src.transform
    rows, cols = mean_rain.shape

with rasterio.open(f_maxd) as src:
    # 75 bands (years) → mean annual max daily
    max_daily = np.nanmean(src.read(), axis=0)

# Cell-center coordinates (origin at SW corner because y-step > 0)
lons = T.c + T.a * (np.arange(cols) + 0.5)   # 13.35 … 16.65
lats = T.f + T.e * (np.arange(rows) + 0.5)   # 45.45 … 46.85

# ── 2. INTERPOLATE TO 0.01° ──────────────────────────────────────────────────
lon_fine = np.arange(lons[0] - 0.05, lons[-1] + 0.05, 0.01)
lat_fine = np.arange(lats[0] - 0.05, lats[-1] + 0.05, 0.01)

def interpolate_cubic(data, lons, lats, lon_fine, lat_fine, sigma=1.5):
    LON2D, LAT2D = np.meshgrid(lons, lats)
    valid = ~np.isnan(data)
    if not valid.all():
        nn = NearestNDInterpolator(
            np.column_stack([LON2D[valid], LAT2D[valid]]), data[valid])
        data = data.copy()
        data[~valid] = nn(LON2D[~valid], LAT2D[~valid])
    spl = RectBivariateSpline(lats, lons, data, kx=3, ky=3)
    return gaussian_filter(spl(lat_fine, lon_fine), sigma=sigma)

mean_fine = interpolate_cubic(mean_rain, lons, lats, lon_fine, lat_fine)
maxd_fine = interpolate_cubic(max_daily, lons, lats, lon_fine, lat_fine)

# ── 3. SLOVENIA MASK ─────────────────────────────────────────────────────────
# Provide your own shapefile or GeoPackage of Slovenia boundary
# slov = gpd.read_file('slovenia_boundary.gpkg')
# geom = [mapping(slov.geometry.values[0])]
# fine_tf = from_bounds(lon_fine[0]-0.005, lat_fine[0]-0.005,
#                       lon_fine[-1]+0.005, lat_fine[-1]+0.005,
#                       len(lon_fine), len(lat_fine))
# mask = geometry_mask(geom, transform=fine_tf, invert=True,
#                      out_shape=(len(lat_fine), len(lon_fine)))
# mean_fine = np.where(mask, mean_fine, np.nan)
# maxd_fine = np.where(mask, maxd_fine, np.nan)

# ── 4. PLOT ──────────────────────────────────────────────────────────────────
LONF, LATF = np.meshgrid(lon_fine, lat_fine)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7.5), facecolor='white')

for ax, data, levels, cmap, title, unit in [
    (ax1, mean_fine,
     [600,800,1000,1200,1400,1600,1800,2000,2200,2400],
     'YlGnBu', 'Mean annual precipitation (mm)', 'mm'),
    (ax2, maxd_fine,
     [30,40,50,60,70,80,90,100,110,120],
     'Blues', 'Mean annual max daily precipitation (mm)', 'mm'),
]:
    ax.set_facecolor('#d0e8f0')
    cf = ax.contourf(LONF, LATF, data, levels=levels, cmap=cmap, extend='both')
    ax.contour(LONF, LATF, data, levels=levels, colors='white',
               linewidths=0.3, alpha=0.4)
    # slov.boundary.plot(ax=ax, color='black', linewidth=1.5)  # uncomment
    ax.set_xlim(13.15, 16.85); ax.set_ylim(45.35, 47.05)
    ax.set_aspect('equal')
    # Legend
    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.BoundaryNorm(levels, cmap_obj.N)
    patches = [Patch(facecolor=cmap_obj(norm(levels[i]+(levels[i+1]-levels[i])*0.5)),
                     edgecolor='black', linewidth=0.5,
                     label=f'{levels[i]}–{levels[i+1]}')
               for i in range(len(levels)-1)]
    ax.legend(handles=patches, title=f'({unit})', loc='lower right',
              fontsize=7.5, title_fontsize=7.5, frameon=True)
    ax.set_title(title, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('slovenia_rainfall_maps.png', dpi=200, bbox_inches='tight')
print("Done → slovenia_rainfall_maps.png")