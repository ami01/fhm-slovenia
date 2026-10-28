"""
prepare_rainfall.py
────────────────────
Extracts rainfall statistics from E-OBS NetCDF data for Slovenia and exports
them as GeoTIFF rasters.

Input:
  - rr_ens_mean_0.1deg_reg_v31.0e.nc   (E-OBS daily rainfall ensemble mean)
    Cropped to Slovenia AOI: lat 45.4–46.9, lon 13.3–16.7

Outputs (raw 0.1° resolution, EPSG:4324):
  - slovenia_mean_annual_rainfall.tif     (climatological mean annual total, mm/year)
  - slovenia_annual_max_daily.tif         (annual max daily rainfall per year, mm/day, 75 bands)

Note: Run Slovenia_Rainfall_Interpolation.py next to reproject to EPSG:3794 @ 12.5 m.

Dependencies: xarray, rioxarray
"""

import xarray as xr
import rioxarray # Required for GeoTIFF export

# 1. Path to your local file
file_path = "/home/arora/Downloads/rr_ens_mean_0.1deg/rr_ens_mean_0.1deg_reg_v31.0e.nc"

# 2. Open the dataset
print("1. Opening local NetCDF file...")
ds = xr.open_dataset(file_path)

# 3. Crop to Slovenia AOI (Area of Interest)
# Note: E-OBS uses 'latitude' and 'longitude' as dimension names
print("2. Cropping to Slovenia AOI...")
ds_cropped = ds.sel(
    latitude=slice(45.4, 46.9),  # South to North
    longitude=slice(13.3, 16.7)  # West to East
)

print("3. Calculating annual metrics...")
# Calculate total rainfall for every year
annual_totals = ds_cropped['rr'].resample(time='YE').sum(skipna=False)

# Calculate the Climatological Annual Average
mean_annual_rainfall = annual_totals.mean(dim='time')

# Calculate the Annual Maximum daily rainfall per year
annual_max_daily = ds_cropped['rr'].resample(time='YE').max(skipna=False)

# 4. Export to .tif
print("4. Exporting to GeoTIFF...")

# Set the Coordinate Reference System (CRS) to WGS84 (standard for E-OBS)
mean_annual_rainfall.rio.write_crs("EPSG:4324", inplace=True)
annual_max_daily.rio.write_crs("EPSG:4324", inplace=True)

# Save files
mean_annual_rainfall.rio.to_raster('slovenia_mean_annual_rainfall.tif')
annual_max_daily.rio.to_raster('slovenia_annual_max_daily.tif')

print("Success! .tif files have been generated.")