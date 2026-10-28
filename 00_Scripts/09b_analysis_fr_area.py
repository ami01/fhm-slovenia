"""
Slovenia Flood Susceptibility Mapping - Analysis Script
Items 3, 4, 5, and 6:
  3. Susceptibility Class Area Statistics
  4. Frequency Ratio Analysis
  5. National Flood Map Comparison
  6. Spatial Agreement
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

warnings.filterwarnings('ignore')

# ─────────────────────────── PATHS ────────────────────────────────────────────
ALIGNED_DIR   = "/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/01_Input_Features"
FEAT_FILE     = "/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/02_Collinearity/selected_features.txt"
MAPS_DIR      = "/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/07_Classified_Maps"
VEC_DIR       = "/home/arora/Documents/FSM_Slovenia/For_Analysis/Vectors"
FLOOD_DIR     = os.path.join(VEC_DIR, "Flooded_Areas")
OUT_DIR       = "/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/09_Analysis"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS        = ["RF", "XGBoost", "CatBoost", "Ensemble_mean", "DNN"]
PIXEL_AREA_KM2 = 12.5 * 12.5 / 1e6   # 156.25 m² → km²
STRIP_ROWS    = 300

CLASS_LABELS  = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
CLASS_COLORS  = {1: "#2166ac", 2: "#74add1", 3: "#ffffbf", 4: "#f46d43", 5: "#d73027"}

# LULC added: it is nominal (codes 1-7). Without this it is binned into 10 quantile
# classes as though continuous, which produces meaningless frequency ratios for a
# variable whose numeric ordering carries no information.
CATEGORICAL_CGFS = {"Hgk_grouped", "Pedoloska_grouped", "lito_poly_250_d96_grouped",
                    "LULC"}
FLOOD_NOFLOOD_TIF = os.path.join(ALIGNED_DIR, "Flooded_NoFlood.tif")

NATIONAL_FLOODS = {
    "Frequent":      os.path.join(FLOOD_DIR, "Frequent_Flood_Slovenia.shp"),
    "Rare":          os.path.join(FLOOD_DIR, "Rare_Flood_Slovenia.shp"),
    "Extreme_Rare":  os.path.join(FLOOD_DIR, "Extreme_Rare_Flood_Slovenia.shp"),
}

# ─────────────────────────── HELPERS ──────────────────────────────────────────

def read_features():
    with open(FEAT_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def get_ref_meta():
    """Return metadata from the first 5-class map (reference grid)."""
    ref = os.path.join(MAPS_DIR, "FSM_RF_5class.tif")
    with rasterio.open(ref) as src:
        return src.meta.copy(), src.transform, src.crs, src.width, src.height


def read_raster_strips(path, strip_rows=STRIP_ROWS):
    """Yield (data_strip, row_off) for each strip."""
    with rasterio.open(path) as src:
        h = src.height
        for row_off in range(0, h, strip_rows):
            rows = min(strip_rows, h - row_off)
            window = rasterio.windows.Window(0, row_off, src.width, rows)
            yield src.read(1, window=window), row_off, src.nodata


def read_full_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
    return data, nodata


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 – SUSCEPTIBILITY CLASS AREA STATISTICS (Item 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 1: Susceptibility Class Area Statistics")
print("="*70)

def compute_area_stats():
    flood_data, flood_nd = read_full_raster(FLOOD_NOFLOOD_TIF)
    flood_mask = (flood_data != flood_nd) & (flood_data != -9999)

    rows_list = []
    for model in MODELS:
        path = os.path.join(MAPS_DIR, f"FSM_{model}_5class.tif")
        cls_data, cls_nd = read_full_raster(path)

        valid_mask = (cls_data != cls_nd) & (cls_data > 0)
        total_valid = valid_mask.sum()

        for c in range(1, 6):
            class_mask = (cls_data == c)
            n_pix = class_mask.sum()
            area_km2 = n_pix * PIXEL_AREA_KM2
            pct_total = 100.0 * n_pix / total_valid if total_valid > 0 else 0.0

            # % of class pixels that are actually flooded
            both = (class_mask & flood_mask & (flood_data == 1)).sum()
            pct_flooded = 100.0 * both / n_pix if n_pix > 0 else 0.0

            rows_list.append({
                "Model": model,
                "Class": c,
                "Class_Label": CLASS_LABELS[c],
                "Pixels": int(n_pix),
                "Area_km2": round(area_km2, 2),
                "Pct_Total": round(pct_total, 2),
                "Pct_Flooded": round(pct_flooded, 3),
            })
        print(f"  {model}: done")

    df = pd.DataFrame(rows_list)
    df.to_csv(os.path.join(OUT_DIR, "area_statistics.csv"), index=False)
    print(f"  Saved area_statistics.csv")
    return df

area_df = compute_area_stats()

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODELS))
bar_w = 0.15
bottoms = np.zeros(len(MODELS))
model_areas = {model: [] for model in MODELS}
for c in range(1, 6):
    sub = area_df[area_df["Class"] == c]
    vals = [sub[sub["Model"] == m]["Area_km2"].values[0] for m in MODELS]
    ax.bar(x, vals, bar_w * 5 / 1.1, bottom=bottoms, color=CLASS_COLORS[c],
           label=CLASS_LABELS[c], edgecolor='white', linewidth=0.5)
    bottoms += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=11)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Area (km²)", fontsize=12)
ax.set_title("Flood Susceptibility Class Areas by Model", fontsize=14, fontweight='bold')
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "area_statistics.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved area_statistics.png")

print("\nArea Statistics Summary:")
print(area_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 – FREQUENCY RATIO ANALYSIS (Item 4)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 2: Frequency Ratio Analysis")
print("="*70)

FEATURES = read_features()
print(f"  Features ({len(FEATURES)}): {FEATURES}")

def compute_frequency_ratio():
    # Read flood label at 1/10 scale
    with rasterio.open(FLOOD_NOFLOOD_TIF) as src:
        h, w = src.height, src.width
        out_h = max(1, h // 10)
        out_w = max(1, w // 10)
        # scalar band index -> 2-D result; out_shape must be 2-D and no [0] strip.
        # The original form sliced off the first ROW, leaving 6 valid pixels.
        flood_small = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest)
        flood_nd = src.nodata

    flood_valid = (flood_small != flood_nd) & (flood_small != -9999)
    is_flood = (flood_small == 1) & flood_valid
    total_flood_px = is_flood.sum()
    total_valid_px = flood_valid.sum()
    print(f"  Downsampled grid: {out_h}x{out_w} = {out_h*out_w:,} pixels")
    print(f"  Total valid: {total_valid_px:,}, Flood: {total_flood_px:,}")
    if flood_small.ndim != 2:
        raise SystemExit(f"flood_small is {flood_small.shape}, expected 2-D")
    if total_valid_px < 100_000 or total_flood_px == 0:
        raise SystemExit(
            f"Frequency-ratio sample is degenerate: {total_valid_px:,} valid / "
            f"{total_flood_px:,} flood pixels on a {out_h}x{out_w} grid. "
            "Refusing to write meaningless FR values.")

    fr_rows = []

    for feat in FEATURES:
        feat_path = os.path.join(ALIGNED_DIR, f"{feat}.tif")
        if not os.path.exists(feat_path):
            print(f"  WARNING: {feat_path} not found, skipping")
            continue

        with rasterio.open(feat_path) as src:
            cgf_small = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.nearest)
            cgf_nd = src.nodata

        cgf_valid_mask = flood_valid.copy()
        if cgf_nd is not None:
            cgf_valid_mask &= (cgf_small != cgf_nd) & (~np.isnan(cgf_small.astype(float)))

        cgf_vals = cgf_small[cgf_valid_mask].astype(np.float64)
        flood_vals = is_flood[cgf_valid_mask]

        is_cat = feat in CATEGORICAL_CGFS

        if is_cat:
            if feat == "Hgk_grouped":
                bins_vals = np.round(cgf_vals).astype(int)
                unique_bins = np.unique(bins_vals)
            else:
                bins_vals = cgf_vals.astype(int)
                unique_bins = np.unique(bins_vals)

            for b in unique_bins:
                mask_b = (bins_vals == b)
                tot_b = mask_b.sum()
                fld_b = (flood_vals & mask_b).sum()
                fr = ((fld_b / total_flood_px) / (tot_b / total_valid_px)) if (tot_b > 0 and total_flood_px > 0) else 0.0
                fr_rows.append({
                    "Feature": feat, "Bin": int(b), "Bin_Label": str(int(b)),
                    "Total_Px": int(tot_b), "Flood_Px": int(fld_b), "FR": round(fr, 4)
                })
        else:
            # 10 quantile bins
            try:
                quantiles = np.nanpercentile(cgf_vals, np.linspace(0, 100, 11))
                quantiles = np.unique(quantiles)
            except Exception:
                quantiles = np.linspace(cgf_vals.min(), cgf_vals.max(), 11)

            bin_ids = np.digitize(cgf_vals, quantiles[1:-1])  # 0..len(q)-2
            for b_idx in np.unique(bin_ids):
                mask_b = (bin_ids == b_idx)
                tot_b = mask_b.sum()
                fld_b = (flood_vals & mask_b).sum()
                if b_idx < len(quantiles) - 1:
                    lo = quantiles[b_idx]
                    hi = quantiles[b_idx + 1] if b_idx + 1 < len(quantiles) else quantiles[-1]
                else:
                    lo = quantiles[b_idx] if b_idx < len(quantiles) else quantiles[-1]
                    hi = cgf_vals.max()
                fr = ((fld_b / total_flood_px) / (tot_b / total_valid_px)) if (tot_b > 0 and total_flood_px > 0) else 0.0
                fr_rows.append({
                    "Feature": feat, "Bin": b_idx, "Bin_Label": f"{lo:.2f}–{hi:.2f}",
                    "Total_Px": int(tot_b), "Flood_Px": int(fld_b), "FR": round(fr, 4)
                })

        print(f"  {feat}: done")

    fr_df = pd.DataFrame(fr_rows)
    fr_df.to_csv(os.path.join(OUT_DIR, "frequency_ratio.csv"), index=False)
    print("  Saved frequency_ratio.csv")
    return fr_df

fr_df = compute_frequency_ratio()

# Plot FR bar charts
n_feats = len(FEATURES)
ncols = 4
nrows = int(np.ceil(n_feats / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
axes = axes.flatten()

for idx, feat in enumerate(FEATURES):
    ax = axes[idx]
    sub = fr_df[fr_df["Feature"] == feat].copy()
    if sub.empty:
        ax.axis('off')
        continue

    colors = ["#d62728" if v > 1 else "#1f77b4" for v in sub["FR"].values]
    bars = ax.bar(range(len(sub)), sub["FR"].values, color=colors, edgecolor='white', linewidth=0.3)
    ax.axhline(1.0, color='black', linewidth=1.0, linestyle='--', alpha=0.8)
    ax.set_title(feat.replace("_", " "), fontsize=8, fontweight='bold')
    ax.set_xlabel("Bin", fontsize=7)
    ax.set_ylabel("FR", fontsize=7)
    ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["Bin_Label"].values, rotation=45, ha='right', fontsize=5)

# Hide unused axes
for idx in range(n_feats, len(axes)):
    axes[idx].axis('off')

# Add legend
legend_patches = [
    mpatches.Patch(color='#d62728', label='FR > 1 (Flood-Prone)'),
    mpatches.Patch(color='#1f77b4', label='FR ≤ 1'),
]
fig.legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.9)
plt.suptitle("Frequency Ratio Analysis — All Conditioning Factors", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "frequency_ratio_plots.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved frequency_ratio_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 – NATIONAL FLOOD MAP COMPARISON (Item 5)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 3: National Flood Map Comparison")
print("="*70)

def rasterize_vector(shp_path, ref_meta, ref_transform, ref_crs):
    """Rasterize a shapefile to the reference grid."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((ref_meta['height'], ref_meta['width']), dtype=np.uint8)

    out = rasterize(
        shapes=shapes,
        out_shape=(ref_meta['height'], ref_meta['width']),
        transform=ref_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return out


def compute_national_comparison():
    meta, transform, crs, width, height = get_ref_meta()

    # Rasterize national flood zones
    flood_zones = {}
    for zone_name, shp_path in NATIONAL_FLOODS.items():
        print(f"  Rasterizing {zone_name}...")
        rast = rasterize_vector(shp_path, meta, transform, crs)
        flood_zones[zone_name] = rast
        print(f"    {zone_name}: {rast.sum():,} flood pixels")

    comp_rows = []
    for model in ["Ensemble_mean", "XGBoost"]:
        path = os.path.join(MAPS_DIR, f"FSM_{model}_5class.tif")
        cls_data, cls_nd = read_full_raster(path)
        valid_mask = (cls_data != cls_nd) & (cls_data > 0)
        cls4_5_mask = valid_mask & ((cls_data == 4) | (cls_data == 5))
        cls5_mask   = valid_mask & (cls_data == 5)

        for zone_name, zone_rast in flood_zones.items():
            zone_mask = (zone_rast == 1)
            total_zone_px = (zone_mask & valid_mask).sum()

            # % of national zone area that falls in class 4 or 5
            overlap_4_5 = (zone_mask & cls4_5_mask).sum()
            pct_zone_in_4_5 = 100.0 * overlap_4_5 / total_zone_px if total_zone_px > 0 else 0.0

            # % of Very High (class 5) that overlaps each national zone
            total_cls5 = cls5_mask.sum()
            overlap_cls5 = (zone_mask & cls5_mask).sum()
            pct_cls5_in_zone = 100.0 * overlap_cls5 / total_cls5 if total_cls5 > 0 else 0.0

            comp_rows.append({
                "Model": model,
                "Flood_Zone": zone_name,
                "Zone_Area_km2": round(total_zone_px * PIXEL_AREA_KM2, 2),
                "Overlap_Cls4_5_km2": round(overlap_4_5 * PIXEL_AREA_KM2, 2),
                "Pct_Zone_in_Cls4_5": round(pct_zone_in_4_5, 2),
                "VeryHigh_Area_km2": round(total_cls5 * PIXEL_AREA_KM2, 2),
                "Pct_VeryHigh_in_Zone": round(pct_cls5_in_zone, 2),
            })
        print(f"  {model}: done")

    df = pd.DataFrame(comp_rows)
    df.to_csv(os.path.join(OUT_DIR, "national_flood_comparison.csv"), index=False)
    print("  Saved national_flood_comparison.csv")
    return df

comp_df = compute_national_comparison()

# Plot grouped bar chart
zones = list(NATIONAL_FLOODS.keys())
models_comp = ["Ensemble_mean", "XGBoost"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, metric, title_suffix in [
    (0, "Pct_Zone_in_Cls4_5", "% of National Zone in High/Very High (Class 4–5)"),
    (1, "Pct_VeryHigh_in_Zone", "% of Very High Class (Class 5) in National Zone"),
]:
    ax = axes[ax_idx]
    x = np.arange(len(zones))
    bar_w = 0.35
    colors_m = ['#e41a1c', '#377eb8']
    for mi, model in enumerate(models_comp):
        sub = comp_df[comp_df["Model"] == model]
        vals = [sub[sub["Flood_Zone"] == z][metric].values[0] for z in zones]
        ax.bar(x + mi * bar_w - bar_w / 2, vals, bar_w, label=model, color=colors_m[mi], alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([z.replace("_", " ") for z in zones], fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title(title_suffix, fontsize=10, fontweight='bold', wrap=True)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(100, ax.get_ylim()[1] * 1.1))
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=7.5)

plt.suptitle("Comparison with National Flood Hazard Zones", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "national_flood_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved national_flood_comparison.png")

print("\nNational Flood Comparison:")
print(comp_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 – SPATIAL AGREEMENT (Item 6)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART 4: Spatial Agreement")
print("="*70)

def compute_spatial_agreement():
    meta, transform, crs, width, height = get_ref_meta()
    nodata_out = -9999

    # Build agreement map strip by strip
    agreement_path = os.path.join(OUT_DIR, "spatial_agreement_map.tif")

    out_meta = meta.copy()
    out_meta.update(dtype='int16', nodata=nodata_out, count=1)

    # Pre-read all valid masks from strip 0 to check valid area
    # First pass: read all 5 models strip by strip, accumulate
    print("  Building agreement map (strip by strip)...")

    model_paths = [os.path.join(MAPS_DIR, f"FSM_{m}_5class.tif") for m in MODELS]
    model_files = [rasterio.open(p) for p in model_paths]
    try:
        ref_nd = model_files[0].nodata

        # agreement_counts[k] = number of pixels where exactly k models agree on class 5
        agreement_counts = np.zeros(6, dtype=np.int64)  # 0..5

        with rasterio.open(agreement_path, 'w', **out_meta) as dst:
            for row_off in range(0, height, STRIP_ROWS):
                rows = min(STRIP_ROWS, height - row_off)
                window = rasterio.windows.Window(0, row_off, width, rows)

                strips = []
                for mf in model_files:
                    s = mf.read(1, window=window)
                    strips.append(s)

                # Valid if all models have valid data
                valid = np.ones((rows, width), dtype=bool)
                for s in strips:
                    nd = ref_nd if ref_nd is not None else -9999
                    valid &= (s != nd) & (s > 0)

                # Count agreement on class 5
                agree = np.zeros((rows, width), dtype=np.int16)
                for s in strips:
                    agree += (s == 5).astype(np.int16)

                out_strip = np.where(valid, agree, nodata_out).astype(np.int16)
                dst.write(out_strip, 1, window=window)

                # Accumulate stats
                for k in range(6):
                    agreement_counts[k] += int((out_strip == k).sum())

    finally:
        for mf in model_files:
            mf.close()

    print(f"  Saved spatial_agreement_map.tif")

    # Compute stats
    total_valid = agreement_counts.sum()
    stats_rows = []
    for k in range(6):
        pct = 100.0 * agreement_counts[k] / total_valid if total_valid > 0 else 0.0
        area_km2 = agreement_counts[k] * PIXEL_AREA_KM2
        stats_rows.append({
            "Models_Agree_VeryHigh": k,
            "Pixels": int(agreement_counts[k]),
            "Area_km2": round(area_km2, 2),
            "Pct_Slovenia": round(pct, 4),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUT_DIR, "spatial_agreement_stats.csv"), index=False)
    print("  Saved spatial_agreement_stats.csv")

    print("\nSpatial Agreement Stats:")
    print(stats_df.to_string(index=False))
    return stats_df


agree_df = compute_spatial_agreement()

# Plot spatial agreement
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart
ax = axes[0]
colors_agree = ['#2166ac', '#74add1', '#abd9e9', '#fee090', '#f46d43', '#d73027']
bars = ax.bar(agree_df["Models_Agree_VeryHigh"], agree_df["Pct_Slovenia"],
              color=colors_agree[:len(agree_df)], edgecolor='white', linewidth=0.5)
ax.set_xlabel("Number of Models Agreeing on 'Very High'", fontsize=11)
ax.set_ylabel("% of Total Slovenia Area", fontsize=11)
ax.set_title("Spatial Agreement: Very High Susceptibility\n(all 5 models)", fontsize=11, fontweight='bold')
ax.set_xticks(agree_df["Models_Agree_VeryHigh"])
for bar in bars:
    ax.annotate(f"{bar.get_height():.2f}%",
                (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                ha='center', va='bottom', fontsize=9)

# Downsampled map visualization
ax2 = axes[1]
agree_map_path = os.path.join(OUT_DIR, "spatial_agreement_map.tif")
with rasterio.open(agree_map_path) as src:
    h_ds = max(1, src.height // 20)
    w_ds = max(1, src.width // 20)
    # A scalar band index returns a 2-D array, so out_shape must be 2-D and there is
    # no leading band axis to strip. The original `read(1, out_shape=(1,h,w))[0]`
    # sliced off the first ROW instead, yielding a 1-D array that imshow rejects.
    ds_map = src.read(1, out_shape=(h_ds, w_ds),
                      resampling=rasterio.enums.Resampling.nearest)
    assert ds_map.ndim == 2, f"expected 2-D downsampled map, got {ds_map.shape}"
    ds_map_float = ds_map.astype(float)
    nd = src.nodata
    if nd is not None:
        ds_map_float[ds_map == nd] = np.nan

cmap_agree = matplotlib.colors.ListedColormap(colors_agree)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap_agree.N)
im = ax2.imshow(ds_map_float, cmap=cmap_agree, norm=norm, interpolation='nearest')
ax2.set_title("Agreement Map\n(downsampled 20×)", fontsize=11, fontweight='bold')
ax2.axis('off')
cbar = plt.colorbar(im, ax=ax2, ticks=[0, 1, 2, 3, 4, 5], fraction=0.04, pad=0.04)
cbar.set_label("Models agreeing on Very High", fontsize=9)
cbar.ax.set_yticklabels(['0', '1', '2', '3', '4', '5'], fontsize=8)

plt.suptitle("Spatial Agreement Analysis", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spatial_agreement.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved spatial_agreement.png")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED WORD REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Generating Word Report...")
print("="*70)

def df_to_docx_table(doc, df, title=None, caption=None):
    if title:
        p = doc.add_heading(title, level=2)
    if caption:
        cp = doc.add_paragraph(caption)
        cp.style = 'Caption'

    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
        hdr_cells[i].paragraphs[0].runs[0].bold = True

    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)

    doc.add_paragraph()


doc = Document()
doc.add_heading("Slovenia Flood Susceptibility Mapping — Analysis Report", level=0)
doc.add_paragraph(
    "This report summarises the results of Items 3–6 of the FSM analysis: "
    "class area statistics, frequency ratio analysis, comparison with national "
    "flood hazard zones, and spatial agreement across models."
)

# --- Part 1 ---
doc.add_heading("1. Susceptibility Class Area Statistics (Item 3)", level=1)
doc.add_paragraph(
    "For each of the five models, pixels were counted per susceptibility class (1–5), "
    "converted to km², and expressed as a percentage of the total valid Slovenia area. "
    "The cross-tabulation column 'Pct_Flooded' shows what percentage of pixels in each class "
    "are historically flooded (from the Flooded_NoFlood reference raster)."
)
df_to_docx_table(doc, area_df, title=None, caption="Table 1: Class area statistics for all models")
doc.add_picture(os.path.join(OUT_DIR, "area_statistics.png"), width=Inches(6))

# --- Part 2 ---
doc.add_heading("2. Frequency Ratio Analysis (Item 4)", level=1)
doc.add_paragraph(
    "Frequency Ratio (FR) was computed for each conditioning factor at 1/10 spatial scale "
    "(~1.3 M pixels). Continuous factors were binned into 10 quantile classes; categorical "
    "factors (Hgk_grouped, Pedoloska_grouped, lito_poly_250_d96_grouped) used their natural "
    "unique values. FR > 1 indicates flood-prone classes; FR < 1 indicates relatively safe classes."
)
# Show first 60 rows to keep table manageable
fr_display = fr_df.head(60)
df_to_docx_table(doc, fr_display, title=None,
                 caption="Table 2: Frequency Ratio values (first 60 rows; full data in frequency_ratio.csv)")
doc.add_picture(os.path.join(OUT_DIR, "frequency_ratio_plots.png"), width=Inches(6.5))

# --- Part 3 ---
doc.add_heading("3. National Flood Map Comparison (Item 5)", level=1)
doc.add_paragraph(
    "National flood zones (Frequent, Rare, Extreme Rare) were rasterized to the model grid "
    "and overlaid with the Ensemble_mean and XGBoost 5-class maps. "
    "Two metrics are reported: (a) % of each national zone that falls in model class 4 or 5, "
    "and (b) % of the Very High class (5) that overlaps each national zone."
)
df_to_docx_table(doc, comp_df, title=None, caption="Table 3: National flood zone comparison")
doc.add_picture(os.path.join(OUT_DIR, "national_flood_comparison.png"), width=Inches(6.5))

# --- Part 4 ---
doc.add_heading("4. Spatial Agreement (Item 6)", level=1)
doc.add_paragraph(
    "The agreement map counts, for each pixel, how many of the five models classify it as "
    "Very High susceptibility (class 5). Values range from 0 (no model) to 5 (all models). "
    "A consensus map of pixels with ≥4 models agreeing on Very High is discussed below."
)
df_to_docx_table(doc, agree_df, title=None, caption="Table 4: Spatial agreement statistics")
doc.add_picture(os.path.join(OUT_DIR, "spatial_agreement.png"), width=Inches(6.5))

# Consensus stats
consensus_pct = agree_df[agree_df["Models_Agree_VeryHigh"] >= 4]["Pct_Slovenia"].sum()
doc.add_paragraph(
    f"Consensus Very High area (≥4 models agreeing): {consensus_pct:.2f}% of Slovenia."
)

report_path = os.path.join(OUT_DIR, "analysis_report.docx")
doc.save(report_path)
print(f"  Saved analysis_report.docx")

# ─────────────────────────── SUMMARY ──────────────────────────────────────────
print("\n" + "="*70)
print("ALL OUTPUTS SAVED TO:", OUT_DIR)
print("="*70)
outputs = [
    "area_statistics.csv",
    "area_statistics.png",
    "frequency_ratio.csv",
    "frequency_ratio_plots.png",
    "national_flood_comparison.csv",
    "national_flood_comparison.png",
    "spatial_agreement_map.tif",
    "spatial_agreement_stats.csv",
    "spatial_agreement.png",
    "analysis_report.docx",
]
for f in outputs:
    fpath = os.path.join(OUT_DIR, f)
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    print(f"  {'OK' if exists else 'MISSING':6s}  {f}  ({size:,} bytes)")
print("="*70)
print("Done.")
