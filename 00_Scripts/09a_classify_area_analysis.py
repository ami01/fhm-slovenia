"""
Classified Map Area Analysis — Slovenia Flood Susceptibility
============================================================
Reads 5-class and binary classified maps for all 5 models,
computes per-class and binary area statistics (pixels, km², %),
and generates summary tables and plots.

Inputs
  classified_maps/  FSM_{model}_5class.tif | FSM_{model}_binary.tif
  validation/       threshold_summary.csv | validation_metrics.csv

Outputs → ML_Results/analysis/
  class_area_statistics.csv      — per-class area for all models
  binary_flood_statistics.csv    — flood/no-flood areas at optimal threshold
  class_area_comparison.png      — stacked bar chart of class areas
  flood_area_comparison.png      — flood area per model (km² and %)
  threshold_probability_dist.png — probability distribution + thresholds per model
"""

# ── Config ────────────────────────────────────────────────────────────────────
# 06b writes probability maps per sample size. The flat directory holds only a stale
# 1x DNN map from stage 05b, so read the 10x set - which is what the flat classified
# maps in 07_Classified_Maps were produced from. Override with MAP_SIZE=<tag>.
import os as _os
PROB_DIR    = _os.path.join(
    '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/06_Probability_Maps',
    _os.environ.get('MAP_SIZE', '10x'))
CLSF_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/07_Classified_Maps'
VALD_DIR    = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/08_Validation'
OUT_DIR     = '/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/09_Analysis'
NODATA      = -9999
PIXEL_AREA_KM2 = (12.5 * 12.5) / 1e6   # 0.000156 km² per pixel

CLASS_LABELS = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
CLASS_COLORS = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
MODELS       = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
MODEL_COLORS = {'RF': '#2196F3', 'XGBoost': '#FF5722', 'CatBoost': '#4CAF50',
                'DNN': '#FF9800', 'Ensemble_mean': '#9C27B0'}

import os, warnings
warnings.filterwarnings('ignore')
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio

# ── Load threshold & validation tables ───────────────────────────────────────
thresh_df = pd.read_csv(os.path.join(VALD_DIR, 'threshold_summary.csv'))
val_df    = pd.read_csv(os.path.join(VALD_DIR, 'validation_metrics.csv'))
thresh_map = dict(zip(thresh_df['Model'], thresh_df['CSI_threshold']))

print("=" * 65)
print("CSI-OPTIMAL THRESHOLDS")
print("=" * 65)
print(f"\n  {'Model':<20} {'CSI Threshold':>15} {'CSI Value':>12} {'Youden Thresh':>15}")
print(f"  {'-'*20} {'-'*15} {'-'*12} {'-'*15}")
for _, row in thresh_df.iterrows():
    print(f"  {row['Model']:<20} {row['CSI_threshold']:>15.4f} {row['CSI_value']:>12.4f} {row['Youden_threshold']:>15.4f}")

# ── 5-class area statistics ───────────────────────────────────────────────────
print()
print("=" * 65)
print("5-CLASS AREA STATISTICS")
print("=" * 65)

class_records = []
for model in MODELS:
    path = os.path.join(CLSF_DIR, f'FSM_{model}_5class.tif')
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    total_valid = int((arr != NODATA).sum())
    print(f"\n  {model}  (total valid pixels: {total_valid:,}  = {total_valid * PIXEL_AREA_KM2:,.1f} km²)")
    print(f"  {'Class':<12} {'Label':<12} {'Pixels':>12} {'Area (km²)':>12} {'%':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for cls_id, label in CLASS_LABELS.items():
        n_pix = int((arr == cls_id).sum())
        area  = n_pix * PIXEL_AREA_KM2
        pct   = 100.0 * n_pix / total_valid if total_valid > 0 else 0.0
        print(f"  {cls_id:<12} {label:<12} {n_pix:>12,} {area:>12.1f} {pct:>8.2f}%")
        class_records.append({'Model': model, 'Class': cls_id, 'Label': label,
                               'Pixels': n_pix, 'Area_km2': round(area, 2), 'Pct': round(pct, 3)})

class_df = pd.DataFrame(class_records)
class_df.to_csv(os.path.join(OUT_DIR, 'class_area_statistics.csv'), index=False)
print(f"\nSaved: class_area_statistics.csv")

# ── Binary flood area statistics ──────────────────────────────────────────────
print()
print("=" * 65)
print("BINARY FLOOD AREA STATISTICS  (CSI-optimal threshold)")
print("=" * 65)
print(f"\n  {'Model':<20} {'Threshold':>10} {'Flood px':>12} {'Flood km²':>12} {'Flood %':>9} {'No-flood km²':>14}")
print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*9} {'-'*14}")

binary_records = []
for model in MODELS:
    path = os.path.join(CLSF_DIR, f'FSM_{model}_binary.tif')
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    total_valid  = int((arr != NODATA).sum())
    n_flood      = int((arr == 1).sum())
    n_noflood    = int((arr == 0).sum())
    flood_km2    = n_flood   * PIXEL_AREA_KM2
    noflood_km2  = n_noflood * PIXEL_AREA_KM2
    flood_pct    = 100.0 * n_flood / total_valid if total_valid > 0 else 0.0
    thresh       = thresh_map.get(model, float('nan'))
    print(f"  {model:<20} {thresh:>10.4f} {n_flood:>12,} {flood_km2:>12.1f} {flood_pct:>8.2f}% {noflood_km2:>14.1f}")
    binary_records.append({'Model': model, 'CSI_Threshold': thresh,
                           'Flood_pixels': n_flood, 'NoFlood_pixels': n_noflood,
                           'Flood_km2': round(flood_km2, 2), 'NoFlood_km2': round(noflood_km2, 2),
                           'Total_km2': round(total_valid * PIXEL_AREA_KM2, 2),
                           'Flood_pct': round(flood_pct, 3)})

binary_df = pd.DataFrame(binary_records)
binary_df.to_csv(os.path.join(OUT_DIR, 'binary_flood_statistics.csv'), index=False)
print(f"\nSaved: binary_flood_statistics.csv")

# ── Plot 1: Stacked bar — 5-class areas per model ─────────────────────────────
print()
print("Generating plots ...")

fig, ax = plt.subplots(figsize=(13, 7))
x      = np.arange(len(MODELS))
bottom = np.zeros(len(MODELS))
for cls_id, label in CLASS_LABELS.items():
    vals = [class_df[(class_df['Model']==m) & (class_df['Class']==cls_id)]['Area_km2'].values[0]
            for m in MODELS]
    bars = ax.bar(x, vals, bottom=bottom, color=CLASS_COLORS[cls_id-1],
                  label=label, edgecolor='white', linewidth=0.5)
    for bar, v, b in zip(bars, vals, bottom):
        mid = b + v / 2
        total = b + v + (bottom[0] if cls_id == 1 else 0)
        pct = 100 * v / sum(vals) * len(MODELS)   # rough
        if v > 200:  # only label if large enough
            ax.text(bar.get_x() + bar.get_width()/2, mid,
                    f'{v:,.0f}', ha='center', va='center', fontsize=8, color='black')
    bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', '\n') for m in MODELS], fontsize=11)
ax.set_ylabel('Area (km²)', fontsize=12)
ax.set_title('Flood Susceptibility Class Areas per Model — Slovenia (10x, 12.5 m)',
             fontsize=13, fontweight='bold')
ax.legend(title='Susceptibility Class', loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'class_area_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Plot 2: Flood area comparison (binary maps) ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

flood_km2s = [binary_df[binary_df['Model']==m]['Flood_km2'].values[0] for m in MODELS]
flood_pcts = [binary_df[binary_df['Model']==m]['Flood_pct'].values[0] for m in MODELS]
colors     = [MODEL_COLORS[m] for m in MODELS]

bars1 = ax1.bar(MODELS, flood_km2s, color=colors, edgecolor='white', linewidth=0.8, alpha=0.9)
for bar, v in zip(bars1, flood_km2s):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{v:,.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylabel('Flood Area (km²)', fontsize=12)
ax1.set_title('Predicted Flood Area', fontsize=12, fontweight='bold')
ax1.set_xticklabels([m.replace('_', '\n') for m in MODELS], fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

bars2 = ax2.bar(MODELS, flood_pcts, color=colors, edgecolor='white', linewidth=0.8, alpha=0.9)
for bar, v in zip(bars2, flood_pcts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{v:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylabel('Flood Area (%)', fontsize=12)
ax2.set_title('Predicted Flood Area (%)', fontsize=12, fontweight='bold')
ax2.set_xticklabels([m.replace('_', '\n') for m in MODELS], fontsize=10)
ax2.grid(axis='y', alpha=0.3)

for ax in [ax1, ax2]:
    for i, (model, thresh) in enumerate(thresh_map.items()):
        if model in MODELS:
            idx = MODELS.index(model)

plt.suptitle('Binary Flood Classification at CSI-Optimal Thresholds — Slovenia',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'flood_area_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Plot 3: Probability distribution + optimal threshold per model ─────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for i, model in enumerate(MODELS):
    ax = axes_flat[i]
    prob_path = os.path.join(PROB_DIR, f'FSM_{model}_probability.tif')
    # Downsample 1/20 for fast histogram
    with rasterio.open(prob_path) as ds:
        scale = 20
        arr_ds = ds.read(1, out_shape=(ds.height // scale, ds.width // scale),
                         resampling=rasterio.enums.Resampling.nearest)
    valid_pix = arr_ds[(arr_ds != -9999.0)].flatten()

    # Get quantile breaks for this model
    qb = np.percentile(valid_pix, [20, 40, 60, 80])
    thresh = thresh_map.get(model, 0.5)

    ax.hist(valid_pix, bins=100, color=MODEL_COLORS[model], alpha=0.7, density=True, log=True)

    # Quantile class boundaries
    for q, lbl in zip(qb, ['q20', 'q40', 'q60', 'q80']):
        ax.axvline(q, color='gray', ls=':', lw=1.2, alpha=0.8)

    # Optimal threshold
    ax.axvline(thresh, color='red', ls='--', lw=2,
               label=f'CSI thresh = {thresh:.4f}')

    # Class region annotations
    boundaries = [0] + list(qb) + [1]
    for cls_id in range(1, 6):
        mid = (boundaries[cls_id-1] + boundaries[cls_id]) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                CLASS_LABELS[cls_id], ha='center', fontsize=7, color='#444',
                transform=ax.get_xaxis_transform(), va='top')

    ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_xlabel('Probability', fontsize=10)
    ax.set_ylabel('Density (log)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 1)

axes_flat[-1].set_visible(False)
plt.suptitle('Probability Distributions with CSI-Optimal Threshold & Class Boundaries — Slovenia',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'threshold_probability_dist.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Final summary table ───────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\n{'Model':<20} {'Threshold':>10} {'Flood km²':>12} {'Flood %':>9}  "
      f"{'VLow%':>7} {'Low%':>7} {'Mod%':>7} {'High%':>7} {'VHigh%':>8}")
print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*9}  {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
for model in MODELS:
    b = binary_df[binary_df['Model'] == model].iloc[0]
    cls_pcts = [class_df[(class_df['Model']==model) & (class_df['Class']==c)]['Pct'].values[0]
                for c in range(1, 6)]
    print(f"{model:<20} {b['CSI_Threshold']:>10.4f} {b['Flood_km2']:>12,.1f} {b['Flood_pct']:>8.2f}%  "
          + '  '.join(f"{p:>6.2f}%" for p in cls_pcts))

print(f"\nOutputs saved to: {OUT_DIR}")
print(f"  class_area_statistics.csv")
print(f"  binary_flood_statistics.csv")
print(f"  class_area_comparison.png")
print(f"  flood_area_comparison.png")
print(f"  threshold_probability_dist.png")
