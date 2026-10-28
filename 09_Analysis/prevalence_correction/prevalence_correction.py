"""
Prevalence-correction analysis for FSM Slovenia (10x / operational baseline).

Question: the CSI-optimal threshold used throughout the manuscript is chosen
against a BALANCED 1:1 held-out test set (~50% flood / ~50% non-flood). That
same threshold is then applied unmodified to the full national probability
grid, whose true prevalence is ~5.6% flood / ~94.4% non-flood. Does choosing
the threshold against the TRUE population prevalence instead change the
threshold, the implied skill, and the resulting flood-prone area estimate?

Method: reuse the exact same held-out test samples and the exact same
full-grid probability extraction procedure as 00_Scripts/07_threshold_classify_validate.py
(so this is directly comparable to the numbers already in the manuscript), but
reweight each held-out sample by the true population count divided by the
observed test-set count for its class (Horvitz-Thompson-style reweighting),
so that the weighted confusion matrix at any threshold is an unbiased estimate
of the confusion matrix on the true, imbalanced population. Search a fine grid
of thresholds for the one that maximises weighted CSI. Then apply that
corrected threshold to the already-computed full-country probability raster
(no re-inference; just re-counting pixels against a different cut) to get a
corrected area estimate, directly comparable to Table 9 (tab:binaryarea) in
the manuscript.

No model is retrained and no new inference is run: this only touches held-out
test predictions (already legitimately out-of-sample) and a fixed, previously
computed probability surface.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

ROOT = "/Users/ami_01/FSM_Pipeline_LULC"
SAMPLES_DIR = os.path.join(ROOT, "03_Samples")
PROB_DIR = os.path.join(ROOT, "06_Probability_Maps", "10x")
OUT_DIR = "/private/tmp/claude-501/-Users-ami-01-FSM-Pipeline-LULC/1200dfe8-390c-4fe4-b5dd-ae6a22e9f16f/scratchpad"

MODELS = ["RF", "XGBoost", "CatBoost", "DNN", "Ensemble_mean"]
NODATA_IN = -9999.0

# True national population counts (from the manuscript's flood-inventory section
# and dataset_inventory.md): 7,326,493 flood cells / 122,423,292 non-flood cells
# among the 129,749,785 valid national cells.
TRUE_FLOOD = 7_326_493
TRUE_NONFLOOD = 122_423_292
TRUE_TOTAL = TRUE_FLOOD + TRUE_NONFLOOD
TRUE_PREVALENCE = TRUE_FLOOD / TRUE_TOTAL

# Existing (balanced-set) CSI-optimal thresholds and areas, from the manuscript
# (Tables tab:holdout / tab:binaryarea) and 08_Validation/threshold_summary.csv,
# reproduced here only for the side-by-side comparison printed at the end.
CURRENT = {
    "RF":            dict(thresh=0.5172, csi=0.9186, area_km2=2278.63, pct=11.25),
    "XGBoost":       dict(thresh=0.5369, csi=0.9406, area_km2=1956.70, pct=9.66),
    "CatBoost":      dict(thresh=0.5222, csi=0.9255, area_km2=2143.06, pct=10.58),
    "DNN":           dict(thresh=0.5320, csi=0.9085, area_km2=2491.56, pct=12.30),
    "Ensemble_mean": dict(thresh=0.5566, csi=0.9280, area_km2=2100.53, pct=10.37),
}

CELL_AREA_KM2 = 12.5 * 12.5 / 1_000_000.0  # 156.25 m^2 -> km^2


def load_test_set():
    df = pd.read_csv(os.path.join(SAMPLES_DIR, "samples_test_10x.csv"))
    y = df["label"].values.astype(int)
    xs = df["x_3794"].values
    ys = df["y_3794"].values
    print(f"Loaded {len(df):,} held-out test samples "
          f"({(y==1).sum():,} flood / {(y==0).sum():,} non-flood)")
    return y, xs, ys


def extract_probs(model, xs, ys):
    path = os.path.join(PROB_DIR, f"FSM_{model}_probability.tif")
    with rasterio.open(path) as ds:
        tf = ds.transform
        cols = np.floor((xs - tf.c) / tf.a).astype(int)
        rows = np.floor((ys - tf.f) / tf.e).astype(int)
        rows = np.clip(rows, 0, ds.height - 1)
        cols = np.clip(cols, 0, ds.width - 1)
        arr = ds.read(1)
        probs = arr[rows, cols].astype(np.float64)
        probs[probs == NODATA_IN] = np.nan
    return probs


def weighted_confusion(y, probs, w, thresholds):
    """Vectorised weighted TP/FP/FN/TN across a grid of thresholds."""
    is_flood = y == 1
    p_flood = probs[is_flood]
    w_flood = w[is_flood]
    p_nonflood = probs[~is_flood]
    w_nonflood = w[~is_flood]

    # For each threshold: TP = weighted flood samples with prob>=t
    tp = np.array([w_flood[p_flood >= t].sum() for t in thresholds])
    fn = w_flood.sum() - tp
    fp = np.array([w_nonflood[p_nonflood >= t].sum() for t in thresholds])
    tn = w_nonflood.sum() - fp
    return tp, fp, fn, tn


def metrics_at(tp, fp, fn, tn):
    csi = tp / (tp + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else np.nan
    total = tp + fp + fn + tn
    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total ** 2)
    kappa = (po - pe) / (1 - pe)
    return dict(CSI=csi, Sensitivity=sens, Specificity=spec, Precision=prec,
                F1=f1, Kappa=kappa)


def count_area_above_threshold(model, thresh):
    path = os.path.join(PROB_DIR, f"FSM_{model}_probability.tif")
    n_above = 0
    n_valid = 0
    with rasterio.open(path) as ds:
        strip = 500
        for row0 in range(0, ds.height, strip):
            row1 = min(row0 + strip, ds.height)
            win = Window(0, row0, ds.width, row1 - row0)
            arr = ds.read(1, window=win)
            valid = arr != NODATA_IN
            n_valid += valid.sum()
            n_above += (valid & (arr >= thresh)).sum()
    area_km2 = n_above * CELL_AREA_KM2
    total_km2 = n_valid * CELL_AREA_KM2
    pct = 100.0 * n_above / n_valid
    return area_km2, pct, total_km2


def main():
    print(f"True national prevalence: {TRUE_FLOOD:,} flood / {TRUE_NONFLOOD:,} "
          f"non-flood = {TRUE_PREVALENCE*100:.3f}% flood\n")

    y, xs, ys = load_test_set()
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    w1 = TRUE_FLOOD / n1
    w0 = TRUE_NONFLOOD / n0
    w = np.where(y == 1, w1, w0)
    print(f"Per-sample weights: flood x{w1:.4f}  non-flood x{w0:.4f}  "
          f"(ratio {w0/w1:.3f} : 1)\n")

    thresholds = np.linspace(0.001, 0.999, 999)

    rows_out = []
    for model in MODELS:
        print(f"=== {model} ===")
        probs = extract_probs(model, xs, ys)
        valid = ~np.isnan(probs)
        y_v, probs_v, w_v = y[valid], probs[valid], w[valid]

        tp, fp, fn, tn = weighted_confusion(y_v, probs_v, w_v, thresholds)
        csi_curve = tp / (tp + fp + fn)
        best_i = np.nanargmax(csi_curve)
        t_star = thresholds[best_i]
        m_star = metrics_at(tp[best_i], fp[best_i], fn[best_i], tn[best_i])

        # CSI achieved at the OLD (balanced-derived) threshold, but measured
        # against the true-prevalence-weighted confusion matrix.
        old_t = CURRENT[model]["thresh"]
        old_i = np.argmin(np.abs(thresholds - old_t))
        m_old_thresh_true_prev = metrics_at(tp[old_i], fp[old_i], fn[old_i], tn[old_i])

        area_new, pct_new, total_km2 = count_area_above_threshold(model, t_star)

        print(f"  balanced-set threshold (current, in manuscript): {old_t:.4f}  "
              f"-> reported CSI (balanced) = {CURRENT[model]['csi']:.4f}")
        print(f"  same threshold, evaluated at TRUE prevalence:     "
              f"CSI = {m_old_thresh_true_prev['CSI']:.4f}  "
              f"Precision = {m_old_thresh_true_prev['Precision']:.4f}")
        print(f"  true-prevalence-OPTIMAL threshold:                {t_star:.4f}  "
              f"-> CSI (true prevalence) = {m_star['CSI']:.4f}  "
              f"Precision = {m_star['Precision']:.4f}  Kappa = {m_star['Kappa']:.4f}")
        print(f"  area @ old threshold (manuscript):   {CURRENT[model]['area_km2']:.2f} km^2 "
              f"({CURRENT[model]['pct']:.2f} %)")
        print(f"  area @ corrected threshold:            {area_new:.2f} km^2 "
              f"({pct_new:.2f} %)")
        print(f"  area delta:                            "
              f"{area_new - CURRENT[model]['area_km2']:+.2f} km^2 "
              f"({pct_new - CURRENT[model]['pct']:+.2f} pp)\n")

        rows_out.append(dict(
            Model=model,
            Threshold_balanced=old_t,
            CSI_balanced_reported=CURRENT[model]["csi"],
            CSI_balancedThresh_atTruePrevalence=m_old_thresh_true_prev["CSI"],
            Precision_balancedThresh_atTruePrevalence=m_old_thresh_true_prev["Precision"],
            Threshold_truePrevalence=t_star,
            CSI_truePrevalence=m_star["CSI"],
            Precision_truePrevalence=m_star["Precision"],
            F1_truePrevalence=m_star["F1"],
            Kappa_truePrevalence=m_star["Kappa"],
            Sensitivity_truePrevalence=m_star["Sensitivity"],
            Specificity_truePrevalence=m_star["Specificity"],
            Area_km2_balancedThresh=CURRENT[model]["area_km2"],
            Pct_balancedThresh=CURRENT[model]["pct"],
            Area_km2_truePrevalenceThresh=area_new,
            Pct_truePrevalenceThresh=pct_new,
            Area_delta_km2=area_new - CURRENT[model]["area_km2"],
            Pct_delta_pp=pct_new - CURRENT[model]["pct"],
        ))

    out_df = pd.DataFrame(rows_out)
    out_path = os.path.join(OUT_DIR, "prevalence_correction_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
