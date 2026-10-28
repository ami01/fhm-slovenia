"""
10v — Write the v2 results into _v2_view/ in the file formats the existing stage and figure scripts read
(review 2026-10-09), so 09d, 09e, 09i and the 10s figure builders can be rerun unchanged on the v2 models.
    python 00_Scripts/10v_prepare_v2_view.py
Writes, inside _v2_view/:
  08_Validation/10x/threshold_summary.csv, validation_metrics.csv     (from 05_Results_v2/v2/holdout_predictions)
  09_Analysis/spatial_cv/10x/mean_metrics.csv, per_fold_metrics.csv  (08e v2 set, NATIONAL threshold rule)
  09_Analysis/shap/shap_importance_all_sizes.csv                      (08v2_shap, short feature names as 08d)
"""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, '_v2_view')
SHORT = {'Distance_to_River': 'Dist_River', 'Distance_to_Road': 'Dist_Road', 'Hgk_grouped': 'Hydro_Soil',
         'Pedoloska_grouped': 'Soil_Type', 'River_Density': 'River_Den', 'aspect': 'Aspect', 'drainage_density': 'Drain_Den',
         'elevation': 'Elevation', 'flow_accumulation': 'Flow_Acc', 'hand': 'HAND', 'lito_poly_250_d96_grouped': 'Lithology',
         'ls_factor': 'LS_Factor', 'plan_curvature': 'Plan_Curv', 'profile_curvature': 'Prof_Curv',
         'slovenia_annual_max_daily_p90_3794_12m5': 'Rain_P90', 'spi': 'SPI', 'tri': 'TRI', 'twi': 'TWI', 'valley_depth': 'Valley_Dep'}
ORDER = ['XGBoost', 'Ensemble_mean', 'CatBoost', 'RF', 'DNN']

thr = json.load(open(os.path.join(ROOT, '04_Models_v2', 'v2', 'thresholds.json')))['thresholds']
P = pd.read_parquet(os.path.join(ROOT, '05_Results_v2', 'v2', 'holdout_predictions.parquet'))
y = P.label.values.astype(int)
vm, ts = [], []
for m in ORDER:
    p = P[m].values; t = thr[m]; q = (p >= t).astype(int)
    tp, tn, fp, fn = int(((q == 1) & (y == 1)).sum()), int(((q == 0) & (y == 0)).sum()), int(((q == 1) & (y == 0)).sum()), int(((q == 0) & (y == 1)).sum())
    po = (tp + tn) / len(y); pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / len(y) ** 2
    vm.append(dict(Model=m, Threshold=t, AUC=round(roc_auc_score(y, p), 4), Avg_Prec=round(average_precision_score(y, p), 4),
                   F1=round(2 * tp / (2 * tp + fp + fn), 4), Accuracy=round(po, 4), Kappa=round((po - pe) / (1 - pe), 4),
                   CSI=round(tp / (tp + fp + fn), 4), Sensitivity=round(tp / (tp + fn), 4), Specificity=round(tn / (tn + fp), 4),
                   Precision=round(tp / (tp + fp), 4), TP=tp, TN=tn, FP=fp, FN=fn))
    fpr, tpr, th = roc_curve(y, p); j = np.argmax(tpr - fpr)
    ts.append(dict(Model=m, CSI_threshold=t, CSI_value=round(tp / (tp + fp + fn), 4), Youden_threshold=round(float(th[j]), 4),
                   Youden_value=round(float(tpr[j] - fpr[j]), 4)))
os.makedirs(os.path.join(V, '08_Validation', '10x'), exist_ok=True)
pd.DataFrame(vm).to_csv(os.path.join(V, '08_Validation', '10x', 'validation_metrics.csv'), index=False)
pd.DataFrame(ts).to_csv(os.path.join(V, '08_Validation', '10x', 'threshold_summary.csv'), index=False)

S = os.path.join(ROOT, '09_Analysis', 'spatial_cv_v2', '10x_v2')
if os.path.exists(os.path.join(S, 'per_fold_metrics.csv')):
    F = pd.read_csv(os.path.join(S, 'per_fold_metrics.csv')); F = F[F.threshold_rule == 'national']
    F = F[F.Model != 'Ensemble_mean']                 # the 10s figures expect the four learners
    rows = []
    for m in ['RF', 'XGBoost', 'CatBoost', 'DNN']:
        s = F[F.Model == m]; r = dict(Model=m)
        for c in ['AUC', 'CSI', 'F1', 'Kappa', 'Sensitivity', 'Specificity']:
            r[f'{c}_mean'] = round(s[c].mean(), 4); r[f'{c}_std'] = round(s[c].std(), 4)
        r['SampleSize'] = '10x'; rows.append(r)
    os.makedirs(os.path.join(V, '09_Analysis', 'spatial_cv', '10x'), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(V, '09_Analysis', 'spatial_cv', '10x', 'mean_metrics.csv'), index=False)
    F.to_csv(os.path.join(V, '09_Analysis', 'spatial_cv', '10x', 'per_fold_metrics.csv'), index=False)

SH = os.path.join(ROOT, '09_Analysis', 'shap_v2', 'shap_importance_all_sizes.csv')
if os.path.exists(SH):
    D = pd.read_csv(SH); D['Feature'] = D.Feature.map(lambda f: SHORT.get(f, f))
    os.makedirs(os.path.join(V, '09_Analysis', 'shap'), exist_ok=True)
    D.to_csv(os.path.join(V, '09_Analysis', 'shap', 'shap_importance_all_sizes.csv'), index=False)
print(pd.DataFrame(vm)[['Model', 'Threshold', 'AUC', 'CSI', 'Kappa']].to_string(index=False))
