"""
10y — Figure 3 (methodological framework) for the JFRM v2 manuscript (review 2026-10-11).
Reuses the layout of 10s_jem_final_figures.fig03 with the v2 workflow: input audit, validation split, transferability
tests, screening and prevalence thresholds, four-learner agreement, area of applicability, warning-unit statistics.
    python 00_Scripts/10y_fig03_v2.py  → Manuscript/JFRM_v2/figures/main/Fig03_methodological_framework.{png,pdf}
"""
import importlib.util, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location('f10s', os.path.join(HERE, '10s_jem_final_figures.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
f.BANDS = [
    ('A', 'Data and common grid', '', 'blue', 'layers', [
        ('Conditioning data', 'DEM, hydrography, roads, soils, geology, rainfall, land use'),
        ('Flood inventory', 'frequent, rare and extremely rare zones; 5.6\u00a0% of cells'),
        ('Input audit', 'flow-path HAND, routed flow accumulation, repaired hydrogeology; log of heavy tails')]),
    ('B', 'Factor screening', '', 'blue', 'funnel', [
        ('24 candidates', 'topographic, hydrological, climatic, substrate, cover'),
        ('Iterative VIF cut', 'remove highest, recompute, stop below 10'),
        ('20 factors retained', 'maximum VIF 3.98')]),
    ('C', 'Sampling and training', '', 'orange', 'models', [
        ('5 nested samples', '50 000–500 000 per class, balanced, 20 % hold-out'),
        ('4 learners + ensemble', 'RF, XGBoost, CatBoost, DNN, unweighted mean'),
        ('Validation split', '10 % of training data: early stopping and every threshold')]),
    ('D', 'Validation and transferability', '', 'orange', 'split', [
        ('Random hold-out', 'AUC, CSI, F1, κ on the balanced test set'),
        ('Spatial block CV', '50 km blocks, national threshold; 5 km buffer'),
        ('Transferability', 'leave one regime out; area of applicability')]),
    ('E', 'Verification and independent evidence', '', 'green', 'target', [
        ('Delivered maps', 'cell by cell at the 5.6\u00a0% base rate; screening and prevalence thresholds'),
        ('Incident reports', '5 019 river-flood sites outside the inventory vs wind, snow and ice sites'),
        ('Operating curve', 'inventory CSI against reports flagged; Curve Number check')]),
    ('F', 'Interpretation and sensitivity', '', 'green', 'chart', [
        ('SHAP, 20 attributions', 'four learners × five samples; rank stability'),
        ('Frequency ratio', 'model-free, every bin of every factor'),
        ('Sensitivity tests', 'land-use ablation, encodings, class ratio, Q500 target')]),
    ('G', 'National products', '', 'purple', 'map', [
        ('Susceptibility maps', 'probability and binary, all models; two thresholds'),
        ('Confidence layers', 'four-learner agreement; area-of-applicability mask'),
        ('Warning-unit statistics', '26 ARSO units; hotspots')]),
]
REFS = {'A': '§2.1–2.4', 'B': '§2.3', 'C': '§3.1', 'D': '§3.2', 'E': '§3.3–3.4', 'F': '§3.5; Suppl.', 'G': '§3.5'}
OUT = os.path.join(os.path.dirname(HERE), 'Manuscript', 'JFRM_v2', 'figures', 'main')
f.fig03(refs=REFS, out_dir=OUT, stem='Fig03_methodological_framework')
print('->', OUT)
