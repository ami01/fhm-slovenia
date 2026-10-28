"""
09o — Sensitivity of the results to output resolution (co-author review 2026-10-22, recommendation 2).
=======================================================================================================
The maps are computed on a 12.5 m grid, but several predictors have much coarser source scales (1:250 000 geology and
soils, 0.1° E-OBS rainfall). This stage aggregates the delivered v2 probability maps to 25, 50 and 100 m without
retraining and asks whether the principal results change.

Aggregation of probability (per coarse cell, over valid 12.5 m cells; a coarse cell is valid if >= 50 % of its cells are):
    mean  — primary;   max — conservative screening alternative.
Reference inventory on the coarse grid:
    fractional — every coarse cell carries its flooded and non-flooded 12.5 m area (area-weighted scores; primary);
    any        — flooded if any 12.5 m cell is flooded;
    half       — flooded if >= 50 % of its valid area is flooded.
Scores (each model, screening threshold of 04_Models_v2/v2/thresholds.json; ensemble also at the prevalence threshold):
    AUC and average precision (fractional, any, half); POD, FAR, CSI, FBI (fractional = area-based, any, half);
    flagged area and change from 12.5 m; Jaccard index and share of cells changing class against the 12.5 m binary map
    (on the 12.5 m support); where the changes lie (inventory margin, narrow corridors, Dinaric karst units);
    detection of narrow (<= 50 m wide) and wide inventory corridors; river-flood reports outside the inventory flagged
    and their AUC against wind/snow/ice sites (sites as in 09i/09m); four-learner disagreement.
    PYTHONPATH=<dir with openpyxl> python 00_Scripts/09o_resolution_sensitivity.py      (FIG_ONLY=1: figure only)
Outputs → 09_Analysis/resolution_v2/{metrics.csv, change_location.csv, corridors.csv, disagreement.csv, fig_resolution.{png,pdf}}
"""
import json, os, time
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.features import rasterize
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt, maximum_filter
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '09_Analysis', 'resolution_v2'); os.makedirs(OUT, exist_ok=True)
MODELS = ['RF', 'XGBoost', 'CatBoost', 'DNN', 'Ensemble_mean']
THR = json.load(open(os.path.join(ROOT, '04_Models_v2', 'v2', 'thresholds.json')))['thresholds']
T_PREV = 0.885                                      # ensemble prevalence threshold (07w)
FACTORS = (1, 2, 4, 8)                              # 12.5, 25, 50, 100 m
KARST = ['Kras', 'Reka', 'Kraška Ljubljanica', 'Krka', 'Kolpa']
t_start = time.time()

FIG_ONLY = bool(os.environ.get('FIG_ONLY'))           # redraw the figure from metrics.csv
if FIG_ONLY:
    R = pd.read_csv(os.path.join(OUT, 'metrics.csv'))
else:
    with rasterio.open(os.path.join(ROOT, '01_Input_Features', 'Flooded_NoFlood.tif')) as d:
        inv = d.read(1); T = d.transform; inv_ok = inv != d.nodata; px = abs(T.a) ** 2 / 1e6
    P = {}
    ok = inv_ok.copy()
    for m in MODELS:
        with rasterio.open(os.path.join(ROOT, '06_Probability_Maps_v2', 'v2', f'FSM_{m}_probability.tif')) as d:
            P[m] = d.read(1); ok &= P[m] != d.nodata
    flood = (inv == 1) & ok
    print(f'valid cells {ok.sum():,}, base rate {flood.sum() / ok.sum():.4f}', flush=True)

    # --- auxiliary layers on the 12.5 m grid: inventory margin, corridor width, karst units -------------------------
    e_in = distance_transform_edt(flood)                          # flood cell -> nearest non-flood (cells)
    e_out = distance_transform_edt(~flood)                        # non-flood cell -> nearest flood
    margin = ok & (((flood) & (e_in <= 4)) | ((~flood) & (e_out <= 4)))       # within 50 m of the inventory edge
    narrow = flood & (maximum_filter(e_in, size=9) <= 2)         # corridor no wider than ~50 m
    del e_out
    u = gpd.read_file(os.path.join(ROOT, 'geo_static.json')).to_crs(3794)
    karst = rasterize(((g, 1) for g in u[u.ime.isin(KARST)].geometry), out_shape=inv.shape, transform=T, dtype='uint8').astype(bool) & ok

    # --- incident sites (same selection as 09m) ----------------------------------------------------------------------
    ev = pd.read_excel(os.path.join(ROOT, 'Events_natural-disasters_2005-2026_all-atts.xlsx'),
                       usecols=['WGSLat', 'WGSLon', 'SkupinaDogodka', 'Dogodek']).dropna(subset=['WGSLat', 'WGSLon'])
    ev['key'] = ev.WGSLat.round(5).astype(str) + ',' + ev.WGSLon.round(5).astype(str)
    fl = ev[ev.Dogodek == 'poplave ob vodotokih']; fl = fl[fl.key.map(ev[ev.Dogodek.isin(
        ['poplave ob vodotokih', 'poplave meteorne vode', 'poplave zaradi dviga podtalnice, kanalizacijskih vod'])].key.value_counts()) < 20]
    s = fl.drop_duplicates('key'); ct = ev[ev.SkupinaDogodka.isin(['Močan veter', 'Visok sneg', 'Žled'])]
    ct = ct[ct.key.map(ct.key.value_counts()) < 20].drop_duplicates('key'); ct = ct[~ct.key.isin(set(ev.loc[ev.SkupinaDogodka == 'Poplave', 'key']))]
    tr = Transformer.from_crs(4326, 3794, always_xy=True)


    def rc(lon, lat):
        x, y = tr.transform(lon, lat); r, c = rasterio.transform.rowcol(T, x, y); r, c = np.asarray(r), np.asarray(c)
        k = (r >= 0) & (r < inv.shape[0]) & (c >= 0) & (c < inv.shape[1]); r, c = r[k], c[k]
        k = ok[r, c]; return r[k], c[k]


    fr, fc = rc(s.WGSLon.values, s.WGSLat.values); o = inv[fr, fc] == 0; fr, fc = fr[o], fc[o]
    cr, cc = rc(ct.WGSLon.values, ct.WGSLat.values)
    print(f'sites: {len(fr):,} river-flood outside inventory, {len(cr):,} comparison', flush=True)


    def hist_scores(s, wp, wn, nb=20000):
        """AUC and average precision from weighted positives/negatives via a fine histogram of scores."""
        b = np.minimum((s * nb).astype(np.int64), nb)
        hp = np.bincount(b, wp, nb + 1); hn = np.bincount(b, wn, nb + 1)
        cn = np.cumsum(hn) - hn                                   # negatives strictly below each bin
        auc = (hp * (cn + 0.5 * hn)).sum() / (hp.sum() * hn.sum())
        tp, fp = np.cumsum(hp[::-1]), np.cumsum(hn[::-1])         # descending thresholds
        prec = tp / np.maximum(tp + fp, 1e-12); rec = tp / hp.sum()
        ap = (np.diff(np.r_[0, rec]) * prec).sum()
        return auc, ap


    def contingency(f, wp, wn):
        tp = (f * wp).sum(); fp = (f * wn).sum(); fn = ((~f) * wp).sum()
        return dict(POD=tp / (tp + fn), FAR=fp / max(tp + fp, 1e-12), CSI=tp / (tp + fp + fn), FBI=(tp + fp) / (tp + fn))


    def block(a, k, H, W):
        return a[:H, :W].reshape(H // k, k, W // k, k)


    rows, chg_rows, cor_rows, dis_rows = [], [], [], []
    fine_bin = {}
    for k in FACTORS:
        H, W = inv.shape[0] // k * k, inv.shape[1] // k * k
        okb = block(ok, k, H, W); n_ok = okb.sum((1, 3)).astype(np.float32)
        n_fl = block(flood, k, H, W).sum((1, 3)).astype(np.float32)
        cv = n_ok >= 0.5 * k * k                                   # valid coarse cells
        wp, wn = n_fl[cv], (n_ok - n_fl)[cv]                       # flooded / non-flooded 12.5 m area (cells)
        any_ = (n_fl > 0)[cv]; half = (n_fl >= 0.5 * n_ok)[cv]
        sr, sc_ = fr // k, fc // k; kr, kc = cr // k, cc // k
        vf = (fr < H) & (fc < W); vf[vf] &= cv[sr[vf], sc_[vf]]
        vc = (cr < H) & (cc < W); vc[vc] &= cv[kr[vc], kc[vc]]
        flags = {}
        for agg in (('mean', 'max') if k > 1 else ('mean',)):
            for m in MODELS:
                pb = block(np.where(ok, P[m], 0 if agg == 'mean' else -1).astype(np.float32), k, H, W)
                pc = (pb.sum((1, 3)) / np.maximum(n_ok, 1)) if agg == 'mean' else pb.max((1, 3))
                s_ = pc[cv]
                auc_f, ap_f = hist_scores(s_, wp, wn)
                auc_a, ap_a = hist_scores(s_, any_.astype(np.float64), (~any_).astype(np.float64))
                auc_h, ap_h = hist_scores(s_, half.astype(np.float64), (~half).astype(np.float64))
                pf, pcmp = pc[sr[vf], sc_[vf]], pc[kr[vc], kc[vc]]
                inc_auc = roc_auc_score(np.r_[np.ones(len(pf)), np.zeros(len(pcmp))], np.r_[pf, pcmp])
                for rule, t in ((('screening', THR[m]), ('prevalence', T_PREV)) if m == 'Ensemble_mean' else (('screening', THR[m]),)):
                    f = s_ >= t
                    r = dict(res_m=12.5 * k, agg=agg, model=m, threshold_rule=rule, threshold=t,
                             AUC_frac=auc_f, AP_frac=ap_f, AUC_any=auc_a, AUC_half=auc_h, AP_any=ap_a, AP_half=ap_h,
                             prevalence_any=(any_ * (wp + wn)).sum() / (wp + wn).sum(), prevalence_half=(half * (wp + wn)).sum() / (wp + wn).sum(),
                             flagged_km2=(f * (wp + wn)).sum() * px,
                             reports_flagged=(pf >= t).mean(), comparison_flagged=(pcmp >= t).mean(), report_AUC=inc_auc,
                             n_reports=len(pf), n_comparison=len(pcmp))
                    r.update({f'{key}_frac': v for key, v in contingency(f, wp, wn).items()})
                    r.update({f'{key}_any': v for key, v in contingency(f, any_.astype(float), (~any_).astype(float)).items()})
                    r.update({f'{key}_half': v for key, v in contingency(f, half.astype(float), (~half).astype(float)).items()})
                    # agreement with the 12.5 m binary map, on the 12.5 m support
                    full = np.zeros(cv.shape, bool); full[cv] = f
                    fine = np.repeat(np.repeat(full, k, 0), k, 1)
                    sup = ok[:H, :W] & np.repeat(np.repeat(cv, k, 0), k, 1)
                    if k == 1:
                        fine_bin[(m, rule)] = fine
                    ref = fine_bin[(m, rule)][:H, :W]
                    a, b = ref[sup], fine[sup]
                    inter, uni = (a & b).sum(), (a | b).sum(); ch = a != b
                    r.update(Jaccard=inter / uni, changed_pct=100 * ch.mean(), gained_pct=100 * (b & ~a).mean(),
                             lost_pct=100 * (a & ~b).mean())
                    rows.append(r)
                    if k > 1 and agg == 'mean':
                        chm = np.zeros((H, W), bool); chm[sup] = ch
                        n = chm.sum()
                        chg_rows.append(dict(res_m=12.5 * k, model=m, threshold_rule=rule, changed_cells=int(n),
                                             in_margin_pct=100 * (chm & margin[:H, :W]).sum() / n,
                                             in_narrow_corridor_pct=100 * (chm & narrow[:H, :W]).sum() / n,
                                             in_karst_units_pct=100 * (chm & karst[:H, :W]).sum() / n,
                                             margin_share_of_area_pct=100 * margin[:H, :W][sup].mean(),
                                             karst_share_of_area_pct=100 * karst[:H, :W][sup].mean()))
                    # detection of narrow vs wide corridors and of karst-unit floods (area-based POD on 12.5 m support)
                    if agg == 'mean':
                        fl_s = flood[:H, :W] & sup
                        for name, msk in (('narrow corridors', narrow[:H, :W]), ('wide floodplains', flood[:H, :W] & ~narrow[:H, :W]),
                                          ('Dinaric karst units', karst[:H, :W])):
                            z = fl_s & msk
                            cor_rows.append(dict(res_m=12.5 * k, model=m, threshold_rule=rule, zone=name,
                                                 flood_cells=int(z.sum()), POD=fine[z].mean()))
                    if rule == 'screening':
                        flags[(agg, m)] = full
                print(f'{12.5 * k:>5} m {agg:4s} {m:13s} AUC_frac {auc_f:.4f} CSI_frac {r["CSI_frac"] if rule == "screening" else float("nan"):.3f} '
                      f'reports AUC {inc_auc:.3f}  [{time.time() - t_start:.0f} s]', flush=True)
            L = np.stack([flags[(agg, m)] for m in ('RF', 'XGBoost', 'CatBoost', 'DNN')]).sum(0)
            area = (n_ok * cv); tot = area.sum()
            dis_rows.append(dict(res_m=12.5 * k, agg=agg, none_pct=100 * (area * (L == 0)).sum() / tot,
                                 all4_pct=100 * (area * (L == 4)).sum() / tot,
                                 disputed_pct=100 * (area * ((L > 0) & (L < 4))).sum() / tot))

    R = pd.DataFrame(rows)
    base = R[R.res_m == 12.5][['model', 'threshold_rule', 'flagged_km2']].rename(columns={'flagged_km2': 'km2_125'})
    R = R.merge(base, on=['model', 'threshold_rule']); R['area_change_pct'] = 100 * (R.flagged_km2 / R.km2_125 - 1)
    R.to_csv(os.path.join(OUT, 'metrics.csv'), index=False)
    pd.DataFrame(chg_rows).to_csv(os.path.join(OUT, 'change_location.csv'), index=False)
    pd.DataFrame(cor_rows).to_csv(os.path.join(OUT, 'corridors.csv'), index=False)
    pd.DataFrame(dis_rows).to_csv(os.path.join(OUT, 'disagreement.csv'), index=False)
    show = ['res_m', 'agg', 'model', 'threshold_rule', 'AUC_frac', 'AP_frac', 'CSI_frac', 'POD_frac', 'FAR_frac', 'FBI_frac',
            'CSI_any', 'CSI_half', 'flagged_km2', 'area_change_pct', 'Jaccard', 'changed_pct', 'reports_flagged', 'report_AUC']
    print(R[show].round(4).to_string(index=False))
    print(pd.DataFrame(chg_rows).round(2).to_string(index=False)); print(pd.DataFrame(cor_rows).round(3).to_string(index=False))
    print(pd.DataFrame(dis_rows).round(2).to_string(index=False))

# --- figure -------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
COL = {'RF': '#2a78d6', 'XGBoost': '#eb6834', 'CatBoost': '#1f9e74', 'DNN': '#8e5cc4', 'Ensemble_mean': '#0b0b0b'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
panels = [('AUC_frac', '(a) AUC, fractional reference'), ('CSI_frac', '(b) CSI, fractional reference'),
          ('flagged_km2', '(c) Flagged area (km$^2$)'), ('Jaccard', '(d) Jaccard index vs 12.5 m map'),
          ('reports_flagged', '(e) Reports outside inventory flagged'), ('report_AUC', '(f) AUC, reports vs comparison sites')]
S = R[R.threshold_rule == 'screening']
for a, (col, title) in zip(axs.ravel(), panels):
    for m in MODELS:
        g = S[(S.model == m) & (S['agg'] == 'mean')].sort_values('res_m')
        a.plot(g.res_m, g[col], '-o', ms=3, lw=1.3 if m != 'Ensemble_mean' else 1.8, color=COL[m],
               label=m.replace('_mean', ' mean'))
        if m == 'Ensemble_mean':
            g2 = pd.concat([g[g.res_m == 12.5], S[(S.model == m) & (S['agg'] == 'max')]]).sort_values('res_m')
            a.plot(g2.res_m, g2[col], ':s', ms=3, lw=1.2, color=COL[m], label='Ensemble, max aggregation')
    a.set_xscale('log', base=2); a.set_xticks([12.5, 25, 50, 100]); a.set_xticklabels(['12.5', '25', '50', '100'])
    a.set_title(title, loc='left', fontsize=7); a.grid(color='#e6e5e0', lw=0.5)
    [a.spines[q].set_visible(False) for q in ('top', 'right')]
for a in axs[1]:
    a.set_xlabel('output cell size (m)')
axs[0, 0].legend(fontsize=5.6, frameon=False)
fig.tight_layout()
for e in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'fig_resolution.{e}'), dpi=300, bbox_inches='tight')
print(f'done in {(time.time() - t_start) / 60:.1f} min -> {OUT}')
