import sys; sys.path.insert(0, "/tmp/claude-424415485/-home-arora-manuscripts/94762bea-e25a-41c4-a1ed-29893d8e277b/scratchpad/pylib")
import numpy as np, pandas as pd, rasterio
from pyproj import Transformer
from sklearn.metrics import roc_auc_score
R="/home/arora/manuscripts/FSM_Pipeline_LULC/"
ev=pd.read_excel(R+"Events_natural-disasters_2005-2026_all-atts.xlsx",usecols=['NastanekDatumCas','WGSLat','WGSLon','SkupinaDogodka','Dogodek']).dropna(subset=['WGSLat','WGSLon'])
ev['key']=ev.WGSLat.round(5).astype(str)+','+ev.WGSLon.round(5).astype(str)
RIV='poplave ob vodotokih'
fl=ev[ev.Dogodek.isin([RIV,'poplave meteorne vode','poplave zaradi dviga podtalnice, kanalizacijskih vod'])].copy()
fl=fl[fl.key.map(fl.key.value_counts())<20]; fl['aug23']=fl.NastanekDatumCas.between('2023-08-03','2023-08-20',inclusive='left')
s=fl[fl.Dogodek==RIV].groupby('key').agg(lat=('WGSLat','first'),lon=('WGSLon','first'),aug23=('aug23','any')).reset_index()
ct=ev[ev.SkupinaDogodka.isin(['Močan veter','Visok sneg','Žled'])]; ct=ct[ct.key.map(ct.key.value_counts())<20].drop_duplicates('key')
ct=ct[~ct.key.isin(set(ev.loc[ev.SkupinaDogodka=='Poplave','key']))]
ct=ct.assign(group=ct.SkupinaDogodka)
tr=Transformer.from_crs(4326,3794,always_xy=True)
s['x'],s['y']=tr.transform(s.lon.values,s.lat.values); cx,cy=tr.transform(ct.WGSLon.values,ct.WGSLat.values)
def samp(path,xs,ys):
    with rasterio.open(path) as d:
        a=d.read(1); nd=d.nodata; r,c=rasterio.transform.rowcol(d.transform,xs,ys); r,c=np.asarray(r),np.asarray(c)
        ok=(r>=0)&(r<d.height)&(c>=0)&(c<d.width); v=np.full(len(r),np.nan); v[ok]=a[r[ok],c[ok]]
        if nd is not None: v[v==nd]=np.nan
        return v
F=R+"01_Input_Features/"
lay={"inv":F+"Flooded_NoFlood.tif","p":R+"06_Probability_Maps/10x/FSM_Ensemble_mean_probability.tif","prf":R+"06_Probability_Maps/10x/FSM_RF_probability.tif",
     "elev":F+"elevation.tif","hand":F+"hand.tif","d2r":F+"Distance_to_River.tif","lulc":F+"LULC.tif"}
S=pd.DataFrame({k:samp(v,s.x.values,s.y.values) for k,v in lay.items()}); S['aug23']=s.aug23.values; S['y']=1
C=pd.DataFrame({k:samp(v,np.asarray(cx),np.asarray(cy)) for k,v in lay.items()}); C['aug23']=False; C['y']=0; C['grp']=ct.group.values
S=S[(S.inv==0)&S.p.notna()]; C=C[C.p.notna()]
print(f"flood sites outside inventory {len(S)} (paper 5019); control {len(C)} (paper 28924)")
D=pd.concat([S,C],ignore_index=True)
auc=lambda d,col,sign=1: roc_auc_score(d.y,sign*d[col])
print("AUC ensemble prob %.3f | RF prob %.3f" % (auc(D,'p'),auc(D,'prf')))
print("single-variable AUCs on the same test: -HAND %.3f, -elevation %.3f, -dist-to-river %.3f" % (auc(D.dropna(subset=['hand']),'hand',-1),auc(D,'elev',-1),auc(D.dropna(subset=['d2r']),'d2r',-1)))
print("\nelevation (m) median: flood %.0f, control %.0f | by control group:" % (S.elev.median(),C.elev.median()), C.groupby('grp').elev.median().round(0).to_dict(), C.grp.value_counts().to_dict())
print("built-up LULC share at site: flood %.2f, control %.2f" % ((S.lulc==5).mean(),(C.lulc==5).mean()))
D['eb']=pd.qcut(D.elev,5,labels=False)
print("\nAUC of ensemble prob within elevation quintiles (pooled):")
for b,g in D.groupby('eb'):
    print(f"  band {b} ({g.elev.min():.0f}-{g.elev.max():.0f} m): n flood {int(g.y.sum())}, control {int((g.y==0).sum())}, AUC {roc_auc_score(g.y,g.p):.3f}, AUC(-HAND) {roc_auc_score(g.dropna(subset=['hand']).y,-g.dropna(subset=['hand']).hand):.3f}")
# reweight controls to the flood elevation distribution (20 bins)
bins=np.quantile(D.elev,np.linspace(0,1,21)); D['b20']=np.clip(np.digitize(D.elev,bins[1:-1]),0,19)
pf=D[D.y==1].b20.value_counts(normalize=True); pc=D[D.y==0].b20.value_counts(normalize=True)
w=np.where(D.y==1,1.0,D.b20.map(pf/pc).fillna(0).values)
print("\nelevation-matched AUC (controls reweighted): ensemble %.3f, RF %.3f, -HAND %.3f" % (roc_auc_score(D.y,D.p,sample_weight=w),roc_auc_score(D.y,D.prf,sample_weight=w),roc_auc_score(D.dropna(subset=['hand']).y,-D.dropna(subset=['hand']).hand,sample_weight=w[D.hand.notna().values])))
for g in ['Močan veter','Visok sneg','Žled']:
    d=D[(D.y==1)|(D.grp==g)]; print(f"AUC vs {g} only: {roc_auc_score(d.y,d.p):.3f} (n={int((d.y==0).sum())})")
# does the map add over HAND? logistic on rank(HAND) + rank(p)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
d=D.dropna(subset=['hand','d2r']).copy()
for cols in (['hand'],['hand','d2r','elev'],['p'],['p','hand','d2r','elev']):
    X=d[cols].rank(pct=True).values
    pr=cross_val_predict(LogisticRegression(max_iter=1000),X,d.y,cv=5,method='predict_proba')[:,1]
    print(f"5-fold logistic AUC with {cols}: {roc_auc_score(d.y,pr):.3f}")
