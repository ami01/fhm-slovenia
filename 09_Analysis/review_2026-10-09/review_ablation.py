import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
R="/home/arora/manuscripts/FSM_Pipeline_LULC/"
F=[l.strip() for l in open(R+"02_Collinearity/selected_features.txt") if l.strip()]
df=pd.read_csv(R+"09_Analysis/spatial_cv/fold_assignments.csv")
SUB=['Pedoloska_grouped','lito_poly_250_d96_grouped','Hgk_grouped']
sets={"all 20":F,"without substrate (soil, lithology, hydrogeology)":[f for f in F if f not in SUB],
      "HAND + distance to river + elevation":['hand','Distance_to_River','elevation']}
for name,cols in sets.items():
    a=[]
    for k in sorted(df.fold.unique()):
        tr,te=df[df.fold!=k],df[df.fold==k]
        rf=RandomForestClassifier(n_estimators=200,max_depth=20,min_samples_leaf=5,max_features="sqrt",class_weight="balanced",n_jobs=32,random_state=42).fit(tr[cols],tr.label)
        a.append(roc_auc_score(te.label,rf.predict_proba(te[cols])[:,1]))
    print(f"{name:52s} spatial AUC {np.mean(a):.4f} ± {np.std(a,ddof=1):.4f}  per fold {np.round(a,3)}")
