import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
R="/home/arora/manuscripts/FSM_Pipeline_LULC/"
F=[l.strip() for l in open(R+"02_Collinearity/selected_features.txt") if l.strip()]
df=pd.read_csv(R+"09_Analysis/spatial_cv/fold_assignments.csv")
def csi(y,p,t):
    q=p>=t; tp=(q&(y==1)).sum(); fp=(q&(y==0)).sum(); fn=(~q&(y==1)).sum(); return tp/(tp+fp+fn)
def best_t(y,p):
    ts=np.linspace(0.05,0.95,100); return ts[np.argmax([csi(y,p,t) for t in ts])]
rows=[]
for k in sorted(df.fold.unique()):
    tr,te=df[df.fold!=k],df[df.fold==k]
    rf=RandomForestClassifier(n_estimators=200,max_depth=20,min_samples_leaf=5,max_features="sqrt",class_weight="balanced",n_jobs=32,random_state=42,oob_score=True)
    rf.fit(tr[F],tr.label); pte=rf.predict_proba(te[F])[:,1]; poob=rf.oob_decision_function_[:,1]
    t_test=best_t(te.label.values,pte); t_train=best_t(tr.label.values,poob)
    rows.append(dict(fold=k,AUC=roc_auc_score(te.label,pte),t_testopt=t_test,CSI_testopt=csi(te.label.values,pte,t_test),
                     t_trainopt=t_train,CSI_trainopt=csi(te.label.values,pte,t_train),CSI_t05=csi(te.label.values,pte,0.5)))
D=pd.DataFrame(rows); print(D.round(4).to_string(index=False))
print("mean:", D[['AUC','CSI_testopt','CSI_trainopt','CSI_t05']].mean().round(4).to_dict())
