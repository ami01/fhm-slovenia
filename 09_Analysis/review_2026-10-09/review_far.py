import numpy as np, rasterio
from scipy.ndimage import distance_transform_edt
R="/home/arora/manuscripts/FSM_Pipeline_LULC/"; F=R+"01_Input_Features/"
with rasterio.open(F+"Flooded_NoFlood.tif") as d: inv=d.read(1); nd=d.nodata
with rasterio.open(R+"07_Classified_Maps/10x/FSM_Ensemble_mean_binary.tif") as d: b=d.read(1); bnd=d.nodata
with rasterio.open(F+"hand.tif") as d: hand=d.read(1)
with rasterio.open(F+"Distance_to_River.tif") as d: d2r=d.read(1)
valid=(inv!=nd)&(b!=bnd); fl=(inv==1)&valid; flag=(b==1)&valid
px=12.5*12.5/1e6
fp=flag&~fl
print(f"inventory {fl.sum()*px:.0f} km2; flagged {flag.sum()*px:.0f} km2; flagged outside inventory (FP) {fp.sum()*px:.0f} km2; FBI {flag.sum()/fl.sum():.2f}")
dist=distance_transform_edt(~fl)*12.5
for a,bb in [(0,100),(100,500),(500,2000),(2000,1e9)]:
    m=fp&(dist>=a)&(dist<bb); print(f"  FP {a:>5}-{bb:<6} m from inventory: {m.sum()*px:7.0f} km2 ({m.sum()/fp.sum():.0%}); median HAND {np.median(hand[m]) if m.any() else np.nan:.1f} m; median dist-to-river {np.median(d2r[m]):.0f} m")
hv=np.where(hand==-9999,np.nan,hand)
print(f"FP with HAND < 2 m: {(fp&(hv<2)).sum()*px:.0f} km2 ({(fp&(hv<2)).sum()/fp.sum():.0%}); inventory cells with HAND < 2 m: {(fl&(hv<2)).sum()/fl.sum():.0%}")
print(f"FP more than 2 km from any inventory cell AND HAND < 3 m: {(fp&(dist>2000)&(hv<3)).sum()*px:.0f} km2 -> candidate unstudied floodplain")
