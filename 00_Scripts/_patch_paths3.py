"""Third patching pass: os.path.join(BASE, '<legacy-folder>') constructions.

Passes 1 and 2 handled literal paths and f-strings. This handles the remaining
form, e.g. os.path.join(BASE, 'probability_maps', size), and sets up
06b_predict_probability_maps_all_sizes.py for the numbered tree.
"""
import re
from pathlib import Path

ROOT = Path("/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC")
DST = ROOT / "00_Scripts"
OLD = "/home/arora/Documents/FSM_Slovenia/For_Analysis"

JOIN_MAP = {
    "probability_maps": "06_Probability_Maps",
    "classified_maps":  "07_Classified_Maps",
    "validation":       "08_Validation",
    "spatial_cv":       "09_Analysis/spatial_cv",
    "analysis":         "09_Analysis",
    "samples":          "03_Samples",
    "shap":             "09_Analysis/shap",
}

targets = sorted(DST.glob("0[4-9]*.py")) + sorted(DST.glob("1*.py"))
changed = []
for p in targets:
    txt = orig = p.read_text()
    for old, new in JOIN_MAP.items():
        txt = re.sub(
            r"os\.path\.join\((BASE|ML_BASE|OUTPUT_DIR|BASE_DIR),\s*'" + old + r"'",
            r"os.path.join(\1, '" + new + "'", txt)
    if txt != orig:
        p.write_text(txt)
        changed.append(p.name)

print(f"pass 3 patched {len(changed)}: {', '.join(changed) or 'none'}")

# ── 06b: per-size probability maps, repointed and extended to 10x ────────────
six_b = DST / "06b_predict_probability_maps_all_sizes.py"
if six_b.exists():
    t = six_b.read_text()
    t = t.replace(f"{OLD}/CgFs_Aligned", f"{ROOT}/01_Input_Features")
    t = t.replace(f"{OLD}/Collinearity", f"{ROOT}/02_Collinearity")
    t = t.replace(f"{OLD}/ML_Results", str(ROOT))
    # models_{size} -> 04_Models/{size}
    t = t.replace("os.path.join(BASE_DIR, f'models_{size}')",
                  "os.path.join(BASE_DIR, '04_Models', size)")
    t = t.replace("os.path.join(BASE_DIR, 'probability_maps', size)",
                  "os.path.join(BASE_DIR, '06_Probability_Maps', size)")
    # the 10x models exist too; include them so every size gets maps
    t = t.replace("SIZES = ['1x', '2x', '3x', '4x']",
                  "SIZES = ['1x', '2x', '3x', '4x', '10x']")
    t = t.replace("Probability Maps — 1x through 4x Models",
                  "Probability Maps — all sample sizes (1x through 10x)")
    six_b.write_text(t)
    print("06b repointed and extended to SIZES = 1x,2x,3x,4x,10x")

# ── Verify ───────────────────────────────────────────────────────────────────
print("\nchecking for legacy folder names still used as paths ...")
LEGACY = re.compile(r"['\"](probability_maps|classified_maps|spatial_cv)['\"]")
issues = []
for p in targets + [six_b]:
    if not p.exists():
        continue
    for ln, line in enumerate(p.read_text().splitlines(), 1):
        if LEGACY.search(line) and ("join" in line or "/" in line) and not line.strip().startswith("#"):
            issues.append(f"  {p.name}:{ln}  {line.strip()[:92]}")
if issues:
    print(f"{len(issues)} line(s) to review:")
    print("\n".join(issues[:20]))
else:
    print("PASS - no legacy folder names remain in path constructions")

print("\nchecking no write path resolves into For_Analysis ...")
WRITE_HINT = re.compile(r"(to_csv|savefig|to_file|makedirs|save\(|dump\(|"
                        r"open\(.*['\"][wa]|rasterio\.open\(.*['\"]w)")
bad = [f"  {p.name}:{ln}  {l.strip()[:92]}"
       for p in targets + [six_b] if p.exists()
       for ln, l in enumerate(p.read_text().splitlines(), 1)
       if OLD in l and WRITE_HINT.search(l)]
# also catch OUT_PATH-style: variable assigned an OLD path anywhere
for p in targets + [six_b]:
    if not p.exists():
        continue
    for ln, l in enumerate(p.read_text().splitlines(), 1):
        if re.search(r"^\s*(OUT|OUTPUT|DST|SAVE)\w*\s*=.*" + re.escape(OLD), l):
            bad.append(f"  {p.name}:{ln}  {l.strip()[:92]}")
print("PASS - no writes into For_Analysis" if not bad
      else f"FAIL:\n" + "\n".join(bad))
