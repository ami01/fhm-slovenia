"""Second patching pass: redirect f-string-derived paths that escaped _patch_paths.py.

The first pass replaced literal absolute paths. Several scripts instead do

    BASE     = '/home/.../For_Analysis'
    PROB_DIR = f'{BASE}/ML_Results/probability_maps'
    OUT_DIR  = f'{BASE}/Manuscript/figures'

so only the BASE line was a literal and the derived paths kept pointing at the
original tree. Left unfixed, stages 09c/10a/10b/10c would OVERWRITE the existing
manuscript figures in For_Analysis.

Read-only inputs (Vectors/) stay in the original tree and are exposed inside the
pipeline root as a symlink, so a single BASE swap is correct for both.
"""
import re
from pathlib import Path

ROOT = Path("/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC")
DST = ROOT / "00_Scripts"
OLD = "/home/arora/Documents/FSM_Slovenia/For_Analysis"

# Derived-subpath rewrites, longest first. Applied to the f-string bodies after the
# BASE literal itself has been repointed at ROOT.
SUBPATHS = [
    ("/ML_Results/probability_maps", "/06_Probability_Maps"),
    ("/ML_Results/classified_maps",  "/07_Classified_Maps"),
    ("/ML_Results/validation",       "/08_Validation"),
    ("/ML_Results/spatial_cv",       "/09_Analysis/spatial_cv"),
    ("/ML_Results/analysis",         "/09_Analysis"),
    ("/ML_Results/samples",          "/03_Samples"),
    ("/ML_Results/shap",             "/09_Analysis/shap"),
    ("/ML_Results/models",           "/04_Models"),
    ("/ML_Results/results",          "/05_Results"),
    ("/ML_Results",                  ""),
    ("/Collinearity",                "/02_Collinearity"),
]

# {VAR}/sub  -> {VAR}/numbered   (ML_BASE already means the results root)
FSTRING_FIXES = [
    (r"\{ML_BASE\}/samples",    "{BASE}/03_Samples"),
    (r"\{ML_BASE\}/shap",       "{BASE}/09_Analysis/shap"),
    (r"\{ML_BASE\}/spatial_cv", "{BASE}/09_Analysis/spatial_cv"),
    (r"\{ML_BASE\}/analysis",   "{BASE}/09_Analysis"),
    (r"\{ML_BASE\}/validation", "{BASE}/08_Validation"),
    (r"\{ML_BASE\}/models",     "{BASE}/04_Models"),
    (r"\{ML_BASE\}/results",    "{BASE}/05_Results"),
    (r"\{ML_BASE\}/probability_maps", "{BASE}/06_Probability_Maps"),
    (r"\{ML_BASE\}/classified_maps",  "{BASE}/07_Classified_Maps"),
    (r"\{BASE\}/samples",       "{BASE}/03_Samples"),
    (r"\{OUTPUT_DIR\}/samples", "{BASE}/03_Samples"),
]

TARGETS = [
    "08c_validate_classify_spatial_cv_multiscale.py",
    "08d_shap_analysis.py",
    "09b_analysis_fr_area.py",
    "09c_subbasin_analysis.py",
    "10a_figures_maps.py",
    "10b_generate_maps_all_sizes.py",
    "10c_fix_comparison_maps.py",
    "10d_setup_manuscript_folder.py",
    "11a_export_codebook.py",
    "11b_export_raba_lulc_conversion.py",
    "11c_export_scripts_summary.py",
    "11d_generate_summary_report.py",
]

changed = []
for name in TARGETS:
    p = DST / name
    if not p.exists():
        print(f"  MISSING {name}")
        continue
    txt = orig = p.read_text()

    # 1. repoint the BASE literal at the pipeline root
    txt = txt.replace(f"'{OLD}'", f"'{ROOT}'").replace(f'"{OLD}"', f'"{ROOT}"')

    # 2. rewrite derived subpaths in f-strings
    for a, b in FSTRING_FIXES:
        txt = re.sub(a, b, txt)

    # 3. rewrite any remaining literal subpaths now rooted at ROOT
    for a, b in SUBPATHS:
        txt = txt.replace(f"{ROOT}{a}", f"{ROOT}{b}")
        txt = re.sub(r"(\{(?:BASE|ML_BASE|OUTPUT_DIR)\})" + re.escape(a),
                     r"\1" + b, txt)

    # 4. ML_BASE is now the pipeline root itself
    txt = re.sub(r"^ML_BASE\s*=\s*f'\{BASE\}'\s*$", "ML_BASE     = BASE", txt, flags=re.M)
    txt = re.sub(r"^(ML_BASE\s*=\s*)f?['\"]\{BASE\}['\"]", r"\1BASE", txt, flags=re.M)

    # 5. documentation outputs belong in 10_Documentation
    txt = txt.replace(f"{ROOT}/Raster_Code_Descriptions.docx",
                      f"{ROOT}/10_Documentation/Raster_Code_Descriptions.docx")
    txt = txt.replace(f"{ROOT}/RABA_LULC_Conversion.docx",
                      f"{ROOT}/10_Documentation/RABA_LULC_Conversion.docx")
    txt = txt.replace(f"{ROOT}/Scripts_Summary.docx",
                      f"{ROOT}/10_Documentation/Scripts_Summary.docx")

    if txt != orig:
        p.write_text(txt)
        changed.append(name)

print(f"patched {len(changed)} scripts: {', '.join(changed)}\n")

# ── Expose read-only inputs inside the root so {BASE}/Vectors resolves ────────
for link_name, target in [("Vectors", Path(OLD) / "Vectors"),
                          ("Rasters", Path(OLD) / "Rasters")]:
    link = ROOT / link_name
    if target.exists() and not link.exists():
        link.symlink_to(target)
        print(f"symlinked {link_name} -> {target}")

# ── Verification ─────────────────────────────────────────────────────────────
print("\nverifying no script writes into the original For_Analysis tree ...")
WRITE_HINT = re.compile(r"(to_csv|savefig|to_file|makedirs|open\(.*['\"][wa]|"
                        r"rasterio\.open\(.*['\"]w|save\(|dump\()")
bad = []
for p in sorted(DST.glob("0[4-9]*.py")) + sorted(DST.glob("1*.py")):
    for ln, line in enumerate(p.read_text().splitlines(), 1):
        if OLD in line and WRITE_HINT.search(line):
            bad.append(f"  {p.name}:{ln}  {line.strip()[:95]}")

remaining = []
for p in sorted(DST.glob("0[4-9]*.py")) + sorted(DST.glob("1*.py")):
    hits = [f"{p.name}:{ln}" for ln, line in enumerate(p.read_text().splitlines(), 1)
            if OLD in line]
    if hits:
        remaining.append((p.name, len(hits)))

if bad:
    print(f"FAIL - {len(bad)} write(s) into the original tree:")
    print("\n".join(bad))
else:
    print("PASS - no write path resolves into For_Analysis")

if remaining:
    print("\nremaining read-only references to For_Analysis (expected: input data):")
    for n, c in remaining:
        print(f"  {n}: {c}")
