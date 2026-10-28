"""Preflight check for a pipeline stage — run automatically before each stage.

Catches the failure classes that actually bit this run:

  1. syntax errors                          -> compile()
  2. names used but never defined           -> AST scan (caught 4 in 08a)
  3. missing imports / modules              -> import each top-level module
  4. input files that do not exist          -> resolve read paths and stat them
  5. reads from a directory that another    -> per-size subfolder mismatch
     stage writes per-size subfolders into     (caught in 07, 08a, 09a, 09c, 10a)

Exit 0 = safe to launch, exit 1 = do not launch.

LIMITS - this is a static check and cannot catch:
  - semantic errors (e.g. appending a duplicate row to a table)
  - paths built at runtime from data rather than literals
  - a stage that runs fine but produces wrong numbers
It reduces the chance of a late crash; it does not certify correctness.

    python3 _preflight.py <script.py> [<script.py> ...]
"""
import ast
import builtins
import importlib.util
import os
import re
import sys
from pathlib import Path

DST = Path(__file__).resolve().parent
ROOT = DST.parent

STAGE_DIRS = ["01_Input_Features", "02_Collinearity", "03_Samples", "04_Models",
              "05_Results", "06_Probability_Maps", "07_Classified_Maps",
              "08_Validation", "09_Analysis", "10_Documentation"]

READ_CALLS = r"(read_csv|read_file|np\.load|joblib\.load|load_model|torch\.load|"
READ_CALLS += r"pickle\.load|read_pickle|imread)"
WRITE_CALLS = (r"(to_csv|savefig|joblib\.dump|save_model|torch\.save|to_file|"
               r"pickle\.dump|makedirs|imwrite|to_pickle|savez|np\.save|json\.dump)")
OPEN_WRITE = re.compile(r"open\(.*,\s*['\"][wax]")


def undefined_names(tree):
    assigned, imported, used = set(), set(), {}

    class V(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Store):
                assigned.add(n.id)
            elif isinstance(n.ctx, ast.Load) and n.id not in used:
                used[n.id] = n.lineno

        def visit_Import(self, n):
            for a in n.names:
                imported.add((a.asname or a.name).split('.')[0])

        def visit_ImportFrom(self, n):
            for a in n.names:
                imported.add(a.asname or a.name)

        def visit_FunctionDef(self, n):
            assigned.add(n.name); self.generic_visit(n)

        def visit_ClassDef(self, n):
            assigned.add(n.name); self.generic_visit(n)

        def visit_comprehension(self, n):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
            self.generic_visit(n)

        def visit_ExceptHandler(self, n):
            # `except Exception as e` binds e via .name (a str), not a Name node
            if n.name:
                assigned.add(n.name)
            self.generic_visit(n)

        def visit_Global(self, n):
            assigned.update(n.names); self.generic_visit(n)

        def visit_Nonlocal(self, n):
            assigned.update(n.names); self.generic_visit(n)

    V().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in a.args + a.kwonlyargs + getattr(a, 'posonlyargs', []):
                assigned.add(arg.arg)
            if a.vararg:
                assigned.add(a.vararg.arg)
            if a.kwarg:
                assigned.add(a.kwarg.arg)
    return {k: v for k, v in used.items()
            if k not in (assigned | imported | set(dir(builtins)))}


def missing_modules(tree):
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split('.')[0])
    out = []
    for m in sorted(mods):
        try:
            if importlib.util.find_spec(m) is None:
                out.append(m)
        except (ImportError, ValueError, ModuleNotFoundError):
            out.append(m)
    return out


def read_targets(text):
    """Directories the script reads from, with the concrete filenames where known."""
    vmap = {}
    for m in re.finditer(r"^(\w+)\s*=\s*.*?(" + "|".join(STAGE_DIRS) + r")([^\s'\"]*)",
                         text, re.M):
        vmap[m.group(1)] = m.group(2) + m.group(3).rstrip("',\")")
    reads = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        hits = {d for d in STAGE_DIRS if d in s}
        for v, d in vmap.items():
            if re.search(rf"\b{re.escape(v)}\b", s):
                hits.add(d)
        if not hits:
            continue
        if OPEN_WRITE.search(s) or re.search(WRITE_CALLS, s):
            continue
        if re.search(r"rasterio\.open\(", s) or re.search(READ_CALLS, s) or 'open(' in s:
            reads |= hits
    return sorted(reads)


def check(path: Path) -> bool:
    print(f"\n=== preflight: {path.name} ===")
    text = path.read_text()
    ok = True

    try:
        tree = ast.parse(text)
        compile(text, str(path), 'exec')
        print("  syntax            OK")
    except SyntaxError as e:
        print(f"  syntax            FAIL  line {e.lineno}: {e.msg}")
        return False

    u = undefined_names(tree)
    if u:
        ok = False
        print(f"  undefined names   FAIL  {', '.join(f'{k} (line {v})' for k, v in sorted(u.items(), key=lambda x: x[1]))}")
    else:
        print("  undefined names   OK")

    mm = missing_modules(tree)
    if mm:
        ok = False
        print(f"  imports           FAIL  not installed: {', '.join(mm)}")
    else:
        print("  imports           OK")

    # resolved read directories must exist and be non-empty
    for d in read_targets(text):
        p = ROOT / d
        if not p.exists():
            ok = False
            print(f"  input dir         FAIL  missing: {d}")
        elif p.is_dir() and not any(p.iterdir()):
            ok = False
            print(f"  input dir         FAIL  empty:   {d}")
        else:
            n = len(list(p.iterdir())) if p.is_dir() else 1
            print(f"  input dir         OK    {d} ({n} entries)")

    # per-size mismatch: reading a flat dir that only holds size subfolders
    for d in ("06_Probability_Maps", "07_Classified_Maps"):
        if re.search(rf"['\"][^'\"]*{d}['\"]", text) and not re.search(rf"{d}[/'\"].*(10x|MAP_SIZE|size)", text):
            p = ROOT / d
            if p.exists() and not any(f.suffix == '.tif' for f in p.iterdir()):
                ok = False
                print(f"  per-size layout   FAIL  reads flat {d}/ but it holds only size subfolders")

    print(f"  --> {'PASS' if ok else 'DO NOT LAUNCH'}")
    return ok


if __name__ == '__main__':
    targets = sys.argv[1:]
    if not targets:
        print(__doc__)
        sys.exit(2)
    results = [check(DST / t if not os.path.isabs(t) else Path(t)) for t in targets]
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
