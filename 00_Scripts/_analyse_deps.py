"""Derive the true execution order from what each script actually reads and writes.

Tracks dependencies at (folder, filename) granularity rather than folder level, so
a script reading back its own output is not mistaken for a dependency on another
script that happens to write into the same folder.
"""
import re
from collections import defaultdict
from pathlib import Path

DST = Path("/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC/00_Scripts")

STAGE_DIRS = [
    "01_Input_Features", "02_Collinearity", "03_Samples", "04_Models",
    "05_Results", "06_Probability_Maps", "07_Classified_Maps",
    "08_Validation", "09_Analysis", "10_Documentation",
]

READ_CALLS = r"(read_csv|read_file|np\.load|joblib\.load|load_model|torch\.load|" \
             r"pickle\.load|read_pickle|imread)"
WRITE_CALLS = r"(to_csv|savefig|joblib\.dump|save_model|torch\.save|to_file|" \
              r"pickle\.dump|makedirs|imwrite|to_pickle|savez|np\.save|json\.dump)"

FILE_RE = re.compile(r"['\"]([\w.\-]+\.(?:tif|csv|pkl|json|npz|pth|png|cbm|shp|txt|docx))['\"]")
# open(...) / rasterio.open(...) with an explicit write mode. `.*` (not [^)]*) so it
# spans a nested os.path.join(...) before reaching the mode argument.
OPEN_WRITE_RE = re.compile(r"open\(.*,\s*['\"][wax]")
OPEN_ANY_RE = re.compile(r"\bopen\(")


def var_map(text):
    """Map variable names to the stage folder they point at."""
    out = {}
    for m in re.finditer(r"^(\w+)\s*=\s*.*?(" + "|".join(STAGE_DIRS) + r")", text, re.M):
        out[m.group(1)] = m.group(2)
    for _ in range(3):
        for m in re.finditer(r"^(\w+)\s*=\s*(?:os\.path\.join\()?\s*[f]?['\"]?\{?(\w+)\}?",
                             text, re.M):
            if m.group(2) in out and m.group(1) not in out:
                out[m.group(1)] = out[m.group(2)]
    return out


def analyse(path):
    """Return (reads, writes) as sets of (stage_dir, filename-or-None)."""
    text = path.read_text()
    vmap = var_map(text)
    # filename -> stage dir, learned from assignments like p = os.path.join(DIR, 'x.tif')
    file_dir = {}
    for m in re.finditer(r"^(\w+)\s*=\s*os\.path\.join\((\w+),\s*['\"]([\w.\-]+)['\"]",
                         text, re.M):
        if m.group(2) in vmap:
            file_dir[m.group(1)] = (vmap[m.group(2)], m.group(3))

    # key -> first line number at which it is read / written
    reads, writes = {}, {}
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("#") or not s:
            continue

        dirs = {d for d in STAGE_DIRS if d in s}
        fname = None
        for v, d in vmap.items():
            if re.search(rf"\b{re.escape(v)}\b", s):
                dirs.add(d)
        for v, (d, f) in file_dir.items():
            if re.search(rf"\b{re.escape(v)}\b", s):
                dirs.add(d)
                fname = f
        if not dirs:
            continue
        if fname is None:
            m = FILE_RE.search(s)
            fname = m.group(1) if m else None

        if OPEN_WRITE_RE.search(s):
            is_w, is_r = True, False
        elif re.search(WRITE_CALLS, s):
            is_w, is_r = True, False
        elif "rasterio.open(" in s or OPEN_ANY_RE.search(s) or re.search(READ_CALLS, s):
            is_w, is_r = False, True
        else:
            continue

        for d in dirs:
            target = writes if is_w else reads
            target.setdefault((d, fname), lineno)
    return reads, writes


scripts = sorted(p for p in DST.glob("[0-9]*.py"))
info = {p.name: analyse(p) for p in scripts}

# producer index at (dir, file) level, with a (dir, None) fallback
producers = defaultdict(set)
for name, (_, w) in info.items():
    for key in w:
        producers[key].add(name)


def resolve(name, key):
    """Who produces this (dir, file)? Ignore the script itself (self-reads)."""
    d, f = key
    exact = producers.get(key, set()) - {name}
    if exact:
        return exact
    if f is not None:
        # file unknown to any producer: fall back to folder-level producers
        return {p for k, v in producers.items() if k[0] == d for p in v} - {name}
    return {p for k, v in producers.items() if k[0] == d for p in v} - {name}


print(f"{'script':<50}{'reads':<46}writes")
print("-" * 132)
for p in scripts:
    r, w = info[p.name]
    fmt = lambda s: ",".join(sorted(f"{d}/{f or chr(42)}" for d, f in s))[:44] or "-"
    print(f"{p.name:<50}{fmt(r):<46}{fmt(w)[:60]}")

print("\n\nDEPENDENCIES")
print("-" * 80)
deps = {}
for name, (r, w) in info.items():
    after = set()
    for key, read_line in r.items():
        # If this script writes the same key BEFORE it reads it, it is reading back
        # its own output - self-contained, no dependency. If it reads before
        # writing (read-modify-write, e.g. validation_metrics.csv), the dependency
        # on whoever creates the file is real.
        own_write = w.get(key)
        if own_write is not None and own_write < read_line:
            continue
        after |= resolve(name, key)
    deps[name] = after
    if after:
        print(f"{name}\n    after: {', '.join(sorted(after))}")

print("\n\nORDERING VIOLATIONS")
print("-" * 80)
bad = [(n, a) for n, aft in deps.items() for a in aft if a > n]
if bad:
    for n, a in sorted(bad):
        print(f"  {n}  must run AFTER  {a}")
else:
    print("  none - numbering is consistent with the real dependency graph")
