"""Repoint hardcoded absolute paths after moving the project to another machine.

Every pipeline script hardcodes the original root
`/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC` (and some reference the
sibling `/home/arora/Documents/FSM_Slovenia/For_Analysis` for read-only inputs).
There is no config layer — this is inherited from the original codebase.

    python3 00_Scripts/_repath.py --new-root /new/path/FSM_Pipeline_LULC --dry-run
    python3 00_Scripts/_repath.py --new-root /new/path/FSM_Pipeline_LULC

Also handles the input tree, which may live somewhere unrelated:

    python3 00_Scripts/_repath.py --new-root /new/path/FSM_Pipeline_LULC \
                                  --new-data-root /data/FSM_Slovenia/For_Analysis

Writes `.bak` copies unless --no-backup. Idempotent: running it twice is a no-op.
"""
import argparse
import shutil
import sys
from pathlib import Path

OLD_ROOT = "/home/arora/Documents/FSM_Slovenia/FSM_Pipeline_LULC"
OLD_DATA = "/home/arora/Documents/FSM_Slovenia/For_Analysis"
OLD_RAW = "/home/arora/Documents/FSM_Slovenia/Raw_Data"

TEXT_SUFFIXES = {".py", ".sh", ".md", ".yml", ".yaml", ".txt", ".csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", required=True,
                    help="absolute path to this repository on the new machine")
    ap.add_argument("--new-data-root", default=None,
                    help=f"replacement for {OLD_DATA} (read-only input rasters/vectors)")
    ap.add_argument("--new-raw-root", default=None,
                    help=f"replacement for {OLD_RAW} (catchment boundaries etc.)")
    ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
    ap.add_argument("--no-backup", action="store_true", help="do not write .bak files")
    args = ap.parse_args(argv)

    new_root = str(Path(args.new_root).resolve())
    if not Path(new_root).is_dir():
        ap.error(f"--new-root does not exist: {new_root}")

    pairs = [(OLD_ROOT, new_root)]
    if args.new_data_root:
        pairs.append((OLD_DATA, str(Path(args.new_data_root).resolve())))
    if args.new_raw_root:
        pairs.append((OLD_RAW, str(Path(args.new_raw_root).resolve())))
    # longest first so the root does not partially match the data path
    pairs.sort(key=lambda p: -len(p[0]))

    repo = Path(__file__).resolve().parent.parent
    files = [p for p in repo.rglob("*")
             if p.is_file() and p.suffix in TEXT_SUFFIXES
             and ".git/" not in str(p) and not p.name.endswith(".bak")]

    changed, total_hits = [], 0
    for f in files:
        try:
            txt = f.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        hits = sum(txt.count(old) for old, _ in pairs)
        if not hits:
            continue
        new_txt = txt
        for old, new in pairs:
            new_txt = new_txt.replace(old, new)
        if new_txt == txt:
            continue
        total_hits += hits
        changed.append((f.relative_to(repo), hits))
        if not args.dry_run:
            if not args.no_backup:
                shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))
            f.write_text(new_txt)

    verb = "would update" if args.dry_run else "updated"
    print(f"{verb} {len(changed)} file(s), {total_hits} path occurrence(s)\n")
    for rel, n in sorted(changed, key=lambda x: -x[1])[:25]:
        print(f"  {n:>4}  {rel}")
    if len(changed) > 25:
        print(f"  ... and {len(changed)-25} more")

    if not args.dry_run and changed:
        print("\nNext:")
        print("  1. Re-link the input rasters:  01_Input_Features/ holds symlinks")
        print("     that point at the OLD machine and are now broken.")
        print("  2. Verify:  python3 00_Scripts/_analyse_deps.py")
        print("  3. Preflight a stage before running it:")
        print("     python3 00_Scripts/_preflight.py 07_threshold_classify_validate.py")
        print("  4. Remove backups when satisfied:  find . -name '*.bak' -delete")

    if args.dry_run:
        print("\n(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
