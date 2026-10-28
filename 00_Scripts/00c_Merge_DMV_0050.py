"""
merge_dmv0050.py
────────────────
Merges all .xyz tiles for the 5 m Digital Elevation Model (DMV0050)
from the four quadrant folders (SZ / SV / JZ / JV) — each with
their own sub-folder structure — into a single merged ASCII YXZ file.

Quadrant map (Slovenian cardinal directions):
    SZ = Severozahod  (NW)   SV = Severovzhod  (NE)
    JZ = Jugozahod    (SW)   JV = Jugovzhod     (SE)

Source folders:
    /home/arora/Documents/FSM_Slovenia/Raw_Data/DMV0050_SZ/
    /home/arora/Documents/FSM_Slovenia/Raw_Data/DMV0050_SV/
    /home/arora/Documents/FSM_Slovenia/Raw_Data/DMV0050_JZ/
    /home/arora/Documents/FSM_Slovenia/Raw_Data/DMV0050_JV/

Output:
    /home/arora/Documents/FSM_Slovenia/Merged/DMV0050_merged.xyz

Usage:
    # Merge all four quadrants into one file (default)
    python merge_dmv0050.py

    # Merge but keep per-quadrant files as well
    python merge_dmv0050.py --per-quadrant

    # Only build per-quadrant files (no combined file)
    python merge_dmv0050.py --per-quadrant --no-combined

    # Compress output(s)
    python merge_dmv0050.py --compress
"""

import os
import sys
import gzip
import time
import argparse
import logging
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DATA_DIR = "/home/arora/Documents/FSM_Slovenia/Raw_Data/"
OUTPUT_DIR    = "/home/arora/Documents/FSM_Slovenia/Merged/"

# Four quadrant folders — tiles are further nested in sub-folders inside each
QUADRANT_DIRS = {
    "SZ": Path(BASE_DATA_DIR) / "DMV0050_SZ",   # NW
    "SV": Path(BASE_DATA_DIR) / "DMV0050_SV",   # NE
    "JZ": Path(BASE_DATA_DIR) / "DMV0050_JZ",   # SW
    "JV": Path(BASE_DATA_DIR) / "DMV0050_JV",   # SE
}

# Merge order: NW → NE → SW → SE  (top-left to bottom-right, west before east)
MERGE_ORDER = ["SZ", "SV", "JZ", "JV"]

FILE_EXTENSION = ".XYZ"
CHUNK_SIZE     = 1024 * 1024 * 8  # 8 MB read buffer

# ─── Logging setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def find_xyz_files(root: Path) -> list[Path]:
    """Recursively collect all .xyz files under root, sorted for reproducibility."""
    return sorted(root.rglob(f"*{FILE_EXTENSION}"))


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def open_output(path: Path, compress: bool):
    """Return an open writable binary file-like object (plain or gzip)."""
    if compress:
        return gzip.open(str(path) + ".gz", "wb")
    return open(path, "wb")


def stream_tiles_into(out, tiles: list[Path], label: str) -> tuple[int, int, int]:
    """
    Stream-write all tiles into an already-open file object `out`.
    Returns (files_written, files_skipped, bytes_written).
    """
    written = skipped = total_bytes = 0

    for i, fpath in enumerate(tiles, 1):
        try:
            fsize = fpath.stat().st_size
            if fsize == 0:
                log.warning("  [%4d/%d] EMPTY  %s — skipped", i, len(tiles), fpath.name)
                skipped += 1
                continue

            log.info("  [%4d/%d] %s  %-45s (%s)",
                     i, len(tiles), label, fpath.name, format_size(fsize))

            with open(fpath, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    # Guarantee newline separation between tiles
                    if total_bytes > 0 and chunk and chunk[0:1] not in (b"\n", b"\r"):
                        pass  # data normally starts a new line; no prefix needed
                    out.write(chunk)
                    total_bytes += len(chunk)

            written += 1

        except OSError as exc:
            log.error("  Could not read %s: %s — skipped", fpath, exc)
            skipped += 1

    return written, skipped, total_bytes


def write_header(out, quadrants: list[str], total_tiles: int, compress: bool):
    header = (
        "# DMV0050  –  Digital Elevation Model of Slovenia, 5 m resolution\n"
        "# Source  : Surveying and Mapping Authority of the Republic of Slovenia\n"
        "# Format  : Y  X  H  (ASCII, space-delimited)\n"
        f"# Quadrants included : {', '.join(quadrants)}\n"
        f"# Total tiles        : {total_tiles}\n"
        f"# Merged             : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    out.write(header.encode())


# ─── Merge logic ─────────────────────────────────────────────────────────────

def collect_tiles() -> dict[str, list[Path]]:
    """Validate directories and collect tile paths per quadrant."""
    tile_map: dict[str, list[Path]] = {}
    missing = []

    for qname in MERGE_ORDER:
        qdir = QUADRANT_DIRS[qname]
        if not qdir.exists():
            log.warning("Quadrant directory not found: %s", qdir)
            missing.append(qname)
            continue
        tiles = find_xyz_files(qdir)
        tile_map[qname] = tiles
        log.info("  %s  →  %4d tiles  in  %s", qname, len(tiles), qdir)

    if missing:
        log.warning("Missing quadrant(s): %s — they will be skipped.", missing)
    if not tile_map:
        log.error("No valid quadrant directories found. Check BASE_DATA_DIR: %s", BASE_DATA_DIR)
        sys.exit(1)

    return tile_map


def merge_per_quadrant(tile_map: dict[str, list[Path]], output_dir: Path,
                       compress: bool, write_hdr: bool):
    """Write one merged .xyz per quadrant."""
    results = {}
    for qname in MERGE_ORDER:
        if qname not in tile_map:
            continue
        tiles = tile_map[qname]
        out_path = output_dir / f"DMV0050_{qname}_merged.xyz"
        final_path = Path(str(out_path) + ".gz") if compress else out_path
        log.info("")
        log.info("── Quadrant %s → %s", qname, out_path.name)

        t0 = time.perf_counter()
        with open_output(out_path, compress) as out:
            if write_hdr:
                write_header(out, [qname], len(tiles), compress)
            w, s, b = stream_tiles_into(out, tiles, qname)

        elapsed = time.perf_counter() - t0
        log.info("  %s done: %d tiles in %.1f s  →  %s",
                 qname, w, elapsed, format_size(final_path.stat().st_size))
        results[qname] = final_path
    return results


def merge_combined(tile_map: dict[str, list[Path]], output_dir: Path,
                   compress: bool, write_hdr: bool):
    """Write a single merged .xyz file from all quadrants."""
    out_path   = output_dir / "DMV0050_merged.xyz"
    final_path = Path(str(out_path) + ".gz") if compress else out_path
    all_tiles  = sum(len(v) for v in tile_map.values())
    present_q  = [q for q in MERGE_ORDER if q in tile_map]

    log.info("")
    log.info("── Combined merge  →  %s", out_path.name)
    log.info("   Quadrants : %s", "  ".join(present_q))
    log.info("   Total tiles: %d", all_tiles)

    t0 = time.perf_counter()
    total_written = total_skipped = total_bytes = 0

    with open_output(out_path, compress) as out:
        if write_hdr:
            write_header(out, present_q, all_tiles, compress)

        for qname in MERGE_ORDER:
            if qname not in tile_map:
                continue
            tiles = tile_map[qname]
            log.info("")
            log.info("   ── Streaming quadrant %s (%d tiles) ──", qname, len(tiles))
            w, s, b = stream_tiles_into(out, tiles, qname)
            total_written  += w
            total_skipped  += s
            total_bytes    += b

    elapsed = time.perf_counter() - t0
    log.info("")
    log.info("─" * 60)
    log.info("Combined merge complete in %.1f s", elapsed)
    log.info("Tiles merged  : %d  (skipped: %d)", total_written, total_skipped)
    log.info("Data written  : %s", format_size(total_bytes))
    log.info("Output        : %s  (%s)", final_path, format_size(final_path.stat().st_size))
    log.info("─" * 60)
    return final_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge DMV0050 (5 m) tiles from all four quadrants into one .xyz file."
    )
    p.add_argument("--base-data-dir", default=BASE_DATA_DIR,
                   help=f"Parent of the four DMV0050_* folders (default: {BASE_DATA_DIR})")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help=f"Output directory (default: {OUTPUT_DIR})")
    p.add_argument("--per-quadrant", action="store_true",
                   help="Also write one merged file per quadrant (SZ / SV / JZ / JV)")
    p.add_argument("--no-combined", action="store_true",
                   help="Skip the single combined output (useful with --per-quadrant)")
    p.add_argument("--compress", action="store_true",
                   help="Gzip-compress output file(s) (adds .gz extension)")
    p.add_argument("--no-header", action="store_true",
                   help="Suppress informational comment header in output(s)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Allow overriding base dir from CLI
    global BASE_DATA_DIR
    BASE_DATA_DIR = args.base_data_dir
    for qname, _ in QUADRANT_DIRS.items():
        QUADRANT_DIRS[qname] = Path(BASE_DATA_DIR) / f"DMV0050_{qname}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_hdr = not args.no_header

    log.info("DMV0050 Merge — 5 m DEM")
    log.info("Base data dir : %s", BASE_DATA_DIR)
    log.info("Output dir    : %s", output_dir)
    log.info("─" * 60)
    log.info("Scanning quadrant directories …")

    tile_map = collect_tiles()

    if args.per_quadrant:
        merge_per_quadrant(tile_map, output_dir, args.compress, write_hdr)

    if not args.no_combined:
        merge_combined(tile_map, output_dir, args.compress, write_hdr)

    if args.no_combined and not args.per_quadrant:
        log.warning("Nothing to do: --no-combined was set without --per-quadrant.")