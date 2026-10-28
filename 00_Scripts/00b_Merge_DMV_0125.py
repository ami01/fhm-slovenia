"""
merge_dmv0125.py
────────────────
Merges all .xyz tiles for the 12.5 m Digital Elevation Model (DMV0125)
into a single ASCII YXZ file.

Source: /home/arora/Documents/FSM_Slovenia/Raw_Data/DMV0125/
Output: /home/arora/Documents/FSM_Slovenia/Merged/DMV0125_merged.xyz

Usage:
    python merge_dmv0125.py [--input-dir DIR] [--output-dir DIR]
                            [--compress] [--no-header]
"""

import os
import sys
import gzip
import time
import argparse
import logging
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR  = "/home/arora/Documents/FSM_Slovenia/Raw_Data/DMV0125/"
DEFAULT_OUTPUT_DIR = "/home/arora/Documents/FSM_Slovenia/Merged"
OUTPUT_FILENAME    = "DMV0125_merged.XYZ"
FILE_EXTENSION     = ".XYZ"
CHUNK_SIZE         = 1024 * 1024 * 8  # 8 MB read buffer

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
    files = sorted(root.rglob(f"*{FILE_EXTENSION}"))
    return files


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def open_output(path: Path, compress: bool):
    """Return an open writable file-like object (plain or gzip)."""
    if compress:
        return gzip.open(str(path) + ".gz", "wb")
    return open(path, "wb")


# ─── Main merge ──────────────────────────────────────────────────────────────

def merge(input_dir: Path, output_path: Path, compress: bool, write_header: bool):

    xyz_files = find_xyz_files(input_dir)
    if not xyz_files:
        log.error("No .xyz files found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d tile files in %s", len(xyz_files), input_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = Path(str(output_path) + ".gz") if compress else output_path

    t0 = time.perf_counter()
    total_bytes_read  = 0
    total_lines       = 0
    skipped_files     = 0

    with open_output(output_path, compress) as out:

        # Optional informational header (comment lines, ignored by GDAL/QGIS)
        if write_header:
            header = (
                "# DMV0125  –  Digital Elevation Model of Slovenia, 12.5 m resolution\n"
                "# Source  : Surveying and Mapping Authority of the Republic of Slovenia\n"
                "# Format  : Y  X  H  (ASCII, space-delimited)\n"
                f"# Tiles   : {len(xyz_files)}\n"
                f"# Merged  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            out.write(header.encode())

        for i, fpath in enumerate(xyz_files, 1):
            try:
                fsize = fpath.stat().st_size
                if fsize == 0:
                    log.warning("[%4d/%d] EMPTY  %s — skipped", i, len(xyz_files), fpath.name)
                    skipped_files += 1
                    continue

                log.info("[%4d/%d] Reading  %-40s  (%s)",
                         i, len(xyz_files), fpath.name, format_size(fsize))

                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        # Ensure file ends with newline before next tile's data
                        out.write(chunk)
                        total_bytes_read += len(chunk)

                # Quick line-count estimate from file size / avg line length
                # (exact count would require re-reading)
                total_lines += fsize // 20  # rough estimate (~20 bytes/line)

            except OSError as exc:
                log.error("Could not read %s: %s — skipped", fpath, exc)
                skipped_files += 1

    elapsed = time.perf_counter() - t0
    log.info("─" * 60)
    log.info("Done in %.1f s", elapsed)
    log.info("Tiles merged  : %d  (skipped: %d)", len(xyz_files) - skipped_files, skipped_files)
    log.info("Data read     : %s", format_size(total_bytes_read))
    log.info("Output        : %s  (%s)", final_path, format_size(final_path.stat().st_size))
    log.info("─" * 60)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge DMV0125 (12.5 m) tiles into one .xyz file.")
    p.add_argument("--input-dir",  default=DEFAULT_INPUT_DIR,
                   help=f"Root folder with .xyz tiles (default: {DEFAULT_INPUT_DIR})")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Output folder (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--compress", action="store_true",
                   help="Gzip-compress the output (adds .gz extension)")
    p.add_argument("--no-header", action="store_true",
                   help="Suppress the informational comment header in the output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir   = Path(args.input_dir)
    output_path = Path(args.output_dir) / OUTPUT_FILENAME

    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    log.info("DMV0125 Merge — 12.5 m DEM")
    log.info("Input  : %s", input_dir)
    log.info("Output : %s%s", output_path, ".gz" if args.compress else "")
    log.info("─" * 60)

    merge(input_dir, output_path, args.compress, write_header=not args.no_header)