"""
dmv0125_to_geotiff.py
─────────────────────
Converts DMV0125 (12.5 m DEM) from raw ASCII XYZ tiles to GeoTIFF.

Handles the "Ungridded dataset / change of Y direction" error that GDAL's
XYZ driver raises: instead of relying on GDAL to parse XYZ, we read each
tile with numpy, reconstruct the regular grid ourselves, and write a proper
GeoTIFF with rasterio.

Pipeline:
  ① Read each .XYZ tile with numpy  →  sort  →  reshape into 2-D grid
  ② Write per-tile Float32 GeoTIFF (native CRS, LZW-compressed)
  ③ Mosaic all per-tile GeoTIFFs into one merged GeoTIFF (rasterio.merge)
  ④ Reproject merged GeoTIFF → EPSG:4326 and EPSG:32633 (UTM 33N)

Source CRS auto-set to EPSG:3794 (D96/TM) — confirmed from:
  • filename suffix "D96" in the tiles
  • sample coordinates  Y≈380000 (Easting), X≈97000 (Northing)
    which match D96/TM's false northing of -5 000 000

Requirements:
  pip install rasterio
  (or: conda install -c conda-forge rasterio)

Usage:
  python dmv0125_to_geotiff.py                      # all outputs, 4 workers
  python dmv0125_to_geotiff.py --workers 8          # more parallelism
  python dmv0125_to_geotiff.py --skip-merge         # per-tile GeoTIFFs only
  python dmv0125_to_geotiff.py --use-d96            # UTM output = D96/TM
  python dmv0125_to_geotiff.py --source-epsg 3912   # if tiles are D48/GK
"""

import sys
import time
import logging
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ─── PROJ conflict fix ───────────────────────────────────────────────────────
# Must be set BEFORE rasterio/pyproj is imported.
# Finds the proj.db that ships with the current rasterio install and forces
# PROJ_DATA / PROJ_LIB to point at it, bypassing the stale rapids_env proj.db.
import os, sqlite3

def _fix_proj_data():
    from pathlib import Path
    env_prefix = os.environ.get("CONDA_PREFIX", "")
    candidates = []
    try:
        import pyproj
        candidates.append(pyproj.datadir.get_data_dir())
    except Exception:
        pass
    if env_prefix:
        for hit in Path(env_prefix).rglob("proj.db"):
            candidates.append(str(hit.parent))
    for cand in candidates:
        db = Path(cand) / "proj.db"
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(str(db))
            row = con.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
            ).fetchone()
            con.close()
            if row and int(row[0]) >= 6:
                os.environ["PROJ_DATA"] = cand
                os.environ["PROJ_LIB"]  = cand
                return cand
        except Exception:
            continue
    return None

_proj_dir = _fix_proj_data()

# ─── Check rasterio early ────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.merge import merge as rio_merge
    from rasterio.warp import (calculate_default_transform,
                                reproject as rio_reproject,
                                Resampling)
    from rasterio.crs import CRS
except ImportError:
    print("ERROR: rasterio is not installed.")
    print("Install with:  pip install rasterio")
    print("          or:  conda install -c conda-forge rasterio")
    sys.exit(1)

# ─── Default paths ───────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR  = "/home/arora/Documents/FSM_Slovenia/Raw_Data/DMV0125"
DEFAULT_OUTPUT_DIR = "/home/arora/Documents/FSM_Slovenia/GeoTIFF/DMV0125"

# ─── CRS — WKT strings avoid proj.db EPSG lookups at import time ─────────────
# crs_from_epsg() triggers a proj.db read; CRS.from_wkt() does not.
# These WKT2 strings are the canonical definitions for each EPSG code.

WKT = {
    # EPSG:3794  D96/TM — modern official Slovenian projection
    3794: """PROJCRS["Slovenia 1996 / Transverse Mercator",
  BASEGEOGCRS["Slovenia 1996",
    DATUM["Slovenia Geodetic Datum 1996",
      ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],
    PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],
    ANGLEUNIT["degree",0.0174532925199433]],
  CONVERSION["Transverse Mercator",
    METHOD["Transverse Mercator",ID["EPSG",9807]],
    PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433]],
    PARAMETER["Longitude of natural origin",15,ANGLEUNIT["degree",0.0174532925199433]],
    PARAMETER["Scale factor at natural origin",0.9999,SCALEUNIT["unity",1]],
    PARAMETER["False easting",500000,LENGTHUNIT["metre",1]],
    PARAMETER["False northing",-5000000,LENGTHUNIT["metre",1]]],
  CS[Cartesian,2],
    AXIS["(Y)",east,ORDER[1],LENGTHUNIT["metre",1]],
    AXIS["(X)",north,ORDER[2],LENGTHUNIT["metre",1]]]""",

    # EPSG:3912  D48/GK — old Slovenian Gauss-Kruger
    3912: """PROJCRS["MGI 1901 / Balkans zone 6",
  BASEGEOGCRS["MGI 1901",
    DATUM["MGI 1901",
      ELLIPSOID["Bessel 1841",6377397.155,299.1528128,LENGTHUNIT["metre",1]]],
    PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],
    ANGLEUNIT["degree",0.0174532925199433]],
  CONVERSION["Gauss-Kruger zone 6",
    METHOD["Transverse Mercator",ID["EPSG",9807]],
    PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433]],
    PARAMETER["Longitude of natural origin",15,ANGLEUNIT["degree",0.0174532925199433]],
    PARAMETER["Scale factor at natural origin",0.9999,SCALEUNIT["unity",1]],
    PARAMETER["False easting",500000,LENGTHUNIT["metre",1]],
    PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],
  CS[Cartesian,2],
    AXIS["(Y)",east,ORDER[1],LENGTHUNIT["metre",1]],
    AXIS["(X)",north,ORDER[2],LENGTHUNIT["metre",1]]]""",

    # EPSG:32633  WGS 84 / UTM zone 33N
    32633: """PROJCRS["WGS 84 / UTM zone 33N",
  BASEGEOGCRS["WGS 84",
    DATUM["World Geodetic System 1984",
      ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],
    PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],
    ANGLEUNIT["degree",0.0174532925199433]],
  CONVERSION["UTM zone 33N",
    METHOD["Transverse Mercator",ID["EPSG",9807]],
    PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433]],
    PARAMETER["Longitude of natural origin",15,ANGLEUNIT["degree",0.0174532925199433]],
    PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1]],
    PARAMETER["False easting",500000,LENGTHUNIT["metre",1]],
    PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],
  CS[Cartesian,2],
    AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],
    AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]]]""",

    # EPSG:4326  WGS 84 geographic
    4326: """GEOGCRS["WGS 84",
  DATUM["World Geodetic System 1984",
    ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],
  PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],
  CS[ellipsoidal,2],
    AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],
    AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]]]""",
}

def crs_from_epsg(epsg: int) -> CRS:
    """Return CRS from embedded WKT (no proj.db lookup). Falls back to from_epsg."""
    if epsg in WKT:
        return CRS.from_wkt(WKT[epsg])
    return crs_from_epsg(epsg)

NODATA = -9999.0

# ─── GeoTIFF creation profile ────────────────────────────────────────────────

BASE_PROFILE = {
    "driver":    "GTiff",
    "dtype":     "float32",
    "count":     1,
    "nodata":    NODATA,
    "compress":  "lzw",
    "predictor": 2,          # horizontal differencing — excellent for elevation
    "tiled":     True,
    "blockxsize": 512,
    "blockysize": 512,
    "bigtiff":   "IF_SAFER",
}

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt_size(path: Path) -> str:
    n = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def find_xyz_files(root: Path) -> list:
    """Case-insensitive glob for .xyz / .XYZ files, recursively."""
    files = list(root.rglob("*.xyz")) + list(root.rglob("*.XYZ"))
    seen, out = set(), []
    for f in sorted(files):
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ─── Step 1: per-tile XYZ → GeoTIFF ─────────────────────────────────────────

def xyz_to_tif(args_tuple):
    """
    Worker: convert one XYZ tile to a GeoTIFF.
    Runs in a subprocess so must import rasterio/numpy itself.
    Returns (success: bool, xyz_path_str: str, info_str: str)
    """
    xyz_path_str, tif_path_str, source_epsg = args_tuple

    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.crs import CRS

        xyz_path = Path(xyz_path_str)
        tif_path = Path(tif_path_str)

        # ── Read ──────────────────────────────────────────────────────
        data = np.loadtxt(xyz_path, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, 3)
        if data.shape[0] == 0:
            return False, xyz_path_str, "empty file"

        # Columns: Y=Easting, X=Northing, H=Elevation  (YXZ Slovenian convention)
        easting   = data[:, 0]
        northing  = data[:, 1]
        elevation = data[:, 2].astype(np.float32)

        # ── Detect pixel resolution (12.5 m expected) ─────────────────
        # Strategy: round coords to 2 dp, compute diffs, filter noise,
        # then sanity-clamp to known DMV0125 resolution (12.5 m).
        #
        # Background on the MemoryError bug:
        #   Coordinates like 380377.70 accumulate float64 error across a tile
        #   (e.g. 380377.70 + 239*12.5 can drift slightly). If the tile has
        #   any irregular spacing or sub-grid offsets, np.unique produces extra
        #   near-duplicate entries. Their tiny diffs (< 0.01 m) then dominate
        #   the median -> res=0.0125 m -> grid shape 300 000×225 000 = 251 GiB.
        easting_r  = np.round(easting,  2)
        northing_r = np.round(northing, 2)
        u_e = np.unique(easting_r)
        u_n = np.unique(northing_r)

        if len(u_e) < 2 or len(u_n) < 2:
            return False, xyz_path_str, "too few unique coordinates for a grid"

        diffs_e = np.diff(u_e)
        diffs_n = np.diff(u_n)
        # Keep only diffs >= 0.5 m (eliminates float noise; real spacing is 12.5 m)
        MIN_RES = 0.5
        pos_e = diffs_e[diffs_e >= MIN_RES]
        pos_n = diffs_n[diffs_n >= MIN_RES]
        if len(pos_e) == 0 or len(pos_n) == 0:
            res = 12.5
        else:
            res = round(float(np.median(np.concatenate([pos_e, pos_n]))), 2)

        # Sanity clamp: if detected res is implausible for DMV0125, use 12.5 m.
        # Accepts anything within ±30 % of 12.5 (i.e. 8.75–16.25 m).
        KNOWN_RES = 12.5
        if not (KNOWN_RES * 0.7 <= res <= KNOWN_RES * 1.3):
            res = KNOWN_RES

        # ── Grid dimensions ───────────────────────────────────────────
        e_min, e_max = float(u_e[0]),  float(u_e[-1])
        n_min, n_max = float(u_n[0]),  float(u_n[-1])

        ncols = round((e_max - e_min) / res) + 1
        nrows = round((n_max - n_min) / res) + 1

        # ── Map each point to its (row, col) index ────────────────────
        # Use rounded coordinates (same as used to compute u_e/u_n) for consistency
        col_idx = np.clip(np.round((easting_r  - e_min) / res).astype(int), 0, ncols - 1)
        row_idx = np.clip(np.round((n_max - northing_r) / res).astype(int), 0, nrows - 1)

        # ── Fill raster ───────────────────────────────────────────────
        grid = np.full((nrows, ncols), -9999.0, dtype=np.float32)
        grid[row_idx, col_idx] = elevation

        # ── Affine transform (top-left corner) ────────────────────────
        transform = from_origin(
            west  = e_min - res / 2,
            north = n_max + res / 2,
            xsize = res,
            ysize = res,
        )

        # ── Write GeoTIFF ─────────────────────────────────────────────
        profile = {
            "driver":     "GTiff",
            "dtype":      "float32",
            "count":      1,
            "nodata":     -9999.0,
            "compress":   "lzw",
            "predictor":  2,
            "tiled":      True,
            "blockxsize": 512,
            "blockysize": 512,
            "bigtiff":    "IF_SAFER",
            "crs":        crs_from_epsg(source_epsg),
            "width":      ncols,
            "height":     nrows,
            "transform":  transform,
        }

        tif_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tif_path, "w", **profile) as dst:
            dst.write(grid, 1)

        return True, xyz_path_str, tif_path_str

    except Exception as exc:
        return False, xyz_path_str, f"{type(exc).__name__}: {exc}"


def convert_tiles(input_dir: Path, tile_dir: Path,
                  source_epsg: int, workers: int) -> list:
    """Convert all XYZ tiles to per-tile GeoTIFFs. Returns list of output Paths."""

    xyz_files = find_xyz_files(input_dir)
    if not xyz_files:
        log.error("No .XYZ files found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d XYZ tiles", len(xyz_files))
    tile_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(xf), str(tile_dir / (xf.stem + ".tif")), source_epsg)
        for xf in xyz_files
    ]

    done_tifs = []
    failed = 0
    t0 = time.perf_counter()

    if workers == 1:
        for i, task in enumerate(tasks, 1):
            ok, src, info = xyz_to_tif(task)
            if ok:
                done_tifs.append(Path(info))
                if i % 100 == 0 or i == len(tasks):
                    log.info("  [%4d/%d] %s", i, len(tasks), Path(src).name)
            else:
                failed += 1
                log.warning("  ✗  %s — %s", Path(src).name, info)
    else:
        log.info("Parallel conversion with %d workers …", workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(xyz_to_tif, t): t for t in tasks}
            completed = 0
            for fut in as_completed(futures):
                completed += 1
                ok, src, info = fut.result()
                if ok:
                    done_tifs.append(Path(info))
                else:
                    failed += 1
                    log.warning("  ✗  %s — %s", Path(src).name, info[:120])
                if completed % 100 == 0 or completed == len(tasks):
                    log.info("  Progress: %d / %d  (failed: %d)",
                             completed, len(tasks), failed)

    elapsed = time.perf_counter() - t0
    log.info("Tile conversion done: %d OK, %d failed in %.1f s",
             len(done_tifs), failed, elapsed)
    return sorted(done_tifs)


# ─── Step 2: Mosaic ──────────────────────────────────────────────────────────

def mosaic_tiles(tile_paths: list, out_path: Path) -> None:
    """Merge per-tile GeoTIFFs into one mosaic using rasterio.merge."""
    log.info("Mosaicking %d tiles → %s", len(tile_paths), out_path.name)
    t0 = time.perf_counter()

    open_files = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = rio_merge(open_files, nodata=NODATA)
    finally:
        for f in open_files:
            f.close()

    profile = {
        **BASE_PROFILE,
        "crs":       open_files[0].crs,
        "width":     mosaic.shape[2],
        "height":    mosaic.shape[1],
        "transform": transform,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)

    log.info("  Mosaic done in %.1f s  →  %s", time.perf_counter() - t0, fmt_size(out_path))


# ─── Step 3: Reproject ───────────────────────────────────────────────────────

def reproject_tif(src_path: Path, dst_path: Path,
                  target_crs: CRS, label: str) -> None:
    log.info("Reprojecting → %s", label)
    t0 = time.perf_counter()

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = {
            **BASE_PROFILE,
            "crs":       target_crs,
            "width":     width,
            "height":    height,
            "transform": transform,
        }
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            rio_reproject(
                source        = rasterio.band(src, 1),
                destination   = rasterio.band(dst, 1),
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = transform,
                dst_crs       = target_crs,
                resampling    = Resampling.bilinear,
                src_nodata    = NODATA,
                dst_nodata    = NODATA,
            )

    log.info("  Done in %.1f s  →  %s", time.perf_counter() - t0, fmt_size(dst_path))


# ─── Overviews ───────────────────────────────────────────────────────────────

def build_overviews(tif_path: Path) -> None:
    log.info("  Building overviews for %s …", tif_path.name)
    with rasterio.open(tif_path, "r+") as ds:
        ds.build_overviews([2, 4, 8, 16, 32, 64], Resampling.average)
        ds.update_tags(ns="rio_overview", resampling="average")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    source_epsg = args.source_epsg
    use_d96     = args.use_d96
    utm_crs     = crs_from_epsg(3794) if use_d96 else crs_from_epsg(32633)
    utm_label   = "D96/TM (EPSG:3794)" if use_d96 else "UTM 33N (EPSG:32633)"
    utm_epsg    = 3794 if use_d96 else 32633

    tile_dir    = output_dir / "tiles_native"
    merged_path = output_dir / f"DMV0125_EPSG{source_epsg}_merged.tif"
    wgs84_path  = output_dir / "DMV0125_EPSG4326.tif"
    utm_path    = output_dir / f"DMV0125_EPSG{utm_epsg}.tif"

    log.info("═" * 62)
    log.info("DMV0125 → GeoTIFF  (rasterio pipeline)")
    log.info("  Source CRS : EPSG:%d", source_epsg)
    log.info("  UTM output : %s", utm_label)
    log.info("  Workers    : %d", args.workers)
    log.info("  Input      : %s", input_dir)
    log.info("  Output     : %s", output_dir)
    log.info("═" * 62)

    # ① Per-tile conversion
    log.info("")
    log.info("── Step 1/3: XYZ tiles → per-tile GeoTIFFs …")
    tile_tifs = convert_tiles(input_dir, tile_dir, source_epsg, args.workers)

    if not tile_tifs:
        log.error("No tiles converted. Aborting.")
        sys.exit(1)

    if args.skip_merge:
        log.info("--skip-merge: stopping after tile conversion.")
        log.info("Per-tile GeoTIFFs: %s", tile_dir)
        return

    # ② Mosaic
    log.info("")
    log.info("── Step 2/3: Mosaicking …")
    mosaic_tiles(tile_tifs, merged_path)
    if not args.skip_overviews:
        build_overviews(merged_path)

    # ③ Reproject
    log.info("")
    log.info("── Step 3/3: Reprojecting …")

    if not args.skip_wgs84:
        reproject_tif(merged_path, wgs84_path, crs_from_epsg(4326), "WGS84 (EPSG:4326)")
        if not args.skip_overviews:
            build_overviews(wgs84_path)

    if not args.skip_utm:
        reproject_tif(merged_path, utm_path, utm_crs, utm_label)
        if not args.skip_overviews:
            build_overviews(utm_path)

    log.info("")
    log.info("═" * 62)
    log.info("Outputs:")
    for p in [merged_path, wgs84_path, utm_path]:
        if p.exists():
            log.info("  %-48s  %s", p.name, fmt_size(p))
    log.info("Per-tile GeoTIFFs: %s  (%d files)", tile_dir, len(tile_tifs))
    log.info("═" * 62)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert DMV0125 XYZ tiles to GeoTIFF (solves GDAL ungridded error).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Install:
  pip install rasterio
  conda install -c conda-forge rasterio   (in rapids_env)

CRS:
  EPSG:3794  D96/TM       -- Default source (D96 tiles)
  EPSG:3912  D48/GK       -- Use --source-epsg 3912 for older tiles
  EPSG:32633 UTM Zone 33N -- Default projected output
  EPSG:4326  WGS84        -- Always produced (unless --skip-wgs84)

Examples:
  python dmv0125_to_geotiff.py
  python dmv0125_to_geotiff.py --workers 8
  python dmv0125_to_geotiff.py --use-d96
  python dmv0125_to_geotiff.py --source-epsg 3912   # D48 tiles
        """
    )
    p.add_argument("--input-dir",      default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--source-epsg",    type=int, default=3794,
                   help="Source CRS EPSG code (default: 3794 = D96/TM)")
    p.add_argument("--use-d96",        action="store_true",
                   help="UTM output = D96/TM (EPSG:3794) instead of UTM 33N")
    p.add_argument("--workers",        type=int, default=4,
                   help="Parallel workers for tile conversion (default: 4)")
    p.add_argument("--skip-merge",     action="store_true")
    p.add_argument("--skip-wgs84",     action="store_true")
    p.add_argument("--skip-utm",       action="store_true")
    p.add_argument("--skip-overviews", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())