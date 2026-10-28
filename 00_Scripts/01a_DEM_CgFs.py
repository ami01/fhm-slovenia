"""
dem_conditioning_factors.py
═══════════════════════════
GPU-accelerated derivation of DEM-based conditioning factors
for flood susceptibility / hazard mapping  (FSM Paper 1, Khagaria district).

Factors produced
────────────────
  Terrain morphology  (GPU)
    01  elevation        — raw DEM (clipped, filled)
    02  slope            — gradient in degrees  (Horn 1981)
    03  aspect           — flow-facing direction, degrees from N
    04  plan_curvature   — convergence/divergence of flow (± convention)
    05  profile_curvature— acceleration of flow along slope
    06  tpi              — Topographic Position Index (3×3 window)
    07  tri              — Terrain Ruggedness Index (Riley et al. 1999)
    08  roughness        — max − min in 3×3 window
    09  valley_depth     — elevation below local ridge (large-window TPI)

  Hydrological  (CPU — D8 flow routing via pysheds, then GPU composites)
    10  flow_direction   — D8 encoded 1–128
    11  flow_accumulation— upstream contributing cells
    12  drainage_density — flow_acc threshold → binary network → density
    13  twi              — ln(a / tan β)  Beven & Kirkby 1979
    14  spi              — As × tan β     stream power index
    15  ls_factor        — (As/22.13)^0.4 × (sin β/0.0896)^1.3   RUSLE

  Height Above Nearest Drainage  (CPU + optional GPU post-process)
    16  hand             — vertical distance to nearest drainage cell

Output
──────
  <output_dir>/factors/          — one Float32 GeoTIFF per factor
  <output_dir>/stack.tif         — 16-band stacked GeoTIFF (ML-ready)
  <output_dir>/stack.npz         — numpy compressed array + metadata

GPU acceleration
────────────────
  CuPy replaces scipy.ndimage for all convolution and element-wise ops.
  Speedup: ~10–30× on morphology factors (slope/curvature/TRI etc.).
  Flow accumulation (D8) is inherently sequential → stays on CPU.
  Falls back gracefully to scipy/numpy if CuPy is unavailable.

Requirements
────────────
  conda install -c conda-forge rasterio scipy scikit-image
  pip install pysheds                 # flow routing
  conda install -c rapidsai cupy      # GPU (already in rapids_env)

Usage
─────
  python dem_conditioning_factors.py --dem /path/to/dem.tif
  python dem_conditioning_factors.py --dem /path/to/dem.tif --force-cpu
  python dem_conditioning_factors.py --dem /path/to/dem.tif --window 5
  python dem_conditioning_factors.py --dem /path/to/dem.tif --drain-thresh 500
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import argparse
import sqlite3
import warnings
from pathlib import Path

# ── scientific ────────────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import (uniform_filter, maximum_filter, minimum_filter,
                            sobel, label as nd_label)

# ── rasterio ──────────────────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    from rasterio.warp import reproject as rio_reproject, Resampling
    from rasterio.fill import fillnodata
except ImportError:
    sys.exit("ERROR: pip install rasterio")

# ── GPU libraries (optional) ─────────────────────────────────────────────────
def _fix_proj_data():
    env_prefix = os.environ.get("CONDA_PREFIX", "")
    candidates = []
    try:
        import pyproj; candidates.append(pyproj.datadir.get_data_dir())
    except Exception: pass
    if env_prefix:
        for hit in Path(env_prefix).rglob("proj.db"):
            candidates.append(str(hit.parent))
    for cand in candidates:
        db = Path(cand) / "proj.db"
        if not db.exists(): continue
        try:
            con = sqlite3.connect(str(db))
            row = con.execute(
                "SELECT value FROM metadata WHERE key='DATABASE.LAYOUT.VERSION.MINOR'"
            ).fetchone()
            con.close()
            if row and int(row[0]) >= 6:
                os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = cand
                return cand
        except Exception: continue
    return None
_fix_proj_data()

try:
    import cupy as cp
    from cupyx.scipy.ndimage import (uniform_filter as cu_uniform_filter,
                                      maximum_filter as cu_maximum_filter,
                                      minimum_filter as cu_minimum_filter)
    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False

# ── pysheds (flow routing) ────────────────────────────────────────────────────
try:
    from pysheds.grid import Grid as PyshedsGrid
    _PYSHEDS_AVAILABLE = True
except ImportError:
    _PYSHEDS_AVAILABLE = False

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── WKT CRS bank (no proj.db lookups) ───────────────────────────────────────
WKT = {
    4326: 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
          'ELLIPSOID["WGS 84",6378137,298.257223563]],'
          'CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east]]',
    32644: 'PROJCRS["WGS 84 / UTM zone 44N",'
           'BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
           'ELLIPSOID["WGS 84",6378137,298.257223563]],'
           'PRIMEM["Greenwich",0],ANGLEUNIT["degree",0.0174532925199433]],'
           'CONVERSION["UTM zone 44N",METHOD["Transverse Mercator"],'
           'PARAMETER["Longitude of natural origin",81,ANGLEUNIT["degree",0.0174532925199433]],'
           'PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1]],'
           'PARAMETER["False easting",500000,LENGTHUNIT["metre",1]],'
           'PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],'
           'CS[Cartesian,2],AXIS["(E)",east],AXIS["(N)",north],LENGTHUNIT["metre",1]]',
}

def crs_from_epsg(epsg):
    return CRS.from_wkt(WKT[epsg]) if epsg in WKT else CRS.from_epsg(epsg)

# ─── Constants ────────────────────────────────────────────────────────────────
NODATA_OUT  = np.float32(-9999.0)
DEG_TO_RAD  = np.float32(np.pi / 180.0)
RAD_TO_DEG  = np.float32(180.0 / np.pi)

FACTOR_NAMES = [
    "elevation", "slope", "aspect", "plan_curvature", "profile_curvature",
    "tpi", "tri", "roughness", "valley_depth",
    "flow_direction", "flow_accumulation", "drainage_density",
    "twi", "spi", "ls_factor", "hand",
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt_size(path: Path) -> str:
    n = path.stat().st_size
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def elapsed(t0): return f"{time.perf_counter()-t0:.1f}s"

def to_gpu(arr: np.ndarray):
    """Send numpy array to GPU, preserving float32."""
    return cp.asarray(arr.astype(np.float32))

def to_cpu(arr) -> np.ndarray:
    """Retrieve array from GPU to CPU float32 numpy."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return arr.get().astype(np.float32)
    return np.asarray(arr, dtype=np.float32)

def mask_nodata(arr, nodata_val=NODATA_OUT):
    """Return array with NODATA replaced by nan (for display/stats)."""
    a = arr.astype(np.float32)
    a[a == nodata_val] = np.nan
    return a

# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_dem(dem_path: Path) -> tuple:
    """
    Load DEM GeoTIFF.
    Returns (dem_float32, profile, cell_size_x, cell_size_y, nodata_src).
    """
    with rasterio.open(dem_path) as src:
        profile  = src.profile.copy()
        nodata   = src.nodata
        dem      = src.read(1).astype(np.float32)
        res_x    = abs(src.transform.a)   # pixel width  in CRS units
        res_y    = abs(src.transform.e)   # pixel height in CRS units

    log.info("  DEM loaded: %d × %d px  |  res: %.2f m  |  nodata: %s",
             dem.shape[1], dem.shape[0], res_x, nodata)

    # Mask nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan

    # Fill nodata via rasterio interpolation (small gaps only)
    nodata_mask = np.isnan(dem).astype(np.uint8)
    if nodata_mask.sum() > 0:
        log.info("  Filling %d nodata pixels …", nodata_mask.sum())
        dem = fillnodata(dem, mask=(~np.isnan(dem)).astype(np.uint8),
                         max_search_distance=100, smoothing_iterations=0)
        dem[np.isnan(dem)] = 0.0   # any remaining → sea level

    return dem.astype(np.float32), profile, float(res_x), float(res_y)


def save_factor(arr: np.ndarray, name: str, profile: dict,
                out_dir: Path) -> Path:
    """Write a float32 factor array as a compressed GeoTIFF."""
    out_path = out_dir / f"{name}.tif"
    p = profile.copy()
    p.update({
        "dtype":     "float32",
        "count":     1,
        "nodata":    float(NODATA_OUT),
        "compress":  "lzw",
        "predictor": 2,
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff":   "IF_SAFER",
    })
    out = arr.copy()
    out[~np.isfinite(out)] = float(NODATA_OUT)
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(out.astype(np.float32), 1)
    return out_path


# ─── GPU-accelerated terrain factors ─────────────────────────────────────────

class TerrainProcessor:
    """
    Computes terrain factors.
    If use_gpu=True and CuPy is available, all convolution/element-wise
    operations run on GPU. Falls back to scipy/numpy transparently.
    """

    def __init__(self, dem: np.ndarray, res_x: float, res_y: float,
                 use_gpu: bool = True, window: int = 3):
        self.res_x  = res_x
        self.res_y  = res_y
        self.window = window
        self.use_gpu = use_gpu and _GPU_AVAILABLE

        if self.use_gpu:
            self._xp    = cp
            self._dem   = to_gpu(dem)
            self._uf    = cu_uniform_filter
            self._maxf  = cu_maximum_filter
            self._minf  = cu_minimum_filter
            log.info("  TerrainProcessor: GPU (CuPy)")
        else:
            self._xp    = np
            self._dem   = dem.astype(np.float32)
            self._uf    = uniform_filter
            self._maxf  = maximum_filter
            self._minf  = minimum_filter
            log.info("  TerrainProcessor: CPU (scipy/numpy)")

    def _sobel(self):
        """
        Horn (1981) 3×3 Sobel-based gradient.
        Returns (dzdx, dzdy) — partial derivatives in x and y.
        """
        xp = self._xp
        d  = self._dem
        if self.use_gpu:
            from cupyx.scipy.ndimage import sobel as cu_sobel
            dzdx = cu_sobel(d, axis=1) / (8.0 * self.res_x)
            dzdy = cu_sobel(d, axis=0) / (8.0 * self.res_y)
        else:
            dzdx = sobel(d, axis=1) / (8.0 * self.res_x)
            dzdy = sobel(d, axis=0) / (8.0 * self.res_y)
        return dzdx, dzdy

    # ── 01 Elevation ──────────────────────────────────────────────────────

    def elevation(self) -> np.ndarray:
        return to_cpu(self._dem)

    # ── 02 Slope ──────────────────────────────────────────────────────────

    def slope(self) -> np.ndarray:
        """Slope in degrees (0–90)."""
        xp = self._xp
        dzdx, dzdy = self._sobel()
        s = xp.arctan(xp.sqrt(dzdx**2 + dzdy**2)) * float(RAD_TO_DEG)
        return to_cpu(s)

    # ── 03 Aspect ─────────────────────────────────────────────────────────

    def aspect(self) -> np.ndarray:
        """
        Aspect in degrees clockwise from North (0–360).
        Flat areas (slope < 0.001°) encoded as -1.
        """
        xp = self._xp
        dzdx, dzdy = self._sobel()
        # atan2 gives angle from East, CCW → convert to bearing from North
        asp = xp.arctan2(-dzdy, dzdx) * float(RAD_TO_DEG)
        asp = xp.where(asp < 0, asp + 360.0, asp)
        # Bearing from North (clockwise): 90 − angle
        bearing = 90.0 - asp
        bearing = xp.where(bearing < 0, bearing + 360.0, bearing)
        # Flat mask
        slope_mag = xp.sqrt(dzdx**2 + dzdy**2) * float(RAD_TO_DEG)
        bearing   = xp.where(slope_mag < 0.001, xp.float32(-1.0), bearing)
        return to_cpu(bearing)

    # ── 04 & 05 Plan / Profile Curvature ──────────────────────────────────

    def curvature(self) -> tuple:
        """
        Plan and profile curvature using Zevenbergen & Thorne (1987).
        Returns (plan_curv, profile_curv) both in 1/100 m units.

        Positive plan_curv  → converging flow (valley)
        Negative plan_curv  → diverging flow  (ridge)
        Positive prof_curv  → decelerating flow (concave up)
        Negative prof_curv  → accelerating flow (convex up)
        """
        xp  = self._xp
        z   = self._dem
        cs  = float(self.res_x)  # assuming square pixels
        cs2 = cs * cs

        # 3×3 neighbourhood labels (Zevenbergen & Thorne notation)
        p  = xp.roll(z, -1, axis=1); q = xp.roll(z, 1, axis=1)
        r  = xp.roll(z, -1, axis=0); s = xp.roll(z, 1, axis=0)
        t  = xp.roll(xp.roll(z, -1, axis=0), -1, axis=1)
        u  = xp.roll(xp.roll(z, -1, axis=0),  1, axis=1)
        v  = xp.roll(xp.roll(z,  1, axis=0), -1, axis=1)
        w  = xp.roll(xp.roll(z,  1, axis=0),  1, axis=1)

        # Second-order coefficients
        D = (p + q - 2*z) / (2 * cs2)
        E = (r + s - 2*z) / (2 * cs2)
        F = (-t + u + v - w) / (4 * cs2)
        G = (-p + q)  / (2 * cs)
        H = (-r + s)  / (2 * cs)

        denom = G*G + H*H
        safe  = xp.where(denom < 1e-10, xp.float32(1e-10), denom)

        plan_curv    = -2.0 * (D * H*H - F * G * H + E * G*G) / safe
        profile_curv =  2.0 * (D * G*G + F * G * H + E * H*H) / safe

        # Edges are unreliable (roll wraps) → zero them out
        plan_curv[:1,:]=plan_curv[-1:,:]=plan_curv[:,:1]=plan_curv[:,-1:]=0
        profile_curv[:1,:]=profile_curv[-1:,:]=profile_curv[:,:1]=profile_curv[:,-1:]=0

        return to_cpu(plan_curv), to_cpu(profile_curv)

    # ── 06 TPI ────────────────────────────────────────────────────────────

    def tpi(self, window: int = None) -> np.ndarray:
        """
        Topographic Position Index.
        TPI = Z - mean(Z in window).  Positive → ridge, negative → valley.
        """
        w   = window or self.window
        avg = self._uf(self._dem, size=w)
        return to_cpu(self._dem - avg)

    # ── 07 TRI ────────────────────────────────────────────────────────────

    def tri(self) -> np.ndarray:
        """
        Terrain Ruggedness Index (Riley et al. 1999).
        TRI = sqrt(sum of squared diff with 8 neighbours).
        """
        xp  = self._xp
        z   = self._dem
        pad = xp.pad(z, 1, mode='edge')

        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        sq_sum  = xp.zeros_like(z)
        for dr, dc in offsets:
            r0, r1 = 1+dr, pad.shape[0]-1+dr
            c0, c1 = 1+dc, pad.shape[1]-1+dc
            sq_sum += (pad[r0:r1, c0:c1] - z) ** 2

        return to_cpu(xp.sqrt(sq_sum))

    # ── 08 Roughness ──────────────────────────────────────────────────────

    def roughness(self, window: int = None) -> np.ndarray:
        """Roughness = max(Z) − min(Z) in local window."""
        w    = window or self.window
        rmax = self._maxf(self._dem, size=w)
        rmin = self._minf(self._dem, size=w)
        return to_cpu(rmax - rmin)

    # ── 09 Valley depth ───────────────────────────────────────────────────

    def valley_depth(self, large_window: int = 51) -> np.ndarray:
        """
        Valley depth = large-window TPI interpreted as depth below ridge.
        Uses a larger window (default 51 px ≈ 1.5 km at 30 m res) to
        capture regional ridge–valley contrast.
        Positive values → below ridge (valley), negative → above (ridge).
        """
        avg  = self._uf(self._dem, size=large_window)
        vd   = to_cpu(avg - self._dem)   # ridge above → positive depth
        return vd


# ─── CPU flow routing (pysheds D8) ───────────────────────────────────────────

class FlowRouter:
    """
    D8 flow routing via pysheds.
    Produces: flow_direction, flow_accumulation.
    """

    def __init__(self, dem: np.ndarray, profile: dict, dem_path: Path,
                 drain_thresh: int = 500):
        self.dem         = dem
        self.profile     = profile
        self.dem_path    = dem_path
        self.drain_thresh = drain_thresh
        self._fdir       = None
        self._facc       = None
        self._grid       = None

    def _build(self):
        """Run D8 routing (called once, results cached)."""
        if not _PYSHEDS_AVAILABLE:
            log.warning("pysheds not installed — flow-based factors skipped.")
            log.warning("Install: pip install pysheds")
            return False

        log.info("  D8 flow routing (pysheds) …")
        t = time.perf_counter()

        grid = PyshedsGrid.from_raster(str(self.dem_path))
        dem  = grid.read_raster(str(self.dem_path))

        # Condition DEM: fill pits → resolve flats
        pit_filled   = grid.fill_pits(dem)
        flooded      = grid.fill_depressions(pit_filled)
        inflated     = grid.resolve_flats(flooded)

        # D8 flow direction and accumulation
        fdir = grid.flowdir(inflated)
        facc = grid.accumulation(fdir)

        self._grid = grid
        self._fdir = np.array(fdir).astype(np.float32)
        self._facc = np.array(facc).astype(np.float32)

        log.info("  Flow routing done in %s", elapsed(t))
        return True

    def flow_direction(self) -> np.ndarray:
        if self._fdir is None: self._build()
        return self._fdir if self._fdir is not None else np.full(self.dem.shape, NODATA_OUT)

    def flow_accumulation(self) -> np.ndarray:
        if self._facc is None: self._build()
        return self._facc if self._facc is not None else np.full(self.dem.shape, NODATA_OUT)

    def drainage_density(self, use_gpu: bool = False) -> np.ndarray:
        """
        Drainage density = fraction of drainage cells in a local window.
        Drainage network = cells where flow_acc ≥ drain_thresh.
        """
        facc = self.flow_accumulation()
        if np.all(facc == NODATA_OUT):
            return np.full(self.dem.shape, NODATA_OUT)

        network = (facc >= self.drain_thresh).astype(np.float32)

        if use_gpu and _GPU_AVAILABLE:
            net_gpu = cp.asarray(network)
            dens    = cu_uniform_filter(net_gpu, size=11)
            return to_cpu(dens)
        else:
            return uniform_filter(network, size=11).astype(np.float32)


# ─── Composite hydrological factors (GPU element-wise after flow routing) ─────

def compute_twi(flow_acc: np.ndarray, slope_deg: np.ndarray,
                res_x: float, use_gpu: bool) -> np.ndarray:
    """
    Topographic Wetness Index: TWI = ln(a / tan β)
    where a = specific catchment area = flow_acc × cell_area / contour_length.
    """
    xp = (cp if use_gpu and _GPU_AVAILABLE else np)

    fa   = xp.asarray(flow_acc.astype(np.float32))
    sl   = xp.asarray(slope_deg.astype(np.float32))

    cell_area = float(res_x) ** 2
    # Specific catchment area (m²/m): (cells + 1) × cell_area / res_x
    sca      = (fa + 1.0) * cell_area / float(res_x)
    slope_rad = sl * float(DEG_TO_RAD)
    tan_beta  = xp.tan(slope_rad)
    tan_beta  = xp.where(tan_beta < 0.001, xp.float32(0.001), tan_beta)

    twi = xp.log(sca / tan_beta)
    twi = xp.where(xp.isfinite(twi), twi, xp.float32(NODATA_OUT))
    return to_cpu(twi)


def compute_spi(flow_acc: np.ndarray, slope_deg: np.ndarray,
                res_x: float, use_gpu: bool) -> np.ndarray:
    """Stream Power Index: SPI = As × tan β"""
    xp = (cp if use_gpu and _GPU_AVAILABLE else np)

    fa   = xp.asarray(flow_acc.astype(np.float32))
    sl   = xp.asarray(slope_deg.astype(np.float32))

    cell_area = float(res_x) ** 2
    sca       = (fa + 1.0) * cell_area / float(res_x)
    tan_beta  = xp.tan(sl * float(DEG_TO_RAD))

    spi = sca * tan_beta
    spi = xp.where(xp.isfinite(spi), spi, xp.float32(NODATA_OUT))
    return to_cpu(spi)


def compute_ls(flow_acc: np.ndarray, slope_deg: np.ndarray,
               res_x: float, use_gpu: bool) -> np.ndarray:
    """
    RUSLE LS factor (Moore & Burch 1986):
    LS = (As/22.13)^0.4 × (sin β / 0.0896)^1.3
    """
    xp = (cp if use_gpu and _GPU_AVAILABLE else np)

    fa   = xp.asarray(flow_acc.astype(np.float32))
    sl   = xp.asarray(slope_deg.astype(np.float32))

    cell_area = float(res_x) ** 2
    sca       = (fa + 1.0) * cell_area / float(res_x)
    sin_beta  = xp.sin(sl * float(DEG_TO_RAD))
    sin_beta  = xp.where(sin_beta < 0.0001, xp.float32(0.0001), sin_beta)

    ls = (sca / 22.13) ** 0.4 * (sin_beta / 0.0896) ** 1.3
    ls = xp.where(xp.isfinite(ls), ls, xp.float32(NODATA_OUT))
    return to_cpu(ls)


# ─── HAND (Height Above Nearest Drainage) ────────────────────────────────────

def compute_hand(dem: np.ndarray, flow_acc: np.ndarray,
                 drain_thresh: int) -> np.ndarray:
    """
    Height Above Nearest Drainage (Nobre et al. 2011).
    For each cell: find nearest drainage cell (flow_acc ≥ thresh),
    HAND = elevation − elevation(nearest drainage cell).

    Uses a fast BFS / distance-transform approach.
    """
    if np.all(flow_acc == NODATA_OUT):
        return np.full(dem.shape, NODATA_OUT, dtype=np.float32)

    from scipy.ndimage import distance_transform_edt

    drainage = (flow_acc >= drain_thresh).astype(bool)

    if not drainage.any():
        log.warning("  No drainage cells at threshold %d — lowering to 1%%ile",
                    drain_thresh)
        thresh_low = np.nanpercentile(flow_acc[flow_acc > 0], 1)
        drainage   = (flow_acc >= thresh_low).astype(bool)

    # For each non-drainage cell find the index of the nearest drainage cell
    # distance_transform_edt with indices=True returns feature index arrays
    _, (row_idx, col_idx) = distance_transform_edt(
        ~drainage, return_indices=True
    )

    # Elevation of nearest drainage cell for every pixel
    drain_elev = dem[row_idx, col_idx]
    hand       = (dem - drain_elev).astype(np.float32)
    hand       = np.where(np.isfinite(hand), hand, NODATA_OUT)
    return hand


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(args):
    dem_path   = Path(args.dem)
    output_dir = Path(args.output_dir)
    factor_dir = output_dir / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)

    use_gpu    = (not args.force_cpu) and _GPU_AVAILABLE
    if not _GPU_AVAILABLE and not args.force_cpu:
        log.warning("CuPy not found — running CPU fallback.")
        log.warning("GPU install: conda install -c rapidsai cupy")
    if use_gpu:
        try:
            mem_free, mem_total = cp.cuda.Device().mem_info
            log.info("GPU: %s | %.1f / %.1f GB free",
                     cp.cuda.Device().name, mem_free/1e9, mem_total/1e9)
        except Exception:
            pass

    log.info("═" * 64)
    log.info("DEM Conditioning Factors  —  FSM Paper 1")
    log.info("  DEM       : %s", dem_path)
    log.info("  Output    : %s", output_dir)
    log.info("  GPU       : %s", "enabled" if use_gpu else "CPU fallback")
    log.info("  Window    : %d px", args.window)
    log.info("  Drain thr : %d cells", args.drain_thresh)
    log.info("═" * 64)

    # ── 0. Load DEM ───────────────────────────────────────────────────────
    log.info("")
    log.info("── Loading DEM …")
    t0 = time.perf_counter()
    dem, profile, res_x, res_y = load_dem(dem_path)
    log.info("  Done %s", elapsed(t0))

    results: dict[str, np.ndarray] = {}

    # ── 1. GPU terrain factors ────────────────────────────────────────────
    log.info("")
    log.info("── Terrain morphology factors (GPU=%s) …", use_gpu)
    t_gpu = time.perf_counter()

    tp = TerrainProcessor(dem, res_x, res_y, use_gpu=use_gpu, window=args.window)

    log.info("  elevation …");         t=time.perf_counter()
    results["elevation"]          = tp.elevation()
    log.info("    %s", elapsed(t))

    log.info("  slope …");             t=time.perf_counter()
    slope_arr = tp.slope()
    results["slope"]              = slope_arr
    log.info("    %s", elapsed(t))

    log.info("  aspect …");            t=time.perf_counter()
    results["aspect"]             = tp.aspect()
    log.info("    %s", elapsed(t))

    log.info("  curvature (plan + profile) …"); t=time.perf_counter()
    plan_curv, prof_curv          = tp.curvature()
    results["plan_curvature"]     = plan_curv
    results["profile_curvature"]  = prof_curv
    log.info("    %s", elapsed(t))

    log.info("  TPI …");               t=time.perf_counter()
    results["tpi"]                = tp.tpi()
    log.info("    %s", elapsed(t))

    log.info("  TRI …");               t=time.perf_counter()
    results["tri"]                = tp.tri()
    log.info("    %s", elapsed(t))

    log.info("  roughness …");         t=time.perf_counter()
    results["roughness"]          = tp.roughness()
    log.info("    %s", elapsed(t))

    log.info("  valley depth …");      t=time.perf_counter()
    results["valley_depth"]       = tp.valley_depth(large_window=args.valley_window)
    log.info("    %s", elapsed(t))

    log.info("  Terrain factors total: %s", elapsed(t_gpu))

    # ── 2. CPU flow routing ───────────────────────────────────────────────
    log.info("")
    log.info("── Flow routing (D8, CPU) …")
    t_flow = time.perf_counter()

    router = FlowRouter(dem, profile, dem_path, drain_thresh=args.drain_thresh)

    log.info("  flow direction …");    t=time.perf_counter()
    fdir = router.flow_direction()
    results["flow_direction"]     = fdir
    log.info("    %s", elapsed(t))

    log.info("  flow accumulation …"); t=time.perf_counter()
    facc = router.flow_accumulation()
    results["flow_accumulation"]  = facc
    log.info("    %s", elapsed(t))

    log.info("  drainage density …");  t=time.perf_counter()
    results["drainage_density"]   = router.drainage_density(use_gpu=use_gpu)
    log.info("    %s", elapsed(t))

    log.info("  Flow routing total: %s", elapsed(t_flow))

    # ── 3. Composite factors (GPU element-wise) ───────────────────────────
    log.info("")
    log.info("── Composite hydrological factors (GPU=%s) …", use_gpu)

    log.info("  TWI …");               t=time.perf_counter()
    results["twi"]                = compute_twi(facc, slope_arr, res_x, use_gpu)
    log.info("    %s", elapsed(t))

    log.info("  SPI …");               t=time.perf_counter()
    results["spi"]                = compute_spi(facc, slope_arr, res_x, use_gpu)
    log.info("    %s", elapsed(t))

    log.info("  LS factor …");         t=time.perf_counter()
    results["ls_factor"]          = compute_ls(facc, slope_arr, res_x, use_gpu)
    log.info("    %s", elapsed(t))

    # ── 4. HAND ───────────────────────────────────────────────────────────
    log.info("")
    log.info("── HAND …")
    t=time.perf_counter()
    results["hand"]               = compute_hand(dem, facc, args.drain_thresh)
    log.info("    %s", elapsed(t))

    # ── 5. Save per-factor GeoTIFFs ───────────────────────────────────────
    log.info("")
    log.info("── Saving per-factor GeoTIFFs …")
    t_save = time.perf_counter()

    saved_paths = {}
    for name in FACTOR_NAMES:
        if name not in results:
            log.warning("  Missing factor: %s", name)
            continue
        path = save_factor(results[name], name, profile, factor_dir)
        saved_paths[name] = path
        log.info("  %-22s  →  %s", name, fmt_size(path))

    log.info("  Saved %d factors in %s", len(saved_paths), elapsed(t_save))

    # ── 6. Stacked outputs ────────────────────────────────────────────────
    if not args.skip_stack:
        log.info("")
        log.info("── Building stacked outputs …")
        _save_stack(results, profile, output_dir, use_gpu)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 64)
    log.info("All factors complete.")
    log.info("  Per-factor GeoTIFFs : %s", factor_dir)
    log.info("  Stacked GeoTIFF     : %s/stack.tif", output_dir)
    log.info("  NumPy array         : %s/stack.npz", output_dir)
    log.info("═" * 64)


def _save_stack(results: dict, profile: dict, output_dir: Path,
                use_gpu: bool):
    """
    Save all factors as:
      (a) multi-band GeoTIFF  — convenient for QGIS / rasterio slicing
      (b) .npz array          — fast numpy load for ML pipelines
    """
    bands   = [name for name in FACTOR_NAMES if name in results]
    n_bands = len(bands)
    H, W    = next(iter(results.values())).shape

    # ── Multi-band GeoTIFF ────────────────────────────────────────────
    stack_path = output_dir / "stack.tif"
    p = profile.copy()
    p.update({
        "dtype":      "float32",
        "count":      n_bands,
        "nodata":     float(NODATA_OUT),
        "compress":   "lzw",
        "predictor":  2,
        "tiled":      True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff":    "IF_SAFER",
    })
    t = time.perf_counter()
    with rasterio.open(stack_path, "w", **p) as dst:
        for i, name in enumerate(bands, 1):
            arr = results[name].copy()
            arr[~np.isfinite(arr)] = float(NODATA_OUT)
            dst.write(arr.astype(np.float32), i)
            dst.update_tags(i, factor=name)

    log.info("  stack.tif  →  %s  (%d bands, %s)", stack_path,
             n_bands, fmt_size(stack_path))

    # ── NumPy .npz  ───────────────────────────────────────────────────
    npz_path = output_dir / "stack.npz"

    # Optionally normalise on GPU before saving
    if use_gpu and _GPU_AVAILABLE:
        log.info("  Normalising stack on GPU (min-max per band) …")
        norm_bands = {}
        for name in bands:
            arr = cp.asarray(results[name])
            valid = arr[arr != float(NODATA_OUT)]
            if len(valid) > 0:
                lo, hi = float(valid.min()), float(valid.max())
                if hi > lo:
                    arr_n = cp.where(arr == float(NODATA_OUT),
                                     cp.float32(NODATA_OUT),
                                     (arr - lo) / (hi - lo))
                else:
                    arr_n = cp.zeros_like(arr)
            else:
                arr_n = arr
            norm_bands[name] = to_cpu(arr_n)
    else:
        norm_bands = results

    stack_np = np.stack(
        [norm_bands[n].astype(np.float32) for n in bands], axis=0
    )  # shape: (C, H, W)

    np.savez_compressed(
        str(npz_path),
        data         = stack_np,
        factor_names = np.array(bands),
        nodata       = np.float32(NODATA_OUT),
        transform    = np.array(profile.get("transform", [])),
        crs          = np.array([str(profile.get("crs", ""))]),
    )
    log.info("  stack.npz  →  %s  shape=%s  %s",
             npz_path, stack_np.shape, fmt_size(npz_path))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GPU-accelerated DEM conditioning factors for flood susceptibility (FSM Paper 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Factors produced (16 total):
  Terrain  : elevation, slope, aspect, plan_curvature, profile_curvature,
             tpi, tri, roughness, valley_depth
  Flow     : flow_direction, flow_accumulation, drainage_density
  Composite: twi, spi, ls_factor
  HAND     : hand

Install requirements (rapids_env):
  pip install pysheds rasterio           # if not already present
  conda install -c rapidsai cupy         # GPU (already in rapids_env)

Examples:
  python dem_conditioning_factors.py --dem /path/to/khagaria_dem.tif
  python dem_conditioning_factors.py --dem /path/to/dem.tif --force-cpu
  python dem_conditioning_factors.py --dem /path/to/dem.tif --drain-thresh 1000
  python dem_conditioning_factors.py --dem /path/to/dem.tif --window 5 --valley-window 101
        """
    )
    p.add_argument("--dem",
                   default="/home/arora/Documents/FSM_Slovenia/Processed_Data/Rasters/DEM_Filled_Slovenia.tif",
                   #default="/home/arora/Documents/FSM_Slovenia/GeoTIFF/DMV0125/DMV0125_EPSG3794_merged.tif",
                   help="Input DEM GeoTIFF path")
    p.add_argument("--output-dir",
                   default="/home/arora/Documents/FSM_Slovenia/Processed_Data/Rasters/DEM_Factors",
                   help="Output directory")
    p.add_argument("--force-cpu",      action="store_true",
                   help="Disable GPU even if CuPy is available (for benchmarking)")
    p.add_argument("--window",         type=int, default=3,
                   help="Local window size in pixels for TPI/TRI/roughness (default: 3)")
    p.add_argument("--valley-window",  type=int, default=51,
                   help="Window size for valley depth large-scale TPI (default: 51)")
    p.add_argument("--drain-thresh",   type=int, default=500,
                   help="Flow accumulation threshold to define drainage network (default: 500)")
    p.add_argument("--skip-stack",     action="store_true",
                   help="Skip stacked GeoTIFF and .npz outputs")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())