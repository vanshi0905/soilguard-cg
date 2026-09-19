"""
SoilGuard-CG: SCORPAN Digital Elevation Model (DEM) & Terrain Covariates Module
================================================================================
Implements:
1. Ingestion of 30m Digital Elevation Model (SRTM / Cartosat DEM) covering
   Chhattisgarh agricultural plains.
2. Realistic, deterministic synthesis of Raipur plain 30m topography matching
   the Mahanadi/Kharun river basin (mean elevation ~280m, elevation range
   240m-380m, natural dendritic drainage networks).
3. Bilinear / bicubic geospatial resampling to the exact 10m Sentinel-2 grid
   (EPSG:32644, shape 2223x2086, affine (10, 0, 562232.68, 0, -10, 2355560.49)).
4. 2nd-order finite difference / Sobel gradient derivation of Slope in degrees:
   slope = arctan(sqrt(p^2 + q^2)) * (180 / pi).
5. Topographic Wetness Index (TWI):
   TWI = ln(a / (tan(beta) + 10^-4)) with multi-flow accumulation proxy.
6. Profile Curvature (curvature along the line of steepest slope).
7. Primary export function: compute_terrain_covariates(dem_raster, cell_size=10.0).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, reproject
from scipy.ndimage import convolve, gaussian_filter, uniform_filter

# Ensure project paths are resolved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (SCRIPT_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Canonical 10m Sentinel-2 Raipur Golden Reference Grid Specifications
TARGET_CRS = "EPSG:32644"
TARGET_HEIGHT = 2223
TARGET_WIDTH = 2086
TARGET_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH)
CELL_SIZE_10M = 10.0

# Bounding box coordinates in EPSG:32644 (UTM Zone 44N)
LEFT_BOUND = 562232.6789541794
BOTTOM_BOUND = 2333330.486155833
RIGHT_BOUND = 583092.6789541794
TOP_BOUND = 2355560.486155833
TARGET_BOUNDS = (LEFT_BOUND, BOTTOM_BOUND, RIGHT_BOUND, TOP_BOUND)

TARGET_AFFINE = Affine(10.0, 0.0, LEFT_BOUND, 0.0, -10.0, TOP_BOUND)

# Default paths
GOLDEN_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "golden")
DEFAULT_DEM_GOLDEN_PATH = os.path.join(GOLDEN_DATA_DIR, "dem_raipur_golden.tif")


# ==============================================================================
# TERRAIN COVARIATE DERIVATION MATHEMATICS
# ==============================================================================

def compute_slope(
    dem_raster: np.ndarray,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    Computes topographic slope in degrees using 2nd-order finite difference /
    Sobel gradient (Horn's method):
        p = dz/dx, q = dz/dy
        slope_deg = arctan(sqrt(p^2 + q^2)) * (180.0 / pi)

    Parameters
    ----------
    dem_raster : np.ndarray
        2D elevation array.
    cell_size : float, default=10.0
        Grid cell resolution in meters.

    Returns
    -------
    np.ndarray
        Slope in degrees, float32, in range [0.0, 90.0].
    """
    dem = np.asarray(dem_raster, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError(f"dem_raster must be 2-dimensional, got shape {dem.shape}")
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")

    # Horn's 2nd-order weighted finite difference kernels
    kx = np.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]], dtype=np.float32) / (8.0 * cell_size)
    ky = np.array([[ 1.0,  2.0,  1.0],
                   [ 0.0,  0.0,  0.0],
                   [-1.0, -2.0, -1.0]], dtype=np.float32) / (8.0 * cell_size)

    p = convolve(dem, kx, mode="nearest")
    q = convolve(dem, ky, mode="nearest")

    # Boundary replication to eliminate border convolution attenuation
    if dem.shape[0] > 2 and dem.shape[1] > 2:
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

        q[:, 0] = q[:, 1]
        q[:, -1] = q[:, -2]
        q[0, :] = q[1, :]
        q[-1, :] = q[-2, :]

    grad_mag = np.sqrt(p**2 + q**2)
    slope_rad = np.arctan(grad_mag)
    slope_deg = slope_rad * (180.0 / np.pi)

    # In natural terrain, slope is non-negative and bounded
    slope_deg = np.clip(slope_deg, 0.0, 89.99).astype(np.float32)
    return slope_deg


def compute_twi(
    dem_raster: np.ndarray,
    cell_size: float = 10.0,
    slope_rad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes Topographic Wetness Index (TWI):
        TWI = ln(a / (tan(beta) + 10^-4))
    where:
        a = specific catchment area per unit contour width (multi-flow accumulation proxy)
        beta = slope in radians

    Parameters
    ----------
    dem_raster : np.ndarray
        2D elevation array.
    cell_size : float, default=10.0
        Grid cell resolution in meters.
    slope_rad : np.ndarray, optional
        Precomputed slope in radians. If None, computed internally.

    Returns
    -------
    np.ndarray
        TWI values, float32, strictly bounded in [2.0, 20.0].
    """
    dem = np.asarray(dem_raster, dtype=np.float32)
    ny, nx = dem.shape

    if slope_rad is None:
        kx = np.array([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], dtype=np.float32) / (8.0 * cell_size)
        ky = np.array([[ 1.0,  2.0,  1.0],
                       [ 0.0,  0.0,  0.0],
                       [-1.0, -2.0, -1.0]], dtype=np.float32) / (8.0 * cell_size)
        p = convolve(dem, kx, mode="nearest")
        q = convolve(dem, ky, mode="nearest")
        if ny > 2 and nx > 2:
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            q[:, 0] = q[:, 1]
            q[:, -1] = q[:, -2]
            q[0, :] = q[1, :]
            q[-1, :] = q[-2, :]
        slope_rad = np.arctan(np.sqrt(p**2 + q**2))

    tan_beta = np.tan(slope_rad)

    # Multi-scale specific catchment area proxy (a)
    # Baseline unit contour width = cell_size (m)
    a_proxy = np.full_like(dem, fill_value=cell_size, dtype=np.float32)

    min_dim = min(ny, nx)
    std_dem = float(np.nanstd(dem))
    norm_scale = max(std_dem, 1.0)

    # Multi-scale hierarchical aggregation of upslope catchment convergence
    # Scales correspond to 30m, 90m, 270m, 810m, 2430m on a 10m grid
    scale_steps = [3, 9, 27, 81, 243]
    valid_scales = [sc for sc in scale_steps if sc <= min_dim]

    for sc in valid_scales:
        local_mean = uniform_filter(dem, size=sc, mode="nearest")
        # Depressions/valleys lower than local neighborhood accumulate flow
        depr = np.maximum(0.0, local_mean - dem)
        weight = float(sc * cell_size)
        a_proxy += weight * (depr / norm_scale)

    # In natural catchment hydrology, TWI = ln(a / (tan(beta) + 10^-4))
    denom = tan_beta + 1e-4
    twi = np.log(a_proxy / denom)

    # Ensure finite, realistic bounds [2.0, 20.0] as mandated by scientific specs
    twi = np.clip(twi, 2.0, 20.0).astype(np.float32)
    return twi


def compute_profile_curvature(
    dem_raster: np.ndarray,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    Computes Profile Curvature: vertical surface curvature along the line of
    steepest slope (flow direction):
        k_prof = - (p^2 * r + 2 * p * q * s + q^2 * t) / ((p^2 + q^2) * (1 + p^2 + q^2)^(3/2))

    Parameters
    ----------
    dem_raster : np.ndarray
        2D elevation array.
    cell_size : float, default=10.0
        Grid cell resolution in meters.

    Returns
    -------
    np.ndarray
        Profile curvature in m^-1, float32, bounded in [-1.0, 1.0].
    """
    dem = np.asarray(dem_raster, dtype=np.float32)
    ny, nx = dem.shape

    # First derivatives
    kx = np.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]], dtype=np.float32) / (8.0 * cell_size)
    ky = np.array([[ 1.0,  2.0,  1.0],
                   [ 0.0,  0.0,  0.0],
                   [-1.0, -2.0, -1.0]], dtype=np.float32) / (8.0 * cell_size)

    # Second derivatives
    kr = np.array([[0.0,  0.0, 0.0],
                   [1.0, -2.0, 1.0],
                   [0.0,  0.0, 0.0]], dtype=np.float32) / (cell_size**2)
    kt = np.array([[0.0,  1.0, 0.0],
                   [0.0, -2.0, 0.0],
                   [0.0,  1.0, 0.0]], dtype=np.float32) / (cell_size**2)
    ks = np.array([[-1.0, 0.0,  1.0],
                   [ 0.0, 0.0,  0.0],
                   [ 1.0, 0.0, -1.0]], dtype=np.float32) / (4.0 * cell_size**2)

    p = convolve(dem, kx, mode="nearest")
    q = convolve(dem, ky, mode="nearest")
    r = convolve(dem, kr, mode="nearest")
    t = convolve(dem, kt, mode="nearest")
    s = convolve(dem, ks, mode="nearest")

    # Boundary handling
    if ny > 2 and nx > 2:
        for arr in (p, q, r, t, s):
            arr[:, 0] = arr[:, 1]
            arr[:, -1] = arr[:, -2]
            arr[0, :] = arr[1, :]
            arr[-1, :] = arr[-2, :]

    p2 = p**2
    q2 = q**2
    pq = p * q
    grad2 = p2 + q2

    denom = grad2 * ((1.0 + grad2) ** 1.5)
    mask = grad2 > 1e-10

    prof_curv = np.zeros_like(dem, dtype=np.float32)
    numerator = -(p2[mask] * r[mask] + 2.0 * pq[mask] * s[mask] + q2[mask] * t[mask])
    prof_curv[mask] = numerator / (denom[mask] + 1e-12)

    # Clip to realistic physical geomorphometric bounds [-1.0, 1.0]
    prof_curv = np.clip(prof_curv, -1.0, 1.0).astype(np.float32)
    return prof_curv


def compute_terrain_covariates(
    dem_raster: np.ndarray,
    cell_size: float = 10.0,
) -> Dict[str, np.ndarray]:
    """
    Primary interface contract for SCORPAN Relief (r) covariate derivation.
    Computes:
      - elevation: float32 2D array
      - slope_deg: float32 2D array (degrees)
      - twi: float32 2D array (Topographic Wetness Index)
      - profile_curvature: float32 2D array (vertical surface curvature)

    Parameters
    ----------
    dem_raster : np.ndarray
        2D digital elevation model array.
    cell_size : float, default=10.0
        Pixel size in meters.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
          'elevation', 'slope_deg', 'twi', 'profile_curvature'
    """
    dem = np.asarray(dem_raster, dtype=np.float32)
    if dem.ndim != 2:
        raise ValueError(f"dem_raster must be 2D array, got ndim={dem.ndim} with shape {dem.shape}")

    slope_d = compute_slope(dem, cell_size=cell_size)
    slope_r = np.deg2rad(slope_d)
    twi_layer = compute_twi(dem, cell_size=cell_size, slope_rad=slope_r)
    curv_layer = compute_profile_curvature(dem, cell_size=cell_size)

    return {
        "elevation": dem,
        "slope_deg": slope_d,
        "twi": twi_layer,
        "profile_curvature": curv_layer,
    }


# ==============================================================================
# DETERMINISTIC 30M DEM SYNTHESIS & INGESTION
# ==============================================================================

def generate_raipur_dem_30m(
    shape_30m: Tuple[int, int] = (741, 696),
    bounds: Tuple[float, float, float, float] = TARGET_BOUNDS,
    seed: int = 42,
) -> Tuple[np.ndarray, Affine]:
    """
    Synthesizes a realistic, deterministic 30m Digital Elevation Model (DEM)
    covering the Raipur agricultural plain in the central Chhattisgarh / Mahanadi basin.

    Topographic characteristics:
      - Regional north-bound gradient draining towards the Shivnath / Mahanadi confluence
      - Meandering Kharun river valley corridor (elevation ~245m - 258m)
      - Dendritic tributary incisions converging into the central drainage stem
      - Lateritic residual ridges / sandstone mounds (Chandi / Charmuria formations)
        reaching ~330m - 375m in the flanks
      - Multi-scale Gaussian Random Field terrain roughness
      - Mean elevation ~280m, elevation range strictly within [240m, 380m]

    Parameters
    ----------
    shape_30m : Tuple[int, int], default=(741, 696)
        Grid dimensions (height, width) at 30m resolution.
    bounds : Tuple[float, float, float, float]
        (min_x, min_y, max_x, max_y) in EPSG:32644.
    seed : int, default=42
        Random seed for 100% deterministic reproducibility.

    Returns
    -------
    dem_30m : np.ndarray
        2D float32 elevation array of shape shape_30m.
    transform_30m : Affine
        Affine geospatial transform for the 30m grid.
    """
    ny, nx = shape_30m
    left, bottom, right, top = bounds
    transform_30m = from_bounds(left, bottom, right, top, nx, ny)

    rng = np.random.RandomState(seed)
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]

    # Normalized spatial coordinates: u (West->East), vn (South->North)
    u = x_idx / (nx - 1.0)
    vn = 1.0 - (y_idx / (ny - 1.0))

    # 1. Regional baseline tilt: drains northward towards Mahanadi confluence
    dem = 285.5 - 18.0 * vn + 3.5 * (u - 0.5)

    # 2. Main meandering Kharun river corridor
    u_river = 0.34 + 0.05 * np.sin(2.0 * np.pi * 1.5 * vn) + 0.02 * np.sin(2.0 * np.pi * 3.7 * vn + 0.8)
    dist_river = np.abs(u - u_river)
    valley_main = -22.0 * np.exp(-((dist_river / 0.055) ** 2)) * (0.8 + 0.3 * vn)
    dem += valley_main

    # 3. Dendritic tributary incisions
    # Tributary 1 (Eastern feeder joining at vn=0.45)
    trib1_path = 0.45 + 0.5 * (u - u_river) + 0.08 * np.sin(2.0 * np.pi * 3.0 * u)
    dist_trib1 = np.where(u > u_river, np.abs(vn - trib1_path), 1.0)
    dem += -11.0 * np.exp(-((dist_trib1 / 0.025) ** 2)) * np.exp(-((np.maximum(0.0, u - u_river) / 0.4) ** 2))

    # Tributary 2 (Western feeder joining at vn=0.65)
    trib2_path = 0.65 - 0.4 * (u_river - u) + 0.04 * np.sin(2.0 * np.pi * 4.0 * u)
    dist_trib2 = np.where(u < u_river, np.abs(vn - trib2_path), 1.0)
    dem += -9.5 * np.exp(-((dist_trib2 / 0.02) ** 2)) * np.exp(-((np.maximum(0.0, u_river - u) / 0.3) ** 2))

    # Tributary 3 (Southern feeder joining at vn=0.20)
    trib3_path = 0.20 + 0.35 * (u - u_river)
    dist_trib3 = np.where(u > u_river, np.abs(vn - trib3_path), 1.0)
    dem += -8.5 * np.exp(-((dist_trib3 / 0.02) ** 2)) * np.exp(-((np.maximum(0.0, u - u_river) / 0.35) ** 2))

    # 4. Residual Raipur sandstone / laterite uplands
    dem += 88.0 * np.exp(-(((u - 0.86) / 0.11) ** 2 + ((vn - 0.12) / 0.12) ** 2))
    dem += 65.0 * np.exp(-(((u - 0.09) / 0.09) ** 2 + ((vn - 0.09) / 0.10) ** 2))
    dem += 35.0 * np.exp(-(((u - 0.90) / 0.13) ** 2 + ((vn - 0.82) / 0.14) ** 2))

    # 5. Multi-scale coherent Gaussian Random Field roughness
    noise1 = gaussian_filter(rng.randn(ny, nx), sigma=40.0) * 140.0
    noise2 = gaussian_filter(rng.randn(ny, nx), sigma=16.0) * 35.0
    noise3 = gaussian_filter(rng.randn(ny, nx), sigma=5.0) * 10.0
    dem += (noise1 + noise2 + noise3)

    # Ensure strict physical range [240.5, 379.5] with mean ~280m
    dem = np.clip(dem, 240.5, 379.5).astype(np.float32)

    return dem, transform_30m


def resample_dem(
    source_dem: np.ndarray,
    src_transform: Affine,
    src_crs: str = TARGET_CRS,
    target_shape: Tuple[int, int] = TARGET_SHAPE,
    target_transform: Affine = TARGET_AFFINE,
    target_crs: str = TARGET_CRS,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Resamples a 2D DEM array from source grid to target grid using bilinear
    or bicubic interpolation.

    Parameters
    ----------
    source_dem : np.ndarray
        Source DEM array (e.g. 30m resolution).
    src_transform : Affine
        Affine transformation matrix of the source DEM.
    src_crs : str, default='EPSG:32644'
        Coordinate Reference System of the source.
    target_shape : Tuple[int, int], default=(2223, 2086)
        Target grid shape (height, width).
    target_transform : Affine
        Target affine transformation.
    target_crs : str, default='EPSG:32644'
        Target CRS.
    method : str, default='bilinear'
        Interpolation method: 'bilinear' or 'bicubic' / 'cubic'.

    Returns
    -------
    np.ndarray
        Resampled 2D float32 elevation array of shape target_shape.
    """
    src_arr = np.asarray(source_dem, dtype=np.float32)
    dst_arr = np.empty(target_shape, dtype=np.float32)

    if method.lower() in ("bicubic", "cubic"):
        resampling_mode = Resampling.cubic
    elif method.lower() == "nearest":
        resampling_mode = Resampling.nearest
    else:
        resampling_mode = Resampling.bilinear

    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling_mode,
    )

    return dst_arr


def export_terrain_covariates_geotiff(
    output_path: str,
    covariates: Dict[str, np.ndarray],
    transform: Affine = TARGET_AFFINE,
    crs: str = TARGET_CRS,
) -> str:
    """
    Exports terrain covariates to a multi-band compressed GeoTIFF.
    Bands:
      1: elevation (m)
      2: slope_deg (degrees)
      3: twi (dimensionless)
      4: profile_curvature (m^-1)

    Parameters
    ----------
    output_path : str
        Target file path.
    covariates : Dict[str, np.ndarray]
        Covariate dictionary containing 'elevation', 'slope_deg', 'twi', 'profile_curvature'.
    transform : Affine
        Raster geospatial transform.
    crs : str
        Spatial reference system.

    Returns
    -------
    str
        Absolute path to the created GeoTIFF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    band_names = ["elevation", "slope_deg", "twi", "profile_curvature"]
    ny, nx = covariates["elevation"].shape

    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": len(band_names),
        "dtype": rasterio.float32,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": None,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, key in enumerate(band_names, start=1):
            arr = covariates[key].astype(np.float32)
            dst.write(arr, idx)
            dst.set_band_description(idx, key)

    return os.path.abspath(output_path)


def load_or_generate_dem(
    dem_path: Optional[str] = None,
    target_shape: Tuple[int, int] = TARGET_SHAPE,
    target_affine: Affine = TARGET_AFFINE,
    target_crs: str = TARGET_CRS,
    resampling_method: str = "bilinear",
    save_golden: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Ingests an existing DEM GeoTIFF or generates the deterministic 30m Raipur DEM
    and resamples it to the exact 10m Sentinel-2 reference grid.

    If dem_path is None or points to a non-existent file:
      1. Checks if DEFAULT_DEM_GOLDEN_PATH exists.
      2. If not, generates the 30m Raipur plain DEM and resamples to 10m.
      3. If save_golden is True, exports to DEFAULT_DEM_GOLDEN_PATH as a 4-band stack.
      4. Returns the 10m elevation raster.

    Parameters
    ----------
    dem_path : str, optional
        Path to an external DEM GeoTIFF.
    target_shape : Tuple[int, int], default=(2223, 2086)
        Required raster dimensions.
    target_affine : Affine
        Required geospatial transform.
    target_crs : str, default='EPSG:32644'
        Required CRS.
    resampling_method : str, default='bilinear'
        Resampling algorithm ('bilinear' or 'bicubic').
    save_golden : bool, default=True
        Whether to persist generated DEM to golden data directory.
    seed : int, default=42
        Deterministic random seed.

    Returns
    -------
    np.ndarray
        2D float32 elevation array of shape target_shape.
    """
    # 1. Check if dem_path was provided and exists
    if dem_path is not None and os.path.isfile(dem_path):
        with rasterio.open(dem_path) as src:
            if (src.height, src.width) == target_shape and str(src.crs) == target_crs and src.transform == target_affine:
                return src.read(1).astype(np.float32)
            else:
                # Resample from source raster to target grid
                src_arr = src.read(1).astype(np.float32)
                return resample_dem(
                    source_dem=src_arr,
                    src_transform=src.transform,
                    src_crs=str(src.crs),
                    target_shape=target_shape,
                    target_transform=target_affine,
                    target_crs=target_crs,
                    method=resampling_method,
                )

    # 2. Check if default golden DEM exists
    if os.path.isfile(DEFAULT_DEM_GOLDEN_PATH):
        with rasterio.open(DEFAULT_DEM_GOLDEN_PATH) as src:
            if (src.height, src.width) == target_shape and str(src.crs) == target_crs and src.transform == target_affine:
                return src.read(1).astype(np.float32)

    # 3. Generate deterministic 30m DEM and resample to 10m
    dem_30m, transform_30m = generate_raipur_dem_30m(
        shape_30m=(741, 696),
        bounds=TARGET_BOUNDS,
        seed=seed,
    )

    dem_10m = resample_dem(
        source_dem=dem_30m,
        src_transform=transform_30m,
        src_crs=TARGET_CRS,
        target_shape=target_shape,
        target_transform=target_affine,
        target_crs=target_crs,
        method=resampling_method,
    )

    # 4. Save golden dataset if requested
    if save_golden:
        covariates = compute_terrain_covariates(dem_10m, cell_size=CELL_SIZE_10M)
        export_terrain_covariates_geotiff(
            output_path=DEFAULT_DEM_GOLDEN_PATH,
            covariates=covariates,
            transform=target_affine,
            crs=target_crs,
        )

    return dem_10m
