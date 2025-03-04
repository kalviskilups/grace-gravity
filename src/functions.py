"""
Core functionality for GRACE gravity field analysis.

This module provides functions for calculating various gravity field metrics,
including anomalies, gradients, and comparative measures from GRACE data.
"""

from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Tuple

import numpy as np
import pyshtools as pysh
from numpy.typing import NDArray

from src.io import find_grace_files_for_period, load_sh_grav_coeffs


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points using the Haversine formula.

    The Haversine formula determines the shortest distance between two points on a
    sphere given their longitudes and latitudes.

    Args:
        lat1: Latitude of the first point in degrees
        lon1: Longitude of the first point in degrees
        lat2: Latitude of the second point in degrees
        lon2: Longitude of the second point in degrees

    Returns:
        Distance between the points in kilometers

    Examples:
        >>> haversine_distance(52.5200, 13.4050, 48.8566, 2.3522)
        878.5699308283413  # Distance between Berlin and Paris
    """
    # Earth radius in kilometers
    r = 6371.0

    # Convert to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return r * c


def compute_gravity_anomaly(coeffs: pysh.SHGravCoeffs, exclude_degrees: int = 5) -> NDArray:
    """
    Compute gravity anomaly while excluding low-degree coefficients.

    This function calculates gravity anomalies from spherical harmonic coefficients,
    zeroing out the low-degree components which typically represent the Earth's
    overall shape rather than local anomalies.

    Args:
        coeffs: Input spherical harmonic coefficients
        exclude_degrees: Number of low degrees to zero out, defaults to 5

    Returns:
        Gravity anomaly grid as a numpy array (in m/s²)

    Examples:
        >>> coeffs = load_sh_grav_coeffs("GRACE_data.gfc")
        >>> anomalies = compute_gravity_anomaly(coeffs, exclude_degrees=5)
        >>> anomalies.shape
        (180, 360)  # Grid covering the Earth
    """
    # Physical constants
    gm = coeffs.gm  # Gravitational constant * mass [m^3/s^2]
    r0 = coeffs.r0  # Reference radius [m]

    # Create a copy of coefficients and zero out low degrees
    modified_coeffs = coeffs.coeffs.copy()
    modified_coeffs[:, 5:exclude_degrees, :] = 0

    # Calculate gravity components
    rad, theta, phi, total, pot = pysh.gravmag.MakeGravGridDH(
        modified_coeffs, gm, r0, lmax=coeffs.lmax, normal_gravity=1
    )

    # Add reference gravity
    g_ref = gm / (r0**2)
    total = total + g_ref

    return total


def compute_gravity_gradient_tensor(
    coeffs: pysh.SHGravCoeffs, max_degree: int = 30, heatmap: bool = False
) -> Tuple[NDArray, NDArray]:
    """
    Compute gravity gradient tensor components from spherical harmonic coefficients.

    This function calculates the gravity gradient tensor components V_xx and V_xz,
    which represent spatial derivatives of the gravitational potential.

    Args:
        coeffs: Spherical harmonic coefficients
        max_degree: Maximum degree to include in calculations
        heatmap: If True, filter coefficients for heatmap visualization

    Returns:
        Tuple containing V_xx and V_xz gradient components in Eötvös units

    Examples:
        >>> coeffs = load_sh_grav_coeffs("GRACE_data.gfc")
        >>> v_xx, v_xz = compute_gravity_gradient_tensor(coeffs, max_degree=60)
        >>> v_xx.shape, v_xz.shape
        ((180, 360), (180, 360))  # Global grid of tensor components
    """
    # Get parameters from coeffs object
    gm = coeffs.gm
    r0 = coeffs.r0

    # Filter coefficients if needed for heatmap
    if heatmap:
        modified_coeffs = coeffs.copy()
        temp_coeffs = modified_coeffs.coeffs.copy()
        temp_coeffs[:, 5:max_degree, :] = 0
        modified_coeffs.coeffs = temp_coeffs
        coeffs = modified_coeffs

    # Convert to array format needed by pyshtools
    coeff_array = coeffs.to_array(normalization="4pi", csphase=-1)

    # Calculate gravity gradient tensor components
    v_xx, v_yy, v_zz, v_xy, v_xz, v_yz = pysh.gravmag.MakeGravGradGridDH(
        coeff_array, gm, r0, lmax=coeffs.lmax, sampling=2, extend=False
    )

    # Convert to Eötvös units (1E9)
    return v_xx * 1e9, v_xz * 1e9


def compute_anomaly_difference(coeffs1: pysh.SHGravCoeffs, coeffs2: pysh.SHGravCoeffs, exclude_degrees: int) -> NDArray:
    """
    Compute the absolute difference between two gravity anomalies.

    This function calculates the difference between two sets of gravity anomalies,
    useful for comparing changes over time or between different models.

    Args:
        coeffs1: First set of coefficients
        coeffs2: Second set of coefficients

    Returns:
        Scaled anomaly difference grid (in microGal units)

    Examples:
        >>> coeffs1 = load_sh_grav_coeffs("GRACE_2010_01.gfc")
        >>> coeffs2 = load_sh_grav_coeffs("GRACE_2010_03.gfc")
        >>> diff = compute_anomaly_difference(coeffs1, coeffs2)
        >>> np.mean(diff)  # Average difference in microGal
    """
    anomaly1 = compute_gravity_anomaly(coeffs1, exclude_degrees)
    anomaly2 = compute_gravity_anomaly(coeffs2, exclude_degrees)

    # Scale to microGal
    return abs(anomaly2 - anomaly1) * 1e8


def taper_coeffs(
    coeffs: pysh.SHGravCoeffs, min_degree: int, max_degree: int, taper_width: int = 2
) -> pysh.SHGravCoeffs:
    """
    Apply a cosine taper to the spherical harmonic coefficients.

    This function applies a cosine taper to spherical harmonic coefficients,
    which helps reduce ringing artifacts in gravity field reconstructions.

    Args:
        coeffs: Spherical harmonic coefficient object
        min_degree: Minimum degree of the band
        max_degree: Maximum degree of the band
        taper_width: Width over which to taper

    Returns:
        Tapered spherical harmonic coefficients

    Examples:
        >>> coeffs = load_sh_grav_coeffs("GRACE_data.gfc")
        >>> tapered = taper_coeffs(coeffs, min_degree=10, max_degree=60, taper_width=3)
    """
    # Make a copy to avoid modifying the original
    tapered_coeffs = coeffs.copy()

    # Get maximum degree
    lmax = coeffs.lmax

    # Create degree weights array
    degrees = np.arange(lmax + 1)
    weights = np.ones(lmax + 1)

    # Calculate taper weights
    for l_degrees in degrees:
        if l_degrees < min_degree:
            if l_degrees < min_degree - taper_width:
                weights[l_degrees] = 0.0
            else:
                weights[l_degrees] = 0.5 * (1 + np.cos(np.pi * (min_degree - l_degrees) / taper_width))
        elif l_degrees > max_degree:
            if l_degrees > max_degree + taper_width:
                weights[l_degrees] = 0.0
            else:
                weights[l_degrees] = 0.5 * (1 + np.cos(np.pi * (l_degrees - max_degree) / taper_width))
        else:
            weights[l_degrees] = 1.0

    # Apply taper weights to coefficients
    for l_degrees in range(lmax + 1):
        tapered_coeffs.coeffs[:, l_degrees, :] *= weights[l_degrees]

    return tapered_coeffs


def filter_earthquake_signal(
    gfc_file: str,
    epicenter_lat: float,
    epicenter_lon: float,
    min_degree: int = 10,
    max_degree: int = 60,
    taper_width: int = 3,
    radius: float = 2.0,
) -> Tuple[float, float]:
    """
    Filter earthquake signal from GRACE data using spectral band filtering.

    This function focuses on extracting earthquake-related gravity signals
    by applying band filtering and analyzing a region around the epicenter.

    Args:
        gfc_file: Path to the GFC file
        epicenter_lat: Latitude of earthquake epicenter
        epicenter_lon: Longitude of earthquake epicenter
        min_degree: Minimum spherical harmonic degree
        max_degree: Maximum spherical harmonic degree
        taper_width: Taper width for filtering
        radius: Radius around epicenter to average (in degrees)

    Returns:
        Tuple of mean V_xx and V_xz values within the specified radius

    Raises:
        ValueError: If no valid grid points within the specified radius

    Examples:
        >>> # For the 2010 Chile earthquake
        >>> v_xx, v_xz = filter_earthquake_signal(
        ...     "GRACE_2010_03.gfc",
        ...     -35.846, -72.719,
        ...     min_degree=10, max_degree=60
        ... )
        >>> v_xx, v_xz
        (-15.23, 8.45)  # Example gravity gradient values
    """
    # Normalize longitude to 0-360 range
    if epicenter_lon < 0:
        epicenter_lon += 360

    # Load coefficients and apply filtering
    coeffs = load_sh_grav_coeffs(gfc_file)
    filtered_coeffs = taper_coeffs(coeffs, min_degree, max_degree, taper_width)
    v_xx, v_xz = compute_gravity_gradient_tensor(filtered_coeffs, heatmap=False)

    # Set up coordinate grid
    nlat, nlon = v_xx.shape
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Calculate distance from epicenter
    distance_grid = np.vectorize(haversine_distance)(epicenter_lat, epicenter_lon, lat_grid, lon_grid)
    distance_threshold = radius * 111  # Convert degrees to km (approximate)
    mask = distance_grid <= distance_threshold

    # Check if any points are within the radius
    if not np.any(mask):
        raise ValueError("No valid grid points within the specified radius")

    # Calculate mean values
    mean_v_xx = np.mean(v_xx[mask])
    mean_v_xz = np.mean(v_xz[mask])

    # Validate results
    if np.isnan(mean_v_xx) or np.isnan(mean_v_xz):
        raise ValueError("NaN values encountered in gravity gradient calculations")

    return mean_v_xx, mean_v_xz


def compute_long_term_average(
    data_dir: str,
    baseline_start_date: datetime,
    baseline_end_date: datetime,
    epicenter_lat: float,
    epicenter_lon: float,
    min_degree: int,
    max_degree: int,
    taper_width: int = 3,
) -> Tuple[float, float]:
    """
    Compute long-term average gravity gradients for baseline comparison.

    This function calculates the average gravity gradients over a baseline period,
    which can be used to detect anomalies when comparing with post-event measurements.

    Args:
        data_dir: Directory containing GFC files
        baseline_start_date: Start date for baseline period
        baseline_end_date: End date for baseline period
        epicenter_lat: Latitude of earthquake epicenter
        epicenter_lon: Longitude of earthquake epicenter
        min_degree: Minimum spherical harmonic degree
        max_degree: Maximum spherical harmonic degree
        taper_width: Taper width for filtering

    Returns:
        Tuple of mean V_xx and V_xz baseline values

    Raises:
        ValueError: If no GRACE files found in the baseline period

    Examples:
        >>> # Calculate baseline for Chile earthquake (2010)
        >>> v_xx_avg, v_xz_avg = compute_long_term_average(
        ...     "data/",
        ...     datetime(2008, 1, 1), datetime(2009, 12, 31),
        ...     -35.846, -72.719,
        ...     min_degree=10, max_degree=60
        ... )
    """
    grace_files = find_grace_files_for_period(data_dir, baseline_start_date, baseline_end_date)
    if not grace_files:
        raise ValueError(f"No GRACE files found between {baseline_start_date} and {baseline_end_date}")

    v_xx_list, v_xz_list = [], []
    for gfc_file in grace_files:
        try:
            mean_v_xx, mean_v_xz = filter_earthquake_signal(
                gfc_file,
                epicenter_lat,
                epicenter_lon,
                min_degree,
                max_degree,
                taper_width,
            )
            v_xx_list.append(mean_v_xx)
            v_xz_list.append(mean_v_xz)
        except Exception as e:
            print(f"Warning: Error processing {gfc_file}: {e}")
            continue

    if not v_xx_list or not v_xz_list:
        raise ValueError("No valid gradient values calculated for baseline period")

    return np.mean(v_xx_list), np.mean(v_xz_list)
