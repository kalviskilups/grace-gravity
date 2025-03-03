"""
Core functionality for GRACE gravity field analysis and visualization.

This module provides functions for loading, processing, and analyzing
gravitational data from the GRACE satellite mission.
"""

import glob
import os
import re
from datetime import datetime, timedelta
from math import atan2, cos, radians, sin, sqrt
from typing import List, Optional, Tuple

import numpy as np
import pyshtools as pysh
from numpy.typing import NDArray


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points using the Haversine formula.

    Args:
        lat1: Latitude of the first point in degrees
        lon1: Longitude of the first point in degrees
        lat2: Latitude of the second point in degrees
        lon2: Longitude of the second point in degrees

    Returns:
        Distance between the points in kilometers
    """
    # Earth radius in kilometers
    r = 6371.0

    # Convert to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return r * c


def load_sh_grav_coeffs(gfc_file: str, format: str = "icgem") -> pysh.SHGravCoeffs:
    """
    Load spherical harmonic coefficients from a GRACE GFC file.

    Args:
        gfc_file: Path to the GFC file
        format: File format, defaults to "icgem"

    Returns:
        Spherical harmonic coefficients object

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(gfc_file):
        raise FileNotFoundError(f"GFC file not found: {gfc_file}")

    try:
        return pysh.SHGravCoeffs.from_file(gfc_file, format=format)
    except Exception as e:
        raise ValueError(f"Failed to load coefficients from {gfc_file}: {str(e)}")


def compute_gravity_anomaly(coeffs: pysh.SHGravCoeffs, exclude_degrees: int = 5) -> NDArray:
    """
    Compute gravity anomaly while excluding low-degree coefficients.

    Args:
        coeffs: Input spherical harmonic coefficients
        exclude_degrees: Number of low degrees to zero out, defaults to 5

    Returns:
        Gravity anomaly grid as a numpy array
    """
    # Physical constants
    gm = 3.9860044150e14  # Gravitational constant * mass [m^3/s^2]
    r0 = 6.3781363000e06  # Reference radius [m]

    # Create a copy of coefficients and zero out low degrees
    modified_coeffs = coeffs.coeffs.copy()
    modified_coeffs[:, 0:exclude_degrees, :] = 0

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

    Args:
        coeffs: Spherical harmonic coefficients
        max_degree: Maximum degree to include in calculations
        heatmap: If True, filter coefficients for heatmap visualization

    Returns:
        Tuple containing V_xx and V_xz gradient components in Eötvös units
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


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date information from GFC filename.

    Args:
        filename: Path to the GFC file

    Returns:
        Extracted date object or None if no date pattern is found
    """
    # Pattern for ITSG files with full date (e.g., ITSG-Grace2014_2004-12-28.gfc)
    itsg_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if itsg_match:
        year, month, day = map(int, itsg_match.groups())
        return datetime(year, month, day)

    # Pattern for CSR files with day of year (e.g., CSR_Release-06_2004336)
    csr_match = re.search(r"(\d{4})(\d{3})", filename)
    if csr_match:
        year, doy = map(int, csr_match.groups())
        return datetime(year, 1, 1) + timedelta(days=doy - 1)

    # Pattern for monthly solutions (e.g., GFZ_RL06_2004_09)
    monthly_match = re.search(r"_(\d{4})[-_](\d{2})", filename)
    if monthly_match:
        year, month = map(int, monthly_match.groups())
        return datetime(year, month, 15)  # Mid-month convention

    # Pattern for ITSG monthly data (e.g., ITSG-Grace2018_n120_2015-12)
    itsg_monthly_match = re.search(r"(\d{4})-(\d{2})$", filename)
    if itsg_monthly_match:
        year, month = map(int, itsg_monthly_match.groups())
        return datetime(year, month, 15)  # Mid-month convention

    return None


def compute_anomaly_difference(coeffs1: pysh.SHGravCoeffs, coeffs2: pysh.SHGravCoeffs) -> NDArray:
    """
    Compute the absolute difference between two gravity anomalies.

    Args:
        coeffs1: First set of coefficients
        coeffs2: Second set of coefficients

    Returns:
        Scaled anomaly difference grid (in microGal units)
    """
    anomaly1 = compute_gravity_anomaly(coeffs1)
    anomaly2 = compute_gravity_anomaly(coeffs2)

    # Scale to microGal
    return abs(anomaly2 - anomaly1) * 1e8


def find_grace_files_for_period(
    data_dir: str,
    start_date: datetime,
    end_date: Optional[datetime] = None,
    days_after: int = 30,
) -> List[str]:
    """
    Find GRACE files within a specified time period.

    Args:
        data_dir: Directory containing GFC files
        start_date: Start date
        end_date: End date (if None, calculated from days_after)
        days_after: Days after start_date (used if end_date is None)

    Returns:
        List of GFC files sorted by date

    Raises:
        FileNotFoundError: If no files found in the date range
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if end_date is None:
        end_date = start_date + timedelta(days=days_after)

    all_files = glob.glob(os.path.join(data_dir, "*.gfc"))
    if not all_files:
        raise FileNotFoundError(f"No GFC files found in directory: {data_dir}")

    period_files = []

    for file in all_files:
        file_date = parse_date_from_filename(file)
        if file_date and start_date <= file_date <= end_date:
            period_files.append((file, file_date))

    if not period_files:
        raise FileNotFoundError(f"No GRACE files found in the date range {start_date} to {end_date}")

    # Sort by date
    period_files.sort(key=lambda x: x[1])
    return [f[0] for f in period_files]


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


def taper_coeffs(
    coeffs: pysh.SHGravCoeffs, min_degree: int, max_degree: int, taper_width: int = 2
) -> pysh.SHGravCoeffs:
    """
    Apply a cosine taper to the spherical harmonic coefficients.

    Args:
        coeffs: Spherical harmonic coefficient object
        min_degree: Minimum degree of the band
        max_degree: Maximum degree of the band
        taper_width: Width over which to taper

    Returns:
        Tapered spherical harmonic coefficients
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


def plot_coefficient_spectrum(gfc_file: str, unit: str = "per_l", xscale: str = "lin", yscale: str = "log") -> None:
    """
    Plot the spectral characteristics of spherical harmonic coefficients.

    Args:
        gfc_file: Path to the GRACE GFC file
        unit: Spectral representation unit
        xscale: X-axis scaling
        yscale: Y-axis scaling
    """
    coeffs = pysh.SHGravCoeffs.from_file(gfc_file, format="icgem")
    coeffs.plot_spectrum(unit=unit, xscale=xscale, yscale=yscale, legend=True)


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
