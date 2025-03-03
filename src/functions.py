import glob
import os
import re
from datetime import datetime, timedelta
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pyshtools as pysh


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Haversine distance function to compute geodesic distances in kilometers
    """
    r = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def load_sh_grav_coeffs(gfc_file, format="icgem"):
    """
    Load spherical harmonic coefficients from a GRACE GFC file.

    Args:
        gfc_file (str): Path to the GFC file
        format (str, optional): File format. Defaults to "icgem"

    Returns:
        pysh.SHGravCoeffs: Spherical harmonic coefficients
    """
    return pysh.SHGravCoeffs.from_file(gfc_file, format=format)


def compute_gravity_anomaly(coeffs, exclude_degrees=5):
    """
    Compute gravity anomaly while excluding low-degree coefficients.

    Args:
        coeffs (pysh.SHGravCoeffs): Input spherical harmonic coefficients
        exclude_degrees (int, optional): Number of low degrees to zero out. Defaults to 5

    Returns:
        numpy.ndarray: Gravity anomaly grid
    """
    gm = 3.9860044150e14  # Gravitational constant * mass [m^3/s^2]
    r0 = 6.3781363000e06  # Reference radius [m]

    modified_coeffs = coeffs.coeffs.copy()
    modified_coeffs[:, 0:exclude_degrees, :] = 0

    rad, theta, phi, total, pot = pysh.gravmag.MakeGravGridDH(
        modified_coeffs, gm, r0, lmax=coeffs.lmax, normal_gravity=1
    )

    # combined_total = np.sqrt(rad**2 + theta**2 + phi**2)

    g_ref = gm / (r0**2)
    total = total + g_ref

    return total


def compute_gravity_gradient_tensor(coeffs, max_degree=30, heatmap=False):
    """
    Compute gravity anomaly from spherical harmonic coefficients.

    Args:
        coeffs (pysh.SHGravCoeffs): Spherical harmonic coefficients
        exclude_degrees (int, optional): Number of low degrees to exclude. Defaults to 0.

    Returns:
        numpy.ndarray: Gravity anomaly grid
    """
    # Get parameters from coeffs object
    gm = coeffs.gm
    r0 = coeffs.r0

    if heatmap:
        modified_coeffs = coeffs.copy()
        temp_coeffs = modified_coeffs.coeffs.copy()
        temp_coeffs[:, 5:max_degree, :] = 0
        modified_coeffs.coeffs = temp_coeffs
        coeffs = modified_coeffs

    coeff_array = coeffs.to_array(normalization="4pi", csphase=-1)

    # Calculate gravity anomaly
    v_xx, v_yy, v_zz, v_xy, v_xz, v_yz = pysh.gravmag.MakeGravGradGridDH(
        coeff_array, gm, r0, lmax=coeffs.lmax, sampling=2, extend=False
    )

    return v_xx * 1e9, v_xz * 1e9


def parse_date_from_filename(filename):
    """
    Extract date information from GFC filename.

    Args:
        filename (str): Path to the GFC file

    Returns:
        datetime or None: Extracted date object
    """
    # Try different date patterns found in common GFC files

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


def compute_anomaly_difference(coeffs1, coeffs2):
    """
    Compute the absolute difference between two gravity anomalies.

    Args:
        coeffs1 (pysh.SHGravCoeffs): First set of coefficients
        coeffs2 (pysh.SHGravCoeffs): Second set of coefficients

    Returns:
        numpy.ndarray: Scaled anomaly difference
    """
    anomaly1 = compute_gravity_anomaly(coeffs1)
    anomaly2 = compute_gravity_anomaly(coeffs2)

    return abs(anomaly2 - anomaly1) * 1e8


def find_grace_files_for_period(data_dir, start_date, end_date=None, days_after=30):
    """
    Find GRACE files within a specified time period.

    Args:
        data_dir (str): Directory containing GFC files
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, calculated from days_after.
        days_after (int, optional): Days after start_date. Defaults to 30.

    Returns:
        list: List of GFC files sorted by date
    """
    if end_date is None:
        end_date = start_date + timedelta(days=days_after)

    all_files = glob.glob(os.path.join(data_dir, "*.gfc"))
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
    gfc_file,
    epicenter_lat,
    epicenter_lon,
    min_degree=10,
    max_degree=60,
    taper_width=3,
    radius=2,
):
    """
    Filter earthquake signal from GRACE data using spectral band filtering.

    Args:
        gfc_file (str): Path to the GFC file
        epicenter_lat (float): Latitude of earthquake epicenter
        epicenter_lon (float): Longitude of earthquake epicenter
        min_degree (int, optional): Minimum spherical harmonic degree. Defaults to 10.
        max_degree (int, optional): Maximum spherical harmonic degree. Defaults to 60.
        taper_width (int, optional): Taper width. Defaults to 3.

    Returns:
        tuple: (float, float) Mean V_xx and V_xz values within the specified radius
    """

    if epicenter_lon < 0:
        epicenter_lon += 360

    coeffs = load_sh_grav_coeffs(gfc_file)
    filtered_coeffs = taper_coeffs(coeffs, min_degree, max_degree, taper_width)
    v_xx, v_xz = compute_gravity_gradient_tensor(filtered_coeffs, heatmap=False)

    nlat, nlon = v_xx.shape
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    distance_grid = np.vectorize(haversine_distance)(epicenter_lat, epicenter_lon, lat_grid, lon_grid)
    distance_threshold = radius * 111  # km
    mask = distance_grid <= distance_threshold

    mean_v_xx = np.mean(v_xx[mask])
    mean_v_xz = np.mean(v_xz[mask])

    if np.isnan(mean_v_xx) or np.isnan(mean_v_xz):
        raise ValueError("No valid grid points within the specified radius")

    return mean_v_xx, mean_v_xz


def taper_coeffs(coeffs, min_degree, max_degree, taper_width=2):
    """
    Apply a cosine taper to the spherical harmonic coefficients.

    Args:
        coeffs (pysh.SHGravCoeffs): Spherical harmonic coefficient object
        min_degree (int): Minimum degree of the band
        max_degree (int): Maximum degree of the band
        taper_width (int, optional): Width over which to taper. Defaults to 2

    Returns:
        pysh.SHGravCoeffs: Tapered spherical harmonic coefficients
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


def plot_coefficient_spectrum(gfc_file, unit="per_l", xscale="lin", yscale="log"):
    """
    Plot the spectral characteristics of spherical harmonic coefficients.

    Args:
        gfc_file (str): Path to the GRACE GFC file
        unit (str, optional): Spectral representation unit. Defaults to "per_l"
        xscale (str, optional): X-axis scaling. Defaults to "lin"
        yscale (str, optional): Y-axis scaling. Defaults to "log"
    """
    coeffs = pysh.SHGravCoeffs.from_file(gfc_file, format="icgem")
    coeffs.plot_spectrum(unit=unit, xscale=xscale, yscale=yscale, legend=True)


def compute_long_term_average(
    data_dir,
    baseline_start_date,
    baseline_end_date,
    epicenter_lat,
    epicenter_lon,
    min_degree,
    max_degree,
    taper_width=3,
):
    grace_files = find_grace_files_for_period(data_dir, baseline_start_date, baseline_end_date)
    if not grace_files:
        raise ValueError(f"No GRACE files found between {baseline_start_date} and {baseline_end_date}")

    v_xx_list, v_xz_list = [], []
    for gfc_file in grace_files:
        mean_v_xx, mean_v_xz = filter_earthquake_signal(
            gfc_file, epicenter_lat, epicenter_lon, min_degree, max_degree, taper_width
        )
        v_xx_list.append(mean_v_xx)
        v_xz_list.append(mean_v_xz)

    return np.mean(v_xx_list), np.mean(v_xz_list)
