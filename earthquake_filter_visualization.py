import glob
import os
import re
from datetime import datetime, timedelta
from math import atan2, cos, radians, sin, sqrt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh
from scipy.interpolate import griddata


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


def compute_gravity_anomaly(coeffs, max_degree=30, heatmap=False):
    """
    Compute gravity anomaly from spherical harmonic coefficients.

    Args:
        coeffs (pysh.SHGravCoeffs): Spherical harmonic coefficients
        exclude_degrees (int, optional): Number of low degrees to exclude. Defaults to 0.

    Returns:
        numpy.ndarray: Gravity anomaly grid
    """
    # Get parameters from coeffs object
    gm = 3.9860044150e14  # Gravitational constant * mass [m^3/s^2]
    r0 = 6.3781363000e06  # Reference radius [m]

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

    # Pattern for ITSG files (e.g., ITSG-Grace2014_2004-12-28.gfc)
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

    return None


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
    v_xx, v_xz = compute_gravity_anomaly(filtered_coeffs, heatmap=False)

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


def plot_earthquake_anomaly(
    gfc_file,
    epicenter_lat,
    epicenter_lon,
    min_degree=10,
    max_degree=60,
    title=None,
    region_size=15,
    output_file=None,
):
    """
    Plot earthquake gravity gradient anomalies.

    Args:
        gfc_file (str): Path to the GFC file
        epicenter_lat (float): Latitude of epicenter
        epicenter_lon (float): Longitude of epicenter
        min_degree (int, optional): Minimum spherical harmonic degree. Defaults to 10.
        max_degree (int, optional): Maximum spherical harmonic degree. Defaults to 60.
        taper_width (int, optional): Taper width. Defaults to 3.
        title (str, optional): Plot title
        region_size (float, optional): Size of region around epicenter in degrees
        output_file (str, optional): Path to save figure
    """

    if epicenter_lon < 0:
        epicenter_lon += 360
    # Load and compute gravity gradients
    coeffs = load_sh_grav_coeffs(gfc_file)
    v_xx, v_xz = compute_gravity_anomaly(coeffs, heatmap=True)

    # Set up the grid
    nlat, nlon = v_xx.shape
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)

    # Set region boundaries
    lat_min = max(-90, epicenter_lat - region_size)
    lat_max = min(90, epicenter_lat + region_size)
    lon_min = epicenter_lon - region_size
    lon_max = epicenter_lon + region_size

    # Convert longitude to 0-360 range if needed
    lon_min = lon_min % 360
    lon_max = lon_max % 360

    # Find indices for the region of interest
    lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]

    # Handle case where the region crosses the date line
    if lon_min > lon_max:
        lon_indices = np.where((lons >= lon_min) | (lons <= lon_max))[0]
    else:
        lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]

    # Extract the region data
    region_v_xx = v_xx[np.ix_(lat_indices, lon_indices)]
    region_v_xz = v_xz[np.ix_(lat_indices, lon_indices)]

    # Extract regional coordinates
    region_lats = lats[lat_indices]
    region_lons = lons[lon_indices]
    region_lon_grid, region_lat_grid = np.meshgrid(region_lons, region_lats)

    # Create higher resolution grid for smooth plotting, but only for the region
    refine_factor = 2
    xi = np.linspace(lon_min, lon_max, len(lon_indices) * refine_factor)
    yi = np.linspace(lat_min, lat_max, len(lat_indices) * refine_factor)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate data to higher resolution grid using only the region data
    zi_v_xx = griddata(
        (region_lon_grid.flatten(), region_lat_grid.flatten()),
        region_v_xx.flatten(),
        (xi, yi),
        method="cubic",
    )

    zi_v_xz = griddata(
        (region_lon_grid.flatten(), region_lat_grid.flatten()),
        region_v_xz.flatten(),
        (xi, yi),
        method="cubic",
    )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    # First subplot: V_xx gradient
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=":")

    # Plot V_xx
    pcm1 = ax1.contourf(
        xi,
        yi,
        zi_v_xx,
        levels=50,
        transform=ccrs.PlateCarree(),
        cmap="RdYlBu_r",
        extend="both",
    )

    # Add epicenter marker
    ax1.plot(
        epicenter_lon,
        epicenter_lat,
        "k*",
        markersize=12,
        transform=ccrs.PlateCarree(),
        label="Epicenter",
    )

    # Add colorbar
    cbar1 = plt.colorbar(pcm1, ax=ax1, orientation="vertical")
    cbar1.set_label("V_xx Gravity Gradient (Eötvös)")
    ax1.set_title("V_xx Gravity Gradient")

    # Second subplot: V_xz gradient
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=":")

    # Plot V_xz
    pcm2 = ax2.contourf(
        xi,
        yi,
        zi_v_xz,
        levels=50,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        extend="both",
    )

    # Add epicenter marker
    ax2.plot(
        epicenter_lon,
        epicenter_lat,
        "k*",
        markersize=12,
        transform=ccrs.PlateCarree(),
        label="Epicenter",
    )

    # Add colorbar
    cbar2 = plt.colorbar(pcm2, ax=ax2, orientation="vertical")
    cbar2.set_label("V_xz Gravity Gradient (Eötvös)")
    ax2.set_title("V_xz Gravity Gradient")

    # Main title
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        file_date = parse_date_from_filename(gfc_file)
        date_str = file_date.strftime("%Y-%m-%d") if file_date else "Unknown date"
        plt.suptitle(f"Gravity Gradients at {date_str}", fontsize=16)

    plt.tight_layout()

    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {output_file}")
    else:
        plt.show()

    return fig


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


def generate_time_series_plots(
    data_dir,
    earthquake_date,
    epicenter_lat,
    epicenter_lon,
    min_degree=10,
    max_degree=60,
    taper_width=3,
    days_before=700,
    days_after=365,
    region_radius=2,
    output_file="time_series.png",
    iqr_k=2.5,
):
    if isinstance(earthquake_date, str):
        earthquake_date = datetime.strptime(earthquake_date, "%Y-%m-%d")

    start_date = earthquake_date - timedelta(days=days_before)
    end_date = earthquake_date + timedelta(days=days_after)
    grace_files = find_grace_files_for_period(data_dir, start_date, end_date)

    if not grace_files:
        print(f"No GRACE files found for {start_date} to {end_date}")
        return

    dates, v_xx_values, v_xz_values = [], [], []
    baseline_start = datetime(2008, 2, 1)
    baseline_end = datetime(2011, 3, 1)

    try:
        v_xx_avg, v_xz_avg = compute_long_term_average(
            data_dir,
            baseline_start,
            baseline_end,
            epicenter_lat,
            epicenter_lon,
            min_degree,
            max_degree,
            taper_width,
        )
        print("Long-term average gradients computed successfully.")
    except ValueError as e:
        print(f"Baseline error: {e}")
        return

    for gfc_file in grace_files:
        file_date = parse_date_from_filename(gfc_file)
        if not file_date:
            continue
        try:
            mean_v_xx, mean_v_xz = filter_earthquake_signal(
                gfc_file,
                epicenter_lat,
                epicenter_lon,
                min_degree,
                max_degree,
                taper_width,
                region_radius,
            )
            dates.append(file_date)
            v_xx_values.append(mean_v_xx - v_xx_avg)
            v_xz_values.append(mean_v_xz - v_xz_avg)
        except Exception as e:
            print(f"Error processing {gfc_file}: {e}")

    if not dates:
        print("No valid data extracted")
        return

    days_relative = [(d - earthquake_date).days for d in dates]

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Process and plot V_xx
    v_xx_array = np.array(v_xx_values)
    q1_xx, q3_xx = np.percentile(v_xx_array, [25, 75])
    iqr_xx = q3_xx - q1_xx
    lower_bound_xx = q1_xx - iqr_k * iqr_xx
    upper_bound_xx = q3_xx + iqr_k * iqr_xx
    anomaly_indices_xx = np.where((v_xx_array < lower_bound_xx) | (v_xx_array > upper_bound_xx))[0]

    ax1.plot(days_relative, v_xx_values, "b-", label="Mean V_xx")
    ax1.axvline(x=0, color="r", linestyle="-", label="Earthquake")
    if anomaly_indices_xx.size > 0:
        ax1.scatter(
            np.array(days_relative)[anomaly_indices_xx],
            v_xx_array[anomaly_indices_xx],
            color="red",
            label="Detected Anomaly",
            zorder=3,
            s=100,
            edgecolors="black",
        )
    ax1.axhline(y=lower_bound_xx, color="gray", linestyle="--", label="Lower IQR")
    ax1.axhline(y=upper_bound_xx, color="gray", linestyle="--", label="Upper IQR")
    ax1.set_ylabel("V_xx (Eötvös)")
    ax1.set_title("V_xx Time Series (±{}° around epicenter)".format(region_radius))
    ax1.legend()
    ax1.grid(True)

    # Process and plot V_xz
    v_xz_array = np.array(v_xz_values)
    q1_xz, q3_xz = np.percentile(v_xz_array, [25, 75])
    iqr_xz = q3_xz - q1_xz
    lower_bound_xz = q1_xz - iqr_k * iqr_xz
    upper_bound_xz = q3_xz + iqr_k * iqr_xz
    anomaly_indices_xz = np.where((v_xz_array < lower_bound_xz) | (v_xz_array > upper_bound_xz))[0]

    ax2.plot(days_relative, v_xz_values, "g-", label="Mean V_xz")
    ax2.axvline(x=0, color="r", linestyle="-", label="Earthquake")
    if anomaly_indices_xz.size > 0:
        ax2.scatter(
            np.array(days_relative)[anomaly_indices_xz],
            v_xz_array[anomaly_indices_xz],
            color="orange",
            label="Detected Anomaly",
            zorder=3,
            s=100,
            edgecolors="black",
        )
    ax2.axhline(y=lower_bound_xz, color="gray", linestyle="--", label="Lower IQR")
    ax2.axhline(y=upper_bound_xz, color="gray", linestyle="--", label="Upper IQR")
    ax2.set_xlabel("Days Relative to Earthquake")
    ax2.set_ylabel("V_xz (Eötvös)")
    ax2.set_title("V_xz Time Series (±{}° around epicenter)".format(region_radius))
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Combined time series plot saved as {output_file}")


if __name__ == "__main__":
    # Configure for the 2010 Chilean earthquake
    DATA_DIR = "time_series/ddk0_2010_daily"
    EARTHQUAKE_DATE = "2010-02-27"  # Chilean earthquake
    EPICENTER_LAT = -35.91  # Latitude of the Chilean earthquake epicenter
    EPICENTER_LON = -72.73  # Longitude of the Chilean earthquake epicenter
    OUTPUT_DIR = "earthquake_analysis_chile"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find a GFC file close to the earthquake date (adjust days_after if needed)
    try:
        earthquake_datetime = datetime.strptime(EARTHQUAKE_DATE, "%Y-%m-%d")
        earthquake_files = find_grace_files_for_period(
            DATA_DIR, earthquake_datetime, earthquake_datetime + timedelta(days=15)
        )

        if earthquake_files:
            selected_file = earthquake_files[0]  # Use the first file found

            # Plot the gravity gradients for this file
            plot_earthquake_anomaly(
                selected_file,
                EPICENTER_LAT,
                EPICENTER_LON,
                min_degree=15,
                max_degree=40,
                region_size=50,  # Adjust region size for better focus on earthquake area
                title=f"Chilean Earthquake ({EARTHQUAKE_DATE}) Gravity Gradients",
                output_file=os.path.join(OUTPUT_DIR, "chile_earthquake_gravity.png"),
            )

            print(f"Gravity gradient map created using file: {selected_file}")
        else:
            print("No GFC file found within 15 days after the earthquake")

    except Exception as e:
        print(f"Error processing earthquake gravity map: {e}")

    # Generate time series plots
    try:
        generate_time_series_plots(
            DATA_DIR,
            EARTHQUAKE_DATE,
            EPICENTER_LAT,
            EPICENTER_LON,
            min_degree=10,
            max_degree=40,
            taper_width=3,
            days_before=600,  # Analyze data from 4 months before earthquake
            days_after=300,  # to 6 months after earthquake
            region_radius=2,  # Adjust radius based on earthquake magnitude
            output_file=os.path.join(OUTPUT_DIR, "chile_time_series.png"),
            iqr_k=2.0,  # Adjust sensitivity for anomaly detection
        )

        print("Time series analysis completed")

    except Exception as e:
        print(f"Error generating time series: {e}")
