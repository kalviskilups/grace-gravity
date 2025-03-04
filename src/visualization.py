"""
Visualization functions for GRACE gravity field data.

This module provides functions for creating visualizations of gravitational
anomalies, gravity gradient fields, and time series analysis.
"""

from datetime import datetime, timedelta
from typing import Optional, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from src.functions import (
    compute_gravity_gradient_tensor,
    compute_long_term_average,
    filter_earthquake_signal,
    taper_coeffs,
)
from src.io import (
    find_grace_files_for_period,
    load_sh_grav_coeffs,
    parse_date_from_filename,
)
from src.processing import detect_anomalies, normalize_longitude


def plot_coefficient_spectrum(
    gfc_file: str,
    unit: str = "per_l",
    xscale: str = "lin",
    yscale: str = "log",
    output_file: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the spectral characteristics of spherical harmonic coefficients.

    Args:
        gfc_file: Path to the GRACE GFC file
        unit: Spectral representation unit
        xscale: X-axis scaling ("lin" or "log")
        yscale: Y-axis scaling ("lin" or "log")
        output_file: Path to save figure (if None, will display)

    Returns:
        Matplotlib figure object

    Examples:
        >>> fig = plot_coefficient_spectrum(
        ...     "GRACE_2010_03.gfc",
        ...     unit="per_l",
        ...     xscale="log",
        ...     output_file="spectrum.png"
        ... )
    """
    coeffs = load_sh_grav_coeffs(gfc_file, format="icgem")
    fig = coeffs.plot_spectrum(unit=unit, xscale=xscale, yscale=yscale, legend=True)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_earthquake_anomaly(
    gfc_file: str,
    epicenter_lat: float,
    epicenter_lon: float,
    min_degree: int = 10,
    max_degree: int = 60,
    title: Optional[str] = None,
    region_size: float = 15.0,
    output_file: Optional[str] = None,
) -> plt.Figure:
    """
    Plot earthquake gravity gradient anomalies.

    Creates a visualization of gravity gradient tensor components
    focused on the region around an earthquake epicenter.

    Args:
        gfc_file: Path to the GFC file
        epicenter_lat: Latitude of epicenter
        epicenter_lon: Longitude of epicenter
        min_degree: Minimum spherical harmonic degree
        max_degree: Maximum spherical harmonic degree
        title: Plot title (if None, will be generated from file date)
        region_size: Size of region around epicenter in degrees
        output_file: Path to save figure (if None, will display)

    Returns:
        Matplotlib figure object

    Examples:
        >>> # Visualize the 2010 Chile earthquake region
        >>> fig = plot_earthquake_anomaly(
        ...     "GRACE_2010_03.gfc",
        ...     -35.846, -72.719,
        ...     min_degree=10, max_degree=60,
        ...     region_size=20.0,
        ...     output_file="chile_earthquake.png"
        ... )
    """
    # Normalize longitude to 0-360 range
    epicenter_lon = normalize_longitude(epicenter_lon)

    # Load and compute gravity gradients
    coeffs = load_sh_grav_coeffs(gfc_file)

    # Apply degree filtering if specified
    if min_degree > 0 or max_degree < coeffs.lmax:
        coeffs = taper_coeffs(coeffs, min_degree, max_degree)

    v_xx, v_xz = compute_gravity_gradient_tensor(coeffs, heatmap=True)

    # Set up the grid
    nlat, nlon = v_xx.shape
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)

    # Set region boundaries
    lat_min = max(-90, epicenter_lat - region_size)
    lat_max = min(90, epicenter_lat + region_size)
    lon_min = epicenter_lon - region_size
    lon_max = epicenter_lon + region_size

    # Normalize longitude boundaries
    lon_min = normalize_longitude(lon_min)
    lon_max = normalize_longitude(lon_max)

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
        plt.close()

    return fig


def generate_time_series_plots(
    data_dir: str,
    earthquake_date: Union[str, datetime],
    epicenter_lat: float,
    epicenter_lon: float,
    min_degree: int = 10,
    max_degree: int = 60,
    taper_width: int = 3,
    days_before: int = 700,
    days_after: int = 365,
    region_radius: float = 2.0,
    output_file: str = "time_series.png",
    iqr_k: float = 2.5,
    baseline_start: Optional[datetime] = None,
    baseline_end: Optional[datetime] = None,
) -> plt.Figure:
    """
    Generate time series plots of gravity gradient anomalies around an earthquake.

    This function creates time series visualizations showing how gravity gradients
    change over time relative to an earthquake event, with anomaly detection.

    Args:
        data_dir: Directory containing GFC files
        earthquake_date: Date of the earthquake (str or datetime)
        epicenter_lat: Latitude of epicenter
        epicenter_lon: Longitude of epicenter
        min_degree: Minimum spherical harmonic degree
        max_degree: Maximum spherical harmonic degree
        taper_width: Taper width for filtering
        days_before: Days before earthquake to analyze
        days_after: Days after earthquake to analyze
        region_radius: Radius around epicenter in degrees
        output_file: Path to save figure
        iqr_k: Multiplier for IQR-based anomaly detection
        baseline_start: Start date for baseline period (defaults to 1 year before earthquake)
        baseline_end: End date for baseline period (defaults to 1 month before earthquake)

    Returns:
        Matplotlib figure object

    Examples:
        >>> # Generate time series for the 2010 Chile earthquake
        >>> fig = generate_time_series_plots(
        ...     "data/",
        ...     "2010-02-27",
        ...     -35.846, -72.719,
        ...     days_before=365, days_after=180,
        ...     output_file="chile_time_series.png"
        ... )
    """
    # Convert string date to datetime if needed
    if isinstance(earthquake_date, str):
        earthquake_date = datetime.strptime(earthquake_date, "%Y-%m-%d")

    # Define time range
    start_date = earthquake_date - timedelta(days=days_before)
    end_date = earthquake_date + timedelta(days=days_after)

    # Set default baseline period if not provided
    if baseline_start is None:
        baseline_start = earthquake_date - timedelta(days=365)  # 1 year before
    if baseline_end is None:
        baseline_end = earthquake_date - timedelta(days=30)  # 1 month before

    try:
        grace_files = find_grace_files_for_period(data_dir, start_date, end_date)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    if not grace_files:
        print(f"No GRACE files found for {start_date} to {end_date}")
        return None

    dates, v_xx_values, v_xz_values = [], [], []

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
        print(f"Long-term average gradients computed successfully: V_xx={v_xx_avg:.2f}, V_xz={v_xz_avg:.2f}")
    except ValueError as e:
        print(f"Baseline error: {e}")
        return None

    # Process each file
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
        return None

    days_relative = [(d - earthquake_date).days for d in dates]

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Process and plot V_xx
    v_xx_array = np.array(v_xx_values)
    anomaly_indices_xx, stats_xx = detect_anomalies(v_xx_array, iqr_k)

    ax1.plot(days_relative, v_xx_values, "b-", label="Mean V_xx")
    ax1.axvline(x=0, color="r", linestyle="-", label="Earthquake")
    if len(anomaly_indices_xx) > 0:
        ax1.scatter(
            np.array(days_relative)[anomaly_indices_xx],
            v_xx_array[anomaly_indices_xx],
            color="red",
            label="Detected Anomaly",
            zorder=3,
            s=100,
            edgecolors="black",
        )
    ax1.axhline(y=stats_xx["lower_bound"], color="gray", linestyle="--", label="Lower IQR")
    ax1.axhline(y=stats_xx["upper_bound"], color="gray", linestyle="--", label="Upper IQR")
    ax1.set_ylabel("V_xx (Eötvös)")
    ax1.set_title(f"V_xx Time Series (±{region_radius}° around epicenter)")
    ax1.legend()
    ax1.grid(True)

    # Process and plot V_xz
    v_xz_array = np.array(v_xz_values)
    anomaly_indices_xz, stats_xz = detect_anomalies(v_xz_array, iqr_k)

    ax2.plot(days_relative, v_xz_values, "g-", label="Mean V_xz")
    ax2.axvline(x=0, color="r", linestyle="-", label="Earthquake")
    if len(anomaly_indices_xz) > 0:
        ax2.scatter(
            np.array(days_relative)[anomaly_indices_xz],
            v_xz_array[anomaly_indices_xz],
            color="orange",
            label="Detected Anomaly",
            zorder=3,
            s=100,
            edgecolors="black",
        )
    ax2.axhline(y=stats_xz["lower_bound"], color="gray", linestyle="--", label="Lower IQR")
    ax2.axhline(y=stats_xz["upper_bound"], color="gray", linestyle="--", label="Upper IQR")
    ax2.set_xlabel("Days Relative to Earthquake")
    ax2.set_ylabel("V_xz (Eötvös)")
    ax2.set_title(f"V_xz Time Series (±{region_radius}° around epicenter)")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Combined time series plot saved as {output_file}")
        plt.close()

    return fig
