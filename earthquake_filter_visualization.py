import os
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from src.functions import (
    compute_gravity_gradient_tensor,
    compute_long_term_average,
    filter_earthquake_signal,
    find_grace_files_for_period,
    load_sh_grav_coeffs,
    parse_date_from_filename,
)


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
    DATA_DIR = "/home/kilups/Downloads/ITSG_ITSG-Grace2018_monthly_120"
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
