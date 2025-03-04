#!/usr/bin/env python3
"""
Script to visualize localized gravitational anomalies.

This script demonstrates how to visualize changes in Earth's gravitational field
in a specific region, typically around an earthquake epicenter.
"""

import argparse
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.functions import compute_anomaly_difference
from src.io import load_sh_grav_coeffs, parse_date_from_filename
from src.processing import normalize_longitude


def plot_localized_anomaly(
    anomaly_difference,
    epicenter_lat,
    epicenter_lon,
    region_size=15.0,
    title=None,
    output_file=None,
):
    """
    Plot localized gravitational anomaly around a specific point.

    Args:
        anomaly_difference: Anomaly difference array
        epicenter_lat: Latitude of the point of interest (epicenter)
        epicenter_lon: Longitude of the point of interest (epicenter)
        region_size: Size of region to display in degrees
        title: Plot title
        output_file: Path to save output file
    """
    # Normalize longitude
    epicenter_lon = normalize_longitude(epicenter_lon)

    # Set up the grid
    nlat, nlon = anomaly_difference.shape
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
    region_data = anomaly_difference[np.ix_(lat_indices, lon_indices)]
    region_lats = lats[lat_indices]
    region_lons = lons[lon_indices]
    region_lon_grid, region_lat_grid = np.meshgrid(region_lons, region_lats)

    # Create higher resolution grid for smooth plotting
    refine_factor = 2
    xi = np.linspace(lon_min, lon_max, len(lon_indices) * refine_factor)
    yi = np.linspace(lat_min, lat_max, len(lat_indices) * refine_factor)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate data to higher resolution grid
    zi = griddata(
        (region_lon_grid.flatten(), region_lat_grid.flatten()),
        region_data.flatten(),
        (xi, yi),
        method="cubic",
    )

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)
    ax.gridlines(draw_labels=True)

    levels = np.linspace(np.nanmin(zi), np.nanmax(zi), 50)
    contour = ax.contourf(
        xi,
        yi,
        zi,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        extend="both",
    )

    ax.plot(
        epicenter_lon,
        epicenter_lat,
        "r*",
        markersize=15,
        transform=ccrs.PlateCarree(),
        label="Epicenter",
    )

    cbar = plt.colorbar(contour, ax=ax, orientation="vertical", pad=0.05)
    cbar.set_label("Gravity Anomaly Difference (microGal)")

    if title:
        plt.title(title)
    else:
        plt.title(f"Localized Gravitational Anomaly (±{region_size}° around epicenter)")

    ax.legend(loc="upper right")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Figure saved as {output_file}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize localized gravitational anomalies.")
    parser.add_argument("file1", help="Path to first GRACE GFC file")
    parser.add_argument("file2", help="Path to second GRACE GFC file")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of epicenter")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of epicenter")
    parser.add_argument(
        "--region",
        type=float,
        default=15.0,
        help="Region size in degrees (default: 15.0)",
    )
    parser.add_argument("--title", help="Title for the plot")
    parser.add_argument("--output", help="Output file path for saving the figure")

    args = parser.parse_args()

    # Load coefficients
    print(f"Loading first GFC file: {args.file1}")
    coeffs1 = load_sh_grav_coeffs(args.file1)

    print(f"Loading second GFC file: {args.file2}")
    coeffs2 = load_sh_grav_coeffs(args.file2)

    # Get file dates for automatic title if needed
    if not args.title:
        date1 = parse_date_from_filename(args.file1)
        date2 = parse_date_from_filename(args.file2)
        date_str1 = date1.strftime("%Y-%m-%d") if date1 else "Unknown"
        date_str2 = date2.strftime("%Y-%m-%d") if date2 else "Unknown"
        title = f"Gravitational Change: {date_str1} to {date_str2}"
    else:
        title = args.title

    # Compute anomaly difference
    print("Computing gravitational anomaly difference...")
    anomaly_difference = compute_anomaly_difference(coeffs1, coeffs2)

    # Plot the localized anomaly
    print("Generating visualization...")
    plot_localized_anomaly(
        anomaly_difference,
        args.lat,
        args.lon,
        region_size=args.region,
        title=title,
        output_file=args.output,
    )

    print("Done!")


if __name__ == "__main__":
    main()
