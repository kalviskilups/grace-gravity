#!/usr/bin/env python3
"""
Script to visualize global gravitational anomalies with enhanced visualization.

This script demonstrates how to visualize changes in Earth's gravitational field
by comparing two GRACE datasets and plotting the global anomaly differences with
improved visualization features including interpolation and optional mass inversion.
"""

import argparse
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.functions import compute_anomaly_difference
from src.io import load_sh_grav_coeffs


def plot_global_anomaly(
    anomaly_difference,
    title=None,
    output_file=None,
    projection=ccrs.Robinson(),
    cmap="RdYlBu_r",
    interpolate=True,
    show_inversion=False,
    conversion_factor=1.0,
    smoothing_sigma=2,
):
    """
    Plot global gravitational anomaly map with enhanced visualization.

    Args:
        anomaly_difference: Anomaly difference array
        title: Plot title
        output_file: Path to save output file
        projection: Cartopy projection (default: Robinson)
        cmap: Colormap to use (default: RdYlBu_r)
        interpolate: Whether to interpolate for smoother visualization (default: True)
        show_inversion: Whether to show equivalent mass change (default: False)
        conversion_factor: Conversion factor for mass inversion (default: 1.0)
        smoothing_sigma: Gaussian smoothing sigma for inversion (default: 2)
    """
    if show_inversion:
        # Create a figure with two subplots: anomaly and inversion
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), subplot_kw={"projection": projection})
    else:
        # Create a figure with a single subplot
        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.axes(projection=projection)

    # Determine grid dimensions
    nlat, nlon = anomaly_difference.shape
    lats = np.linspace(-90, 90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Plot anomaly on the first subplot
    ax1.set_global()
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=":")
    ax1.add_feature(cfeature.LAND, alpha=0.1)
    ax1.add_feature(cfeature.OCEAN, alpha=0.1)
    ax1.gridlines(draw_labels=True)

    if interpolate:
        # Create higher resolution grid for smoother visualization
        refine_factor = 2
        xi = np.linspace(0, 360, nlon * refine_factor)
        yi = np.linspace(-90, 90, nlat * refine_factor)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate data to higher resolution grid
        zi = griddata(
            (lon_grid.flatten(), lat_grid.flatten()),
            anomaly_difference.flatten(),
            (xi, yi),
            method="cubic",
        )

        # Plot using contourf for smoother appearance
        levels = np.linspace(
            np.nanpercentile(zi, 2),  # 2nd percentile to avoid outliers
            np.nanpercentile(zi, 98),  # 98th percentile to avoid outliers
            50,
        )
        pcm1 = ax1.contourf(
            xi,
            yi,
            zi,
            levels=levels,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            extend="both",
        )
    else:
        # Use traditional imshow approach
        img_extent = (0, 360, -90, 90)
        pcm1 = ax1.imshow(
            anomaly_difference,
            extent=img_extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            origin="upper",
        )

    # Add colorbar to first subplot
    cbar1 = plt.colorbar(pcm1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar1.set_label("Gravity Anomaly Difference (microGal)")

    # Add title to the first subplot
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title("Global Gravitational Anomaly Difference")

    # If showing inversion, add second subplot
    if show_inversion:
        # Apply inversion to estimate equivalent water thickness
        inversion = gaussian_filter(anomaly_difference, sigma=smoothing_sigma) * conversion_factor

        ax2.set_global()
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS, linestyle=":")
        ax2.add_feature(cfeature.LAND, alpha=0.1)
        ax2.add_feature(cfeature.OCEAN, alpha=0.1)
        ax2.gridlines(draw_labels=True)

        if interpolate:
            # Interpolate inversion data
            zi_inv = griddata(
                (lon_grid.flatten(), lat_grid.flatten()),
                inversion.flatten(),
                (xi, yi),
                method="cubic",
            )

            # Plot using contourf
            levels_inv = np.linspace(
                np.nanpercentile(zi_inv, 2),
                np.nanpercentile(zi_inv, 98),
                50,
            )
            pcm2 = ax2.contourf(
                xi,
                yi,
                zi_inv,
                levels=levels_inv,
                transform=ccrs.PlateCarree(),
                cmap="viridis",
                extend="both",
            )
        else:
            # Use traditional imshow approach
            pcm2 = ax2.imshow(
                inversion,
                extent=img_extent,
                transform=ccrs.PlateCarree(),
                cmap="viridis",
                origin="upper",
            )

        # Add colorbar to second subplot
        cbar2 = plt.colorbar(pcm2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8)
        cbar2.set_label("Equivalent Water Thickness (cm)")

        # Add title to the second subplot
        ax2.set_title("Inverted Mass Change")

        # Add overall title
        fig.suptitle(title if title else "Global Gravitational Analysis", fontsize=16, y=0.98)

    plt.tight_layout()

    # Save figure if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Figure saved as {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize global gravitational anomalies.")
    parser.add_argument("file1", help="Path to first GRACE GFC file")
    parser.add_argument("file2", help="Path to second GRACE GFC file")
    parser.add_argument("--title", help="Title for the plot")
    parser.add_argument("--output", help="Output file path for saving the figure")
    parser.add_argument(
        "--projection",
        default="platecarree",
        choices=["robinson", "platecarree", "mollweide", "orthographic"],
        help="Map projection to use (default: robinson)",
    )
    parser.add_argument("--cmap", default="RdYlBu_r", help="Colormap to use (default: RdYlBu_r)")
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable interpolation for faster plotting",
    )
    parser.add_argument("--show-inversion", action="store_true", help="Show equivalent mass change")
    parser.add_argument(
        "--conversion-factor",
        type=float,
        default=1.0,
        help="Conversion factor for gravity to mass (default: 1.0)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma for inversion (default: 2.0)",
    )

    args = parser.parse_args()

    # Load coefficients
    print(f"Loading first GFC file: {args.file1}")
    coeffs1 = load_sh_grav_coeffs(args.file1)

    print(f"Loading second GFC file: {args.file2}")
    coeffs2 = load_sh_grav_coeffs(args.file2)

    # Compute anomaly difference
    print("Computing gravitational anomaly difference...")
    anomaly_difference = compute_anomaly_difference(coeffs1, coeffs2, exclude_degrees=10)

    # Set projection based on user input
    projections = {
        "robinson": ccrs.Robinson(),
        "platecarree": ccrs.PlateCarree(),
        "mollweide": ccrs.Mollweide(),
        "orthographic": ccrs.Orthographic(central_longitude=0, central_latitude=0),
    }
    projection = projections[args.projection]

    # Plot the anomaly
    print("Generating visualization...")
    plot_global_anomaly(
        anomaly_difference,
        title=args.title,
        output_file=args.output,
        projection=projection,
        cmap=args.cmap,
        interpolate=not args.no_interpolate,
        show_inversion=args.show_inversion,
        conversion_factor=args.conversion_factor,
        smoothing_sigma=args.smoothing,
    )

    print("Done!")


if __name__ == "__main__":
    main()
