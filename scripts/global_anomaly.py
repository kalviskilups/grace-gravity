#!/usr/bin/env python3
"""
Script to visualize global gravitational anomalies.

This script demonstrates how to visualize changes in Earth's gravitational field
by comparing two GRACE datasets and plotting the global anomaly differences.
"""

import argparse
import os
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.functions import compute_anomaly_difference
from src.io import load_sh_grav_coeffs


def plot_global_anomaly(anomaly_difference, title=None, output_file=None):
    """
    Plot global gravitational anomaly map.

    Args:
        anomaly_difference: Anomaly difference array
        title: Plot title
        output_file: Path to save output file
    """
    # Set up the projection
    ax = plt.axes(projection=ccrs.Robinson())

    # Add coastlines and grid
    ax.coastlines()
    ax.gridlines()

    img_extent = (-180, 180, -90, 90)
    im = ax.imshow(anomaly_difference, extent=img_extent, transform=ccrs.PlateCarree(), cmap="viridis", origin="upper")

    # Set global view
    ax.set_global()

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, orientation="horizontal", pad=0.05)
    cbar.set_label("Gravity Anomaly Difference (microGal)")

    # Add title
    if title:
        plt.title(title)
    else:
        plt.title("Global Gravitational Anomaly Difference")

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

    args = parser.parse_args()

    # Load coefficients
    print(f"Loading first GFC file: {args.file1}")
    coeffs1 = load_sh_grav_coeffs(args.file1)

    print(f"Loading second GFC file: {args.file2}")
    coeffs2 = load_sh_grav_coeffs(args.file2)

    # Compute anomaly difference
    print("Computing gravitational anomaly difference...")
    anomaly_difference = compute_anomaly_difference(coeffs1, coeffs2, coeffs1.lmax)

    # Plot the anomaly
    print("Generating visualization...")
    plot_global_anomaly(anomaly_difference, title=args.title, output_file=args.output)

    print("Done!")


if __name__ == "__main__":
    main()
