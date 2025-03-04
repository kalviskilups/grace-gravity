#!/usr/bin/env python3
"""
Script for earthquake gravity time series analysis.

This script demonstrates how to generate time series visualizations of gravity gradient
anomalies before and after earthquakes, identifying potential anomalies.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.io import find_grace_files_for_period
from src.visualization import generate_time_series_plots, plot_earthquake_anomaly


def parse_date(date_str):
    """Parse date string into datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(description="Analyze time series of gravity gradients around earthquakes.")
    parser.add_argument("--data-dir", required=True, help="Directory containing GRACE GFC files")
    parser.add_argument("--eq-date", required=True, type=parse_date, help="Earthquake date (YYYY-MM-DD)")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of earthquake epicenter")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of earthquake epicenter")
    parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    parser.add_argument(
        "--min-degree",
        type=int,
        default=10,
        help="Minimum harmonic degree (default: 10)",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=60,
        help="Maximum harmonic degree (default: 60)",
    )
    parser.add_argument(
        "--days-before",
        type=int,
        default=365,
        help="Days before earthquake to analyze (default: 365)",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=365,
        help="Days after earthquake to analyze (default: 365)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Analysis radius in degrees (default: 2.0)",
    )
    parser.add_argument(
        "--map-region",
        type=float,
        default=15.0,
        help="Map region size in degrees (default: 15.0)",
    )
    parser.add_argument(
        "--iqr-k",
        type=float,
        default=2.5,
        help="IQR multiplier for anomaly detection (default: 2.5)",
    )
    parser.add_argument("--event-name", help="Name of the event (for file naming)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up file naming
    event_name = args.event_name if args.event_name else f"event_{args.eq_date.strftime('%Y%m%d')}"

    # Generate output file paths
    map_file = os.path.join(args.output_dir, f"{event_name}_gravity_map.png")
    timeseries_file = os.path.join(args.output_dir, f"{event_name}_time_series.png")

    # 1. Plot gravity gradient map around earthquake date
    print(f"Generating gravity gradient map for {args.eq_date}...")
    try:
        # Find a GFC file close to the earthquake date
        earthquake_files = find_grace_files_for_period(args.data_dir, args.eq_date, args.eq_date + timedelta(days=15))

        if earthquake_files:
            selected_file = earthquake_files[0]  # Use the first file found
            print(f"Using GFC file: {selected_file}")

            plot_earthquake_anomaly(
                selected_file,
                args.lat,
                args.lon,
                min_degree=args.min_degree,
                max_degree=args.max_degree,
                region_size=args.map_region,
                title=f"Earthquake ({args.eq_date.strftime('%Y-%m-%d')}) Gravity Gradients",
                output_file=map_file,
            )
        print(f"Gravity gradient map saved as {map_file}")
    except Exception as e:
        print(f"Error generating gravity map: {e}")

    # 2. Generate time series analysis
    print("Generating time series analysis...")
    try:
        generate_time_series_plots(
            args.data_dir,
            args.eq_date,
            args.lat,
            args.lon,
            min_degree=args.min_degree,
            max_degree=args.max_degree,
            days_before=args.days_before,
            days_after=args.days_after,
            region_radius=args.radius,
            output_file=timeseries_file,
            iqr_k=args.iqr_k,
        )
        print(f"Time series plot saved as {timeseries_file}")
    except Exception as e:
        print(f"Error generating time series: {e}")

    print("Analysis complete!")


if __name__ == "__main__":
    main()
