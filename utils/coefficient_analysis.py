#!/usr/bin/env python3
"""
Script for analyzing GRACE spherical harmonic coefficients.

This script provides tools to visualize and analyze the spectral characteristics
of GRACE gravitational coefficients.
"""

import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.io import parse_date_from_filename
from src.visualization import plot_coefficient_spectrum


def main():
    parser = argparse.ArgumentParser(description="Analyze GRACE spherical harmonic coefficients.")
    parser.add_argument("gfc_file", help="Path to the GRACE GFC file")
    parser.add_argument(
        "--unit",
        default="per_l",
        choices=["per_l", "per_lm", "total"],
        help="Spectral representation unit (default: per_l)",
    )
    parser.add_argument(
        "--xscale",
        default="lin",
        choices=["lin", "log"],
        help="X-axis scaling (default: lin)",
    )
    parser.add_argument(
        "--yscale",
        default="log",
        choices=["lin", "log"],
        help="Y-axis scaling (default: log)",
    )
    parser.add_argument("--output", help="Output file path for saving the figure")

    args = parser.parse_args()

    if not os.path.exists(args.gfc_file):
        print(f"Error: GFC file not found: {args.gfc_file}")
        return

    if args.output is None:
        file_date = parse_date_from_filename(args.gfc_file)
        date_str = file_date.strftime("%Y%m%d") if file_date else "unknown_date"
        base_dir = os.path.dirname(args.gfc_file)
        base_name = os.path.basename(args.gfc_file).split(".")[0]
        args.output = os.path.join(base_dir, f"{base_name}_spectrum_{date_str}.png")

    print(f"Analyzing coefficient spectrum of {args.gfc_file}...")
    print(f"  - Unit: {args.unit}")
    print(f"  - X-axis scale: {args.xscale}")
    print(f"  - Y-axis scale: {args.yscale}")

    try:
        plot_coefficient_spectrum(
            args.gfc_file,
            unit=args.unit,
            xscale=args.xscale,
            yscale=args.yscale,
            output_file=args.output,
        )
        print(f"Spectrum analysis saved to {args.output}")
    except Exception as e:
        print(f"Error analyzing coefficients: {e}")

    print("Analysis complete!")


if __name__ == "__main__":
    main()
