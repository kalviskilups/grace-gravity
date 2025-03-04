"""
IO operations for GRACE gravity field data.

This module provides functions for loading spherical harmonic coefficients
from GRACE GFC files and handling file operations.
"""

import glob
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional

import pyshtools as pysh


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


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date information from GFC filename.

    Args:
        filename: Path to the GFC file

    Returns:
        Extracted date object or None if no date pattern is found

    Examples:
        >>> parse_date_from_filename("ITSG-Grace2014_2004-12-28.gfc")
        datetime.datetime(2004, 12, 28, 0, 0)
        >>> parse_date_from_filename("CSR_Release-06_2004336")
        datetime.datetime(2004, 12, 1, 0, 0)  # Dec 1, 2004 (day 336)
        >>> parse_date_from_filename("GFZ_RL06_2004_09")
        datetime.datetime(2004, 9, 15, 0, 0)  # Sep 15, 2004 (mid-month)
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

    Examples:
        >>> files = find_grace_files_for_period(
        ...     "data/",
        ...     datetime(2010, 1, 1),
        ...     datetime(2010, 3, 1)
        ... )
        >>> files  # List of files between Jan 1 and Mar 1, 2010
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
