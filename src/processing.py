"""
Processing utilities for GRACE gravity field data.

This module provides helper functions for data processing tasks such as
coordinate normalization, region filtering, and anomaly detection.
"""

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


def normalize_longitude(lon: float) -> float:
    """
    Normalize longitude to 0-360 range.

    Args:
        lon: Longitude in degrees (any range)

    Returns:
        Normalized longitude in range [0, 360)

    Examples:
        >>> normalize_longitude(-180.0)
        180.0
        >>> normalize_longitude(370.0)
        10.0
    """
    return lon % 360


def filter_by_region(
    data: NDArray, lats: NDArray, lons: NDArray, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Filter data to include only points within specified region.

    Args:
        data: Grid data to be filtered
        lats: Latitude array
        lons: Longitude array
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        lon_min: Minimum longitude
        lon_max: Maximum longitude

    Returns:
        Tuple of (filtered_data, region_lats, region_lons)

    Examples:
        >>> data = np.random.random((180, 360))
        >>> lats = np.linspace(-90, 90, 180)
        >>> lons = np.linspace(0, 359, 360)
        >>> region_data, reg_lats, reg_lons = filter_by_region(
        ...     data, lats, lons, 30, 45, 10, 30
        ... )
        >>> reg_lats.min(), reg_lats.max()
        (30.0, 45.0)
    """
    # Normalize longitudes
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
    region_data = data[np.ix_(lat_indices, lon_indices)]
    region_lats = lats[lat_indices]
    region_lons = lons[lon_indices]

    return region_data, region_lats, region_lons


def detect_anomalies(values: NDArray, iqr_k: float = 2.5) -> Tuple[NDArray, Dict[str, float]]:
    """
    Detect anomalies in a dataset using interquartile range (IQR) method.

    Args:
        values: Array of values to analyze
        iqr_k: Multiplier for IQR bounds (default: 2.5)

    Returns:
        Tuple of (anomaly_indices, stats) where stats is a dictionary
        containing boundary values and statistical summaries

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5, 20, 6, 7, 8, 9])
        >>> indices, stats = detect_anomalies(data, iqr_k=1.5)
        >>> indices  # Index of the outlier (20)
        array([5])
        >>> stats["upper_bound"] > 9  # Upper bound should detect the outlier
        True
    """
    # Calculate quartiles
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    # Define bounds
    lower_bound = q1 - iqr_k * iqr
    upper_bound = q3 + iqr_k * iqr

    # Find anomalies
    anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]

    # Create stats dictionary
    stats = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "anomalies_count": len(anomaly_indices),
    }

    return anomaly_indices, stats
