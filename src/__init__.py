"""
GRACE gravity field analysis package.

This package provides functionality for loading, processing, analyzing, and visualizing
gravitational data from the GRACE satellite mission.
"""

from src.functions import (
    compute_anomaly_difference,
    compute_gravity_anomaly,
    compute_gravity_gradient_tensor,
    compute_long_term_average,
    filter_earthquake_signal,
    haversine_distance,
    taper_coeffs,
)
from src.io import (
    find_grace_files_for_period,
    load_sh_grav_coeffs,
    parse_date_from_filename,
)
from src.processing import (
    detect_anomalies,
    filter_by_region,
    normalize_longitude,
)
from src.visualization import (
    generate_time_series_plots,
    plot_coefficient_spectrum,
    plot_earthquake_anomaly,
)

__all__ = [
    # Functions module
    "haversine_distance",
    "compute_gravity_anomaly",
    "compute_gravity_gradient_tensor",
    "compute_anomaly_difference",
    "taper_coeffs",
    "filter_earthquake_signal",
    "compute_long_term_average",
    # IO module
    "load_sh_grav_coeffs",
    "parse_date_from_filename",
    "find_grace_files_for_period",
    # Visualization module
    "plot_coefficient_spectrum",
    "plot_earthquake_anomaly",
    "generate_time_series_plots",
    # Processing module
    "normalize_longitude",
    "filter_by_region",
    "detect_anomalies",
]
