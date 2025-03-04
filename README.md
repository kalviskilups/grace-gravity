# GRACE Gravity Analysis

## Overview

This package provides tools for analyzing gravitational data from the Gravity Recovery and Climate Experiment (GRACE) satellite mission. It focuses on detecting and visualizing gravitational anomalies, with particular emphasis on identifying earthquake-related signals in Earth's gravitational field.

The codebase enables researchers to:
- Load and process GRACE spherical harmonic coefficients
- Calculate gravity anomalies and gradient tensors
- Apply specialized filtering techniques for signal isolation
- Generate visualizations of temporal and spatial gravitational changes
- Perform statistical analysis to identify significant anomalies

## Table of Contents

- [Scientific Background](#scientific-background)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Package Structure](#package-structure)
- [Usage Guide](#usage-guide)
  - [Basic Workflow](#basic-workflow)
  - [Core Functions](#core-functions)
  - [Command-line Scripts](#command-line-scripts)
- [Examples](#examples)
  - [Global Anomaly Visualization](#global-anomaly-visualization)
  - [Local Earthquake Analysis](#local-earthquake-analysis)
  - [Time Series Analysis](#time-series-analysis)
- [References](#references)

## Scientific Background

### Gravitational Anomalies and Earthquakes

Large-scale mass redistributions within the Earth, such as those caused by major earthquakes, can create measurable changes in the local gravitational field. The GRACE mission, which measured Earth's gravity field with unprecedented precision, has enabled the detection of these subtle gravitational signatures.

Key scientific principles:

1. **Gravitational Perturbations**: Earthquakes displace significant masses within Earth's crust and mantle, altering the local gravitational field.

2. **Spherical Harmonic Analysis**: GRACE data is provided as spherical harmonic coefficients, which represent the Earth's gravitational field at different spatial frequencies.

3. **Gravity Gradients**: The spatial derivatives of the gravitational potential (particularly V_xx and V_xz components) have been shown to be sensitive to earthquake-related mass changes.

4. **Temporal Analysis**: By analyzing time series of gravitational measurements around seismic events, precursory and post-event anomalies can be identified.

The detection process involves:
1. Filtering the GRACE spherical harmonic coefficients to isolate specific spatial frequencies
2. Computing gravity gradients or anomalies from these coefficients
3. Analyzing time series of these measurements around earthquake events
4. Identifying statistically significant anomalies that may correlate with seismic activity

## Installation

### Requirements

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib
- Cartopy
- PyShTools

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/grace-gravity.git
   cd grace-gravity
   ```

2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2. Install required dependencies:
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

3. For development, install additional tools:
   ```bash
   uv pip install -r requirements-dev.txt
   pre-commit install
   ```

## Data Sources

This package works with GRACE data in the form of spherical harmonic coefficients, typically provided in ICGEM (International Centre for Global Earth Models) GFC file format. Primary sources include:

1. **ITSG-Grace2018**: 
   - High-resolution gravity field model from TU Graz
   - Offers daily, weekly, and monthly solutions
   - Website: [https://www.tugraz.at/institute/ifg/downloads/gravity-field-models/itsg-grace2018](https://www.tugraz.at/institute/ifg/downloads/gravity-field-models/itsg-grace2018)

2. **ICGEM Data Portal**: 
   - Provides various GRACE and GRACE-FO gravity field models
   - Website: [https://icgem.gfz-potsdam.de/home](https://icgem.gfz-potsdam.de/home)

## Package Structure

The package is organized into the following structure:

```
grace-gravity/
│
├── src/                        # Core source code
│   ├── __init__.py             # Package initialization
│   ├── functions.py            # Core mathematical functions
│   ├── io.py                   # File I/O and data loading
│   ├── processing.py           # Data processing utilities
│   └── visualization.py        # Visualization functions
│
├── scripts/                    # Command-line scripts
│   ├── global_anomaly.py       # Global anomaly visualization
│   ├── local_anomaly.py        # Local anomaly visualization
│   └── time_series_analysis.py # Time series analysis
│
├── utils/                      # Utility scripts
│   ├── clean_script.py         # GFC file cleaning utility
│   └── coefficient_analysis.py # Coefficient analysis utility
│
├── .gitignore                  # Git ignore file
├── .pre-commit-config.yaml     # Pre-commit configuration
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Package dependencies
├── requirements-dev.txt        # Development dependencies
└── README.md                   # Package documentation
```

## Usage Guide

### Basic Workflow

The typical workflow for analyzing gravitational anomalies involves:

1. **Load GRACE data**: Import spherical harmonic coefficients from GFC files
2. **Apply filtering**: Filter coefficients to focus on specific spatial frequencies
3. **Compute gravity metrics**: Calculate anomalies or gradient tensors
4. **Analyze spatially**: Examine regional changes around points of interest
5. **Analyze temporally**: Create time series to detect anomalies
6. **Visualize results**: Generate maps and plots to interpret findings

### Core Functions

#### Loading Data

```python
from src.io import load_sh_grav_coeffs, find_grace_files_for_period
from datetime import datetime, timedelta

# Load specific file
coeffs = load_sh_grav_coeffs("path/to/GRACE_data.gfc")

# Find files within a date range
earthquake_date = datetime(2010, 2, 27)  # Chile earthquake
files = find_grace_files_for_period(
    "data_directory/",
    earthquake_date - timedelta(days=30),
    earthquake_date + timedelta(days=30)
)
```

#### Computing Gravity Metrics

```python
from src.functions import compute_gravity_anomaly, compute_gravity_gradient_tensor, compute_anomaly_difference

# Compute gravity anomaly (excluding low degrees)
anomaly = compute_gravity_anomaly(coeffs, exclude_degrees=5)

# Compute gravity gradient tensor components (V_xx and V_xz)
v_xx, v_xz = compute_gravity_gradient_tensor(coeffs, max_degree=60)

# Compare two time periods
anomaly_diff = compute_anomaly_difference(coeffs1, coeffs2, exclude_degrees=5)
```

#### Filtering and Regional Analysis

```python
from src.functions import taper_coeffs, filter_earthquake_signal

# Apply band-pass filtering (degrees 10-60)
filtered_coeffs = taper_coeffs(coeffs, min_degree=10, max_degree=60, taper_width=3)

# Extract earthquake signal for specific location
v_xx, v_xz = filter_earthquake_signal(
    "GRACE_data.gfc", 
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    min_degree=10, 
    max_degree=60,
    radius=2.0  # Degrees
)
```

#### Visualization

```python
from src.visualization import plot_earthquake_anomaly, generate_time_series_plots

# Plot earthquake gravity gradient anomalies
plot_earthquake_anomaly(
    "GRACE_data.gfc",
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    min_degree=10, 
    max_degree=60,
    region_size=15.0,
    output_file="earthquake_anomaly.png"
)

# Generate time series plots
generate_time_series_plots(
    "data_directory/", 
    earthquake_date, 
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    days_before=365, 
    days_after=180,
    output_file="timeseries.png"
)
```

### Command-line Scripts

For convenient analysis, several command-line scripts are provided:

#### Global Anomaly Analysis

Visualize differences between two global gravity fields:

```bash
python scripts/global_anomaly.py path/to/first_file.gfc path/to/second_file.gfc \
    --title "Global Changes" \
    --output global.png
```

#### Local Anomaly Analysis

Focus on gravitational changes in a region of interest:

```bash
python scripts/local_anomaly.py path/to/first_file.gfc path/to/second_file.gfc \
    --lat -35.846 \
    --lon -72.719 \
    --region 15 \
    --output local.png
```

#### Time Series Analysis

Analyze gravitational changes over time:

```bash
python scripts/time_series_analysis.py \
    --data-dir data/ \
    --eq-date 2010-02-27 \
    --lat -35.846 \
    --lon -72.719 \
    --output-dir results/ \
    --min-degree 10 \
    --max-degree 60 \
    --days-before 365 \
    --days-after 180 \
    --event-name chile_earthquake
```

#### Coefficient Analysis

Analyze the spectral characteristics of GRACE coefficients:

```bash
python utils/coefficient_analysis.py path/to/file.gfc \
    --unit per_l \
    --yscale log \
    --output spectrum.png
```

#### GFC File Cleaning

Clean GRACE GFC files by removing unnecessary reference blocks:

```bash
python utils/clean_script.py \
    --directory data/ \
    --pattern "*.gfc" \
    --verbose
```

## Examples

### Global Anomaly Visualization

This example shows how to visualize global gravitational changes between two time periods:

```python
from src.io import load_sh_grav_coeffs
from src.functions import compute_anomaly_difference
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Load coefficients from two different time periods
coeffs1 = load_sh_grav_coeffs("GRACE_2010_01.gfc")
coeffs2 = load_sh_grav_coeffs("GRACE_2010_03.gfc")

# Compute the difference (January to March 2010)
anomaly_diff = compute_anomaly_difference(coeffs1, coeffs2, exclude_degrees=5)

# Create visualization
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines()
ax.gridlines()

img_extent = (-180, 180, -90, 90)
im = ax.imshow(
    anomaly_diff, 
    extent=img_extent, 
    transform=ccrs.PlateCarree(), 
    cmap="viridis", 
    origin="upper"
)

ax.set_global()
cbar = plt.colorbar(im, ax=ax, shrink=0.8, orientation="horizontal", pad=0.05)
cbar.set_label("Gravity Anomaly Difference (microGal)")
plt.title("Global Gravitational Changes (Jan-Mar 2010)")
plt.savefig("global_changes.png", dpi=300, bbox_inches="tight")
```

### Local Earthquake Analysis

This example focuses on the 2010 Chile earthquake:

```python
from src.io import load_sh_grav_coeffs, find_grace_files_for_period
from src.visualization import plot_earthquake_anomaly
from datetime import datetime, timedelta

# Define the earthquake parameters
earthquake_date = datetime(2010, 2, 27)
epicenter_lat = -35.846
epicenter_lon = -72.719

# Find GRACE data closest to the earthquake
grace_files = find_grace_files_for_period(
    "data/", 
    earthquake_date, 
    earthquake_date + timedelta(days=30)
)

# Visualize gravity gradients
if grace_files:
    plot_earthquake_anomaly(
        grace_files[0],
        epicenter_lat, 
        epicenter_lon,
        min_degree=10, 
        max_degree=60,
        region_size=20.0,
        title=f"Chile Earthquake ({earthquake_date.strftime('%Y-%m-%d')}) Gravity Gradients",
        output_file="chile_eq_gradients.png"
    )
```

### Time Series Analysis

This example detects gravitational anomalies over time:

```python
from src.visualization import generate_time_series_plots
from datetime import datetime

# Generate time series for the 2010 Chile earthquake
generate_time_series_plots(
    "data/", 
    datetime(2010, 2, 27),  # Earthquake date
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    min_degree=10,
    max_degree=60,
    days_before=365,  # 1 year before
    days_after=180,   # 6 months after
    region_radius=2.0,
    iqr_k=2.5,  # IQR multiplier for anomaly detection
    output_file="chile_timeseries.png"
)
```


## References

1. **Gravity field analysis after the GRACE mission**  
   *Acta Geodaetica et Geophysica*, 2016  
   [https://link.springer.com/article/10.1515/acgeo-2016-0034](https://link.springer.com/article/10.1515/acgeo-2016-0034)

2. **ITSG-Grace2018 gravity field model**  
   Institute of Geodesy, TU Graz  
   [https://www.tugraz.at/institute/ifg/downloads/gravity-field-models/itsg-grace2018](https://www.tugraz.at/institute/ifg/downloads/gravity-field-models/itsg-grace2018)

3. **International Centre for Global Earth Models (ICGEM)**  
   [https://icgem.gfz-potsdam.de/home](https://icgem.gfz-potsdam.de/home)

4. **Detection of gravity changes before powerful earthquakes in GRACE satellite observations**  
   *ResearchGate*, 2015  
   [https://www.researchgate.net/publication/272825751_Detection_of_gravity_changes_before_powerful_earthquakes_in_GRACE_satellite_observations](https://www.researchgate.net/publication/272825751_Detection_of_gravity_changes_before_powerful_earthquakes_in_GRACE_satellite_observations)

5. **L3Py Filtering Documentation**  
   [https://l3py.readthedocs.io/en/latest/_generated/l3py.filter.html#module-l3py.filter](https://l3py.readthedocs.io/en/latest/_generated/l3py.filter.html#module-l3py.filter)

6. **GLDAS: Global Land Data Assimilation System**  
   NASA Goddard Earth Sciences Data and Information Services Center  
   [https://disc.gsfc.nasa.gov/datasets?keywords=GLDAS&page=1&project=GLDAS](https://disc.gsfc.nasa.gov/datasets?keywords=GLDAS&page=1&project=GLDAS)