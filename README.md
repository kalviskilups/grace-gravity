# GRACE Gravity Analysis Documentation

## Overview

This package provides tools for analyzing gravitational data from the Gravity Recovery and Climate Experiment (GRACE) satellite mission. It focuses on detecting and visualizing gravitational anomalies, with particular emphasis on earthquake-related signals.

The codebase supports loading GRACE spherical harmonic coefficients, calculating gravity anomalies and gradients, and generating visualizations to identify temporal and spatial changes in Earth's gravitational field.

## Table of Contents

1. [Gravitational Anomalies and Earthquakes](#gravitational-anomalies-and-earthquakes)
2. [Data Sources](#data-sources)
3. [Package Structure](#package-structure)
4. [Core Functionality](#core-functionality)
5. [Visualization Tools](#visualization-tools)
6. [Command-line Scripts](#command-line-scripts)
7. [Examples](#examples)
8. [References](#references)


## Gravitational Anomalies and Earthquakes

Research has shown that powerful earthquakes can cause detectable changes in Earth's gravitational field. These changes can be observed in GRACE data both before and after significant seismic events. By analyzing temporal changes in the gravitational field near earthquake epicenters, researchers can gain insights into the geophysical processes associated with these events.

The detection process involves:
1. Filtering the GRACE spherical harmonic coefficients to isolate specific spatial frequencies
2. Computing gravity gradients or anomalies from these coefficients
3. Analyzing time series of these measurements around earthquake events
4. Identifying statistically significant anomalies that may correlate with seismic activity

## Data Sources

This package works with GRACE data in the form of spherical harmonic coefficients, typically provided in ICGEM (International Centre for Global Earth Models) GFC file format. Primary sources include:

1. **ITSG-Grace2018**: A high-resolution gravity field model from TU Graz, offering daily, weekly, and monthly solutions
   - Website: https://www.tugraz.at/institute/ifg/downloads/gravity-field-models/itsg-grace2018

2. **ICGEM Data Portal**: Provides various GRACE and GRACE-FO gravity field models
   - Website: https://icgem.gfz-potsdam.de/home

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
│   ├── time_series_analysis.py # Time series analysis
│   └── coefficient_analysis.py # Coefficient spectrum analysis
│
├── utils/                      # Utility scripts
│   ├── __init__.py             # Package initialization
│   ├── clean_script.py         # GFC file cleaning utility
│   └── coefficient_analysis.py # Coefficient analysis utility
│
├── requirements.txt            # Package dependencies
└── README.md                   # Package overview
```

## Core Functionality

### Gravity Field Calculations

The package provides functions for computing various gravity field metrics:

#### Gravity Anomalies

Gravity anomalies represent deviations from a reference gravity field model. They are typically computed by:

1. Loading spherical harmonic coefficients
2. Filtering out low-degree components (which represent Earth's overall shape)
3. Computing the gravity anomaly grid using spherical harmonic synthesis

```python
from src.io import load_sh_grav_coeffs
from src.functions import compute_gravity_anomaly

# Load coefficients
coeffs = load_sh_grav_coeffs("GRACE_data.gfc")

# Compute anomaly (excluding degrees 0-5)
anomaly = compute_gravity_anomaly(coeffs, exclude_degrees=5)
```

#### Gravity Gradients

Gravity gradients are spatial derivatives of the gravitational potential. The package focuses on the V_xx and V_xz components, which have been shown to be sensitive to earthquake signals:

```python
from src.functions import compute_gravity_gradient_tensor

# Compute gravity gradient tensor components
v_xx, v_xz = compute_gravity_gradient_tensor(coeffs, max_degree=60)
```

### Filtering and Analysis

The package includes functions for filtering and analyzing gravity field data:

#### Coefficient Filtering

Spherical harmonic coefficients can be filtered to focus on specific spatial scales:

```python
from src.functions import taper_coeffs

# Apply band-pass filtering (degrees 10-60)
filtered_coeffs = taper_coeffs(coeffs, min_degree=10, max_degree=60, taper_width=3)
```

#### Earthquake Signal Analysis

The package provides specialized functions for analyzing gravity changes around earthquake epicenters:

```python
from src.functions import filter_earthquake_signal

# Extract earthquake signal for the 2010 Chile earthquake
v_xx, v_xz = filter_earthquake_signal(
    "GRACE_data.gfc", 
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    min_degree=10, 
    max_degree=60
)
```

#### Time Series Analysis

For detecting temporal anomalies, the package includes functions for time series analysis:

```python
from src.functions import compute_long_term_average
from datetime import datetime

# Compute baseline gravitational values
v_xx_avg, v_xz_avg = compute_long_term_average(
    "data/", 
    datetime(2008, 1, 1), 
    datetime(2009, 12, 31),
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    min_degree=10, 
    max_degree=60
)
```

## Visualization Tools

The package provides several visualization functions:

### Global Anomaly Maps

```python
from src.visualization import plot_global_anomaly

# Plot global gravitational changes
plot_global_anomaly(anomaly_difference, title="Global Gravity Changes", output_file="global.png")
```

### Localized Anomaly Maps

```python
from src.visualization import plot_localized_anomaly

# Plot localized gravitational changes around an earthquake epicenter
plot_localized_anomaly(
    anomaly_difference, 
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    region_size=15.0, 
    output_file="local.png"
)
```

### Time Series Plots

```python
from src.visualization import generate_time_series_plots

# Generate time series plots for the 2010 Chile earthquake
generate_time_series_plots(
    "data/", 
    "2010-02-27", 
    epicenter_lat=-35.846, 
    epicenter_lon=-72.719,
    days_before=365, 
    days_after=180,
    output_file="timeseries.png"
)
```

### Coefficient Spectrum Plots

```python
from src.visualization import plot_coefficient_spectrum

# Plot the spectral characteristics of spherical harmonic coefficients
plot_coefficient_spectrum("GRACE_data.gfc", unit="per_l", xscale="log", output_file="spectrum.png")
```

## Scripts

### Global Anomaly Analysis

```bash
python scripts/global_anomaly.py path/to/first_file.gfc path/to/second_file.gfc --title "Global Changes" --output global.png
```

### Local Anomaly Analysis

```bash
python scripts/local_anomaly.py path/to/first_file.gfc path/to/second_file.gfc --lat -35.846 --lon -72.719 --region 15 --output local.png
```

### Time Series Analysis

```bash
python scripts/time_series_analysis.py --data-dir data/ --eq-date 2010-02-27 --lat -35.846 --lon -72.719 --output-dir results/ --event-name chile_earthquake
```

### Coefficient Analysis

```bash
python scripts/coefficient_analysis.py path/to/file.gfc --unit per_l --yscale log --output spectrum.png
```

### GFC File Cleaning

```bash
python utils/clean_script.py --directory data/ --pattern "*.gfc" --verbose
```

## Example: Analyzing the 2010 Chile Earthquake

```python
from src.io import load_sh_grav_coeffs, find_grace_files_for_period
from src.functions import compute_anomaly_difference
from src.visualization import plot_localized_anomaly, generate_time_series_plots
from datetime import datetime

# Define the earthquake parameters
earthquake_date = datetime(2010, 2, 27)
epicenter_lat = -35.846
epicenter_lon = -72.719

# Find GRACE files before and after the earthquake
files_before = find_grace_files_for_period("data/", earthquake_date - timedelta(days=30), earthquake_date)
files_after = find_grace_files_for_period("data/", earthquake_date, earthquake_date + timedelta(days=30))

# Compare gravity fields
coeffs_before = load_sh_grav_coeffs(files_before[0])
coeffs_after = load_sh_grav_coeffs(files_after[0])
anomaly_diff = compute_anomaly_difference(coeffs_before, coeffs_after)

# Visualize local changes
plot_localized_anomaly(
    anomaly_diff, 
    epicenter_lat, 
    epicenter_lon, 
    region_size=20, 
    title="Chile Earthquake Gravity Changes",
    output_file="chile_anomaly.png"
)

# Generate time series
generate_time_series_plots(
    "data/", 
    earthquake_date, 
    epicenter_lat, 
    epicenter_lon,
    days_before=365, 
    days_after=180,
    output_file="chile_timeseries.png"
)
```

## References

1. **Gravity field analysis after the GRACE mission**  
   *Acta Geodaetica et Geophysica*, 2016  
   https://link.springer.com/article/10.1515/acgeo-2016-0034

2. **ITSG-Grace2018 gravity field model**  
   Institute of Geodesy, TU Graz  
   https://www.tugraz.at/institute/ifg/downloads/gravity-field-models/itsg-grace2018

3. **International Centre for Global Earth Models (ICGEM)**  
   https://icgem.gfz-potsdam.de/home

4. **Detection of gravity changes before powerful earthquakes in GRACE satellite observations**  
   *ResearchGate*, 2015  
   https://www.researchgate.net/publication/272825751_Detection_of_gravity_changes_before_powerful_earthquakes_in_GRACE_satellite_observations

5. **L3Py Filtering Documentation**  
   https://l3py.readthedocs.io/en/latest/_generated/l3py.filter.html#module-l3py.filter

6. **GLDAS: Global Land Data Assimilation System**  
   NASA Goddard Earth Sciences Data and Information Services Center  
   https://disc.gsfc.nasa.gov/datasets?keywords=GLDAS&page=1&project=GLDAS