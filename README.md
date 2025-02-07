# GRACE Gravity Anomaly Detection Project

## Overview

This project focuses on detecting and visualizing gravitational anomalies using data from the Gravity Recovery and Climate Experiment (GRACE) satellite mission. The analysis involves processing spherical harmonic coefficients to identify subtle changes in Earth's gravitational field.

## Project Structure

```
grace-gravity/
│
├── src/                         # Source code directory
│   ├── __init__.py
│   ├── functions.py             # Core functions for anomaly detection
│
├── data/                        # Folder for GRACE data files
│   ├── ...            
│
├── legacy_notebooks/            # Jupyter notebooks for experimentation
│   ├── main.ipynb               
│   ├── new_approach.ipynb
│
├── coefficient_analysis.py  # Script for coefficient spectrum analysis
├── global_anomaly.py        # Script for global anomaly visualization
├── local_anomaly.py         # Script for localized anomaly visualization
├── .pre-commit-config.yaml      # Pre-commit configuration
├── pyproject.toml               # Ruff configuration
├── requirements-dev.txt         # Development dependencies
├── requirements.txt             # Production dependencies
└── README.md                    # Project documentation
```

## Dependencies

### Core Scientific Libraries
- `pyshtools`: Spherical harmonic analysis
- `numpy`: Numerical computing
- `matplotlib`: Visualization
- `cartopy`: Geospatial data visualization

### Development Tools
- `ruff`: Code formatting and linting
- `pre-commit`: Git hook management

## Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/grace-anomaly-detection.git
cd grace-anomaly-detection
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install uv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

4. Install pre-commit hooks
```bash
pre-commit install
```

## Key Functions

### `load_sh_grav_coeffs(gfc_file, format="icgem")`
- Loads spherical harmonic coefficients from GRACE GFC files
- Supports ICGEM (International Centre for Global Earth Models) format

### `compute_gravity_anomaly(coeffs, exclude_degrees=5)`
- Computes gravity anomalies by filtering out low-degree coefficients
- Allows customization of excluded degrees
- Applies reference gravity correction

### `compute_anomaly_difference(coeffs1, coeffs2)`
- Calculates the absolute difference between two gravity anomaly grids
- Scales results to microGals for easier interpretation

### `plot_global_anomaly(anomaly_difference, title)`
- Generates a global map of gravitational anomaly differences
- Uses cartopy for geospatial rendering
- Applies color gradient for visual distinction

### `plot_localized_anomaly(anomaly_difference, lat, lon, region_size)`
- Creates a localized visualization of gravitational anomalies
- Allows focusing on specific geographic regions
- Supports adjustable region size

### `plot_coefficient_spectrum(gfc_file)`
- Visualizes spectral characteristics of spherical harmonic coefficients
- Supports different scaling and representation options

## Usage Examples

### Global Anomaly Detection
```python
from grace_anomaly import (
    load_sh_grav_coeffs, 
    compute_anomaly_difference, 
    plot_global_anomaly
)

# Load GRACE datasets
coeffs1 = load_sh_grav_coeffs('dataset1.gfc')
coeffs2 = load_sh_grav_coeffs('dataset2.gfc')

# Compute and visualize anomalies
anomaly_diff = compute_anomaly_difference(coeffs1, coeffs2)
plot_global_anomaly(anomaly_diff, title='Global Gravitational Changes')
```

### Localized Anomaly Analysis
```python
from grace_anomaly import (
    load_sh_grav_coeffs, 
    compute_anomaly_difference, 
    plot_localized_anomaly
)

# Load GRACE datasets
coeffs1 = load_sh_grav_coeffs('dataset1.gfc')
coeffs2 = load_sh_grav_coeffs('dataset2.gfc')

# Analyze specific region
anomaly_diff = compute_anomaly_difference(coeffs1, coeffs2)
plot_localized_anomaly(
    anomaly_diff, 
    epicenter_lat=3.316, 
    epicenter_lon=95.854,
    region_size=15
)
```

## Limitations and Considerations

- Accuracy depends on GRACE dataset quality
- Low-degree coefficients are filtered out by default
- Visualization resolution limited by input data
- Requires precise GRACE GFC files

## Future Improvements

- Implement more advanced filtering techniques
- Add support for multiple file formats
- Enhance visualization options
- Develop machine learning models for anomaly detection

## References

1. GRACE Mission: https://grace.jpl.nasa.gov/
2. pyshtools Documentation: https://pyshtools.readthedocs.io/
3. ICGEM Data Portal: http://icgem.gfz-potsdam.de/