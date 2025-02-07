import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh


def load_sh_grav_coeffs(gfc_file, format="icgem"):
    """
    Load spherical harmonic coefficients from a GRACE GFC file.

    Args:
        gfc_file (str): Path to the GFC file
        format (str, optional): File format. Defaults to "icgem"

    Returns:
        pysh.SHGravCoeffs: Spherical harmonic coefficients
    """
    return pysh.SHGravCoeffs.from_file(gfc_file, format=format)


def compute_gravity_anomaly(coeffs, exclude_degrees=5):
    """
    Compute gravity anomaly while excluding low-degree coefficients.

    Args:
        coeffs (pysh.SHGravCoeffs): Input spherical harmonic coefficients
        exclude_degrees (int, optional): Number of low degrees to zero out. Defaults to 5

    Returns:
        numpy.ndarray: Gravity anomaly grid
    """
    gm = 3.9860044150e14  # Gravitational constant * mass [m^3/s^2]
    r0 = 6.3781363000e06  # Reference radius [m]

    modified_coeffs = coeffs.coeffs.copy()
    modified_coeffs[:, 0:exclude_degrees, :] = 0

    rad, theta, phi, total, pot = pysh.gravmag.MakeGravGridDH(
        modified_coeffs, gm, r0, lmax=coeffs.lmax, normal_gravity=1
    )

    # combined_total = np.sqrt(rad**2 + theta**2 + phi**2)

    g_ref = gm / (r0**2)
    total = total + g_ref

    return total


def compute_anomaly_difference(coeffs1, coeffs2):
    """
    Compute the absolute difference between two gravity anomalies.

    Args:
        coeffs1 (pysh.SHGravCoeffs): First set of coefficients
        coeffs2 (pysh.SHGravCoeffs): Second set of coefficients

    Returns:
        numpy.ndarray: Scaled anomaly difference
    """
    anomaly1 = compute_gravity_anomaly(coeffs1)
    anomaly2 = compute_gravity_anomaly(coeffs2)

    return abs(anomaly2 - anomaly1) * 1e8


def plot_global_anomaly(anomaly_difference, title="Global Gravitational Anomaly Difference", headless=False):
    """
    Create a global map of gravitational anomaly difference.

    Args:
        anomaly_difference (numpy.ndarray): Anomaly difference grid
        title (str, optional): Plot title
        headless (bool, optional): Save plot as PNG if True. Defaults to False
    """
    grid_size = anomaly_difference.shape[0]
    latitudes = np.linspace(-90, 90, grid_size)
    longitudes = np.linspace(0, 360, 2 * grid_size, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)

    pcm = ax.contourf(
        lon_grid,
        lat_grid,
        anomaly_difference,
        transform=ccrs.PlateCarree(),
        cmap="RdYlBu_r",
        levels=20,
    )
    plt.colorbar(pcm, ax=ax, label="Gravitational Anomaly Difference (µGal)")
    plt.title(title)
    if not headless:
        plt.show()
    else:
        plt.savefig("global_anomaly.png")
        print("Plot saved as global_anomaly.png")


def plot_localized_anomaly(anomaly_difference, epicenter_lat, epicenter_lon, region_size=15, headless=False):
    """
    Create a localized map of gravitational anomaly difference.

    Args:
        anomaly_difference (numpy.ndarray): Anomaly difference grid
        epicenter_lat (float): Latitude of the epicenter
        epicenter_lon (float): Longitude of the epicenter
        region_size (float, optional): Size of the region around epicenter. Defaults to 15
        headless (bool, optional): Save plot as PNG if True. Defaults to False
    """
    grid_size = anomaly_difference.shape[0]
    latitudes = np.linspace(-90, 90, grid_size)
    longitudes = np.linspace(0, 360, 2 * grid_size, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    lat_min = epicenter_lat - region_size
    lat_max = epicenter_lat + region_size
    lon_min = epicenter_lon - region_size
    lon_max = epicenter_lon + region_size

    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)
    local_anomaly = anomaly_difference[np.ix_(lat_mask, lon_mask)]

    local_min, local_max = np.min(local_anomaly), np.max(local_anomaly)

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    pcm = ax.contourf(
        lon_grid,
        lat_grid,
        anomaly_difference,
        transform=ccrs.PlateCarree(),
        cmap="RdYlBu_r",
        levels=np.linspace(local_min, local_max, 20),
    )

    plt.colorbar(pcm, ax=ax, label="Gravitational Anomaly Difference (µGal)")
    plt.title(f"Localized Gravitational Anomaly Around Epicenter\n(Lat: {epicenter_lat}, Lon: {epicenter_lon})")

    if not headless:
        plt.show()
    else:
        plt.savefig("local_anomaly.png")
        print("Plot saved as local_anomaly.png")


def plot_coefficient_spectrum(gfc_file, unit="per_l", xscale="lin", yscale="log"):
    """
    Plot the spectral characteristics of spherical harmonic coefficients.

    Args:
        gfc_file (str): Path to the GRACE GFC file
        unit (str, optional): Spectral representation unit. Defaults to "per_l"
        xscale (str, optional): X-axis scaling. Defaults to "lin"
        yscale (str, optional): Y-axis scaling. Defaults to "log"
    """
    coeffs = pysh.SHGravCoeffs.from_file(gfc_file, format="icgem")
    coeffs.plot_spectrum(unit=unit, xscale=xscale, yscale=yscale, legend=True)
