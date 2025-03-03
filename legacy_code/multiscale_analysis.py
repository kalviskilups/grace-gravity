import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from src.functions import load_sh_grav_coeffs


def taper_coeffs(sht_coeffs, lmax, min_degree, max_degree, taper_width=2):
    """
    Apply a cosine taper to the spherical harmonic coefficients.

    Args:
        sht_coeffs (numpy.ndarray): Array of spherical harmonic coefficients with shape (2, lmax+1, m+1).
        lmax (int): Maximum degree of the coefficients.
        min_degree (int): Minimum degree of the band.
        max_degree (int): Maximum degree of the band.
        taper_width (int, optional): Width (in degrees) over which to taper the coefficients. Defaults to 2.

    Returns:
        numpy.ndarray: Tapered spherical harmonic coefficients.
    """
    degrees = np.arange(lmax + 1)
    weights = np.ones(lmax + 1)

    for l_degree in degrees:
        if l_degree < min_degree:
            if l_degree < min_degree - taper_width:
                weights[l_degree] = 0.0
            else:
                weights[l_degree] = 0.5 * (1 + np.cos(np.pi * (min_degree - l_degree) / taper_width))
        elif l_degree > max_degree:
            if l_degree > max_degree + taper_width:
                weights[l_degree] = 0.0
            else:
                weights[l_degree] = 0.5 * (1 + np.cos(np.pi * (l_degree - max_degree) / taper_width))
        else:
            weights[l_degree] = 1.0

    taper = weights[np.newaxis, :, np.newaxis]
    tapered_coeffs = sht_coeffs * taper
    return tapered_coeffs


def compute_anomaly_band(coeffs, min_degree, max_degree, taper_width=2):
    """
    Compute the gravity anomaly for a given spherical harmonic degree band using a smooth taper.

    Args:
        coeffs (pysh.SHGravCoeffs): GRACE spherical harmonic coefficients.
        min_degree (int): Minimum spherical harmonic degree to retain.
        max_degree (int): Maximum spherical harmonic degree to retain.
        taper_width (int, optional): Width for tapering near band edges. Defaults to 2.

    Returns:
        numpy.ndarray: The gravity anomaly grid for the specified band.
    """
    lmax = coeffs.lmax
    tapered_coeffs = taper_coeffs(coeffs.coeffs.copy(), lmax, min_degree, max_degree, taper_width)

    gm = 3.9860044150e14  # Gravitational constant * mass [m^3/s^2]
    r0 = 6.3781363000e06  # Reference radius [m]

    rad, theta, phi, total, pot = pysh.gravmag.MakeGravGridDH(tapered_coeffs, gm, r0, lmax=lmax, normal_gravity=1)
    # Add reference gravity for consistency:
    g_ref = gm / (r0**2)
    total = total + g_ref
    return total


def compute_multiscale_anomaly(coeffs, degree_bands, taper_width=2):
    """
    Compute filtered gravity anomalies for multiple degree bands from a single GRACE file.

    Args:
        coeffs (pysh.SHGravCoeffs): GRACE spherical harmonic coefficients.
        degree_bands (list of tuple): List of (min_degree, max_degree) tuples.
        taper_width (int, optional): Width for tapering near band edges. Defaults to 2.

    Returns:
        dict: Dictionary mapping band labels (e.g., "6-40") to gravity anomaly grids (in µGal).
    """
    anomalies = {}
    for band in degree_bands:
        min_degree, max_degree = band
        anomaly = compute_anomaly_band(coeffs, min_degree, max_degree, taper_width)
        anomaly_scaled = anomaly * 1e8  # scale to µGal
        anomalies[f"{min_degree}-{max_degree}"] = anomaly_scaled
    return anomalies


def perform_localized_inversion(anomaly, conversion_factor=1.0, smoothing_sigma=2):
    """
    Perform a simple inversion to convert the gravity anomaly (µGal)
    into an equivalent mass change (e.g., equivalent water thickness in cm).

    Args:
        anomaly (numpy.ndarray): Gravity anomaly field.
        conversion_factor (float, optional): Conversion factor from µGal to mass change units.
                                             (Assume 1 µGal ≈ 1 cm water thickness for demonstration.)
        smoothing_sigma (float, optional): Sigma for the Gaussian filter.

    Returns:
        numpy.ndarray: Inverted mass change field.
    """
    smoothed = gaussian_filter(anomaly, sigma=smoothing_sigma)
    mass_change = smoothed * conversion_factor
    return mass_change


def plot_localized_multiscale_anomalies(anomaly_dict, epicenter_lat, epicenter_lon, region_size=40, headless=False):
    """
    Plot localized maps for each degree band's gravity anomalies and inverted mass changes.

    The figure will have two rows per degree band:
      - Top row: Localized gravity anomalies (µGal)
      - Bottom row: Localized inverted mass change (cm EWT)

    Args:
        anomaly_dict (dict): Dictionary with band labels as keys and anomaly grids as values.
        epicenter_lat (float): Latitude of the earthquake epicenter.
        epicenter_lon (float): Longitude of the earthquake epicenter.
        region_size (float, optional): Extent (in degrees) around the epicenter. Defaults to 10.
        headless (bool, optional): If True, the plot is saved to a file.
    """
    if epicenter_lon < 0:
        epicenter_lon += 360

    num_bands = len(anomaly_dict)
    fig, axs = plt.subplots(
        2,
        num_bands,
        figsize=(6 * num_bands, 16),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if num_bands == 1:
        axs = axs.reshape(2, 1)

    # Build full lat-lon grid from anomaly dimensions.
    grid_size = next(iter(anomaly_dict.values())).shape[0]
    latitudes = np.linspace(-90, 90, grid_size)
    longitudes = np.linspace(0, 360, 2 * grid_size, endpoint=False)

    # Define localized region boundaries.
    lat_min = epicenter_lat - region_size
    lat_max = epicenter_lat + region_size
    lon_min = epicenter_lon - region_size
    lon_max = epicenter_lon + region_size

    # Create masks for the localized region.
    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)

    local_lats = np.linspace(lat_min, lat_max, lat_mask.sum())
    local_lons = np.linspace(lon_min, lon_max, lon_mask.sum())
    local_lon_grid, local_lat_grid = np.meshgrid(local_lons, local_lats)

    # Refine grid resolution by interpolation.
    refine_factor = 6
    xi = np.linspace(lon_min, lon_max, local_lon_grid.shape[1] * refine_factor)
    yi = np.linspace(lat_min, lat_max, local_lat_grid.shape[0] * refine_factor)
    xi, yi = np.meshgrid(xi, yi)

    for idx, (band, anomaly) in enumerate(anomaly_dict.items()):
        local_anomaly = anomaly[np.ix_(lat_mask, lon_mask)]
        zi = griddata(
            (local_lon_grid.flatten(), local_lat_grid.flatten()),
            local_anomaly.flatten(),
            (xi, yi),
            method="cubic",
        )

        ax_top = axs[0, idx]
        ax_top.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax_top.add_feature(cfeature.COASTLINE)
        ax_top.add_feature(cfeature.BORDERS, linestyle=":")
        pcm_top = ax_top.contourf(
            xi,
            yi,
            zi,
            levels=50,
            transform=ccrs.PlateCarree(),
            cmap="RdYlBu_r",
            extend="both",
        )
        ax_top.set_title(f"Band {band}\nAnomaly (µGal)")
        plt.colorbar(pcm_top, ax=ax_top, orientation="vertical")
        ax_top.plot(
            epicenter_lon,
            epicenter_lat,
            marker="*",
            markersize=12,
            color="red",
            transform=ccrs.PlateCarree(),
        )

        inversion = perform_localized_inversion(local_anomaly, conversion_factor=1.0, smoothing_sigma=2)
        zi_inv = griddata(
            (local_lon_grid.flatten(), local_lat_grid.flatten()),
            inversion.flatten(),
            (xi, yi),
            method="cubic",
        )
        ax_bot = axs[1, idx]
        ax_bot.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax_bot.add_feature(cfeature.COASTLINE)
        ax_bot.add_feature(cfeature.BORDERS, linestyle=":")
        pcm_bot = ax_bot.contourf(
            xi,
            yi,
            zi_inv,
            levels=50,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            extend="both",
        )
        ax_bot.set_title(f"Band {band}\nInverted Mass Change (cm EWT)")
        plt.colorbar(pcm_bot, ax=ax_bot, orientation="vertical")
        ax_bot.plot(
            epicenter_lon,
            epicenter_lat,
            marker="*",
            markersize=12,
            color="red",
            transform=ccrs.PlateCarree(),
        )

    plt.suptitle(
        f"Localized Gravity Anomalies and Inversion\nEpicenter: ({epicenter_lat}, {epicenter_lon})",
        fontsize=18,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if not headless:
        plt.show()
    else:
        plt.savefig("localized_anomalies_filtered.png")
        print("Plot saved as localized_anomalies_filtered.png")


if __name__ == "__main__":
    # Use a single GRACE file that contains earthquake-related data.
    gfc_file = "data/2010_chile/GFZ_GFZ-Release-05_weekly_unfiltered_GX-OG-_2-GSM+GFZ-GSM-2_2010059-2010065_0007_EIGEN_GW30_0005.gfc"

    # Load spherical harmonic coefficients from the file.
    coeffs = load_sh_grav_coeffs(gfc_file)

    # Define degree bands of interest (adjust as needed).
    degree_bands = [(5, 30)]

    # Compute filtered gravity anomalies (scaled to µGal) from the single file.
    anomalies = compute_multiscale_anomaly(coeffs, degree_bands, taper_width=3)

    # Define the epicenter (example coordinates).
    epicenter_lat = -35.846
    epicenter_lon = -72.719  # Will be converted to 287.281 in plotting if needed.

    # Plot localized anomalies and inverted mass change maps.
    plot_localized_multiscale_anomalies(anomalies, epicenter_lat, epicenter_lon, region_size=20, headless=True)
