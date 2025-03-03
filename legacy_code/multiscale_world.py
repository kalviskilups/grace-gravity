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
        taper_width (int, optional): Width over which to taper. Defaults to 2.

    Returns:
        numpy.ndarray: Tapered spherical harmonic coefficients.
    """
    degrees = np.arange(lmax + 1)
    weights = np.ones(lmax + 1)

    for l_degrees in degrees:
        if l_degrees < min_degree:
            if l_degrees < min_degree - taper_width:
                weights[l_degrees] = 0.0
            else:
                weights[l_degrees] = 0.5 * (1 + np.cos(np.pi * (min_degree - l_degrees) / taper_width))
        elif l_degrees > max_degree:
            if l_degrees > max_degree + taper_width:
                weights[l_degrees] = 0.0
            else:
                weights[l_degrees] = 0.5 * (1 + np.cos(np.pi * (l_degrees - max_degree) / taper_width))
        else:
            weights[l_degrees] = 1.0

    taper = weights[np.newaxis, :, np.newaxis]
    tapered_coeffs = sht_coeffs * taper
    return tapered_coeffs


def compute_anomaly_band(coeffs, min_degree, max_degree, taper_width=2):
    """
    Compute the gravity anomaly for a given spherical harmonic degree band using a smooth taper.

    Args:
        coeffs (pysh.SHGravCoeffs): GRACE spherical harmonic coefficients.
        min_degree (int): Minimum degree to retain.
        max_degree (int): Maximum degree to retain.
        taper_width (int, optional): Taper width. Defaults to 2.

    Returns:
        numpy.ndarray: The gravity anomaly grid for the specified band.
    """
    filtered_coeffs = coeffs.coeffs.copy()

    # Zero out coefficients below lmin and above lmax
    # Note: the coeffs array is organized as [cosine/sine, degree, order]
    filtered_coeffs[:, :min_degree, :] = 0
    filtered_coeffs[:, max_degree + 1 :, :] = 0

    gm = 3.9860044150e14  # [m^3/s^2]
    r0 = 6.3781363000e06  # [m]

    rad, theta, phi, total, pot = pysh.gravmag.MakeGravGridDH(
        filtered_coeffs, gm, r0, lmax=coeffs.lmax, normal_gravity=1
    )
    # Adding a reference gravity (g_ref) for consistency:
    g_ref = gm / (r0**2)
    total = total - g_ref
    return total


def compute_filtered_anomaly(coeffs, degree_bands, taper_width=2):
    """
    Compute filtered gravity anomalies for each degree band from a single GRACE file.

    Args:
        coeffs (pysh.SHGravCoeffs): GRACE spherical harmonic coefficients.
        degree_bands (list of tuple): List of (min_degree, max_degree) tuples.
        taper_width (int, optional): Taper width. Defaults to 2.

    Returns:
        dict: Dictionary mapping band labels to filtered anomaly grids (in µGal).
    """
    filtered_anomalies = {}
    for band in degree_bands:
        min_degree, max_degree = band
        anomaly = compute_anomaly_band(coeffs, min_degree, max_degree, taper_width)
        # Scale to µGal:
        anomaly_scaled = anomaly * 1e8
        filtered_anomalies[f"{min_degree}-{max_degree}"] = anomaly_scaled
    return filtered_anomalies


def perform_localized_inversion(anomaly, conversion_factor=1.0, smoothing_sigma=2):
    """
    Convert the gravity anomaly (µGal) into an equivalent mass change field using Gaussian smoothing.

    Args:
        anomaly (numpy.ndarray): Gravity anomaly field.
        conversion_factor (float, optional): Conversion factor (assume 1 µGal ≈ 1 cm water thickness).
        smoothing_sigma (float, optional): Gaussian sigma. Defaults to 2.

    Returns:
        numpy.ndarray: Inverted mass change field.
    """
    smoothed = gaussian_filter(anomaly, sigma=smoothing_sigma)
    mass_change = smoothed * conversion_factor
    return mass_change


def plot_global_multiscale_anomalies(anomaly_dict, headless=False):
    """
    Plot global maps of filtered gravity anomalies and inverted mass change fields.

    The figure has two rows per degree band:
      - Top row: Gravity anomaly (µGal)
      - Bottom row: Inverted mass change (cm EWT)

    Args:
        anomaly_dict (dict): Dictionary with band labels as keys and anomaly grids as values.
        headless (bool, optional): If True, save the plot instead of showing it.
    """
    num_bands = len(anomaly_dict)
    fig, axs = plt.subplots(
        2,
        num_bands,
        figsize=(6 * num_bands, 16),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if num_bands == 1:
        axs = axs.reshape(2, 1)

    grid_size = next(iter(anomaly_dict.values())).shape[0]
    lats = np.linspace(-90, 90, grid_size)
    lons = np.linspace(0, 360, 2 * grid_size, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    refine_factor = 2
    xi = np.linspace(0, 360, lon_grid.shape[1] * refine_factor)
    yi = np.linspace(-90, 90, lat_grid.shape[0] * refine_factor)
    xi, yi = np.meshgrid(xi, yi)

    for idx, (band, anomaly) in enumerate(anomaly_dict.items()):
        zi = griddata(
            (lon_grid.flatten(), lat_grid.flatten()),
            anomaly.flatten(),
            (xi, yi),
            method="cubic",
        )

        ax_top = axs[0, idx]
        ax_top.set_global()
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

        inversion = perform_localized_inversion(anomaly, conversion_factor=1.0, smoothing_sigma=2)
        zi_inv = griddata(
            (lon_grid.flatten(), lat_grid.flatten()),
            inversion.flatten(),
            (xi, yi),
            method="cubic",
        )
        ax_bot = axs[1, idx]
        ax_bot.set_global()
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

    plt.suptitle("Global Filtered Gravity Anomalies and Inversion", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if not headless:
        plt.show()
    else:
        plt.savefig("global_filtered_anomalies.png")
        print("Plot saved as global_filtered_anomalies.png")


if __name__ == "__main__":
    # Use a single GRACE file for filtering.
    gfc_file = "data/CSR_CSR-Release-06_96x96_unfiltered_GSM-2_2004306-2004335_GRAC_UTCSR_BB01_0600.gfc"

    # Load spherical harmonic coefficients from the single file.
    coeffs = load_sh_grav_coeffs(gfc_file)

    # Define degree bands of interest; adjust as needed.
    degree_bands = [(6, 96)]

    # Compute filtered anomalies from the single file.
    anomalies = compute_filtered_anomaly(coeffs, degree_bands, taper_width=3)

    # Optionally, you can apply additional filters (e.g., high-pass filtering) here.
    # For example:
    # for key in anomalies:
    #     anomalies[key] = high_pass_filter(anomalies[key], sigma=5)

    # Plot the filtered gravity anomalies and their inversion.
    plot_global_multiscale_anomalies(anomalies, headless=True)
