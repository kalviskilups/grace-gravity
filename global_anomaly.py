"""Script to visualize global gravitational anomalies."""

from src.functions import (
    compute_anomaly_difference,
    load_sh_grav_coeffs,
    plot_global_anomaly,
)

if __name__ == "__main__":
    # Replace with your actual GRACE dataset file paths
    gfc_file1 = "data/CSR_CSR-Release-06_60x60_unfiltered_GSM-2_2004336-2004366_GRAC_UTCSR_BA01_0600.gfc"
    gfc_file2 = "data/CSR_CSR-Release-06_60x60_unfiltered_GSM-2_2004306-2004335_GRAC_UTCSR_BA01_0600.gfc"

    # Load coefficients
    coeffs1 = load_sh_grav_coeffs(gfc_file1)
    coeffs2 = load_sh_grav_coeffs(gfc_file2)

    # Compute and plot global anomaly
    anomaly_difference = compute_anomaly_difference(coeffs1, coeffs2)
    plot_global_anomaly(anomaly_difference, headless=True)
