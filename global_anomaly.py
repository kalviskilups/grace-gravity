"""Script to visualize global gravitational anomalies."""

from src.functions import (
    compute_anomaly_difference,
    load_sh_grav_coeffs,
    plot_global_anomaly,
)

if __name__ == "__main__":
    # Replace with your actual GRACE dataset file paths
    gfc_file1 = "data/CSR_CSR-Release-06_96x96_unfiltered_GSM-2_2010001-2010031_GRAC_UTCSR_BB01_0600.gfc"
    gfc_file2 = "data/CSR_CSR-Release-06_96x96_unfiltered_GSM-2_2010060-2010090_GRAC_UTCSR_BB01_0600.gfc"

    # Load coefficients
    coeffs1 = load_sh_grav_coeffs(gfc_file1)
    coeffs2 = load_sh_grav_coeffs(gfc_file2)

    # Compute and plot global anomaly
    anomaly_difference = compute_anomaly_difference(coeffs1, coeffs2)
    plot_global_anomaly(anomaly_difference, headless=True)
