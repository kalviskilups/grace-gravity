"""Analyze the contribution of each coefficient to the total anomaly."""

from src.functions import plot_coefficient_spectrum

if __name__ == "__main__":
    gfc_file = "data/ITSG_ITSG-Grace2014_daily_2006_ITSG-Grace2014_2006-11-15.gfc"
    plot_coefficient_spectrum(gfc_file)
