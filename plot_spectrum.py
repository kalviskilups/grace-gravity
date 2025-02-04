import pyshtools

# This data is from International Centre for Global Earth Models (ICGEM) temporal ISTG model

coeffs = pyshtools.SHGravCoeffs.from_file(
    "ITSG_ITSG-Grace2014_daily_2006_ITSG-Grace2014_2006-11-15.gfc", format="icgem"
)

coeffs.plot_spectrum(unit="per_l", xscale="lin", yscale="log", legend=True)
