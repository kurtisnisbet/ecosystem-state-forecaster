"""Feature engineering for the cube.

Responsibilities:
- Compute NDVI and NDVI anomaly (vs training-only seasonal climatology).
- Build dynamic lags (t-1..t-3) and seasonal encodings (month-of-year).
- Align drivers and terrain; scale/normalise features.
"""

# TODO: compute_ndvi(), compute_anomaly(), add_lags(), add_seasonal_encoding()
