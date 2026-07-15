"""Conformal prediction intervals for the forecasts.

Split-conformal, rolled through the walk-forward folds: the interval half-width
for a test fold is calibrated on the absolute out-of-sample residuals from the
folds *before* it, so the calibration data is genuinely unseen at that point.
With exchangeable residuals this gives a marginal coverage guarantee of about
(1 - alpha); under distribution shift (a drought year, say) coverage can drift,
which is worth stating rather than hiding.

Works on any model's stitched out-of-sample cube (`oos` from the walk_forward_*
functions), so it is model-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def _residuals(obs, oos, time_mask, space_mask):
    o = obs.sel(time=time_mask)
    p = oos.sel(time=time_mask)
    if space_mask is not None:
        o = o.where(space_mask)
        p = p.where(space_mask)
    r = np.abs((o - p).values).ravel()
    return r[np.isfinite(r)]


def _finite_sample_quantile(residuals: np.ndarray, alpha: float) -> float:
    """The conformal quantile with the finite-sample (n+1) correction."""
    n = residuals.size
    if n == 0:
        return np.nan
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level))


def conformal_intervals(
    obs: xr.DataArray,
    oos: xr.DataArray,
    folds: list[dict],
    space_mask: xr.DataArray | None = None,
    alpha: float = 0.1,
):
    """Rolling split-conformal intervals over the walk-forward folds.

    For each fold after the first, calibrate the half-width `q` on the absolute
    residuals from all earlier folds and apply it to this fold. Returns a
    per-fold table (q, empirical coverage, mean interval width, target) and the
    lower/upper interval cubes (NaN outside the calibrated test folds).
    """
    lower = xr.full_like(obs, np.nan)
    upper = xr.full_like(obs, np.nan)
    rows = []
    for k in range(1, len(folds)):
        calib_mask = folds[0]["test"].copy()
        for j in range(1, k):
            calib_mask = calib_mask | folds[j]["test"]
        q = _finite_sample_quantile(_residuals(obs, oos, calib_mask, space_mask), alpha)

        test_mask = folds[k]["test"]
        lower = xr.where(test_mask, oos - q, lower)
        upper = xr.where(test_mask, oos + q, upper)

        resid = _residuals(obs, oos, test_mask, space_mask)
        coverage = float((resid <= q).mean()) if resid.size else np.nan
        rows.append({
            "fold": k, "label": folds[k]["label"], "q": q,
            "coverage": coverage, "width": 2 * q, "target": 1 - alpha,
        })

    lower.name, upper.name = "lower", "upper"
    return pd.DataFrame(rows), lower, upper
