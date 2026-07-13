"""Baselines the models must beat.

- Persistence: next = current.
- Seasonal climatology: next = training mean for that calendar month.

Both return forecasts aligned to the NDVI time axis so evaluate.py can score
them on the same folds as any model.
"""

from __future__ import annotations

import xarray as xr

from .features import seasonal_climatology


def persistence(ndvi: xr.DataArray) -> xr.DataArray:
    """Forecast for month t = observed NDVI at t-1."""
    forecast = ndvi.shift(time=1)
    forecast.name = "persistence"
    return forecast


def climatology_forecast(
    ndvi: xr.DataArray,
    climatology: xr.DataArray | None = None,
) -> xr.DataArray:
    """Forecast for month t = training-mean NDVI for that calendar month.

    Pass a training-only `climatology`; the default computes it from `ndvi`
    itself, which is fine for a quick look but leaks if used across a test fold.
    """
    if climatology is None:
        climatology = seasonal_climatology(ndvi)

    months = ndvi["time"].dt.month
    forecast = climatology.sel(month=months).drop_vars("month")
    forecast.name = "climatology"
    return forecast
