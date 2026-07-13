"""Unit tests for features and baselines on a tiny synthetic cube."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.features import (
    add_lags,
    build_feature_table,
    compute_anomaly,
    seasonal_climatology,
)


@pytest.fixture
def cube():
    rng = np.random.default_rng(0)
    time = pd.date_range("2019-01-01", periods=36, freq="MS")
    month = time.month.values
    season = 0.2 * np.sin(2 * np.pi * (month - 1) / 12)
    data = 0.5 + season[:, None, None] + rng.normal(0, 0.02, (36, 4, 4))
    return xr.DataArray(
        data.astype("float32"),
        dims=("time", "y", "x"),
        coords={"time": time, "y": range(4), "x": range(4)},
        name="ndvi",
    )


def test_climatology_has_twelve_months(cube):
    clim = seasonal_climatology(cube)
    assert clim.sizes["month"] == 12


def test_anomaly_centres_on_training_climatology(cube):
    train = cube["time"] < np.datetime64("2021-01-01")
    clim = seasonal_climatology(cube.sel(time=train))
    anom = compute_anomaly(cube, clim)
    monthly_mean = anom.sel(time=train).groupby("time.month").mean("time")
    assert np.nanmax(np.abs(monthly_mean.values)) < 1e-5


def test_lag_equals_shifted_series(cube):
    lag1 = add_lags(cube)["ndvi_lag1"]
    np.testing.assert_allclose(
        lag1.isel(time=slice(1, None)).values,
        cube.isel(time=slice(0, -1)).values,
    )


def test_feature_table_has_no_nans(cube):
    table = build_feature_table(cube)
    assert len(table) == 33 * 4 * 4  # 36 months minus 3 dropped for max lag
    assert not table.isna().any().any()


def test_persistence_is_previous_step(cube):
    fc = persistence(cube)
    np.testing.assert_allclose(
        fc.isel(time=slice(1, None)).values,
        cube.isel(time=slice(0, -1)).values,
    )
    assert bool(np.isnan(fc.isel(time=0)).all())


def test_climatology_forecast_matches_month_means(cube):
    clim = seasonal_climatology(cube)
    fc = climatology_forecast(cube, climatology=clim)
    assert fc.sizes == cube.sizes
    # forecast for a January equals the January climatology
    jan = cube["time"].dt.month == 1
    np.testing.assert_allclose(
        fc.sel(time=jan).isel(time=0).values,
        clim.sel(month=1).values,
    )
