"""Unit tests for the gradient-boosted-trees model on a tiny synthetic cube."""

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from ecoforecast.evaluate import spatial_blocks, walk_forward_splits
from ecoforecast.models.gbt import fit_predict_fold, make_gbt, walk_forward_gbt


@pytest.fixture
def cube():
    time = pd.date_range("2019-01-01", periods=36, freq="MS")
    season = 0.2 * np.sin(2 * np.pi * (time.month.values - 1) / 12)
    data = 0.5 + season[:, None, None] + np.random.default_rng(0).normal(0, 0.02, (36, 8, 8))
    return xr.DataArray(
        data.astype("float32"), dims=("time", "y", "x"),
        coords={"time": time, "y": range(8), "x": range(8)}, name="ndvi",
    )


def test_make_gbt_applies_overrides():
    assert make_gbt(n_estimators=10).n_estimators == 10


def test_fit_predict_fold_is_aligned_and_finite(cube):
    folds = walk_forward_splits(cube["time"], n_test_folds=2)
    train, _test, _ = spatial_blocks(cube, block_size=4, n_test_blocks=1, seed=0)
    pred, model = fit_predict_fold(cube, folds[-1]["train"], train, params=dict(n_estimators=20))
    assert pred.dims == cube.dims and pred.sizes == cube.sizes
    later = pred.isel(time=slice(3, None)).values          # lags available from month 3
    assert np.isfinite(later).mean() > 0.9
    assert np.isnan(pred.isel(time=0)).all()               # no lags yet at t0


def test_walk_forward_gbt_returns_expected_schema(cube):
    folds = walk_forward_splits(cube["time"], n_test_folds=2)
    train, test, _ = spatial_blocks(cube, block_size=4, n_test_blocks=1, seed=0)
    res, oos, importance = walk_forward_gbt(cube, folds, train, test, params=dict(n_estimators=20))
    assert "gbt" in set(res["predictor"])
    assert {"skill_vs_persistence", "skill_vs_climatology"} <= set(res.columns)
    assert oos.sizes == cube.sizes
    assert not importance.empty


def test_walk_forward_gbt_handles_descending_float_coords():
    # Real DEA cubes have a descending, floating-point y (projected metres);
    # the prediction must stay aligned to the source grid, not re-sorted.
    time = pd.date_range("2020-01-01", periods=36, freq="MS")
    season = 0.2 * np.sin(2 * np.pi * (time.month.values - 1) / 12)
    data = 0.5 + season[:, None, None] + np.random.default_rng(0).normal(0, 0.02, (36, 12, 10))
    cube = xr.DataArray(
        data.astype("float32"), dims=("time", "y", "x"),
        coords={
            "time": time,
            "y": np.linspace(-3_000_000.0, -3_003_000.0, 12),  # descending
            "x": np.linspace(1_900_000.0, 1_902_000.0, 10),
        },
        name="ndvi",
    )
    folds = walk_forward_splits(cube["time"], n_test_folds=2)
    train, test, _ = spatial_blocks(cube, block_size=4, n_test_blocks=1, seed=0)
    res, oos, _ = walk_forward_gbt(cube, folds, train, test, params=dict(n_estimators=20))
    assert "gbt" in set(res["predictor"])
    assert np.array_equal(oos["y"].values, cube["y"].values)


def test_walk_forward_gbt_uses_driver_features():
    time = pd.date_range("2020-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(1)
    rain = 50 + rng.normal(0, 10, (36, 8, 8))
    season = 0.2 * np.sin(2 * np.pi * (time.month.values - 1) / 12)
    data = 0.5 + season[:, None, None] + 0.001 * rain + rng.normal(0, 0.02, (36, 8, 8))
    coords = {"time": time, "y": range(8), "x": range(8)}
    cube = xr.DataArray(data.astype("float32"), dims=("time", "y", "x"), coords=coords, name="ndvi")
    drivers = xr.Dataset({"rain": xr.DataArray(rain.astype("float32"), dims=("time", "y", "x"), coords=coords)})
    folds = walk_forward_splits(cube["time"], n_test_folds=2)
    train, test, _ = spatial_blocks(cube, block_size=4, n_test_blocks=1, seed=0)
    _res, _oos, importance = walk_forward_gbt(cube, folds, train, test, drivers=drivers, params=dict(n_estimators=20))
    assert "rain" in importance.index
