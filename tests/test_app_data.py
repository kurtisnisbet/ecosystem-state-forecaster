"""Tests for the precomputed demo artifact.

The hosted app has no model libraries, so anything it needs must survive the
round trip through NetCDF. These check shape, content and the quantile table
that keeps the confidence slider interactive.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecoforecast.app_data import LEVELS, headline_rmse, save_app_data
from ecoforecast.evaluate import spatial_blocks, walk_forward_splits


def _fixture(seed=0, t=60, h=20, w=16):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2018-01-01", periods=t, freq="MS")
    season = 0.5 + 0.15 * np.sin(2 * np.pi * (time.month.values - 1) / 12)
    obs = (season[:, None, None] + rng.normal(0, 0.03, (t, h, w))).astype("float32")
    coords = {"time": time, "y": np.linspace(-3e6, -3.002e6, h), "x": np.linspace(1.9e6, 1.9016e6, w)}
    ndvi = xr.DataArray(obs, dims=("time", "y", "x"), coords=coords, name="ndvi")
    folds = walk_forward_splits(ndvi["time"], block_months=3, n_test_folds=4, embargo_months=3)
    strain, _stest, _ = spatial_blocks(ndvi, block_size=5, n_test_blocks=2, buffer=1, seed=1)
    test = folds[0]["test"].copy()
    for fo in folds[1:]:
        test = test | fo["test"]
    oos = {
        "gbt": (ndvi + rng.normal(0, 0.02, ndvi.shape)).where(test),
        "gnn": (ndvi + rng.normal(0, 0.05, ndvi.shape)).where(test),
    }
    return ndvi, oos, folds, strain, test


def test_artifact_contents_and_shapes(tmp_path):
    ndvi, oos, folds, strain, _ = _fixture()
    path = save_app_data(ndvi, oos, folds, strain, "test_biome", "sentinel2_100m", tmp_path,
                         display_months=24, coarsen=2)
    assert path.name == "sentinel2_100m_test_biome.nc"

    ds = xr.open_dataset(path)
    for name in ("ndvi", "climatology", "train_mask", "pred_gbt", "pred_gnn", "conformal_q"):
        assert name in ds, name
    assert ds.sizes["y"] == 10 and ds.sizes["x"] == 8       # coarsened by two
    assert ds.sizes["time"] == 24                            # display window
    assert ds["climatology"].sizes["month"] == 12
    assert ds["train_mask"].dtype == bool


def test_forecasts_are_nan_outside_the_test_window(tmp_path):
    ndvi, oos, folds, strain, test = _fixture()
    ds = xr.open_dataset(save_app_data(ndvi, oos, folds, strain, "b", "tag", tmp_path))
    scored = ds["time"].isin(ndvi["time"].sel(time=test).values)
    assert bool(ds["pred_gbt"].where(~scored).notnull().sum() == 0)
    assert bool(ds["pred_gbt"].where(scored).notnull().any())


def test_quantile_table_supports_the_confidence_slider(tmp_path):
    ndvi, oos, folds, strain, _ = _fixture()
    ds = xr.open_dataset(save_app_data(ndvi, oos, folds, strain, "b", "tag", tmp_path))
    q = ds["conformal_q"]
    assert q.sizes == {"model": 2, "level": len(LEVELS)}
    for model in ("gbt", "gnn"):
        vals = q.sel(model=model).values
        assert np.all(vals >= 0)
        assert np.all(np.diff(vals) >= -1e-9)               # monotonic in confidence
    # the noisier model needs the wider interval
    assert float(q.sel(model="gnn", level=0.9)) > float(q.sel(model="gbt", level=0.9))


def test_headline_rmse_is_stored_for_the_skill_bars(tmp_path):
    ndvi, oos, folds, strain, _ = _fixture()
    res = pd.DataFrame({
        "predictor": ["gbt", "climatology", "persistence", "gbt"],
        "time": ["future", "future", "future", "past"],
        "space": ["seen", "seen", "seen", "seen"],
        "rmse": [0.06, 0.09, 0.08, 0.99],
    })
    ds = xr.open_dataset(save_app_data(ndvi, oos, folds, strain, "b", "tag", tmp_path, results=res))
    stored = {str(p): float(v) for p, v in zip(ds["predictor"].values, ds["headline_rmse"].values)}
    assert stored["gbt"] == pytest.approx(0.06)               # the 'past' row is excluded
    assert 1 - stored["gbt"] / stored["climatology"] > 0      # positive skill vs climatology


def test_headline_rmse_averages_the_headline_cell_only():
    res = pd.DataFrame({
        "predictor": ["gbt", "gbt", "gbt"],
        "time": ["future", "future", "past"],
        "space": ["seen", "unseen", "seen"],
        "rmse": [0.10, 0.50, 0.90],
    })
    assert float(headline_rmse(res).sel(predictor="gbt")) == pytest.approx(0.10)


def test_persistence_is_derivable_in_the_app(tmp_path):
    """The app draws persistence from the stored NDVI window, so it is not saved."""
    ndvi, oos, folds, strain, _ = _fixture()
    ds = xr.open_dataset(save_app_data(ndvi, oos, folds, strain, "b", "tag", tmp_path))
    assert "pred_persistence" not in ds
    pers = ds["ndvi"].shift(time=1)
    assert bool(pers.isel(time=slice(1, None)).notnull().all())
