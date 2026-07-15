"""Tests for conformal prediction intervals."""

import numpy as np
import pandas as pd
import xarray as xr

from ecoforecast.evaluate import walk_forward_splits
from ecoforecast.uncertainty import _finite_sample_quantile, conformal_intervals


def test_finite_sample_quantile():
    r = np.arange(1, 101, dtype=float)  # 1..100
    q = _finite_sample_quantile(r, alpha=0.1)
    assert 90 <= q <= 92                # ~90th percentile with the (n+1) correction
    assert np.isnan(_finite_sample_quantile(np.array([]), 0.1))


def _cube_and_oos(seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2018-01-01", periods=48, freq="MS")
    obs = rng.normal(0.5, 0.1, (48, 10, 10)).astype("float32")
    oos = obs + rng.normal(0, 0.05, (48, 10, 10)).astype("float32")   # noisy forecast
    coords = {"time": time, "y": range(10), "x": range(10)}
    da = lambda a: xr.DataArray(a, dims=("time", "y", "x"), coords=coords)
    return da(obs), da(oos)


def test_conformal_coverage_near_target():
    obs, oos = _cube_and_oos()
    folds = walk_forward_splits(obs["time"], block_months=3, n_test_folds=4, embargo_months=3)
    table, lower, upper = conformal_intervals(obs, oos, folds, alpha=0.1)
    assert len(table) == 3                                   # folds 1..3 get calibrated
    assert 0.82 <= table["coverage"].mean() <= 0.98          # ~0.90 marginal coverage


def test_intervals_are_ordered():
    obs, oos = _cube_and_oos(seed=1)
    folds = walk_forward_splits(obs["time"], n_test_folds=4)
    _table, lower, upper = conformal_intervals(obs, oos, folds, alpha=0.1)
    diff = (upper - lower).values
    assert np.nanmin(diff) >= 0                              # upper >= lower everywhere defined
