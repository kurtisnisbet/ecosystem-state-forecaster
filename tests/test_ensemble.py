"""Tests for the stacked ensemble."""

import numpy as np
import pandas as pd
import xarray as xr

from ecoforecast.evaluate import rmse, spatial_blocks, walk_forward_splits
from ecoforecast.ensemble import score_ensemble, stack_ensemble


def _obs_and_members(seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2018-01-01", periods=72, freq="MS")
    base = 0.5 + 0.15 * np.sin(2 * np.pi * (time.month.values - 1) / 12)[:, None, None]
    obs = np.clip(base + rng.normal(0, 0.02, (72, 16, 16)), 0.05, 0.95).astype("float32")
    obs = xr.DataArray(obs, dims=("time", "y", "x"),
                       coords={"time": time, "y": range(16), "x": range(16)}, name="ndvi")
    folds = walk_forward_splits(obs["time"], block_months=3, n_test_folds=4, embargo_months=3)
    test_all = folds[0]["test"].copy()
    for fo in folds[1:]:
        test_all = test_all | fo["test"]
    member = lambda s: (obs + rng.normal(0, s, obs.shape).astype("float32")).where(test_all)
    return obs, {"good": member(0.02), "mid": member(0.06), "bad": member(0.12)}, folds


def test_weights_nonnegative_and_favour_best():
    obs, models, folds = _obs_and_members()
    _ens, weights = stack_ensemble(obs, models, folds)
    assert len(weights) == 3                          # folds 1..3 calibrated
    wcols = [c for c in weights.columns if c.startswith("w_")]
    assert (weights[wcols].values >= -1e-9).all()     # non-negative
    assert weights[wcols].mean().idxmax() == "w_good"  # least-noisy member weighted most


def test_ensemble_at_least_as_good_as_best_member():
    obs, models, folds = _obs_and_members()
    ens, _ = stack_ensemble(obs, models, folds)
    test = folds[1]["test"].copy()
    for fo in folds[2:]:
        test = test | fo["test"]

    def r(pred):
        o = obs.sel(time=test); p = pred.sel(time=test)
        valid = o.notnull() & p.notnull()
        return rmse(p.where(valid), o.where(valid))

    assert r(ens) <= min(r(m) for m in models.values()) * 1.02


def test_score_ensemble_schema():
    obs, models, folds = _obs_and_members()
    strain, stest, _ = spatial_blocks(obs, block_size=4, n_test_blocks=2, seed=0)
    ens, _ = stack_ensemble(obs, models, folds, space_mask=strain)
    res = score_ensemble(obs, ens, folds, strain, stest)
    assert "ensemble" in set(res["predictor"])
    assert {"skill_vs_persistence", "skill_vs_climatology"} <= set(res.columns)
