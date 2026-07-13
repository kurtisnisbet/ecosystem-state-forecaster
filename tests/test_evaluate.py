"""Unit tests for the evaluation splits and metrics."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecoforecast.evaluate import (
    evaluate_folds,
    r2,
    rmse,
    skill_score,
    spatial_blocks,
    summarise_2x2,
    walk_forward_splits,
)


@pytest.fixture
def cube():
    time = pd.date_range("2019-01-01", periods=48, freq="MS")
    data = np.random.default_rng(0).random((48, 12, 12)).astype("float32")
    return xr.DataArray(
        data, dims=("time", "y", "x"),
        coords={"time": time, "y": range(12), "x": range(12)}, name="ndvi",
    )


def test_metrics_on_perfect_prediction(cube):
    assert rmse(cube, cube) == 0.0
    assert r2(cube, cube) == pytest.approx(1.0)
    assert skill_score(0.0, 0.5) == pytest.approx(1.0)
    assert np.isnan(skill_score(0.5, 0.0))  # guard against divide-by-zero


def test_walk_forward_expanding_with_embargo(cube):
    folds = walk_forward_splits(cube["time"], block_months=3, n_test_folds=4, embargo_months=3)
    assert len(folds) == 4
    sizes = []
    for fo in folds:
        ti = np.where(fo["train"].values)[0]
        te = np.where(fo["test"].values)[0]
        assert te.min() > ti.max()                 # test strictly after train
        assert te.min() - ti.max() - 1 == 3        # embargo gap
        assert te.size == 3                         # block length
        sizes.append(ti.size)
    assert sizes == sorted(sizes)                  # expanding window


def test_walk_forward_skips_folds_without_training():
    time = xr.DataArray(pd.date_range("2020-01-01", periods=9, freq="MS"), dims="time")
    folds = walk_forward_splits(time, block_months=3, n_test_folds=4, embargo_months=3)
    assert 0 < len(folds) < 4                       # short record -> some folds skipped
    assert all(int(fo["train"].sum()) > 0 for fo in folds)


def test_spatial_blocks_disjoint_with_buffer(cube):
    train, test, _ = spatial_blocks(cube, block_size=4, n_test_blocks=2, buffer=1, seed=0)
    assert not bool((train & test).any())          # no overlap
    # every test pixel is separated from training by the buffer
    assert not bool((train.shift(y=1, fill_value=False) & test).any())
    assert not bool((train.shift(x=1, fill_value=False) & test).any())


def test_evaluate_folds_shape_and_columns(cube):
    folds = walk_forward_splits(cube["time"], n_test_folds=2)
    train, test, _ = spatial_blocks(cube, block_size=4, n_test_blocks=2, seed=0)
    preds = {"persistence": cube.shift(time=1), "climatology": cube.mean("time").broadcast_like(cube)}
    res = evaluate_folds(cube, preds, folds, train, test)
    assert len(res) == len(folds) * 4 * 2
    assert {"rmse", "r2", "skill_vs_persistence", "skill_vs_climatology"} <= set(res.columns)
    # a predictor scored against itself has zero skill
    persistence_rows = res[res["predictor"] == "persistence"]["skill_vs_persistence"]
    assert np.allclose(persistence_rows.dropna(), 0.0)


def test_summarise_2x2_layout(cube):
    folds = walk_forward_splits(cube["time"], n_test_folds=2)
    train, test, _ = spatial_blocks(cube, block_size=4, n_test_blocks=2, seed=0)
    preds = {"persistence": cube.shift(time=1), "climatology": cube.mean("time").broadcast_like(cube)}
    res = evaluate_folds(cube, preds, folds, train, test)
    table = summarise_2x2(res, "climatology")
    assert list(table.index) == ["future", "seen"]
    assert list(table.columns) == ["seen", "unseen"]
