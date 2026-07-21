"""Unit tests for the evaluation splits and metrics."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecoforecast.evaluate import (
    blocks_in_pixels,
    evaluate_folds,
    pixel_size,
    r2,
    rmse,
    skill_score,
    spatial_blocks,
    summarise_2x2,
    walk_forward_splits,
)


def _grid(res_m, n=40):
    """A cube on a projected grid with `res_m` pixels, y descending as DEA's are."""
    time = pd.date_range("2019-01-01", periods=6, freq="MS")
    data = np.zeros((6, n, n), dtype="float32")
    return xr.DataArray(data, dims=("time", "y", "x"), coords={
        "time": time,
        "y": -3_000_000 - np.arange(n) * res_m,
        "x": 1_900_000 + np.arange(n) * res_m,
    })


def test_pixel_size_reads_the_grid_spacing():
    assert pixel_size(_grid(100)) == 100
    assert pixel_size(_grid(10)) == 10
    with pytest.raises(ValueError):
        pixel_size(_grid(100).isel(x=slice(0, 1)))


def test_metre_blocks_are_the_same_ground_area_at_any_resolution():
    """The point of the change: 2 km stays 2 km when the pixels get smaller."""
    at100 = blocks_in_pixels(_grid(100), block_size_m=2000, buffer_m=200)
    at10 = blocks_in_pixels(_grid(10), block_size_m=2000, buffer_m=200)
    assert at100 == (20, 2)
    assert at10 == (200, 20)
    assert at100[0] * 100 == at10[0] * 10        # identical ground distance


def test_metre_settings_reproduce_the_published_pixel_settings():
    """2000 m / 200 m at 100 m must equal the 20 px / 2 px used for every result."""
    assert blocks_in_pixels(_grid(100), block_size_m=2000, buffer_m=200) == (20, 2)


def test_pixel_settings_are_used_when_no_metres_given():
    assert blocks_in_pixels(_grid(100), None, None, block_size_px=8, buffer_px=3) == (8, 3)
    with pytest.raises(ValueError):
        blocks_in_pixels(_grid(100))


def test_blocks_never_round_down_to_zero():
    block, buff = blocks_in_pixels(_grid(500), block_size_m=600, buffer_m=100)
    assert block >= 1 and buff >= 1


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
