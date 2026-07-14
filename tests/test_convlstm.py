"""Tests for the ConvLSTM.

The data-plumbing tests run everywhere; the model/training tests require PyTorch
and are skipped if it isn't installed.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecoforecast.models.convlstm import channel_stack, make_sequences, sequences_to_cube


def _cube(t=12, h=6, w=6, cloud=True):
    time = pd.date_range("2020-01-01", periods=t, freq="MS")
    v = 0.5 + 0.1 * np.sin(2 * np.pi * (time.month.values - 1) / 12)[:, None, None] + np.zeros((t, h, w))
    v = v.astype("float32")
    if cloud:
        v[2, 0, 0] = np.nan
    return xr.DataArray(v, dims=("time", "y", "x"), coords={"time": time, "y": range(h), "x": range(w)}, name="ndvi")


def _terrain(h=6, w=6):
    layer = xr.DataArray(np.zeros((h, w), "float32"), dims=("y", "x"), coords={"y": range(h), "x": range(w)})
    return xr.Dataset({"elevation": layer})


def test_channel_stack_shapes_and_channels():
    ndvi = _cube()
    stack = channel_stack(ndvi, _terrain())
    assert stack.shape == (12, 5, 6, 6)                       # 4 base + 1 static
    assert np.array_equal(stack[:, 0], np.nan_to_num(ndvi.values, nan=0.0))   # ch0 filled NDVI
    assert stack[2, 1, 0, 0] == 0.0 and stack[0, 1, 0, 0] == 1.0              # ch1 validity mask


def test_make_sequences_and_cube_roundtrip():
    ndvi = _cube()
    stack = channel_stack(ndvi)
    x, y, idx = make_sequences(stack, ndvi.values, seq_len=3)
    assert x.shape == (9, 3, 4, 6, 6) and y.shape == (9, 6, 6)
    assert list(idx) == list(range(3, 12))
    cube = sequences_to_cube(np.ones((9, 6, 6), "float32"), idx, ndvi)
    assert np.isnan(cube.isel(time=slice(0, 3))).all()       # first seq_len months empty
    assert bool((cube.isel(time=slice(3, None)) == 1).all())  # rest aligned to target months


def test_channel_stack_appends_driver_channels():
    ndvi = _cube()
    coords = {"time": ndvi["time"], "y": ndvi["y"], "x": ndvi["x"]}
    drivers = xr.Dataset({"rain": xr.DataArray(np.zeros((12, 6, 6), "float32"), dims=("time", "y", "x"), coords=coords)})
    base = channel_stack(ndvi)
    with_driver = channel_stack(ndvi, drivers=drivers)
    assert with_driver.shape[1] == base.shape[1] + 1


def test_model_forward_and_fit_predict():
    torch = pytest.importorskip("torch")
    from ecoforecast.evaluate import spatial_blocks, walk_forward_splits
    from ecoforecast.models.convlstm import fit_predict_fold, make_convlstm

    model = make_convlstm(in_ch=4, hidden=8)
    out = model(torch.zeros(2, 3, 4, 8, 8))
    assert tuple(out.shape) == (2, 8, 8)

    ndvi = _cube(t=24, h=8, w=8, cloud=False)
    folds = walk_forward_splits(ndvi["time"], n_test_folds=1)
    strain, _stest, _ = spatial_blocks(ndvi, block_size=4, n_test_blocks=1, seed=0)
    pred, history = fit_predict_fold(ndvi, folds[-1]["train"], strain, seq_len=4, hidden=8, epochs=2)
    assert pred.dims == ndvi.dims and pred.sizes == ndvi.sizes
    assert len(history) == 2 and np.isfinite(history).all()
