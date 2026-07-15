"""Tests for the GNN.

Graph and node-tensor construction run everywhere; the model/training tests
require PyTorch and are skipped if it is not installed.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecoforecast.models.gnn import build_node_tensors, grid_edge_index


def _cube(t=12, h=4, w=5):
    time = pd.date_range("2020-01-01", periods=t, freq="MS")
    data = 0.5 + 0.1 * np.random.default_rng(0).normal(0, 1, (t, h, w))
    return xr.DataArray(data.astype("float32"), dims=("time", "y", "x"),
                        coords={"time": time, "y": range(h), "x": range(w)}, name="ndvi")


def test_grid_edge_index_is_valid():
    ei = grid_edge_index(3, 4)
    assert ei.shape == (2, 34)               # 18 horizontal + 16 vertical directed edges
    assert (ei[0] != ei[1]).all()            # no self-loops
    assert ei.min() >= 0 and ei.max() < 12   # ids within the 3x4 grid


def test_build_node_tensors_shapes_and_lags():
    ndvi = _cube()
    static = xr.Dataset({"elev": xr.DataArray(np.zeros((4, 5), "float32"), dims=("y", "x"),
                                              coords={"y": range(4), "x": range(5)})})
    x, y, last, valid, names = build_node_tensors(ndvi, static)
    t, h, w = ndvi.values.shape
    n = h * w
    assert x.shape == (t, n, 3 + 2 + 1) and y.shape == (t, n) and last.shape == (t, n)
    assert np.isnan(x[0, :, 0]).all()                          # lag1 undefined at t0
    assert np.allclose(x[3, :, 0], ndvi.values[2].reshape(n))  # lag1[t=3] == NDVI[t=2]
    assert np.allclose(x[3, :, 0].reshape(h, w), ndvi.values[2])  # node id = y*W + x
    assert not valid[:3].all() and bool(valid[3:].all())


def test_gnn_forward_and_fit_predict():
    torch = pytest.importorskip("torch")
    from ecoforecast.evaluate import spatial_blocks, walk_forward_splits
    from ecoforecast.models.gnn import fit_predict_fold, make_gnn

    model = make_gnn(in_dim=5, hidden=8, rounds=2)
    ei = torch.from_numpy(grid_edge_index(4, 5))
    out = model(torch.zeros(20, 5), ei, torch.zeros(20))
    assert tuple(out.shape) == (20,)

    ndvi = _cube(t=24, h=8, w=8)
    folds = walk_forward_splits(ndvi["time"], n_test_folds=1)
    strain, _stest, _ = spatial_blocks(ndvi, block_size=4, n_test_blocks=1, seed=0)
    pred, history = fit_predict_fold(ndvi, folds[-1]["train"], strain, hidden=8, rounds=2, epochs=2)
    assert pred.dims == ndvi.dims and pred.sizes == ndvi.sizes
    assert len(history) == 2
