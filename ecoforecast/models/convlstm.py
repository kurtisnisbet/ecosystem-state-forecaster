"""ConvLSTM — the spatiotemporal deep-learning centrepiece.

A scaled-back ConvLSTM (per the brief: prove the pipeline locally on CPU with a
small model before scaling to GPU). It ingests a short sequence of monthly
frames — channels: cloud-filled NDVI, a validity mask, month sin/cos, and any
static layers — and predicts the next month's NDVI as a residual from the most
recent frame. Trained with a loss masked to training months x training blocks x
valid pixels, then used to predict the whole grid so it scores in the same
space x time 2x2 as the baselines and the GBT.

Honest caveat: a convolutional model still *sees* held-out blocks as input
context (only the loss excludes them, with a buffer between). So "unseen
location" skill here is a softer test of spatial transfer than it is for the
per-pixel GBT — state this in the README.

The data-plumbing helpers (channel_stack, make_sequences, sequences_to_cube)
are pure NumPy so they can be tested without PyTorch installed.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..baselines import climatology_forecast, persistence
from ..features import seasonal_climatology
from ..evaluate import score_cells

DEFAULTS = dict(seq_len=6, hidden=16, epochs=25, lr=5e-3, seed=0)


# ── data plumbing (no torch) ────────────────────────────────────────────────
def _standardize(a: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance (NaN -> 0), so off-scale drivers or terrain do not
    swamp the 0-1 NDVI channels in the ConvLSTM."""
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a)) or 1.0
    return np.nan_to_num((a - mean) / std, nan=0.0).astype("float32")


def channel_stack(ndvi: xr.DataArray, static: xr.Dataset | None = None,
                  drivers: xr.Dataset | None = None) -> np.ndarray:
    """(time, y, x) NDVI -> (time, channel, y, x) input stack.

    Channels: cloud-filled NDVI, validity mask, month sin, month cos, any static
    layers broadcast over time, then any time-varying drivers already aligned to
    the grid.
    """
    v = ndvi.values
    t, h, w = v.shape
    month = ndvi["time"].dt.month.values
    chans = [
        np.nan_to_num(v, nan=0.0).astype("float32"),
        (~np.isnan(v)).astype("float32"),
        np.broadcast_to(np.sin(2 * np.pi * month / 12)[:, None, None], (t, h, w)).astype("float32"),
        np.broadcast_to(np.cos(2 * np.pi * month / 12)[:, None, None], (t, h, w)).astype("float32"),
    ]
    if static is not None:
        for name in static.data_vars:
            layer = _standardize(static[name].values.astype("float32"))
            chans.append(np.broadcast_to(layer[None], (t, h, w)).astype("float32"))
    if drivers is not None:
        for name in drivers.data_vars:
            chans.append(_standardize(drivers[name].values))
    return np.stack(chans, axis=1)


def make_sequences(stack: np.ndarray, target: np.ndarray, seq_len: int):
    """Sliding windows: X (N, seq_len, C, H, W), Y (N, H, W), target time index."""
    n = stack.shape[0]
    xs, ys, idx = [], [], []
    for i in range(n - seq_len):
        xs.append(stack[i : i + seq_len])
        ys.append(target[i + seq_len])
        idx.append(i + seq_len)
    return np.stack(xs), np.stack(ys), np.array(idx)


def sequences_to_cube(pred: np.ndarray, target_idx: np.ndarray, template: xr.DataArray) -> xr.DataArray:
    """Scatter per-sequence predictions back onto a (time, y, x) grid (NaN elsewhere)."""
    cube = np.full(template.shape, np.nan, dtype="float32")
    cube[target_idx] = pred
    return xr.DataArray(cube, dims=template.dims, coords=template.coords, name="convlstm")


# ── model (torch) ───────────────────────────────────────────────────────────
def _torch():
    import torch
    return torch


def _build_model_classes():
    import torch
    import torch.nn as nn

    class ConvLSTMCell(nn.Module):
        def __init__(self, in_ch, hidden, kernel=3):
            super().__init__()
            self.hidden = hidden
            self.conv = nn.Conv2d(in_ch + hidden, 4 * hidden, kernel, padding=kernel // 2)

        def forward(self, x, h, c):
            i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, c

    class NDVIConvLSTM(nn.Module):
        def __init__(self, in_ch, hidden=16, kernel=3):
            super().__init__()
            self.hidden = hidden
            self.cell = ConvLSTMCell(in_ch, hidden, kernel)
            self.head = nn.Conv2d(hidden, 1, 1)
            # Zero-init the head so the model starts as pure persistence
            # (delta = 0) and learns the correction from there.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):  # x: (B, T, C, H, W)
            b, t, c, h, w = x.shape
            hs = x.new_zeros(b, self.hidden, h, w)
            cs = x.new_zeros(b, self.hidden, h, w)
            for step in range(t):
                hs, cs = self.cell(x[:, step], hs, cs)
            delta = self.head(hs).squeeze(1)     # (B, H, W)
            return x[:, -1, 0] + delta           # residual from last filled NDVI frame

    return ConvLSTMCell, NDVIConvLSTM


def make_convlstm(in_ch: int, hidden: int = 16, kernel: int = 3):
    """Build an NDVIConvLSTM (imports torch lazily)."""
    _, NDVIConvLSTM = _build_model_classes()
    return NDVIConvLSTM(in_ch, hidden, kernel)


# ── training / prediction ───────────────────────────────────────────────────
def fit_predict_fold(
    ndvi: xr.DataArray,
    train_time: xr.DataArray,
    spatial_train: xr.DataArray,
    static: xr.Dataset | None = None,
    drivers: xr.Dataset | None = None,
    seq_len: int = DEFAULTS["seq_len"],
    hidden: int = DEFAULTS["hidden"],
    epochs: int = DEFAULTS["epochs"],
    lr: float = DEFAULTS["lr"],
    seed: int = DEFAULTS["seed"],
    device: str | None = None,
):
    """Train on train-time x train-space x valid pixels; predict the whole cube.

    Uses the GPU automatically when one is available (pass `device` to force
    "cuda" or "cpu"). Returns the prediction cube (NaN in the first `seq_len`
    months) and the per-epoch training loss.
    """
    torch = _torch()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    stack = channel_stack(ndvi, static, drivers)
    target = ndvi.values.astype("float32")
    x, y, idx = make_sequences(stack, target, seq_len)

    tr = np.where(train_time.values[idx])[0]
    space = spatial_train.values.astype("float32")
    valid = (~np.isnan(y)).astype("float32")
    y_filled = np.nan_to_num(y, nan=0.0)

    x_tr = torch.from_numpy(x[tr]).to(dev)
    y_tr = torch.from_numpy(y_filled[tr]).to(dev)
    w_tr = torch.from_numpy(valid[tr] * space[None]).to(dev)

    model = make_convlstm(stack.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_tr)
        loss = ((pred - y_tr) ** 2 * w_tr).sum() / w_tr.sum().clamp(min=1.0)
        loss.backward()
        opt.step()
        history.append(loss.item())

    model.eval()
    with torch.no_grad():
        pred_all = model(torch.from_numpy(x).to(dev)).cpu().numpy()
    return sequences_to_cube(pred_all, idx, ndvi), history


def walk_forward_convlstm(
    ndvi: xr.DataArray,
    folds: list[dict],
    spatial_train: xr.DataArray,
    spatial_test: xr.DataArray,
    static: xr.Dataset | None = None,
    drivers: xr.Dataset | None = None,
    **fit_kwargs,
):
    """Retrain ConvLSTM per fold, score vs baselines, stitch OOS forecasts.

    Returns (results DataFrame, out-of-sample cube, list of per-fold loss curves).
    """
    import torch  # fail early with a clear message if torch is missing

    pers = persistence(ndvi)
    oos = xr.full_like(ndvi, np.nan).rename("convlstm_oos")
    rows, histories = [], []

    for fi, fold in enumerate(folds):
        pred, history = fit_predict_fold(ndvi, fold["train"], spatial_train, static, drivers, **fit_kwargs)
        clim = climatology_forecast(ndvi, seasonal_climatology(ndvi.sel(time=fold["train"])))
        preds = {"convlstm": pred, "persistence": pers, "climatology": clim}
        rows.extend(score_cells(
            ndvi, preds, fold["train"], fold["test"],
            spatial_train, spatial_test, fold=fi, label=fold["label"],
        ))
        oos = xr.where(fold["test"], pred, oos)
        histories.append(history)

    import pandas as pd
    return pd.DataFrame(rows), oos, histories
