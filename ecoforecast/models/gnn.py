"""GraphCast-style GNN — the v2 differentiator.

An encode-process-decode graph network over the pixel grid. Each pixel is a node
with the same per-pixel features as the tabular model (lags, month encoding,
static layers, optional drivers); nodes are joined to their four grid neighbours;
a stack of message-passing rounds lets information move across space; and a
residual decoder predicts next month's NDVI as a correction to the last frame.
The decoder is zero-initialised, so the model starts at persistence.

Trained per fold with a loss masked to training months x training blocks x valid
pixels, and scored in the same space x time 2x2 as the other models.

The graph and node-tensor builders are pure NumPy so they can be tested without
PyTorch; the model and training need torch (imported lazily) and a GPU is used
when available.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..baselines import climatology_forecast, persistence
from ..features import seasonal_climatology
from ..evaluate import score_cells

DEFAULTS = dict(hidden=32, rounds=3, epochs=30, lr=5e-3, seed=0)
LAGS = (1, 2, 3)


# ── graph + node tensors (no torch) ─────────────────────────────────────────
def grid_edge_index(h: int, w: int) -> np.ndarray:
    """4-neighbour edges for an h x w grid, node id = y * w + x. Shape (2, E)."""
    idx = np.arange(h * w).reshape(h, w)
    pairs = [
        (idx[:, :-1].ravel(), idx[:, 1:].ravel()),   # left -> right
        (idx[:, 1:].ravel(), idx[:, :-1].ravel()),   # right -> left
        (idx[:-1, :].ravel(), idx[1:, :].ravel()),   # up -> down
        (idx[1:, :].ravel(), idx[:-1, :].ravel()),   # down -> up
    ]
    src = np.concatenate([p[0] for p in pairs])
    dst = np.concatenate([p[1] for p in pairs])
    return np.stack([src, dst]).astype("int64")


def _standardize(a: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(a))
    std = float(np.nanstd(a)) or 1.0
    return ((a - mean) / std).astype("float32")


def build_node_tensors(ndvi, static=None, drivers=None, lags=LAGS):
    """Per-(time, node) features for the graph.

    Returns X (T, N, F), Y (T, N) target, last (T, N) last-observed NDVI for the
    residual, valid (T, N) where features and target are all present, and the
    feature names. N = H * W with node id = y * W + x.
    """
    v = ndvi.values.astype("float32")
    t, h, w = v.shape
    n = h * w
    month = ndvi["time"].dt.month.values

    feats, names = [], []
    for k in lags:
        lag = np.roll(v, k, axis=0)
        lag[:k] = np.nan
        feats.append(lag.reshape(t, n)); names.append(f"lag{k}")
    feats.append(np.broadcast_to(np.sin(2 * np.pi * month / 12)[:, None], (t, n)).astype("float32")); names.append("month_sin")
    feats.append(np.broadcast_to(np.cos(2 * np.pi * month / 12)[:, None], (t, n)).astype("float32")); names.append("month_cos")
    if static is not None:
        for name in static.data_vars:
            layer = _standardize(static[name].values.astype("float32")).reshape(n)
            feats.append(np.broadcast_to(layer[None], (t, n)).astype("float32")); names.append(name)
    if drivers is not None:
        for name in drivers.data_vars:
            feats.append(_standardize(drivers[name].values).reshape(t, n)); names.append(name)

    x = np.stack(feats, axis=-1)                        # (T, N, F)
    y = v.reshape(t, n)                                 # (T, N)
    last = np.roll(v, 1, axis=0); last[0] = np.nan
    last = last.reshape(t, n)
    valid = (~np.isnan(x).any(axis=-1)) & (~np.isnan(y))
    return x, y, last, valid, names


# ── model (torch) ───────────────────────────────────────────────────────────
def _torch():
    import torch
    return torch


def _build_model_classes():
    import torch
    import torch.nn as nn

    class MessagePassing(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.msg = nn.Linear(2 * dim, dim)
            self.upd = nn.Linear(2 * dim, dim)

        def forward(self, h, edge_index):
            src, dst = edge_index[0], edge_index[1]
            m = torch.relu(self.msg(torch.cat([h[dst], h[src]], dim=-1)))
            agg = torch.zeros_like(h).index_add(0, dst, m)
            ones = torch.ones(dst.size(0), device=h.device, dtype=h.dtype)
            deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype).index_add(0, dst, ones).clamp(min=1).unsqueeze(-1)
            return h + torch.relu(self.upd(torch.cat([h, agg / deg], dim=-1)))

    class NDVIGNN(nn.Module):
        def __init__(self, in_dim, hidden=32, rounds=3):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.processor = nn.ModuleList([MessagePassing(hidden) for _ in range(rounds)])
            self.decoder = nn.Linear(hidden, 1)
            nn.init.zeros_(self.decoder.weight)          # start at persistence
            nn.init.zeros_(self.decoder.bias)

        def forward(self, x, edge_index, last_ndvi):     # x: (N, F)
            h = self.encoder(x)
            for layer in self.processor:
                h = layer(h, edge_index)
            return last_ndvi + self.decoder(h).squeeze(-1)

    return MessagePassing, NDVIGNN


def make_gnn(in_dim: int, hidden: int = 32, rounds: int = 3):
    """Build an NDVIGNN (imports torch lazily)."""
    _, NDVIGNN = _build_model_classes()
    return NDVIGNN(in_dim, hidden, rounds)


# ── training / prediction ───────────────────────────────────────────────────
def fit_predict_fold(
    ndvi: xr.DataArray,
    train_time: xr.DataArray,
    spatial_train: xr.DataArray,
    static: xr.Dataset | None = None,
    drivers: xr.Dataset | None = None,
    hidden: int = DEFAULTS["hidden"],
    rounds: int = DEFAULTS["rounds"],
    epochs: int = DEFAULTS["epochs"],
    lr: float = DEFAULTS["lr"],
    seed: int = DEFAULTS["seed"],
    device: str | None = None,
):
    """Train on train-time x train-space x valid nodes; predict the whole cube.

    SGD over timesteps (each month is one graph). Uses the GPU when available.
    Returns the prediction cube (NaN where lags are unavailable) and the
    per-epoch mean training loss.
    """
    torch = _torch()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    x, y, last, valid, names = build_node_tensors(ndvi, static, drivers)
    t, n, f = x.shape
    h, w = ndvi.sizes["y"], ndvi.sizes["x"]

    edge_index = torch.from_numpy(grid_edge_index(h, w)).to(dev)
    space = torch.from_numpy(spatial_train.values.reshape(n).astype("float32")).to(dev)
    xt = torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(dev)
    yt = torch.from_numpy(np.nan_to_num(y, nan=0.0)).to(dev)
    lastt = torch.from_numpy(np.nan_to_num(last, nan=0.0)).to(dev)
    validt = torch.from_numpy(valid.astype("float32")).to(dev)

    tt = train_time.values
    train_steps = [i for i in range(t) if tt[i] and valid[i].any()]

    model = make_gnn(f, hidden, rounds).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        losses = []
        for i in rng.permutation(train_steps):
            opt.zero_grad()
            pred = model(xt[i], edge_index, lastt[i])
            weight = validt[i] * space
            loss = ((pred - yt[i]) ** 2 * weight).sum() / weight.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)) if losses else float("nan"))

    model.eval()
    cube = np.full((t, n), np.nan, dtype="float32")
    with torch.no_grad():
        for i in range(t):
            if valid[i].any():
                pred = model(xt[i], edge_index, lastt[i]).cpu().numpy()
                cube[i] = np.where(valid[i], pred, np.nan)

    pred_da = xr.DataArray(cube.reshape(t, h, w), dims=ndvi.dims, coords=ndvi.coords, name="gnn")
    return pred_da, history


def walk_forward_gnn(
    ndvi: xr.DataArray,
    folds: list[dict],
    spatial_train: xr.DataArray,
    spatial_test: xr.DataArray,
    static: xr.Dataset | None = None,
    drivers: xr.Dataset | None = None,
    **fit_kwargs,
):
    """Retrain the GNN per fold, score vs baselines, stitch OOS forecasts."""
    import torch  # fail early with a clear message if torch is missing

    pers = persistence(ndvi)
    oos = xr.full_like(ndvi, np.nan).rename("gnn_oos")
    rows, histories = [], []
    for fi, fold in enumerate(folds):
        pred, history = fit_predict_fold(ndvi, fold["train"], spatial_train, static, drivers, **fit_kwargs)
        clim = climatology_forecast(ndvi, seasonal_climatology(ndvi.sel(time=fold["train"])))
        preds = {"gnn": pred, "persistence": pers, "climatology": clim}
        rows.extend(score_cells(
            ndvi, preds, fold["train"], fold["test"],
            spatial_train, spatial_test, fold=fi, label=fold["label"],
        ))
        oos = xr.where(fold["test"], pred, oos)
        histories.append(history)

    import pandas as pd
    return pd.DataFrame(rows), oos, histories
