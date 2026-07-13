"""Honest spatio-temporal evaluation.

Walk-forward temporal folds with an embargo gap, spatial blocks with a buffer,
metrics (RMSE / R²) and skill relative to the persistence and climatology
baselines, reported across the space x time 2x2 (see brief section 10).

All predictions and observations are xarray DataArrays with dims (time, y, x),
aligned on the same grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


# ── metrics ────────────────────────────────────────────────────────────────
def rmse(pred: xr.DataArray, obs: xr.DataArray) -> float:
    return float(np.sqrt(((pred - obs) ** 2).mean()))


def r2(pred: xr.DataArray, obs: xr.DataArray) -> float:
    ss_res = float(((obs - pred) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot else np.nan


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """Fractional RMSE reduction vs a reference (>0 = better than reference)."""
    if not rmse_reference or np.isnan(rmse_reference):
        return np.nan
    return 1 - rmse_model / rmse_reference


# ── temporal splits: expanding-window walk-forward ──────────────────────────
def walk_forward_splits(
    time: xr.DataArray,
    block_months: int = 3,
    n_test_folds: int = 4,
    embargo_months: int = 3,
) -> list[dict]:
    """Expanding-window folds over a contiguous monthly `time` axis.

    Each fold tests a `block_months` block near the end of the record and trains
    on everything up to `embargo_months` before it (the embargo drops the most
    autocorrelated months so the test isn't artificially easy). Folds whose
    training window would be empty are skipped. Returns dicts with boolean
    `train`/`test` masks over time and a human-readable `label`.
    """
    times = np.asarray(time.values)
    n = times.size
    pos = np.arange(n)
    folds = []
    for f in range(n_test_folds):
        test_start = n - (n_test_folds - f) * block_months
        test_stop = test_start + block_months
        train_stop = test_start - embargo_months
        if train_stop <= 0:
            continue
        train = xr.DataArray(pos < train_stop, coords={"time": times}, dims="time")
        test = xr.DataArray(
            (pos >= test_start) & (pos < test_stop), coords={"time": times}, dims="time"
        )
        label = (
            f"{np.datetime_as_string(times[test_start], unit='M')}"
            f"..{np.datetime_as_string(times[test_stop - 1], unit='M')}"
        )
        folds.append({"train": train, "test": test, "label": label})
    return folds


# ── spatial splits: blocks with a buffer ────────────────────────────────────
def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square (Chebyshev) dilation by r pixels, no edge wrap."""
    if r <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, r, constant_values=False)
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= padded[r + dy : r + dy + h, r + dx : r + dx + w]
    return out


def spatial_blocks(
    da: xr.DataArray,
    block_size: int,
    n_test_blocks: int = 3,
    buffer: int = 1,
    seed: int = 0,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Tile (y, x) into square blocks; hold out `n_test_blocks` as test.

    A `buffer`-pixel ring around each test block is excluded from training
    (spatial embargo) so autocorrelation can't leak across the split. `block_size`
    and `buffer` are in pixels — set them from the NDVI variogram range post-EDA.
    Returns (train_mask, test_mask, block_id) as (y, x) DataArrays.
    """
    ny, nx = da.sizes["y"], da.sizes["x"]
    by = np.arange(ny) // block_size
    bx = np.arange(nx) // block_size
    block_id = by[:, None] * (bx.max() + 1) + bx[None, :]

    ids = np.unique(block_id)
    rng = np.random.default_rng(seed)
    test_ids = rng.choice(ids, size=min(n_test_blocks, len(ids)), replace=False)
    test = np.isin(block_id, test_ids)
    train = ~_dilate(test, buffer)

    coords = {"y": da["y"], "x": da["x"]}
    to_da = lambda a: xr.DataArray(a, coords=coords, dims=("y", "x"))
    return to_da(train), to_da(test), to_da(block_id)


# ── scoring across the space x time 2x2 ─────────────────────────────────────
def _paired(obs, pred, time_mask, space_mask):
    o = obs.where(space_mask).sel(time=time_mask)
    p = pred.where(space_mask).sel(time=time_mask)
    valid = o.notnull() & p.notnull()
    return o.where(valid), p.where(valid), int(valid.sum())


def score_cells(
    obs: xr.DataArray,
    predictions: dict[str, xr.DataArray],
    train_time: xr.DataArray,
    test_time: xr.DataArray,
    spatial_train: xr.DataArray,
    spatial_test: xr.DataArray,
    baselines: tuple[str, ...] = ("persistence", "climatology"),
    fold: int = 0,
    label: str = "",
) -> list[dict]:
    """Score every predictor in the four space x time cells for one fold.

    Cells: future/seen time (holdout vs training months) x seen/unseen locations
    (spatial train vs held-out blocks). `predictions` must include the baseline
    keys. Returns row dicts (RMSE, R², n, skill vs each baseline). Kept separate
    from `evaluate_folds` so a per-fold-retrained model can score each fold with
    its own prediction cube.
    """
    cells = [
        ("future", "seen", test_time, spatial_train),
        ("future", "unseen", test_time, spatial_test),
        ("seen", "seen", train_time, spatial_train),
        ("seen", "unseen", train_time, spatial_test),
    ]
    rows = []
    for time_lbl, space_lbl, tmask, smask in cells:
        base_rmse = {}
        for b in baselines:
            bo, bp, _ = _paired(obs, predictions[b], tmask, smask)
            base_rmse[b] = rmse(bp, bo)
        for name, pred in predictions.items():
            o, p, n = _paired(obs, pred, tmask, smask)
            row = {
                "fold": fold, "label": label,
                "time": time_lbl, "space": space_lbl,
                "predictor": name, "n": n,
                "rmse": rmse(p, o), "r2": r2(p, o),
            }
            for b in baselines:
                row[f"skill_vs_{b}"] = skill_score(row["rmse"], base_rmse[b])
            rows.append(row)
    return rows


def evaluate_folds(
    obs: xr.DataArray,
    predictions: dict[str, xr.DataArray],
    folds: list[dict],
    spatial_train: xr.DataArray,
    spatial_test: xr.DataArray,
    baselines: tuple[str, ...] = ("persistence", "climatology"),
) -> pd.DataFrame:
    """Score fixed predictions across every walk-forward fold (see `score_cells`)."""
    rows = []
    for fi, fold in enumerate(folds):
        rows.extend(score_cells(
            obs, predictions, fold["train"], fold["test"],
            spatial_train, spatial_test, baselines, fold=fi, label=fold["label"],
        ))
    return pd.DataFrame(rows)


def summarise_2x2(
    results: pd.DataFrame,
    predictor: str,
    metric: str = "skill_vs_persistence",
) -> pd.DataFrame:
    """Pivot one predictor's `metric` into the space x time 2x2 (mean over folds)."""
    sub = results[results["predictor"] == predictor]
    table = sub.pivot_table(index="time", columns="space", values=metric, aggfunc="mean")
    return table.reindex(index=["future", "seen"], columns=["seen", "unseen"])
