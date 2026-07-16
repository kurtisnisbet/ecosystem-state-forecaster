"""Stacked ensemble of the forecasting models.

Combines the models' out-of-sample forecasts with non-negative weights fit by
rolling calibration: for each walk-forward fold the weights are learned on the
folds before it (so the calibration data is unseen at that point) and applied to
it. Non-negative least squares keeps the weights interpretable as each model's
contribution and stops the stack from over-fitting negative combinations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression

from .baselines import climatology_forecast, persistence
from .features import seasonal_climatology
from .evaluate import score_cells


def _design(obs, model_oos, time_mask, space_mask):
    names = list(model_oos)
    o = obs.sel(time=time_mask)
    if space_mask is not None:
        o = o.where(space_mask)
    cols = []
    for name in names:
        p = model_oos[name].sel(time=time_mask)
        if space_mask is not None:
            p = p.where(space_mask)
        cols.append(p.values.ravel())
    y = o.values.ravel()
    x = np.column_stack(cols)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[keep], y[keep], names


def stack_ensemble(obs, model_oos: dict[str, xr.DataArray], folds, space_mask=None):
    """Rolling convex stack of the model out-of-sample cubes.

    The first fold has nothing earlier to calibrate on, so it uses an equal-weight
    average; later folds fit non-negative weights on the earlier folds and
    normalise them to sum to one. The convex combination keeps the ensemble
    bounded by its members, so it stays robust under a regime shift instead of
    extrapolating past them. Returns the ensemble OOS cube (defined on every
    fold) and a per-fold table of weights.
    """
    names = list(model_oos)
    ens = xr.full_like(obs, np.nan).rename("ensemble")
    equal = sum(model_oos[n] for n in names) / len(names)
    rows = []
    for k in range(len(folds)):
        weights = np.full(len(names), 1.0 / len(names))
        if k >= 1:
            calib = folds[0]["test"].copy()
            for j in range(1, k):
                calib = calib | folds[j]["test"]
            x, y, _ = _design(obs, model_oos, calib, space_mask)
            if y.size >= len(names) + 1:
                fit = LinearRegression(positive=True, fit_intercept=False).fit(x, y).coef_
                if fit.sum() > 0:
                    weights = fit / fit.sum()

        blended = sum(weights[i] * model_oos[names[i]] for i in range(len(names)))
        ens = xr.where(folds[k]["test"], blended, ens)
        row = {"fold": k, "label": folds[k]["label"]}
        row.update({f"w_{name}": float(weights[i]) for i, name in enumerate(names)})
        rows.append(row)
    return ens, pd.DataFrame(rows)


def score_ensemble(obs, ens, folds, spatial_train, spatial_test):
    """Score the ensemble across the space x time 2x2, like the other models."""
    pers = persistence(obs)
    rows = []
    for fi, fold in enumerate(folds):
        clim = climatology_forecast(obs, seasonal_climatology(obs.sel(time=fold["train"])))
        preds = {"ensemble": ens, "persistence": pers, "climatology": clim}
        rows.extend(score_cells(
            obs, preds, fold["train"], fold["test"],
            spatial_train, spatial_test, fold=fi, label=fold["label"],
        ))
    return pd.DataFrame(rows)
