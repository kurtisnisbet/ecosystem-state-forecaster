"""Gradient-boosted trees on per-pixel features + temporal lags.

A LightGBM regressor over the tidy feature table (lag1..3 + month sin/cos +
optional static layers). Trained per walk-forward fold on the training months
and training locations only, then used to predict the whole grid so it can be
scored in the same space x time 2x2 as the baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from lightgbm import LGBMRegressor

from ..baselines import climatology_forecast, persistence
from ..features import build_feature_table, seasonal_climatology
from ..evaluate import score_cells

DEFAULT_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=-1,
    verbose=-1,
)
LAG_SEASON_COLS = ["lag1", "lag2", "lag3", "month_sin", "month_cos"]


def make_gbt(**overrides) -> LGBMRegressor:
    """LightGBM regressor with sensible defaults for this problem."""
    return LGBMRegressor(**{**DEFAULT_PARAMS, **overrides})


def _feature_cols(static: xr.Dataset | None) -> list[str]:
    static_cols = list(static.data_vars) if static is not None else []
    return LAG_SEASON_COLS + static_cols


def fit_predict_fold(
    ndvi: xr.DataArray,
    train_time: xr.DataArray,
    spatial_train: xr.DataArray,
    static: xr.Dataset | None = None,
    params: dict | None = None,
) -> tuple[xr.DataArray, LGBMRegressor]:
    """Train on (train-time x train-space) rows, predict the full (time, y, x) cube.

    Returns the prediction cube (NaN where lag features are unavailable) and the
    fitted model.
    """
    cols = _feature_cols(static)

    layers = xr.Dataset() if static is None else static.copy()
    layers["in_space_train"] = spatial_train
    table = build_feature_table(ndvi, static=layers, dropna=False)

    train_times = ndvi["time"].values[train_time.values]
    is_train = (
        table["time"].isin(train_times)
        & table["in_space_train"].astype(bool)
        & table[cols + ["target"]].notna().all(axis=1)
    )
    model = make_gbt(**(params or {}))
    model.fit(table.loc[is_train, cols], table.loc[is_train, "target"])

    has_features = table[cols].notna().all(axis=1)
    table["pred"] = np.nan
    table.loc[has_features, "pred"] = model.predict(table.loc[has_features, cols])
    pred = table.set_index(["time", "y", "x"])["pred"].to_xarray()
    return pred.rename("gbt").transpose(*ndvi.dims), model


def walk_forward_gbt(
    ndvi: xr.DataArray,
    folds: list[dict],
    spatial_train: xr.DataArray,
    spatial_test: xr.DataArray,
    static: xr.Dataset | None = None,
    params: dict | None = None,
) -> tuple[pd.DataFrame, xr.DataArray, pd.Series]:
    """Retrain GBT per fold, score it against the baselines, stitch OOS forecasts.

    For each fold the climatology baseline is refit on that fold's training
    months (no leakage). Returns (results DataFrame, out-of-sample test-forecast
    cube, mean feature importances).
    """
    pers = persistence(ndvi)
    oos = xr.full_like(ndvi, np.nan).rename("gbt_oos")
    rows, importances = [], []

    for fi, fold in enumerate(folds):
        gbt_pred, model = fit_predict_fold(ndvi, fold["train"], spatial_train, static, params)
        clim = climatology_forecast(ndvi, seasonal_climatology(ndvi.sel(time=fold["train"])))
        preds = {"gbt": gbt_pred, "persistence": pers, "climatology": clim}

        rows.extend(score_cells(
            ndvi, preds, fold["train"], fold["test"],
            spatial_train, spatial_test, fold=fi, label=fold["label"],
        ))
        oos = xr.where(fold["test"], gbt_pred, oos)
        importances.append(pd.Series(model.feature_importances_, index=_feature_cols(static)))

    mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return pd.DataFrame(rows), oos, mean_importance
