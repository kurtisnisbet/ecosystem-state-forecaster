"""Feature engineering for the cube.

NDVI anomaly (vs a training-only seasonal climatology), short dynamic lags,
month-of-year encoding, and a per-pixel feature table for the tabular models.
Seasonality is encoded explicitly (climatology + month sin/cos), never as a
12-month lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

LAGS = (1, 2, 3)


def seasonal_climatology(ndvi: xr.DataArray) -> xr.DataArray:
    """Mean NDVI per calendar month, dims (month, y, x).

    Pass TRAINING data only — computing this over all years leaks the test
    period into the anomaly and the climatology baseline.
    """
    return ndvi.groupby("time.month").mean("time")


def compute_anomaly(ndvi: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    """NDVI minus its seasonal climatology, matched by calendar month."""
    anom = ndvi.groupby("time.month") - climatology
    anom.name = "ndvi_anomaly"
    return anom


def add_lags(da: xr.DataArray, lags: tuple[int, ...] = LAGS) -> xr.Dataset:
    """Dynamic lag features: da shifted forward by each k in `lags`."""
    name = da.name or "ndvi"
    return xr.Dataset({f"{name}_lag{k}": da.shift(time=k) for k in lags})


def add_seasonal_encoding(da: xr.DataArray) -> xr.Dataset:
    """Cyclic month-of-year encoding (sin/cos), one value per timestep."""
    month = da["time"].dt.month
    angle = 2 * np.pi * month / 12
    return xr.Dataset({"month_sin": np.sin(angle), "month_cos": np.cos(angle)})


def build_feature_table(
    target: xr.DataArray,
    lags: tuple[int, ...] = LAGS,
    static: xr.Dataset | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Tidy (time, y, x) design matrix for the tabular models.

    Columns: `target`, `lag{k}` (target at t-k), `month_sin`/`month_cos`, and
    any static layers (terrain, biome id) broadcast over time. With dropna, the
    first max(lags) steps and cloud-masked pixels are removed.
    """
    feats = xr.Dataset({"target": target})
    for k in lags:
        feats[f"lag{k}"] = target.shift(time=k)

    angle = 2 * np.pi * target["time"].dt.month / 12
    feats["month_sin"] = np.sin(angle)
    feats["month_cos"] = np.cos(angle)

    if static is not None:
        for name, layer in static.items():
            feats[name] = layer

    df = feats.to_dataframe().reset_index()
    if dropna:
        df = df.dropna().reset_index(drop=True)
    return df
