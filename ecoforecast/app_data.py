"""Compact per-biome artifacts for the Streamlit demo.

The hosted demo runs in a small container with no model libraries, so the
pipeline precomputes everything it needs: a spatially coarsened NDVI window, the
month-of-year climatology, each model's out-of-sample forecasts over the test
window, the training mask, and a table of conformal quantiles.

Storing the quantiles (rather than one fixed band) is what keeps the demo
interactive: the app can redraw the uncertainty band at any confidence level by
looking one up, with no recomputation. Persistence is derived in the app from the
NDVI window, so it does not need storing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# Confidence levels the app's slider can choose between.
LEVELS = np.round(np.arange(0.50, 0.991, 0.01), 3)


def _coarsen(da: xr.DataArray, factor: int) -> xr.DataArray:
    if factor <= 1:
        return da
    return da.coarsen(y=factor, x=factor, boundary="trim").mean()


def _union(folds, keys):
    mask = folds[keys[0]]["test"].copy()
    for k in keys[1:]:
        mask = mask | folds[k]["test"]
    return mask


def headline_rmse(results) -> xr.DataArray:
    """Headline cell (future time, seen locations) RMSE per predictor."""
    head = results[(results["time"] == "future") & (results["space"] == "seen")]
    mean = head.groupby("predictor")["rmse"].mean()
    return xr.DataArray(mean.values.astype("float32"), dims=("predictor",),
                        coords={"predictor": list(mean.index)})


def save_app_data(
    ndvi: xr.DataArray,
    oos: dict[str, xr.DataArray],
    folds: list[dict],
    spatial_train: xr.DataArray,
    biome: str,
    tag: str,
    out_dir: Path,
    results=None,
    display_months: int = 24,
    coarsen: int = 2,
) -> Path:
    """Write {out_dir}/{tag}_{biome}.nc for the demo. Returns the path.

    `results` is the scored DataFrame for this biome; its headline cell is stored
    so the app reports the same numbers as the README rather than recomputing
    them from the coarsened grid.
    """
    test = _union(folds, list(range(len(folds))))
    calib = _union(folds, list(range(max(1, len(folds) - 1))))

    nd = _coarsen(ndvi, coarsen)
    train = _coarsen(spatial_train.astype("float32"), coarsen) > 0.5

    data = {
        "ndvi": nd.isel(time=slice(-display_months, None)),
        "climatology": nd.groupby("time.month").mean("time"),
        "train_mask": train,
    }

    quantiles = {}
    for name, cube in oos.items():
        pred = _coarsen(cube, coarsen)
        data[f"pred_{name}"] = pred.sel(time=test)
        resid = np.abs((nd - pred).sel(time=calib).where(train).values)
        resid = resid[np.isfinite(resid)]
        quantiles[name] = np.quantile(resid, LEVELS) if resid.size else np.full(LEVELS.size, np.nan)

    ds = xr.Dataset(data)
    ds["conformal_q"] = xr.DataArray(
        np.stack([quantiles[m] for m in oos]),
        dims=("model", "level"),
        coords={"model": list(oos), "level": LEVELS},
    )

    if results is not None:
        ds["headline_rmse"] = headline_rmse(results)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_{biome}.nc"
    # Forecasts are NaN outside their test window once xarray aligns them onto the
    # display axis, so compression pays for itself several times over.
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 5} for v in ds.data_vars})
    return path
