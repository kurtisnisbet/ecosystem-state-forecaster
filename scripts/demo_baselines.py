"""Validate `features` + `baselines` on a synthetic monthly NDVI cube.

Runs with no network or real data: builds a small seasonal + autocorrelated
cube (with an injected 2021 drought and some cloud gaps), exercises the feature
and baseline functions, asserts the key invariants (train-only climatology,
correct lags, no NaNs leaking into the design matrix), scores the baselines on
a held-out future year, and writes figures to docs/figures/ for the README.

Run:  python scripts/demo_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoforecast.features import (
    add_lags,
    build_feature_table,
    compute_anomaly,
    seasonal_climatology,
)
from ecoforecast.baselines import climatology_forecast, persistence

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
RNG = np.random.default_rng(42)


def make_synthetic_cube(n_months=72, size=24):
    """Seasonal + trend + AR(1) noise, a 2021 drought dip, and 5% cloud gaps."""
    time = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    base = 0.30 + 0.40 * (0.5 * yy + 0.5 * (1 - xx))     # greener in one corner
    amp = 0.08 + 0.12 * yy                               # stronger seasonality one way
    month = time.month.values

    cube = np.empty((n_months, size, size), dtype="float32")
    noise = np.zeros((size, size), dtype="float32")
    for t in range(n_months):
        season = amp * np.sin(2 * np.pi * (month[t] - 1) / 12 + np.pi)   # SH summer peak
        trend = 0.02 * (t / n_months) * (2 * xx - 1)                     # slow, spatial
        noise = 0.6 * noise + 0.4 * RNG.normal(0, 0.03, (size, size))    # AR(1)
        cube[t] = base + season + trend + noise

    drought = np.exp(-((np.arange(n_months) - 43) ** 2) / (2 * 3.0 ** 2))  # ~2021-08
    cube -= (0.12 * drought)[:, None, None]
    cube = np.clip(cube, 0.05, 0.95)

    da = xr.DataArray(
        cube,
        dims=("time", "y", "x"),
        coords={"time": time, "y": np.arange(size), "x": np.arange(size)},
        name="ndvi",
    )
    return da.where(RNG.random(da.shape) >= 0.05)


def rmse(pred, obs, time_mask):
    err = (pred - obs).sel(time=time_mask)
    return float(np.sqrt((err ** 2).mean()))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ndvi = make_synthetic_cube()

    # Honest split: climatology / training sees the past only.
    train = ndvi["time"] < np.datetime64("2023-01-01")
    test = ~train
    clim = seasonal_climatology(ndvi.sel(time=train))
    anom = compute_anomaly(ndvi, clim)

    # --- invariants ---------------------------------------------------------
    assert clim.sizes["month"] == 12
    train_monthly_mean = anom.sel(time=train).groupby("time.month").mean("time")
    assert np.nanmax(np.abs(train_monthly_mean.values)) < 1e-5, "train anomaly not centred"

    l1 = add_lags(ndvi)["ndvi_lag1"]
    a = l1.isel(time=slice(1, None)).values
    b = ndvi.isel(time=slice(0, -1)).values
    valid = ~np.isnan(a)
    assert bool((a[valid] == b[valid]).all()), "lag1 != NDVI shifted by one"

    table = build_feature_table(ndvi)
    assert not table[["target", "lag1", "lag2", "lag3", "month_sin", "month_cos"]].isna().any().any()

    # --- baselines scored on the unseen future year -------------------------
    pers = persistence(ndvi)
    climfc = climatology_forecast(ndvi, climatology=clim)
    rmse_pers = rmse(pers, ndvi, test)
    rmse_clim = rmse(climfc, ndvi, test)
    skill = 1 - rmse_clim / rmse_pers

    print("Synthetic validation")
    print(f"  cube: {dict(ndvi.sizes)}   clouds NaN: {float(np.isnan(ndvi).mean()):.1%}")
    print(f"  feature table rows: {len(table):,}")
    print(f"  test RMSE  persistence: {rmse_pers:.4f}")
    print(f"  test RMSE  climatology: {rmse_clim:.4f}")
    print(f"  climatology skill vs persistence: {skill:+.1%}")

    _plot_anomaly(ndvi, clim, anom)
    _plot_forecast_vs_actual(ndvi, pers, climfc, test, rmse_pers, rmse_clim)
    _plot_error_maps(ndvi, pers, climfc)
    print(f"  figures -> {FIG_DIR}")


def _plot_anomaly(ndvi, clim, anom, py=6, px=6):
    t = ndvi["time"].values
    obs = ndvi.isel(y=py, x=px)
    expected = clim.sel(month=ndvi["time"].dt.month).drop_vars("month").isel(y=py, x=px)
    av = anom.isel(y=py, x=px).values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax1.plot(t, obs, color="#1b7837", lw=1.6, label="NDVI")
    ax1.plot(t, expected, color="#999999", lw=1.4, ls="--", label="seasonal climatology (train)")
    ax1.set_ylabel("NDVI"); ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(f"Synthetic pixel ({py},{px}): NDVI vs climatology, and anomaly")

    pos = np.where(~np.isnan(av) & (av >= 0), av, 0.0)
    neg = np.where(~np.isnan(av) & (av < 0), av, 0.0)
    ax2.axhline(0, color="#666666", lw=0.8)
    ax2.fill_between(t, 0, pos, color="#2166ac", alpha=0.75, label="above climatology")
    ax2.fill_between(t, 0, neg, color="#b2182b", alpha=0.75, label="below (e.g. drought)")
    ax2.set_ylabel("anomaly"); ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "synthetic_ndvi_climatology_anomaly.png", dpi=130)
    plt.close(fig)


def _plot_forecast_vs_actual(ndvi, pers, climfc, test, rmse_pers, rmse_clim, py=6, px=6):
    t = ndvi["time"].sel(time=test).values
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, ndvi.sel(time=test).isel(y=py, x=px), color="black", lw=2, marker="o", ms=3, label="actual")
    ax.plot(t, pers.sel(time=test).isel(y=py, x=px), color="#d95f02", lw=1.5, label=f"persistence (RMSE {rmse_pers:.3f})")
    ax.plot(t, climfc.sel(time=test).isel(y=py, x=px), color="#7570b3", lw=1.5, label=f"climatology (RMSE {rmse_clim:.3f})")
    ax.set_title("Baseline forecasts vs actual — held-out future year")
    ax.set_ylabel("NDVI"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "baseline_forecast_vs_actual.png", dpi=130)
    plt.close(fig)


def _plot_error_maps(ndvi, pers, climfc, month="2023-06"):
    actual = ndvi.sel(time=month).isel(time=0)
    e_pers = np.abs(pers.sel(time=month).isel(time=0) - actual)
    e_clim = np.abs(climfc.sel(time=month).isel(time=0) - actual)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    im0 = axes[0].imshow(actual, cmap="YlGn", vmin=0, vmax=0.9); axes[0].set_title(f"actual NDVI ({month})")
    im1 = axes[1].imshow(e_pers, cmap="magma", vmin=0, vmax=0.15); axes[1].set_title("|persistence error|")
    im2 = axes[2].imshow(e_clim, cmap="magma", vmin=0, vmax=0.15); axes[2].set_title("|climatology error|")
    for ax, im in zip(axes, (im0, im1, im2)):
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / "baseline_error_maps.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
