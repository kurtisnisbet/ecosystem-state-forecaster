"""Run baselines, GBT and ConvLSTM on the cached real NDVI cube.

Loads the NetCDF built by build_cube.py, runs the walk-forward + spatial-block
evaluation, prints a results table, and writes figures to docs/figures/real_*.
ConvLSTM is skipped with a message if PyTorch isn't installed, so you can get
baseline + GBT results without it.

Run:  python scripts/run_pipeline.py
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.features import seasonal_climatology
from ecoforecast.evaluate import spatial_blocks, summarise_2x2, walk_forward_splits
from ecoforecast.models.gbt import walk_forward_gbt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "ecoforecast" / "config.yaml").read_text())
    cache = ROOT / cfg["build"]["cache"]
    if not cache.exists():
        raise SystemExit(f"No cube at {cache} — run scripts/build_cube.py first.")

    ndvi = xr.open_dataarray(cache)
    drivers = None
    dcfg = cfg.get("drivers_build", {})
    dpath = ROOT / dcfg.get("cache", "data/__no_drivers__.nc")
    if dcfg.get("use", False) and dpath.exists():
        from ecoforecast.drivers import lag_drivers
        drivers = lag_drivers(xr.open_dataset(dpath))  # prior-month values only

    tsp, ssp = cfg["splits"]["temporal"], cfg["splits"]["spatial"]
    folds = walk_forward_splits(ndvi["time"], tsp["block_months"], tsp["n_test_folds"], tsp["embargo_months"])
    strain, stest, _ = spatial_blocks(ndvi, ssp["block_size_px"], ssp["n_test_blocks"], ssp["buffer_px"], seed=1)

    print(f"cube {dict(ndvi.sizes)}  cloud/gap NaN {float(ndvi.isnull().mean()):.1%}")
    print(f"folds: {[f['label'] for f in folds]}")
    print(f"spatial px  train {int(strain.sum())}  test {int(stest.sum())}  buffer {int((~strain & ~stest).sum())}")
    print(f"drivers: {list(drivers.data_vars) if drivers is not None else 'none'}")

    # GBT scores itself + both baselines; take those rows.
    res_gbt, oos_gbt, importance = walk_forward_gbt(ndvi, folds, strain, stest, drivers=drivers)
    frames, oos = [res_gbt], {"gbt": oos_gbt}

    try:
        import torch
        from ecoforecast.models.convlstm import walk_forward_convlstm
        print(f"ConvLSTM on {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}...")
        res_cl, oos_cl, _ = walk_forward_convlstm(ndvi, folds, strain, stest, drivers=drivers, seq_len=6, hidden=16, epochs=100)
        frames.append(res_cl[res_cl["predictor"] == "convlstm"])
        oos["convlstm"] = oos_cl
    except Exception as exc:
        print(f"ConvLSTM skipped ({type(exc).__name__}) — install torch to include it.")

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(ROOT / "docs" / "real_results.csv", index=False)

    head = res[(res.time == "future") & (res.space == "seen")].groupby("predictor")["rmse"].mean().sort_values()
    print("\nheadline (future / seen) RMSE:")
    print("  " + head.round(4).to_string().replace("\n", "\n  "))
    for model in oos:
        sk = summarise_2x2(res, model, "skill_vs_persistence")
        print(f"\n{model} skill vs persistence (2x2 mean):")
        print("  " + sk.round(3).to_string().replace("\n", "\n  "))

    _plot_skill(res, list(oos))
    _plot_forecast(ndvi, folds, oos)
    print(f"\nfigures -> {FIG_DIR}   results -> docs/real_results.csv")


def _best_pixel(ndvi):
    valid = ndvi.notnull().sum("time")
    flat = int(np.argmax(valid.values))
    return np.unravel_index(flat, valid.shape)


def _test_union(folds):
    mask = folds[0]["test"].copy()
    for fo in folds[1:]:
        mask = mask | fo["test"]
    return mask


def _plot_skill(res, models):
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        tab = summarise_2x2(res, model, "skill_vs_persistence").values
        im = ax.imshow(tab, cmap="RdBu", vmin=-0.4, vmax=0.4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["seen loc", "unseen loc"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["future", "seen"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{tab[i, j]:+.2f}", ha="center", va="center", fontweight="bold")
        ax.set_title(f"{model} skill vs persistence")
    fig.tight_layout(); fig.savefig(FIG_DIR / "real_skill_2x2.png", dpi=130); plt.close(fig)


def _plot_forecast(ndvi, folds, oos):
    py, px = _best_pixel(ndvi)
    test_all = _test_union(folds)
    t = ndvi["time"].sel(time=test_all).values
    clim = climatology_forecast(ndvi, seasonal_climatology(ndvi))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, ndvi.sel(time=test_all).isel(y=py, x=px), "k-o", lw=2, ms=3, label="actual")
    ax.plot(t, persistence(ndvi).sel(time=test_all).isel(y=py, x=px), color="#d95f02", label="persistence")
    ax.plot(t, clim.sel(time=test_all).isel(y=py, x=px), color="#7570b3", label="climatology")
    for model, colour in zip(oos, ("#1b7837", "#e7298a")):
        ax.plot(t, oos[model].sel(time=test_all).isel(y=py, x=px), color=colour, lw=2, label=model)
    ax.set_title(f"Real NDVI — forecasts vs actual, test months (pixel {py},{px})")
    ax.set_ylabel("NDVI"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "real_forecast_vs_baselines.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
