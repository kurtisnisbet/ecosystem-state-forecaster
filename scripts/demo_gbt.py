"""Validate models/gbt.py on the synthetic cube and render the GBT figures.

Runs the walk-forward gradient-boosted-trees model (retrained per fold on
train-time x train-space rows), scores it against the persistence and
climatology baselines across the space x time 2x2, and writes figures to
docs/figures/. On the synthetic cube the GBT beats both baselines — on real
NDVI that is genuinely hard and not guaranteed (see the brief).

Run:  python scripts/demo_gbt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_baselines import make_synthetic_cube  # reuse the same synthetic cube
from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.features import seasonal_climatology
from ecoforecast.evaluate import spatial_blocks, summarise_2x2, walk_forward_splits
from ecoforecast.models.gbt import walk_forward_gbt

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def _static_terrain(ndvi):
    ny, nx = ndvi.sizes["y"], ndvi.sizes["x"]
    yy, xx = np.mgrid[0:ny, 0:nx] / (ny - 1)
    elev = xr.DataArray((0.5 * yy + 0.3 * xx).astype("float32"),
                        dims=("y", "x"), coords={"y": ndvi["y"], "x": ndvi["x"]})
    return xr.Dataset({"elevation": elev})


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ndvi = make_synthetic_cube()
    static = _static_terrain(ndvi)

    folds = walk_forward_splits(ndvi["time"], block_months=3, n_test_folds=4, embargo_months=3)
    strain, stest, _ = spatial_blocks(ndvi, block_size=6, n_test_blocks=3, buffer=1, seed=1)
    res, oos, importance = walk_forward_gbt(ndvi, folds, strain, stest, static=static)

    headline = res[(res.time == "future") & (res.space == "seen")].groupby("predictor")["rmse"].mean()
    print("GBT walk-forward validation")
    print("  predictors scored:", sorted(res["predictor"].unique()))
    print("  headline (future/seen) RMSE:")
    print("    " + headline.round(4).to_string().replace("\n", "\n    "))
    print("  GBT skill vs persistence (2x2 mean):")
    print("    " + summarise_2x2(res, "gbt", "skill_vs_persistence").round(3).to_string().replace("\n", "\n    "))
    print("  feature importance:", {k: int(v) for k, v in importance.items()})

    _plot_forecast(ndvi, folds, oos)
    _plot_skill_2x2(res)
    _plot_importance(importance)
    print(f"  figures -> {FIG_DIR}")


def _test_union(folds):
    mask = folds[0]["test"].copy()
    for fo in folds[1:]:
        mask = mask | fo["test"]
    return mask


def _plot_forecast(ndvi, folds, oos, py=6, px=6):
    test_all = _test_union(folds)
    t = ndvi["time"].sel(time=test_all).values
    clim = climatology_forecast(ndvi, seasonal_climatology(ndvi))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, ndvi.sel(time=test_all).isel(y=py, x=px), "k-o", lw=2, ms=3, label="actual")
    ax.plot(t, persistence(ndvi).sel(time=test_all).isel(y=py, x=px), color="#d95f02", label="persistence")
    ax.plot(t, clim.sel(time=test_all).isel(y=py, x=px), color="#7570b3", label="climatology")
    ax.plot(t, oos.sel(time=test_all).isel(y=py, x=px), color="#1b7837", lw=2, label="GBT (out-of-sample)")
    ax.set_title("GBT vs baselines — stitched walk-forward test months (pixel 6,6)")
    ax.set_ylabel("NDVI"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "gbt_forecast_vs_baselines.png", dpi=130); plt.close(fig)


def _plot_skill_2x2(res):
    tab = summarise_2x2(res, "gbt", "skill_vs_persistence").values
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(tab, cmap="RdBu", vmin=-0.4, vmax=0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["seen locations", "unseen locations"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["future time\n(holdout)", "seen time"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab[i, j]:+.2f}", ha="center", va="center", fontweight="bold")
    ax.set_title("GBT skill vs persistence\n(headline cell = top-left)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / "gbt_skill_2x2.png", dpi=130); plt.close(fig)


def _plot_importance(importance):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    importance.sort_values().plot.barh(ax=ax, color="#1b7837")
    ax.set_title("GBT feature importance (mean over folds)")
    ax.set_xlabel("LightGBM gain-split importance")
    fig.tight_layout(); fig.savefig(FIG_DIR / "gbt_feature_importance.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
