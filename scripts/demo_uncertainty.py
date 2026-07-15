"""Conformal prediction intervals on the synthetic cube.

Uses the gradient-boosted-trees out-of-sample forecasts (no torch needed), wraps
them in rolling split-conformal 90% intervals, checks empirical coverage against
the 0.90 target, and writes an interval-fan figure and a coverage-by-fold figure.

Run:  python scripts/demo_uncertainty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_baselines import make_synthetic_cube
from ecoforecast.evaluate import spatial_blocks, walk_forward_splits
from ecoforecast.models.gbt import walk_forward_gbt
from ecoforecast.uncertainty import conformal_intervals

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ndvi = make_synthetic_cube()
    folds = walk_forward_splits(ndvi["time"], block_months=3, n_test_folds=4, embargo_months=3)
    strain, stest, _ = spatial_blocks(ndvi, block_size=6, n_test_blocks=3, buffer=1, seed=1)

    _, oos, _ = walk_forward_gbt(ndvi, folds, strain, stest)
    table, lower, upper = conformal_intervals(ndvi, oos, folds, space_mask=strain, alpha=0.1)

    print("Conformal 90% intervals by fold:")
    print("  " + table.round(3).to_string(index=False).replace("\n", "\n  "))
    print(f"  mean coverage: {table['coverage'].mean():.3f}  (target 0.90)")

    _plot_fan(ndvi, oos, lower, upper, folds)
    _plot_coverage(table)
    print(f"  figures -> {FIG_DIR}")


def _calibrated_mask(folds):
    mask = folds[1]["test"].copy()
    for fo in folds[2:]:
        mask = mask | fo["test"]
    return mask


def _plot_fan(ndvi, oos, lower, upper, folds, py=6, px=6):
    m = _calibrated_mask(folds)
    t = ndvi["time"].sel(time=m).values
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(t, lower.sel(time=m).isel(y=py, x=px), upper.sel(time=m).isel(y=py, x=px),
                    color="#1b7837", alpha=0.2, label="90% interval")
    ax.plot(t, oos.sel(time=m).isel(y=py, x=px), color="#1b7837", lw=2, label="GBT forecast")
    ax.plot(t, ndvi.sel(time=m).isel(y=py, x=px), "k-o", lw=1.5, ms=3, label="actual")
    ax.set_title("Conformal 90% prediction interval (pixel 6,6)")
    ax.set_ylabel("NDVI"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "uncertainty_interval_fan.png", dpi=130); plt.close(fig)


def _plot_coverage(table):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(table["label"], table["coverage"], color="#1b7837")
    ax.axhline(table["target"].iloc[0], color="#b2182b", lw=1.5, ls="--", label="target 0.90")
    ax.set_ylim(0, 1); ax.set_ylabel("empirical coverage")
    ax.set_title("Conformal coverage by fold"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(FIG_DIR / "uncertainty_coverage.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
