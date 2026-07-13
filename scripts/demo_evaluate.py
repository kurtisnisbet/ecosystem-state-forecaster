"""Validate `evaluate` on the synthetic cube and render the split/skill figures.

Builds the walk-forward temporal folds and buffered spatial blocks, scores the
persistence and climatology baselines across the space x time 2x2, checks the
key invariants (embargo gap, expanding window, disjoint spatial split), and
writes figures to docs/figures/ for the README.

Run:  python scripts/demo_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_baselines import make_synthetic_cube  # reuse the same synthetic cube
from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.features import seasonal_climatology
from ecoforecast.evaluate import (
    evaluate_folds,
    spatial_blocks,
    summarise_2x2,
    walk_forward_splits,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ndvi = make_synthetic_cube()

    folds = walk_forward_splits(ndvi["time"], block_months=3, n_test_folds=4, embargo_months=3)
    strain, stest, _ = spatial_blocks(ndvi, block_size=6, n_test_blocks=3, buffer=1, seed=1)

    # --- invariants ---------------------------------------------------------
    for fo in folds:
        ti = np.where(fo["train"].values)[0]
        te = np.where(fo["test"].values)[0]
        assert te.min() - ti.max() - 1 == 3, "embargo gap != 3 months"
    train_sizes = [int(fo["train"].sum()) for fo in folds]
    assert train_sizes == sorted(train_sizes) and len(set(train_sizes)) == len(train_sizes), "not expanding"
    assert not bool((strain & stest).any()), "spatial train/test overlap"

    # Baselines as the predictions to score (a real model slots in here later).
    preds = {
        "persistence": persistence(ndvi),
        "climatology": climatology_forecast(ndvi, seasonal_climatology(ndvi)),
    }
    res = evaluate_folds(ndvi, preds, folds, strain, stest)

    print("Evaluation validation")
    print(f"  folds: {[f['label'] for f in folds]}")
    print(f"  train sizes (months): {train_sizes}")
    print(f"  spatial px  test: {int(stest.sum())}  train: {int(strain.sum())}"
          f"  buffer: {int((~strain & ~stest).sum())}")
    print(f"  result rows: {len(res)} (= {len(folds)} folds x 4 cells x {len(preds)} predictors)")
    print("  2x2 mean skill_vs_persistence (climatology):")
    print(summarise_2x2(res, "climatology").round(3).to_string().replace("\n", "\n    "))

    _plot_folds(ndvi, folds)
    _plot_spatial_blocks(strain, stest)
    _plot_skill_2x2(res, "climatology")
    print(f"  figures -> {FIG_DIR}")


def _plot_folds(ndvi, folds):
    t = ndvi["time"].values
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, fo in enumerate(folds):
        ti = np.where(fo["train"].values)[0]
        te = np.where(fo["test"].values)[0]
        ax.barh(i, t[ti.max()] - t[0], left=t[0], color="#1b7837", height=0.6, label="train" if i == 0 else "")
        ax.barh(i, t[te.min()] - t[ti.max()], left=t[ti.max()], color="#bbbbbb", height=0.6, label="embargo" if i == 0 else "")
        ax.barh(i, t[te.max()] - t[te.min()] + np.timedelta64(30, "D"), left=t[te.min()], color="#d95f02", height=0.6, label="test" if i == 0 else "")
    ax.set_yticks(range(len(folds))); ax.set_yticklabels([f"fold {i}" for i in range(len(folds))])
    ax.set_title("Expanding-window walk-forward folds (train / embargo / test)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "walk_forward_folds.png", dpi=130); plt.close(fig)


def _plot_spatial_blocks(strain, stest):
    role = np.where(stest.values, 2, np.where(strain.values, 0, 1))  # 0 train, 1 buffer, 2 test
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(role, cmap=ListedColormap(["#1b7837", "#dddddd", "#d95f02"]), vmin=0, vmax=2)
    ax.set_title("Spatial blocks: train / buffer / held-out test")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
    cb.ax.set_yticklabels(["train", "buffer", "test"])
    fig.tight_layout(); fig.savefig(FIG_DIR / "spatial_blocks.png", dpi=130); plt.close(fig)


def _plot_skill_2x2(res, predictor):
    tab = summarise_2x2(res, predictor).values
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(tab, cmap="RdBu", vmin=-0.6, vmax=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["seen locations", "unseen locations"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["future time\n(holdout)", "seen time"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab[i, j]:+.2f}", ha="center", va="center", fontweight="bold")
    ax.set_title(f"Skill vs persistence — {predictor}\n(headline cell = top-left)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / "skill_2x2.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
