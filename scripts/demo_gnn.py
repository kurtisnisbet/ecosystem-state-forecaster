"""Train the GraphCast-style GNN on the synthetic cube and report its skill.

Needs PyTorch. Scaled back for local dev: small hidden size, three message-passing
rounds, few epochs, two walk-forward folds. Writes figures to docs/figures/.

Run:  python scripts/demo_gnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_baselines import make_synthetic_cube
from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.features import seasonal_climatology
from ecoforecast.evaluate import spatial_blocks, summarise_2x2, walk_forward_splits
from ecoforecast.models.gnn import walk_forward_gnn

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ndvi = make_synthetic_cube()
    folds = walk_forward_splits(ndvi["time"], block_months=3, n_test_folds=2, embargo_months=3)
    strain, stest, _ = spatial_blocks(ndvi, block_size=6, n_test_blocks=3, buffer=1, seed=1)

    res, oos, histories = walk_forward_gnn(ndvi, folds, strain, stest, hidden=32, rounds=3, epochs=30)

    headline = res[(res.time == "future") & (res.space == "seen")].groupby("predictor")["rmse"].mean()
    print("GNN walk-forward validation")
    print("  headline (future/seen) RMSE:")
    print("    " + headline.round(4).to_string().replace("\n", "\n    "))
    print("  GNN skill vs persistence (2x2 mean):")
    print("    " + summarise_2x2(res, "gnn", "skill_vs_persistence").round(3).to_string().replace("\n", "\n    "))

    _plot_loss(histories)
    _plot_forecast(ndvi, folds, oos)
    _plot_skill_2x2(res)
    print(f"  figures -> {FIG_DIR}")


def _test_union(folds):
    mask = folds[0]["test"].copy()
    for fo in folds[1:]:
        mask = mask | fo["test"]
    return mask


def _plot_loss(histories):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, h in enumerate(histories):
        ax.plot(range(1, len(h) + 1), h, label=f"fold {i}")
    ax.set_title("GNN training loss (masked MSE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "gnn_training_loss.png", dpi=130); plt.close(fig)


def _plot_forecast(ndvi, folds, oos, py=6, px=6):
    test_all = _test_union(folds)
    t = ndvi["time"].sel(time=test_all).values
    clim = climatology_forecast(ndvi, seasonal_climatology(ndvi))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, ndvi.sel(time=test_all).isel(y=py, x=px), "k-o", lw=2, ms=3, label="actual")
    ax.plot(t, persistence(ndvi).sel(time=test_all).isel(y=py, x=px), color="#d95f02", label="persistence")
    ax.plot(t, clim.sel(time=test_all).isel(y=py, x=px), color="#7570b3", label="climatology")
    ax.plot(t, oos.sel(time=test_all).isel(y=py, x=px), color="#1b7837", lw=2, label="GNN (out-of-sample)")
    ax.set_title("GNN vs baselines — walk-forward test months (pixel 6,6)")
    ax.set_ylabel("NDVI"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "gnn_forecast_vs_baselines.png", dpi=130); plt.close(fig)


def _plot_skill_2x2(res):
    tab = summarise_2x2(res, "gnn", "skill_vs_persistence").values
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(tab, cmap="RdBu", vmin=-0.4, vmax=0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["seen locations", "unseen locations"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["future time\n(holdout)", "seen time"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab[i, j]:+.2f}", ha="center", va="center", fontweight="bold")
    ax.set_title("GNN skill vs persistence\n(headline cell = top-left)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / "gnn_skill_2x2.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
