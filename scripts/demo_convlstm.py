"""Train the scaled-back ConvLSTM on the synthetic cube and render its figures.

Needs PyTorch (`pip install torch`). Scaled back for CPU per the brief: small
hidden size, short sequences, few epochs, and 2 walk-forward folds — enough to
prove the pipeline end to end before scaling to GPU. Writes to docs/figures/:
architecture diagram, training-loss curves, forecast vs baselines, skill 2x2.

Run:  python scripts/demo_convlstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_baselines import make_synthetic_cube  # reuse the same synthetic cube
from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.features import seasonal_climatology
from ecoforecast.evaluate import spatial_blocks, summarise_2x2, walk_forward_splits
from ecoforecast.models.convlstm import walk_forward_convlstm

FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


def _static_terrain(ndvi):
    ny, nx = ndvi.sizes["y"], ndvi.sizes["x"]
    yy, xx = np.mgrid[0:ny, 0:nx] / (ny - 1)
    elev = xr.DataArray((0.5 * yy + 0.3 * xx).astype("float32"),
                        dims=("y", "x"), coords={"y": ndvi["y"], "x": ndvi["x"]})
    return xr.Dataset({"elevation": elev})


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _draw_architecture()

    ndvi = make_synthetic_cube()
    static = _static_terrain(ndvi)
    folds = walk_forward_splits(ndvi["time"], block_months=3, n_test_folds=2, embargo_months=3)
    strain, stest, _ = spatial_blocks(ndvi, block_size=6, n_test_blocks=3, buffer=1, seed=1)

    res, oos, histories = walk_forward_convlstm(
        ndvi, folds, strain, stest, static=static, seq_len=6, hidden=16, epochs=150,
    )

    headline = res[(res.time == "future") & (res.space == "seen")].groupby("predictor")["rmse"].mean()
    print("ConvLSTM walk-forward validation")
    print("  headline (future/seen) RMSE:")
    print("    " + headline.round(4).to_string().replace("\n", "\n    "))
    print("  ConvLSTM skill vs persistence (2x2 mean):")
    print("    " + summarise_2x2(res, "convlstm", "skill_vs_persistence").round(3).to_string().replace("\n", "\n    "))

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
    ax.set_title("ConvLSTM training loss (masked MSE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "convlstm_training_loss.png", dpi=130); plt.close(fig)


def _plot_forecast(ndvi, folds, oos, py=6, px=6):
    test_all = _test_union(folds)
    t = ndvi["time"].sel(time=test_all).values
    clim = climatology_forecast(ndvi, seasonal_climatology(ndvi))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, ndvi.sel(time=test_all).isel(y=py, x=px), "k-o", lw=2, ms=3, label="actual")
    ax.plot(t, persistence(ndvi).sel(time=test_all).isel(y=py, x=px), color="#d95f02", label="persistence")
    ax.plot(t, clim.sel(time=test_all).isel(y=py, x=px), color="#7570b3", label="climatology")
    ax.plot(t, oos.sel(time=test_all).isel(y=py, x=px), color="#1b7837", lw=2, label="ConvLSTM (out-of-sample)")
    ax.set_title("ConvLSTM vs baselines — walk-forward test months (pixel 6,6)")
    ax.set_ylabel("NDVI"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "convlstm_forecast_vs_baselines.png", dpi=130); plt.close(fig)


def _plot_skill_2x2(res):
    tab = summarise_2x2(res, "convlstm", "skill_vs_persistence").values
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(tab, cmap="RdBu", vmin=-0.4, vmax=0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["seen locations", "unseen locations"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["future time\n(holdout)", "seen time"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{tab[i, j]:+.2f}", ha="center", va="center", fontweight="bold")
    ax.set_title("ConvLSTM skill vs persistence\n(headline cell = top-left)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(FIG_DIR / "convlstm_skill_2x2.png", dpi=130); plt.close(fig)


def _draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    def box(x, y, w, h, label, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.2))

    for k in range(3):
        box(0.2 + k * 0.55, 1.4 + k * 0.18, 1.5, 0.9, "", "#e8f3ec")
    ax.text(1.15, 1.15, "input sequence\n(t-5 … t)\n5 channels", ha="center", va="top", fontsize=8)
    ax.text(1.15, 3.0, "NDVI · mask ·\nmonth sin/cos · terrain", ha="center", va="bottom", fontsize=7.5, color="#555")
    arrow(2.3, 2.0, 3.2, 2.0)
    box(3.2, 1.4, 2.1, 1.2, "ConvLSTM cell\n(recurrent over T)", "#cfe8d8")
    ax.add_patch(FancyArrowPatch((4.25, 2.6), (4.25, 3.0), arrowstyle="-|>", mutation_scale=10, color="#999"))
    ax.add_patch(FancyArrowPatch((4.6, 3.0), (4.6, 2.6), arrowstyle="-|>", mutation_scale=10, color="#999"))
    ax.text(4.4, 3.1, "hidden / cell state", ha="center", fontsize=7, color="#999")
    arrow(5.3, 2.0, 6.1, 2.0)
    box(6.1, 1.5, 1.7, 1.0, "1×1 Conv\nhead → Δ", "#cfe8d8")
    arrow(7.8, 2.0, 8.4, 2.0)
    box(8.4, 1.5, 1.4, 1.0, "next-month\nNDVI", "#f2d9b8")
    ax.add_patch(FancyArrowPatch((1.9, 1.5), (9.0, 1.35), connectionstyle="arc3,rad=-0.25",
                                 arrowstyle="-|>", mutation_scale=12, color="#d95f02", lw=1.2, ls="--"))
    ax.text(5.4, 0.75, "residual skip: prediction = last observed NDVI + Δ", ha="center", fontsize=7.5, color="#d95f02")
    ax.set_title("NDVIConvLSTM — one-step-ahead architecture", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_DIR / "convlstm_architecture.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
