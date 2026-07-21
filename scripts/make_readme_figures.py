"""Figures that explain the project to a reader who has not seen it before.

These are the orientation figures for the README (where the sites are, what they
look like, why the baselines are hard to beat) rather than the result figures,
which run_pipeline.py writes. Everything here is drawn from the cached cubes, so
it needs data/ to be populated but no network and no models.

Run:  python scripts/make_readme_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA = ROOT / "data"

SITES = {
    "daintree": ("Daintree", "Tropical rainforest", "#1b7837"),
    "sunshine_coast_hinterland": ("Sunshine Coast", "Subtropical", "#66bd63"),
    "alice_springs": ("Alice Springs", "Arid desert", "#d8801f"),
    "kosciuszko": ("Kosciuszko", "Alpine", "#3182bd"),
}
LABEL_POS = {                      # where to put each label on the locator map
    "daintree": (134.5, -14.5),
    "sunshine_coast_hinterland": (157.5, -25.0),
    "alice_springs": (121.0, -23.0),
    "kosciuszko": (155.5, -38.5),
}
PERSIST, CLIM = "#d95f02", "#7570b3"


def cube(biome, profile="landsat", res=100):
    path = DATA / f"cube_{profile}_{res}m_{biome}.nc"
    return xr.open_dataarray(path) if path.exists() else None


def _bboxes():
    cfg = yaml.safe_load((ROOT / "ecoforecast" / "config.yaml").read_text())
    return {b: cfg["biomes"][b]["bbox"] for b in SITES}


# ── 1. where the sites are ──────────────────────────────────────────────────
def locator_map():
    rings = json.loads((ROOT / "docs" / "australia_outline.json").read_text())["rings"]
    boxes = _bboxes()

    fig, ax = plt.subplots(figsize=(9, 7))
    for ring in rings.values():
        xy = np.array(ring)
        ax.fill(xy[:, 0], xy[:, 1], facecolor="#eceae4", edgecolor="#9a9a93", lw=1.0, zorder=1)

    for biome, (name, climate, colour) in SITES.items():
        x0, y0, x1, y1 = boxes[biome]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lx, ly = LABEL_POS[biome]
        ax.annotate(
            f"{name}\n{climate}", xy=(cx, cy), xytext=(lx, ly),
            fontsize=10, ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=colour, lw=1.4),
            arrowprops=dict(arrowstyle="-", color="#666", lw=1.0,
                            connectionstyle="arc3,rad=0.12"),
        )
        ax.scatter([cx], [cy], s=110, c="#d62728", edgecolors="white", linewidths=1.6, zorder=5)

    ax.set_xlim(112, 168)
    ax.set_ylim(-46, -8)
    ax.set_aspect(1 / np.cos(np.radians(25)))
    ax.set_title("Four study areas, chosen to span Australia's climate range", fontsize=13, pad=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG / "study_areas_map.png", dpi=140)
    plt.close(fig)


# ── 2. what the sites look like ─────────────────────────────────────────────
def ndvi_panels():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (biome, (name, climate, _)) in zip(axes, SITES.items()):
        d = cube(biome)
        if d is None:
            continue
        mean = d.mean("time", skipna=True)
        im = ax.imshow(mean.values, cmap="YlGn", vmin=0.05, vmax=0.85, origin="upper")
        ax.set_title(f"{name}\n{climate}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.suptitle("Average greenness at each site, 1988 to 2026 (green is more vegetation)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "study_areas_ndvi.png", dpi=140)
    plt.close(fig)


# ── 3. how seasonal each site is ────────────────────────────────────────────
def seasonality():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    months = np.arange(1, 13)
    for biome, (name, climate, colour) in SITES.items():
        d = cube(biome)
        if d is None:
            continue
        clim = d.mean(dim=("y", "x"), skipna=True).groupby("time.month").mean("time")
        vals = clim.reindex(month=months).values
        axes[0].plot(months, vals, "-o", ms=4, color=colour, label=f"{name} ({climate.lower()})")
        axes[1].plot(months, vals - np.nanmean(vals), "-o", ms=4, color=colour, label=name)

    axes[0].set_title("Greenness through the year", fontsize=11)
    axes[0].set_ylabel("NDVI")
    axes[1].set_title("Same curves, centred, so the swing is comparable", fontsize=11)
    axes[1].set_ylabel("NDVI relative to the site's own average")
    axes[1].axhline(0, color="#333", lw=1)
    for ax in axes:
        ax.set_xticks(months)
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_xlabel("month")
        ax.legend(fontsize=8)
    fig.suptitle("Alice Springs barely follows the calendar. That is what makes it the hard case.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "study_areas_seasonality.png", dpi=140)
    plt.close(fig)


# ── 4. why these two baselines ──────────────────────────────────────────────
def _best_pixel(d):
    valid = d.notnull().sum("time")
    return np.unravel_index(int(np.argmax(valid.values)), valid.shape)


def why_baselines(years=8):
    """Site averages, not single pixels: one pixel is too noisy to read the point off."""
    pairs = [("kosciuszko", "Kosciuszko (alpine): greenness follows the seasons, so the calendar guesses well"),
             ("alice_springs", "Alice Springs (arid): greenness follows rainfall, so the calendar guesses badly")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8))
    for ax, (biome, title) in zip(axes, pairs):
        d = cube(biome)
        if d is None:
            continue
        series = d.mean(dim=("y", "x"), skipna=True)
        clim = series.groupby("time.month").mean("time")
        series = series.isel(time=slice(-years * 12, None))
        t = series["time"].values
        months = series["time"].dt.month.values
        ax.plot(t, series.values, "-o", color="black", lw=1.8, ms=3, label="what actually happened")
        ax.plot(t, np.roll(series.values, 1), color=PERSIST, lw=1.5, ls="--",
                label="persistence: next month = this month")
        ax.plot(t, clim.sel(month=months).values, color=CLIM, lw=1.5, ls="--",
                label="climatology: next month = the usual value for that calendar month")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("NDVI (site average)")
    axes[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.suptitle("The two baselines every model has to beat", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "why_baselines.png", dpi=140)
    plt.close(fig)


# ── 5. per-site appendix profiles ───────────────────────────────────────────
def site_profiles():
    boxes = _bboxes()
    for biome, (name, climate, colour) in SITES.items():
        d = cube(biome)
        if d is None:
            continue
        fig = plt.figure(figsize=(12, 3.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.6])

        ax0 = fig.add_subplot(gs[0])
        im = ax0.imshow(d.mean("time", skipna=True).values, cmap="YlGn", vmin=0.05, vmax=0.85)
        ax0.set_title("average greenness", fontsize=10)
        ax0.set_xticks([]); ax0.set_yticks([])
        fig.colorbar(im, ax=ax0, fraction=0.046)

        ax1 = fig.add_subplot(gs[1])
        spatial = d.mean(dim=("y", "x"), skipna=True)
        clim = spatial.groupby("time.month").mean("time")
        ax1.plot(np.arange(1, 13), clim.reindex(month=np.arange(1, 13)).values, "-o",
                 color=colour, ms=4)
        ax1.set_title("through the year", fontsize=10)
        ax1.set_xticks(np.arange(1, 13))
        ax1.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=7)
        ax1.set_ylabel("NDVI")

        ax2 = fig.add_subplot(gs[2])
        ax2.plot(spatial["time"].values, spatial.values, color=colour, lw=0.9)
        ax2.set_title("the full 40-year record", fontsize=10)
        ax2.set_ylabel("NDVI")

        x0, y0, x1, y1 = boxes[biome]
        fig.suptitle(f"{name}, {climate.lower()}   |   {x0}, {y0} to {x1}, {y1}   |   "
                     f"{d.sizes['y']} by {d.sizes['x']} pixels at 100 m", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / f"site_{biome}.png", dpi=140)
        plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    locator_map(); print("  study_areas_map.png")
    ndvi_panels(); print("  study_areas_ndvi.png")
    seasonality(); print("  study_areas_seasonality.png")
    why_baselines(); print("  why_baselines.png")
    site_profiles(); print("  site_*.png")


if __name__ == "__main__":
    main()
