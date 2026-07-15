"""Streamlit demo for the Ecosystem State Forecaster.

Pick a biome and a pixel and see next-month NDVI forecast against the persistence
and climatology baselines, with a conformal prediction band. Reads the cached
cubes built by scripts/build_cube.py (data/cube_*.nc). Baselines and the
gradient-boosted-trees forecast are computed on load and cached; the deep-learning
models live in the full pipeline because they need a GPU.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.evaluate import rmse, spatial_blocks, walk_forward_splits
from ecoforecast.features import seasonal_climatology
from ecoforecast.models.gbt import walk_forward_gbt
from ecoforecast.uncertainty import conformal_intervals

DATA = ROOT / "data"


def list_biomes():
    return sorted(p.stem.replace("cube_", "") for p in DATA.glob("cube_*.nc"))


def test_union(folds):
    mask = folds[0]["test"].copy()
    for fo in folds[1:]:
        mask = mask | fo["test"]
    return mask


@st.cache_data(show_spinner="Training the model for this biome...")
def forecast(biome: str, alpha: float):
    ndvi = xr.open_dataarray(DATA / f"cube_{biome}.nc")
    folds = walk_forward_splits(ndvi["time"], block_months=3, n_test_folds=4, embargo_months=3)
    strain, _stest, _ = spatial_blocks(ndvi, block_size=20, n_test_blocks=3, buffer=2, seed=1)

    _, oos, _ = walk_forward_gbt(ndvi, folds, strain, _stest)
    _table, lower, upper = conformal_intervals(ndvi, oos, folds, space_mask=strain, alpha=alpha)

    test = test_union(folds)
    pers = persistence(ndvi)
    clim = climatology_forecast(ndvi, seasonal_climatology(ndvi))

    def cell_rmse(pred):
        o = ndvi.where(strain).sel(time=test)
        p = pred.where(strain).sel(time=test)
        valid = o.notnull() & p.notnull()
        return rmse(p.where(valid), o.where(valid))

    metrics = {"persistence": cell_rmse(pers), "climatology": cell_rmse(clim), "GBT": cell_rmse(oos)}
    return ndvi, oos, lower, upper, folds, metrics


def _best_pixel(ndvi):
    valid = ndvi.notnull().sum("time")
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(valid.values)), valid.shape))


def main():
    st.set_page_config(page_title="Ecosystem State Forecaster", layout="wide")
    st.title("Ecosystem State Forecaster")
    st.caption("Next-month NDVI forecast with honest baselines and a conformal uncertainty band.")

    biomes = list_biomes()
    if not biomes:
        st.warning("No cubes found in `data/`. Run `python scripts/build_cube.py` first.")
        return

    biome = st.sidebar.selectbox("Biome", biomes)
    confidence = st.sidebar.slider("Interval confidence", 0.50, 0.99, 0.90, 0.01)
    ndvi, oos, lower, upper, folds, metrics = forecast(biome, 1 - confidence)

    cols = st.sidebar.columns(2)
    py = cols[0].number_input("pixel y", 0, ndvi.sizes["y"] - 1, _best_pixel(ndvi)[0])
    px = cols[1].number_input("pixel x", 0, ndvi.sizes["x"] - 1, _best_pixel(ndvi)[1])
    months = [str(np.datetime_as_string(t, unit="M")) for t in ndvi["time"].values]
    month = st.sidebar.select_slider("Map month", months, value=months[-1])

    m1, m2, m3 = st.columns(3)
    m1.metric("Persistence RMSE", f"{metrics['persistence']:.3f}")
    m2.metric("Climatology RMSE", f"{metrics['climatology']:.3f}")
    skill = 1 - metrics["GBT"] / metrics["climatology"] if metrics["climatology"] else float("nan")
    m3.metric("GBT RMSE", f"{metrics['GBT']:.3f}", f"{skill:+.0%} vs climatology")

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader(f"NDVI, {month}")
        frame = ndvi.sel(time=month).squeeze()
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(frame, cmap="YlGn", vmin=0, vmax=0.9)
        ax.plot(px, py, "o", ms=10, mfc="none", mec="#d62728", mew=2)
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    with right:
        st.subheader(f"Forecast at pixel ({py}, {px})")
        test = test_union(folds)
        t = ndvi["time"].sel(time=test).values
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.fill_between(t, lower.sel(time=test).isel(y=py, x=px), upper.sel(time=test).isel(y=py, x=px),
                        color="#1b7837", alpha=0.2, label=f"{confidence:.0%} interval")
        ax.plot(t, persistence(ndvi).sel(time=test).isel(y=py, x=px), color="#d95f02", lw=1, label="persistence")
        ax.plot(t, climatology_forecast(ndvi, seasonal_climatology(ndvi)).sel(time=test).isel(y=py, x=px),
                color="#7570b3", lw=1, label="climatology")
        ax.plot(t, oos.sel(time=test).isel(y=py, x=px), color="#1b7837", lw=2, label="GBT")
        ax.plot(t, ndvi.sel(time=test).isel(y=py, x=px), "k-o", lw=1.5, ms=3, label="actual")
        ax.set_ylabel("NDVI"); ax.legend(fontsize=8, ncol=2)
        st.pyplot(fig)


main()
