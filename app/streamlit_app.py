"""Interactive demo for the Ecosystem State Forecaster.

Everything shown here is precomputed by scripts/run_pipeline.py into
docs/app_data/*.nc, so this app imports no model libraries and trains nothing on
load. It reads one small NetCDF per biome per satellite record, which keeps it
inside a free hosting tier's memory limit.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "app_data"

RECORDS = {
    "Sentinel-2, 2015 to 2026 (11 years)": "sentinel2_100m",
    "Landsat, 1988 to 2026 (40 years)": "landsat_100m",
}
BIOME_LABELS = {
    "sunshine_coast_hinterland": "Sunshine Coast (subtropical)",
    "daintree": "Daintree (tropical rainforest)",
    "alice_springs": "Alice Springs (arid)",
    "kosciuszko": "Kosciuszko (alpine)",
}
MODEL_COLORS = {"gbt": "#1b7837", "convlstm": "#e7298a", "gnn": "#3182bd", "ensemble": "#222222"}
BASELINE_COLORS = {"persistence": "#d95f02", "climatology": "#7570b3"}


# ── loading ─────────────────────────────────────────────────────────────────
def biomes_for(tag: str) -> list[str]:
    found = [p.stem[len(tag) + 1 :] for p in sorted(DATA_DIR.glob(f"{tag}_*.nc"))]
    return [b for b in BIOME_LABELS if b in found] + [b for b in found if b not in BIOME_LABELS]


@st.cache_data(show_spinner=False)
def load_biome(tag: str, biome: str) -> xr.Dataset:
    return xr.open_dataset(DATA_DIR / f"{tag}_{biome}.nc").load()


@st.cache_data(show_spinner=False)
def load_headlines(tag: str) -> dict[str, dict[str, float]]:
    out = {}
    for biome in biomes_for(tag):
        ds = load_biome(tag, biome)
        if "headline_rmse" in ds:
            out[biome] = {str(p): float(v) for p, v in zip(ds["predictor"].values, ds["headline_rmse"].values)}
    return out


def models_in(ds: xr.Dataset) -> list[str]:
    order = ["gbt", "convlstm", "gnn", "ensemble"]
    present = [str(m) for m in ds["model"].values]
    return [m for m in order if m in present] + [m for m in present if m not in order]


def climatology_for(ds: xr.Dataset) -> xr.DataArray:
    """Broadcast the month-of-year climatology onto the display time axis."""
    months = ds["time"].dt.month
    return ds["climatology"].sel(month=months).drop_vars("month")


# ── figures ─────────────────────────────────────────────────────────────────
def map_row(actual, pred, model, title, py, px):
    err = np.abs(actual - pred)
    lo, hi = np.nanpercentile(actual, [2, 98]) if np.isfinite(actual).any() else (0.0, 1.0)
    top = np.nanpercentile(err, 98) if np.isfinite(err).any() else 1.0
    panels = [
        (actual, "actual NDVI", dict(cmap="YlGn", vmin=lo, vmax=hi)),
        (pred, f"{model} forecast", dict(cmap="YlGn", vmin=lo, vmax=hi)),
        (err, "absolute error", dict(cmap="magma_r", vmin=0, vmax=top)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (arr, sub, kw) in zip(axes, panels):
        im = ax.imshow(arr, origin="upper", **kw)
        ax.plot(px, py, "o", ms=9, mfc="none", mec="#d62728", mew=2)
        ax.set_title(sub, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def series_fig(ds, chosen, primary, py, px, level, test_mask):
    t = ds["time"].values
    fig, ax = plt.subplots(figsize=(11, 3.8))

    band = float(ds["conformal_q"].sel(model=primary).interp(level=level))
    lead = ds[f"pred_{primary}"].isel(y=py, x=px).values
    ax.fill_between(t, lead - band, lead + band, color=MODEL_COLORS.get(primary, "#888888"),
                    alpha=0.18, label=f"{primary}, {int(level * 100)}% interval")
    ax.plot(t, ds["ndvi"].isel(y=py, x=px).values, "k-o", lw=2, ms=3.5, label="actual", zorder=5)
    ax.plot(t, ds["ndvi"].shift(time=1).isel(y=py, x=px).values,
            color=BASELINE_COLORS["persistence"], lw=1.2, ls="--", label="persistence")
    ax.plot(t, climatology_for(ds).isel(y=py, x=px).values,
            color=BASELINE_COLORS["climatology"], lw=1.2, ls="--", label="climatology")
    for m in chosen:
        ax.plot(t, ds[f"pred_{m}"].isel(y=py, x=px).values,
                color=MODEL_COLORS.get(m, "#888888"), lw=2, label=m)
    if test_mask.any():
        ax.axvline(t[test_mask][0], color="#999", lw=1)
    ax.set_ylabel("NDVI")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def skill_fig(headlines, biomes, models, highlight):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    width = 0.8 / max(len(models), 1)
    xs = np.arange(len(biomes))
    for i, m in enumerate(models):
        vals = []
        for b in biomes:
            row = headlines.get(b, {})
            clim, mv = row.get("climatology"), row.get(m)
            vals.append(1 - mv / clim if clim and mv else np.nan)
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width, label=m,
               color=MODEL_COLORS.get(m, "#888888"), alpha=1.0 if m == highlight else 0.55)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([BIOME_LABELS.get(b, b).split(" (")[0] for b in biomes], fontsize=9)
    ax.set_ylabel("skill vs climatology")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    return fig


# ── app ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ecosystem State Forecaster", layout="wide")
st.title("Ecosystem State Forecaster")
st.caption(
    "Forecasting next month's vegetation greenness (NDVI) across Australian biomes, scored "
    "against persistence and seasonal climatology on splits that do not leak in space or time. "
    "Every forecast below is out of sample."
)

if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.nc")):
    st.error(
        "No demo data found in `docs/app_data`. Build a cube and run the pipeline first:\n\n"
        "```\npython scripts/build_cube.py\npython scripts/run_pipeline.py\n```"
    )
    st.stop()

available = {label: tag for label, tag in RECORDS.items() if biomes_for(tag)}
if not available:
    st.error("Demo files are present but none match a known record tag.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    record_label = st.radio("Satellite record", list(available),
                            help="The headline conclusion flips between these two.")
    tag = available[record_label]
    biomes = biomes_for(tag)
    biome = st.selectbox("Biome", biomes, format_func=lambda b: BIOME_LABELS.get(b, b))
    ds = load_biome(tag, biome)
    all_models = models_in(ds)
    chosen = st.multiselect("Models", all_models, default=all_models)
    primary = st.selectbox("Model shown on the map", all_models,
                           index=all_models.index(chosen[0]) if chosen else 0)
    level = st.slider("Confidence level (%)", 50, 99, 90, 1,
                      help="Redrawn from a stored table of conformal quantiles, so it updates instantly.") / 100

test_mask = ds[f"pred_{primary}"].notnull().any(dim=("y", "x")).values
test_times = ds["time"].values[test_mask]
labels = [str(np.datetime_as_string(t, unit="M")) for t in test_times]

with st.sidebar:
    month_label = st.select_slider("Forecast month", labels, value=labels[-1])
    st.markdown("**Pixel**")
    py = st.slider("row", 0, ds.sizes["y"] - 1, ds.sizes["y"] // 2)
    px = st.slider("column", 0, ds.sizes["x"] - 1, ds.sizes["x"] // 2)
    st.caption("Rows and columns index the coarsened grid. The red circle marks the pixel.")

when = test_times[labels.index(month_label)]
headlines = load_headlines(tag)

if headlines:
    st.subheader("Do the models beat climatology?")
    st.pyplot(skill_fig(headlines, biomes, [m for m in all_models if m in chosen] or all_models, primary))
    st.caption(
        "Above zero means the model beats the seasonal climatology baseline. "
        + ("On the 40-year Landsat record the models win in every biome. They get about 3.5 times more "
           "training data, and climatology itself weakens, because over four decades a typical month has "
           "to absorb far more year-to-year variability. "
           if tag.startswith("landsat") else
           "On the 11-year Sentinel-2 record climatology is very hard to beat, except in arid Alice "
           "Springs, where the seasonal cycle is weak because desert vegetation responds to episodic rain "
           "rather than to the calendar. ")
        + "Switch the satellite record in the sidebar to watch the conclusion change."
    )

st.subheader(f"{BIOME_LABELS.get(biome, biome)}: forecast for {month_label}")
actual = ds["ndvi"].sel(time=when).values
pred = ds[f"pred_{primary}"].sel(time=when).values
st.pyplot(map_row(actual, pred, primary, f"{month_label}, {record_label.split(',')[0]}", py, px))

resid = np.abs(actual - pred)
c1, c2, c3 = st.columns(3)
c1.metric("RMSE this month", f"{np.sqrt(np.nanmean(resid ** 2)):.4f}")
c2.metric("Median absolute error", f"{np.nanmedian(resid):.4f}")
head = headlines.get(biome, {})
if head.get("climatology") and head.get(primary):
    c3.metric(f"Headline skill vs climatology ({primary})",
              f"{1 - head[primary] / head['climatology']:+.1%}",
              help="Across the whole test period, future time at seen locations. The number in the README.")

st.subheader("One pixel through time")
st.pyplot(series_fig(ds, [m for m in all_models if m in chosen], primary, py, px, level, test_mask))

band = float(ds["conformal_q"].sel(model=primary).interp(level=level))
truth = ds["ndvi"].sel(time=test_times)
lead = ds[f"pred_{primary}"].sel(time=test_times)
inside = (np.abs(truth - lead) <= band).where(np.isfinite(truth) & np.isfinite(lead))
cov = float(inside.mean(skipna=True))
d1, d2 = st.columns(2)
d1.metric(f"Interval half width at {int(level * 100)}%", f"{band:.4f} NDVI")
d2.metric("Observed coverage over the test period", f"{cov:.1%}", delta=f"{cov - level:+.1%} vs nominal",
          help="Calibrated on earlier folds, measured on later ones, so it is a real test.")

with st.expander("How to read this"):
    st.markdown(
        """
- **Persistence** says next month equals this month. **Climatology** says next month equals the
  training-period average for that calendar month. A model earns its place only by beating both.
- Forecasts are **out of sample**. Each fold trains only on months before its test block, with an
  embargo gap, and a buffer ring separates training pixels from held-out pixel blocks.
- The interval is **conformal**. Its width is the chosen quantile of absolute errors on earlier
  folds, so observed coverage near the nominal level means the intervals are calibrated.
- Absolute errors are not comparable between Sentinel-2 and Landsat, since they are different
  instruments with different compositing. Compare each model to its own baselines within one record.
- The grid is coarsened by a factor of two for this demo, so map numbers differ slightly from the
  full-resolution results in the README.
        """
    )
