"""Evaluate baselines, GBT and ConvLSTM on each biome's cached cube.

Loops over build.biomes, runs the walk-forward + spatial-block evaluation for
each, prints a per-biome table, and writes a cross-biome comparison
figure, per-biome forecast figures, and a results CSV. Every output is suffixed
with the build tag (profile and resolution, for example sentinel2_100m), so
running a second profile adds files rather than overwriting the first one's
results. ConvLSTM is skipped with a message if PyTorch is not
installed. Drivers are used only when drivers_build.use is true and a per-biome
driver cache exists.

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
from ecoforecast.app_data import save_app_data
from ecoforecast.baselines import climatology_forecast, persistence
from ecoforecast.features import seasonal_climatology
from ecoforecast.evaluate import spatial_blocks, walk_forward_splits
from ecoforecast.models.gbt import walk_forward_gbt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
MODEL_COLORS = {"gbt": "#1b7837", "convlstm": "#e7298a", "gnn": "#3182bd", "ensemble": "#222222"}


def _load_drivers(cfg, biome):
    dcfg = cfg.get("drivers_build", {})
    if not dcfg.get("use", False):
        return None
    dpath = ROOT / dcfg.get("cache_dir", "data") / f"drivers_{biome}.nc"
    if not dpath.exists():
        return None
    from ecoforecast.drivers import lag_drivers
    return lag_drivers(xr.open_dataset(dpath))


def evaluate_biome(cfg, biome, cube_path):
    ndvi = xr.open_dataarray(cube_path)
    tsp, ssp = cfg["splits"]["temporal"], cfg["splits"]["spatial"]
    folds = walk_forward_splits(ndvi["time"], tsp["block_months"], tsp["n_test_folds"], tsp["embargo_months"])
    strain, stest, _ = spatial_blocks(ndvi, ssp["block_size_px"], ssp["n_test_blocks"], ssp["buffer_px"], seed=1)
    drivers = _load_drivers(cfg, biome)

    res_gbt, oos_gbt, _ = walk_forward_gbt(ndvi, folds, strain, stest, drivers=drivers, max_train_rows=cfg["build"].get("max_train_rows"))
    frames, oos = [res_gbt], {"gbt": oos_gbt}
    try:
        import torch
        from ecoforecast.models.convlstm import walk_forward_convlstm
        device = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
        print(f"  ConvLSTM on {device}...")
        res_cl, oos_cl, _ = walk_forward_convlstm(ndvi, folds, strain, stest, drivers=drivers, seq_len=6, hidden=16, epochs=100)
        frames.append(res_cl[res_cl["predictor"] == "convlstm"])
        oos["convlstm"] = oos_cl
    except Exception as exc:
        print(f"  ConvLSTM skipped ({type(exc).__name__}) — install torch to include it.")

    try:
        from ecoforecast.models.gnn import walk_forward_gnn
        print("  GNN...")
        res_gnn, oos_gnn, _ = walk_forward_gnn(ndvi, folds, strain, stest, drivers=drivers, hidden=32, rounds=3, epochs=30)
        frames.append(res_gnn[res_gnn["predictor"] == "gnn"])
        oos["gnn"] = oos_gnn
    except Exception as exc:
        print(f"  GNN skipped ({type(exc).__name__}) — install torch to include it.")

    if len(oos) >= 2:
        from ecoforecast.ensemble import score_ensemble, stack_ensemble
        ens, _weights = stack_ensemble(ndvi, oos, folds, space_mask=strain)
        res_ens = score_ensemble(ndvi, ens, folds, strain, stest)
        frames.append(res_ens[res_ens["predictor"] == "ensemble"])
        oos["ensemble"] = ens

    res = pd.concat(frames, ignore_index=True)
    res["biome"] = biome
    return res, ndvi, folds, oos, strain


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "ecoforecast" / "config.yaml").read_text())
    build = cfg["build"]
    cache_dir = ROOT / build["cache_dir"]
    tag = f"{build['profile']}_{build['resolution_m']}m"

    all_res = []
    for biome in build["biomes"]:
        cube_path = cache_dir / f"cube_{tag}_{biome}.nc"
        if not cube_path.exists():
            print(f"[{biome}] no cube ({cube_path.name}), skip — run build_cube.py")
            continue
        print(f"[{biome}]")
        res, ndvi, folds, oos, strain = evaluate_biome(cfg, biome, cube_path)
        all_res.append(res)
        head = res[(res.time == "future") & (res.space == "seen")].groupby("predictor")["rmse"].mean().sort_values()
        print("  headline RMSE:", {k: round(v, 4) for k, v in head.items()})
        _plot_forecast(ndvi, folds, oos, biome, tag)
        save_app_data(ndvi, oos, folds, strain, biome, tag, ROOT / "docs" / "app_data", results=res)

    if not all_res:
        raise SystemExit("No cubes found — run scripts/build_cube.py first.")

    combined = pd.concat(all_res, ignore_index=True)
    results_csv = ROOT / "docs" / f"biome_results_{tag}.csv"
    combined.to_csv(results_csv, index=False)
    _print_summary(combined, build["biomes"])
    _plot_biome_comparison(combined, build["biomes"], tag)
    print(f"\nfigures -> {FIG_DIR}   results -> docs/{results_csv.name}")


def _best_pixel(ndvi):
    valid = ndvi.notnull().sum("time")
    return np.unravel_index(int(np.argmax(valid.values)), valid.shape)


def _test_union(folds):
    mask = folds[0]["test"].copy()
    for fo in folds[1:]:
        mask = mask | fo["test"]
    return mask


def _print_summary(res, biomes):
    head = res[(res.time == "future") & (res.space == "seen")]
    table = head.pivot_table(index="biome", columns="predictor", values="rmse", aggfunc="mean").reindex(biomes)
    print("\nHeadline (future / seen) RMSE by biome:")
    print("  " + table.round(4).to_string().replace("\n", "\n  "))


def _plot_forecast(ndvi, folds, oos, biome, tag):
    py, px = _best_pixel(ndvi)
    test_all = _test_union(folds)
    t = ndvi["time"].sel(time=test_all).values
    clim = climatology_forecast(ndvi, seasonal_climatology(ndvi))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(t, ndvi.sel(time=test_all).isel(y=py, x=px), "k-o", lw=2, ms=3, label="actual")
    ax.plot(t, persistence(ndvi).sel(time=test_all).isel(y=py, x=px), color="#d95f02", label="persistence")
    ax.plot(t, clim.sel(time=test_all).isel(y=py, x=px), color="#7570b3", label="climatology")
    for model in oos:
        ax.plot(t, oos[model].sel(time=test_all).isel(y=py, x=px), color=MODEL_COLORS.get(model, "#888888"), lw=2, label=model)
    ax.set_title(f"{biome}: forecasts vs actual (pixel {py},{px})")
    ax.set_ylabel("NDVI"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"biome_{biome}_forecast_{tag}.png", dpi=120); plt.close(fig)


def _plot_biome_comparison(res, biomes, tag):
    head = res[(res.time == "future") & (res.space == "seen")]
    models = [m for m in ("gbt", "convlstm", "gnn", "ensemble") if m in set(head["predictor"])]
    table = (head[head["predictor"].isin(models)]
             .pivot_table(index="biome", columns="predictor", values="skill_vs_climatology", aggfunc="mean")
             .reindex(biomes)[models])
    fig, ax = plt.subplots(figsize=(9, 4))
    table.plot.bar(ax=ax, color=[MODEL_COLORS[m] for m in table.columns])
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylabel("skill vs climatology")
    ax.set_xlabel("")
    ax.set_title("Do the models beat climatology? Skill vs climatology by biome (headline cell)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"biome_skill_vs_climatology_{tag}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
