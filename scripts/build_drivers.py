"""Fetch SILO drivers, align them to the cached NDVI cube, and cache the result.

Needs internet and s3fs + rioxarray. Reads the same area and date range as the
NDVI cube from config.yaml, so the driver cube lands on the exact model grid.

Run:  python scripts/build_drivers.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoforecast.drivers import align_to_grid, load_silo

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "ecoforecast" / "config.yaml").read_text())
    build = cfg["build"]
    biome = build["biome"]
    bbox = cfg["biomes"][biome]["bbox"]

    cube_path = ROOT / build["cache"]
    if not cube_path.exists():
        raise SystemExit(f"Build the NDVI cube first (scripts/build_cube.py): {cube_path}")
    ndvi = xr.open_dataarray(cube_path)

    variables = tuple(cfg["drivers_build"]["silo_variables"])
    out = ROOT / cfg["drivers_build"]["cache"]
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching SILO {list(variables)} for {biome}  {build['time_range']}")
    t0 = time.time()
    silo = load_silo(bbox, build["time_range"], variables)
    aligned = align_to_grid(silo, ndvi)
    aligned.to_netcdf(out)
    print(f"  drivers {dict(aligned.sizes)} vars={list(aligned.data_vars)} -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
