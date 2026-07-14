"""Build and cache the real monthly NDVI cube from DEA Sentinel-2.

Needs internet and the geospatial deps (`pip install -r requirements.txt`).
Reads config.yaml, searches DEA's STAC, loads cloud-masked surface reflectance,
computes NDVI, composites to monthly means, and writes a NetCDF cache under
data/ (gitignored). This is the slow, network-heavy step — run it once, then
iterate with scripts/run_pipeline.py against the cache.

Run:  python scripts/build_cube.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoforecast.data import compute_ndvi, load_cube, search_scenes, to_monthly

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "ecoforecast" / "config.yaml").read_text())
    build = cfg["build"]
    biome = build["biome"]
    bbox = cfg["biomes"][biome]["bbox"]
    if bbox is None:
        raise SystemExit(f"Set biomes.{biome}.bbox in config.yaml first.")

    time_range, res = build["time_range"], build["resolution_m"]
    cache = ROOT / build["cache"]
    cache.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building NDVI cube: {biome}  bbox={bbox}  {time_range}  {res} m")
    t0 = time.time()

    items = search_scenes(bbox, time_range)
    print(f"  found {len(items)} Sentinel-2 scenes")
    if not items:
        raise SystemExit("No scenes found — check the bbox and date range.")

    ds = load_cube(items, bbox, resolution=res)
    ndvi = compute_ndvi(ds)
    monthly = to_monthly(ndvi)

    print("  loading + compositing (downloads pixels — this can take a while)...")
    monthly = monthly.compute()
    monthly.name = "ndvi"

    monthly.to_netcdf(cache)
    nan_pct = float(monthly.isnull().mean())
    print(f"  cube {dict(monthly.sizes)}  cloud/gap NaN {nan_pct:.1%}")
    print(f"  saved -> {cache}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
