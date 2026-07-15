"""Build and cache the monthly NDVI cube for each biome from DEA.

Reads config.yaml: the imagery profile (`sentinel2` or `landsat`) sets the
products, NIR band, and date range, and build.resolution_m sets the grid. One
cube per biome is cached at {cache_dir}/cube_{profile}_{resolution}m_{biome}.nc,
skipping any already built. This is the slow, network-heavy step; run it once,
then iterate with run_pipeline.py.

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


def build_one(biome, bbox, time_range, res, products, nir, cache):
    print(f"[{biome}] bbox={bbox}  {time_range}  {res} m")
    t0 = time.time()
    items = search_scenes(bbox, time_range, products)
    print(f"  {len(items)} scenes")
    if not items:
        print("  no scenes found, skipping")
        return
    ds = load_cube(items, bbox, bands=["nbart_red", nir, "oa_fmask"], resolution=res)
    monthly = to_monthly(compute_ndvi(ds, nir_band=nir))
    print("  loading + compositing (downloads pixels)...")
    monthly = monthly.compute()
    monthly.name = "ndvi"
    monthly.to_netcdf(cache)
    print(f"  cube {dict(monthly.sizes)}  cloud/gap NaN {float(monthly.isnull().mean()):.1%}"
          f" -> {cache.name}  ({time.time() - t0:.0f}s)")


def main():
    cfg = yaml.safe_load((ROOT / "ecoforecast" / "config.yaml").read_text())
    build = cfg["build"]
    profile = cfg["imagery"][build["profile"]]
    products, nir, time_range = profile["products"], profile["nir_band"], profile["time_range"]
    res = build["resolution_m"]
    cache_dir = ROOT / build["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{build['profile']}_{res}m"

    for biome in build["biomes"]:
        bbox = cfg["biomes"][biome]["bbox"]
        if bbox is None:
            print(f"[{biome}] no bbox in config, skipping")
            continue
        cache = cache_dir / f"cube_{tag}_{biome}.nc"
        if cache.exists():
            print(f"[{biome}] already cached: {cache.name}")
            continue
        build_one(biome, bbox, time_range, res, products, nir, cache)


if __name__ == "__main__":
    main()
