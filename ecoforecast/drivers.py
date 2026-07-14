"""Environmental drivers aligned to the NDVI grid.

SILO gridded climate data (rainfall, temperature) from the AWS open-data bucket
`s3://silo-open-data`, subset to the area of interest, aggregated to monthly, and
reprojected onto the NDVI cube's grid so it slots into the models as extra
channels. Needs internet and `s3fs` + `rioxarray`.

SILO stores one NetCDF per variable per year at
`Official/annual/{variable}/{year}.{variable}.nc`. `monthly_rain` is already
monthly and small (~14 MB/year); `max_temp` / `min_temp` are daily and much
larger, and are averaged to monthly here.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

SILO_PATH = "silo-open-data/Official/annual/{var}/{year}.{var}.nc"


def load_silo(bbox, time_range, variables=("monthly_rain",)) -> xr.Dataset:
    """Load SILO variables over a bbox and date range, as a monthly (time, lat, lon) set.

    Each annual file is downloaded to a temp path and opened with netcdf4 (reading
    a NetCDF from an S3 file handle would need the h5netcdf engine). The temp cache
    means re-runs do not re-download.
    """
    import os
    import tempfile

    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    minx, miny, maxx, maxy = bbox
    y0 = int(time_range[:4])
    y1 = int(time_range.split("/")[1][:4])
    cache_dir = tempfile.gettempdir()

    out = {}
    for var in variables:
        years = []
        for year in range(y0, y1 + 1):
            local = os.path.join(cache_dir, f"silo_{year}_{var}.nc")
            if not os.path.exists(local):
                fs.get(SILO_PATH.format(var=var, year=year), local)
            ds = xr.open_dataset(local)
            da = ds[var].sortby("lat").sel(
                lat=slice(miny - 0.15, maxy + 0.15),
                lon=slice(minx - 0.15, maxx + 0.15),
            ).resample(time="MS").mean().load()  # snap to month-start; monthly is idempotent
            ds.close()
            years.append(da)
        out[var] = xr.concat(years, dim="time")
    return xr.Dataset(out)


def align_to_grid(drivers: xr.Dataset, target: xr.DataArray) -> xr.Dataset:
    """Reproject coarse lat/lon drivers onto the target cube's grid and time axis."""
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    ref = target.rio.write_crs(target.rio.crs or "EPSG:3577")
    aligned = {}
    for name, da in drivers.items():
        grid = da.rename({"lon": "x", "lat": "y"}).rio.write_crs("EPSG:4326")
        aligned[name] = grid.rio.reproject_match(ref)

    ds = xr.Dataset(aligned).drop_vars("spatial_ref", errors="ignore")
    return ds.reindex(time=target["time"], method="nearest", tolerance=pd.Timedelta("20D"))


def lag_drivers(drivers: xr.Dataset, lags=(1, 2, 3)) -> xr.Dataset:
    """Shift each driver into the past as `{name}_lag{k}`.

    A one-step-ahead forecast of month t may only use drivers observed before t,
    so drivers enter the models lagged, never at the concurrent month. Lagged
    rainfall is also the physically sensible signal: vegetation greens up in the
    weeks to months after rain, not the same month.
    """
    out = {}
    for name, da in drivers.items():
        if "time" not in da.dims:  # skip CRS / grid-mapping vars like spatial_ref
            continue
        for k in lags:
            out[f"{name}_lag{k}"] = da.shift(time=k)
    return xr.Dataset(out)
