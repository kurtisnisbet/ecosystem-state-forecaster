"""Build and load the spatio-temporal data cube.

Query DEA's STAC catalogue for Sentinel-2 (Collection 3), load cloud-masked
surface reflectance into an xarray cube, composite to monthly means, and
derive NDVI.
"""

import pystac_client
import xarray as xr
from odc.stac import load, configure_rio
from pystac_client.stac_api_io import StacApiIO
from urllib3.util.retry import Retry

# Retry transient DEA server errors (502/503/504) with exponential backoff.
_RETRY = Retry(total=6, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])

NDVI_BANDS = ["nbart_red", "nbart_nir_1", "oa_fmask"]
DEA_STAC_URL = "https://explorer.dea.ga.gov.au/stac"
S2_COLLECTIONS = ["ga_s2am_ard_3", "ga_s2bm_ard_3"]

def load_cube(
        items: list,
        bbox: list[float],
        bands: list[str] = NDVI_BANDS,
        crs: str = "EPSG:3577",
        resolution: int = 10,
        groupby: str = "solar_day"
) -> xr.Dataset:
    """Load STAC items into a lazy (Dask-backed) xarray cube.
    
    Uses public access to DEA's public S3 bucket. No credentials needed.
    
    Parameters:
    items: STAC items from 'search_scenes'.
    bbox: [min_lon, min_lat, max_lon, max_lat] in EPSG: 4326
    bands: measurement names to load
    crs, resolution: output grid (Australian Albers, metres).
    groupby: how to merge scenes into timesteps
    
    Returns:
    xarray.Dataset with one lazy data variable per band.
    """

    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    ds = load(
        items,
        bands=bands,
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        groupby=groupby,
        chunks={},
    )
    return ds

def search_scenes(
        bbox: list[float],
        time_range: str,
        collections: list[str] = S2_COLLECTIONS,
) -> list:
    """Search DEA's STAC catalogue for Sentinel-2 scenes.

    DEA's STAC caps the items returned by a single search, silently truncating
    long date ranges. To get the full record, I split the request into
    one-year windows and concatenate the results.

    Parameters
    ----------
    bbox : [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
    time_range : ISO range "YYYY-MM-DD/YYYY-MM-DD".
    collections : STAC collection ids to search.

    Returns
    -------
    list of STAC items across the full range.
    """
    catalog = pystac_client.Client.open(DEA_STAC_URL, stac_io=StacApiIO(max_retries=_RETRY))
    start_year = int(time_range[:4])
    end_year = int(time_range.split("/")[1][:4])

    items = []
    for year in range(start_year, end_year + 1):
        try:
            search = catalog.search(
                collections=collections,
                bbox=bbox,
                datetime=f"{year}-01-01/{year}-12-31",
            )
            items.extend(search.items())
        except Exception as exc:
            print(f"  warning: {year} search failed ({type(exc).__name__}); skipping that year")
    return items

def compute_ndvi(
    ds: xr.Dataset,
    nir_band: str = "nbart_nir_1",
    mask_band: str = "oa_fmask",
    clear_value: int = 1,
) -> xr.DataArray:
    """Cloud-mask a cube and compute NDVI.

    Keeps only clear-land pixels (fmask == clear_value); all others -> NaN.
    `nir_band` is nbart_nir_1 for Sentinel-2, nbart_nir for Landsat.

    Returns
    -------
    xarray.DataArray named "ndvi", values in [-1, 1].
    """
    clear = ds[mask_band] == clear_value
    red = ds["nbart_red"].where(clear)
    nir = ds[nir_band].where(clear)
    ndvi = (nir - red) / (nir + red)
    ndvi.name = "ndvi"
    return ndvi

def to_monthly(ndvi: xr.DataArray, freq: str = "MS") -> xr.DataArray:
    """Composite to monthly means, skipping cloud-masked NaNs.

    freq="MS" groups by calendar month (month-start).
    """
    return ndvi.resample(time=freq).mean()