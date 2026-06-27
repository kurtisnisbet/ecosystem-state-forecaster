"""Build and load the spatio-temporal data cube.

Responsibilities:
- Query DEA's STAC catalogue for Sentinel-2 (Collection 3) over an AOI + dates.
- Load cloud-masked surface reflectance into an xarray cube (x, y, time, band).
- Composite to monthly means; assemble drivers (SILO, ERA5-Land, MODIS) and
  static terrain (Copernicus GLO-30 DEM) onto the same grid.
"""

# TODO: load_s2_cube(), apply_cloud_mask(), composite_monthly(), build_cube()
