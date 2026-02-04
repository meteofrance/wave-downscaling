from pathlib import Path

import xarray as xr


def load_grib(path_grib: Path) -> xr.Dataset:
    """Load grib ww3 and rename unknown variable"""
    ds = xr.open_dataset(path_grib, engine="cfgrib", chunks={"step": 1})
    if "unknown" in ds.keys():
        ds = ds.rename({"unknown": "shps"})
        ds.shps.attrs["long_name"] = "Significant height of primary wave"
        ds.shps.attrs["units"] = "m"
    return ds
