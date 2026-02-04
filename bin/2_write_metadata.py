"""Scan GRIB files to get weather parameters dictionary"""

from typing import Dict

import xarray as xr
import yaml
from tqdm import tqdm

from ww3.data.grib_utils import load_grib
from ww3.settings import GRIDS, METADATA_PATH, SCRATCH_PATH


def add_param2dict(ds: xr.Dataset, param_dict: Dict, model: str) -> Dict:
    """Add parameters from a dataset to a dictionary"""
    for var in ds.data_vars:
        print("----->", var)
        levels = ds[var][ds[var].GRIB_typeOfLevel].values
        key = f"{model}_{var}"
        name = f"{ds[var].long_name}"
        param_dict[key] = {
            "name": key,
            "long_name": name,
            "unit": ds[var].units,
            "cumulative": ds[var].GRIB_stepType == "accum",
            "type_level": ds[var].GRIB_typeOfLevel,
            "levels": levels.astype(int).tolist(),
            "grid": list(GRIDS.keys()),
            "shape": [GRIDS[grid_name]["size"] for grid_name in list(GRIDS.keys())],
            "extent": [GRIDS[grid_name]["extent"] for grid_name in list(GRIDS.keys())],
            "model": model,
        }
        # add the model for each param ? ww3 or arpege ?
        print(param_dict[key])
    return param_dict


# --------------------------------------------------------------------------

if __name__ == "__main__":
    param_dict = {}
    model = ["ww3", "mfwam", "arpege"]
    for model in tqdm(model):
        print("--------------")
        grib_dir = SCRATCH_PATH / model / "BRETAGNE0002" / "grib"
        path_grib = next(grib_dir.iterdir(), None)
        if path_grib:
            ds = load_grib(path_grib)
            param_dict = add_param2dict(ds, param_dict, model)

    yaml_dict = {
        "GRIDS": GRIDS,
        "WEATHER_PARAMS": param_dict,
    }
    with open(METADATA_PATH, "w") as file:
        documents = yaml.dump(yaml_dict, file)
