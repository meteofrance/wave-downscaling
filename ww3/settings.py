from pathlib import Path
from typing import Literal

import numpy as np
import yaml

FORMATSTR = "%Y%m%d%H"
SCRATCH_PATH = Path("/scratch/shared/ww3/datas")

DEFAULT_CONFIG = Path(__file__).parents[1] / "config/base_config.yaml"

if (Path(__file__).parents[0] / "metadata.yaml").exists():
    with open(Path(__file__).parents[0] / "metadata.yaml", "r") as file:
        METADATA = yaml.safe_load(file)


WW3_PATH = SCRATCH_PATH / "ww3"
MFWAM_PATH = SCRATCH_PATH / "mfwam"
ARP_PATH = SCRATCH_PATH / "arpege"

OBS_PATH = (
    SCRATCH_PATH
    / "obs/BRETAGNE0002/jason3/cmems_obs-wave_glo_phy-swh_nrt_j3-l3_PT1S_VAVH_5.80W-1.13W_47.00N-49.86N_0.00m_2023-12-19-2023-12-31.csv"
)

METADATA_PATH = Path(__file__).parents[1] / "ww3/metadata.yaml"
STATS_PATH = SCRATCH_PATH / "stats.pt"

DOWNSCALING_FACTOR = 2
MFWAM_NATIVE_GRID_RES = 0.1

GRIDS = {
    "BRETAGNE0002": {
        "size": [1430, 2336],
        "resolution": 0.002,
        "extent": [49.858, 47.0, -5.8, -1.13],
        "prefix": "bret",
    },
    "CORSE0002": {
        "size": [901, 651],
        "resolution": 0.002,
        "extent": [43.08, 41.2, 8.4, 9.7],
        "prefix": "corse",
    },
    "MANCHE0002": {
        "size": [1501, 2916],
        "resolution": 0.002,
        "extent": [51.5, 48.5, -2.5, 3.33],
        "prefix": "manche",
    },
    "GASCOGNE0002": {
        "size": [2151, 1526],
        "resolution": 0.002,
        "extent": [47.5, 43.2, -3.5, -0.45],
        "prefix": "gas",
    },
    "LIONAZUR0002": {
        "size": [1481, 2456],
        "resolution": 0.002,
        "extent": [44.96, 42.0, 3.0, 7.91],
        "prefix": "lion",
    },
}

for grid_name, values in GRIDS.items():
    downscale_step = int(
        np.ceil(MFWAM_NATIVE_GRID_RES / values["resolution"] / DOWNSCALING_FACTOR)
    )
    lat_size = int(np.ceil(values["size"][0] / downscale_step))
    lon_size = int(np.ceil(values["size"][1] / downscale_step))

    GRIDS[grid_name]["downscale_size"] = [lat_size, lon_size]
    GRIDS[grid_name]["downscale_step"] = downscale_step


GridType = Literal[
    "BRETAGNE0002",
    "CORSE0002",
    "MANCHE0002",
    "GASCOGNE0002",
    "LIONAZUR0002",
]

WaveParamType = Literal[
    "cos_mdps",
    "sin_mdps",
    "cos_mdww",
    "sin_mdww",
    "mpps",
    "mpww",
    "shps",
    "shww",
    "swh",
]

WindParamType = Literal["u10", "v10"]
