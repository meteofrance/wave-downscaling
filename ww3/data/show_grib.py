from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from mftools.utils_log import get_logger
from ww3.data.grib_utils import load_grib

this_module_name = Path(__file__).name
LOGGER = get_logger(this_module_name)


parser = ArgumentParser()
parser.add_argument("path_grib", type=str, help="path of grib file")
args = parser.parse_args()

LOGGER.info("Plot data...")

grib_ww3 = load_grib(args.path_grib)

print(grib_ww3)

plt.figure(figsize=(18, 6))
for i, (key, data) in enumerate(grib_ww3.data_vars.items()):
    plt.subplot(2, 4, i + 1)
    grib_ww3[key].plot()
    plt.title(key)
plt.tight_layout()
plt.show()
