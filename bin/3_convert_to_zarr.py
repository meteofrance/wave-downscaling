from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal

import joblib
import numpy as np
import xarray as xr
from mftools.utils_log import get_logger
from tqdm import tqdm

from ww3.data.grib_utils import load_grib
from ww3.settings import GRIDS, SCRATCH_PATH, GridType

this_module_name = Path(__file__).name
LOGGER = get_logger(this_module_name)


def check_date_done(date: datetime, model: Literal["ww3", "mfwam", "arpege"]) -> bool:
    """Check if the conversion has already been done for the date.

    Args:
        date: Model run date.
        model: Name of the model.

    Return:
        bool: True if the conversion is already done, False otherwise.
    """
    date_str = date.strftime("%Y%m%d%H")
    zarr_dir = SCRATCH_PATH / model / "BRETAGNE0002" / "zarr"
    daily_zarr_dir = zarr_dir / f"{date_str}.zarr"
    return daily_zarr_dir.exists()


def load_grib_from_date(
    date: datetime, model: Literal["ww3", "mfwam", "arpege"], grid: GridType
) -> xr.Dataset | None:
    """Load the grib corresponding to the date.

    Args:
        date: Model run date.
        model: Name of the model.
        grid: Name of the grid.

    Returns:
        xr.DataArray: Data array corresponding to the grib at the date and on the grid.
        None: If there are no grib files or if there are several files for the same date.
    """
    hour_str = date.strftime("%Y%m%d%H")
    grib_dir = SCRATCH_PATH / model / grid / "grib"
    list_grib = list(grib_dir.glob(f"{hour_str}_*.grib"))
    if len(list_grib) > 1:
        LOGGER.warning(f"{date} : several gribs for the date !")
        return None
    elif len(list_grib) == 0:
        LOGGER.warning(f"{date} : grib does not exist !")
        return None
    else:
        grib = load_grib(list_grib[0])
        if model == "arpege":
            # change to negative longitude
            grib["longitude"] = grib["longitude"] - 360

        # delete idx file
        for idx_path in grib_dir.glob(f"{list_grib[0].name}.*.idx"):
            idx_path.unlink()
        return grib


def process_date(
    date: datetime,
    model: Literal["ww3", "mfwam", "arpege"],
    area: List[GridType],
    interpol: bool = False,
) -> None:
    """Converted the grib for a given date to zarr.

    Args:
        date: Model run date.
        model: Name of the model.
        area: Names of the grids.
        interpol: True if the grids need to be interpolated, otherwise False.
    """
    print(f"{datetime.now()} - {model} - {area} - Processing date {date}...")
    date_str = date.strftime("%Y%m%d%H")
    for grid in area:
        dataset = load_grib_from_date(date, model, grid)
        if dataset is not None:
            zarr_dir = SCRATCH_PATH / model / grid / "zarr"
            if interpol:
                dataset = interpolation(dataset, grid)
            dataset.to_zarr(zarr_dir / f"{date_str}.zarr", mode="w", consolidated=True)
        print(f"{datetime.now()} - {model} - {area} - converted date {date}...")


def interpolation(dataset: xr.DataArray, grid: str) -> xr.DataArray:
    """Interpolate a xarray to match a grid.

    Args:
        dataset: Data array to interpolate.
        grid: Name of the grid.

    Return:
        xr.DataArray: Data array interpolated.
    """
    extent = GRIDS[grid]["extent"]
    res = GRIDS[grid]["resolution"]

    nb_lat = round((extent[0] - extent[1]) / res) + 1
    nb_lon = round((extent[3] - extent[2]) / res) + 1

    lat_new = np.linspace(extent[0], extent[1], nb_lat)
    lon_new = np.linspace(extent[2], extent[3], nb_lon)

    dataset = dataset.interp(latitude=lat_new, longitude=lon_new, method="nearest")
    return dataset


def load_zarr_from_date(
    date: datetime, model: Literal["ww3", "mfwam", "arpege"], grid: GridType
) -> xr.Dataset | None:
    """Load the zarr corresponding to the date.

    Args:
        date: Model run date.
        model: Name of the model.
        grid: Name of the grid.

    Returns:
        xr.DataArray: data array at the date and from the grid.
        None: If there are no zarr files or if there are several files for the same date.
    """
    hour_str = date.strftime("%Y%m%d%H")
    zarr_dir = SCRATCH_PATH / model / grid / "zarr"
    list_zarr = list(zarr_dir.glob(f"{hour_str}.zarr"))

    if len(list_zarr) > 1:
        LOGGER.warning(f"{date} : several zarrs for the date !")
        return None
    elif len(list_zarr) == 0:
        LOGGER.warning(f"{date} : zarr does not exist !")
        return None
    else:
        zarr = xr.open_zarr(list_zarr[0], decode_timedelta=False)
        return zarr


def check_zarr_valid(
    date: datetime, model: Literal["ww3", "mfwam", "arpege"], area: List[GridType]
) -> bool:
    """Checks that the zarr file for the given date does not contain parameters consisting only of nan.

    Args:
        date: Model run date.
        model: Name of the model.
        area: Names of the grids.

    Returns:
        bool: True if the zarrs are valid, otherwise False.
    """
    for grid in area:
        dataset = load_zarr_from_date(date, model, grid)
        if dataset:
            for param in dataset.var():
                if np.isnan(dataset[param]).all().values:
                    LOGGER.warning(f"[{model}] {date} : this zarr is corrupt !")
                    return False
    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        nargs="+",
        choices=["mfwam", "ww3", "arpege"],
        help="Model to download (e.g., mfwam, ww3, arpege)",
    )
    parser.add_argument("start", type=str, help="Start date, format YYYYMMDDHH")
    parser.add_argument("end", type=str, help="End date, format YYYYMMDDHH")
    parser.add_argument("--workers", type=int, default=5, dest="workers")
    parser.add_argument(
        "--area",
        nargs="+",
        choices=[
            "BRETAGNE0002",
            "CORSE0002",
            "MANCHE0002",
            "GASCOGNE0002",
            "LIONAZUR0002",
        ],
        default=["BRETAGNE0002"],
        help="Grids to download",
    )
    parser.add_argument(
        "--check_zarr", type=bool, help="Verify if the dataset is valid", default=False
    )
    args = parser.parse_args()

    # Get list of dates
    start_date = datetime.strptime(args.start, "%Y%m%d%H")
    end_date = datetime.strptime(args.end, "%Y%m%d%H")

    diff_date = end_date - start_date
    nb_hours = int(diff_date.total_seconds() // 3600)

    for model in args.model:
        if model == "ww3":
            list_dates = [start_date + i * timedelta(hours=1) for i in range(nb_hours)]
            interpol = False
        elif model == "mfwam":
            list_dates = [
                start_date + timedelta(hours=i)
                for i in range(nb_hours)
                if (start_date + timedelta(hours=i)).hour % 6 == 0
            ]
            interpol = True
        elif model == "arpege":
            list_dates = [
                start_date + timedelta(hours=i)
                for i in range(nb_hours)
                if (start_date + timedelta(hours=i)).hour % 6 == 0
            ]
            interpol = True

        list_dates_to_download = [
            date for date in list_dates if not check_date_done(date, model)
        ]
        LOGGER.info(
            f"Converting {model} data to zarr for {len(list_dates_to_download)} hours : {start_date} to {end_date}"
        )
        joblib.Parallel(n_jobs=args.workers)(
            joblib.delayed(process_date)(date, model, args.area, interpol)
            for date in list_dates_to_download
        )

        if args.check_zarr:
            LOGGER.info(
                f"Check zarr of {model} data for {len(list_dates)} hours : {start_date} to {end_date}"
            )
            joblib.Parallel(n_jobs=args.workers)(
                joblib.delayed(check_zarr_valid)(date, model, args.area)
                for date in tqdm(list_dates)
            )

    print("Done !")
