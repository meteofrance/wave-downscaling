import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import torch
import xarray as xr
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor

from ww3.settings import FORMATSTR, SCRATCH_PATH, GridType, WaveParamType, WindParamType


def get_model_filepath(
    model: str,
    grid_name: str,
    run_date: dt.datetime,
) -> Path:
    filename = f"{run_date.strftime(FORMATSTR)}.zarr"
    return SCRATCH_PATH / model / grid_name / "zarr" / filename


class Sample:
    def __init__(
        self,
        mfwam_run_date: dt.datetime,
        mfwam_leadtime: dt.timedelta,
        downscaling_stride: int,
        grid_name: GridType = "BRETAGNE0002",
        wave_params: list[WaveParamType] = ["shww", "cos_mdps"],
        include_arpege: bool = False,
        include_forcings: bool = False,
    ) -> None:
        """A Sample class to manage training samples data.

        Args:
            mfwam_run_date (dt.datetime): Run date of MFWAM modle
            mfwam_leadtime (dt.timedelta): Desired leadtime of MFWAM model
            downscaling_stride (int): Inside [1;50]. Stride of pixel to keep between full resolution data (200m) and working resolution. Lower is a better resolution. Defaults to 2.
            grid_name (GridType): Name of the model grid / area.
            wave_params (list[WaveParamType], optional): List of wave model parameters. Defaults to ["shww", "cos_mdps"].
            include_arpege (bool, optional): Whether to include Arpege data in the inputs. Defaults to False.
            include_forcings (bool, optional): Whether to include forcing data in the inputs. Defaults to False.
        """
        # Input validation
        if grid_name not in ["BRETAGNE0002"]:
            raise NotImplementedError("Grid must be in ['BRETAGNE0002'].")
        self.mfwam_run_date = mfwam_run_date
        if mfwam_leadtime < dt.timedelta(hours=1):
            raise ValueError(f"MFWAM leadtime must be >= 1h, got {mfwam_leadtime}.")

        self.mfwam_leadtime = mfwam_leadtime
        self.downscaling_stride = downscaling_stride
        self.grid_name = grid_name
        self.wave_params = wave_params
        self.include_arpege = include_arpege
        self.include_forcings = include_forcings

    @classmethod
    def from_npz_path(
        cls,
        npz_path: Path,
        wave_params: list[WaveParamType] = ["shww", "cos_mdps"],
        include_arpege: bool = False,
        include_forcings: bool = False,
    ) -> "Sample":
        """Instantiate a Sample from a npz file."""
        grid_name = cast(GridType, npz_path.parent.stem.split("_")[1])
        downscaling_stride = int(npz_path.parent.stem.split("_")[2])
        mfwam_run_date = dt.datetime.strptime(npz_path.stem.split("_")[0], FORMATSTR)
        mfwam_leadtime = dt.timedelta(hours=int(npz_path.stem.split("_")[1]))
        return cls(
            mfwam_run_date,
            mfwam_leadtime,
            downscaling_stride,
            grid_name,
            wave_params,
            include_arpege,
            include_forcings,
        )

    def __str__(self) -> str:
        return (
            f"Sample("
            f"ww3_analysis_date={self.ww3_analysis_date}, "
            f"mfwam_run_date={self.mfwam_run_date}, "
            f"mfwam_leadtime={self.mfwam_leadtime}, "
            f"downscaling_stride={self.downscaling_stride}, "
            f"grid_name='{self.grid_name}', "
            f"wave_params={self.wave_params}, "
            f"include_arpege={self.include_arpege}, "
            f"include_forcings={self.include_forcings}"
            f")"
        )

    @property
    def ww3_analysis_date(self) -> dt.datetime:
        """Date of corresponding WW3 analysis, equal to MFWAM forecast valid time."""
        return self.mfwam_run_date + self.mfwam_leadtime

    @property
    def path_ww3(self) -> Path:
        return get_model_filepath("ww3", self.grid_name, self.ww3_analysis_date)

    @property
    def path_mfwam(self) -> Path:
        return get_model_filepath("mfwam", self.grid_name, self.mfwam_run_date)

    @property
    def path_arpege(self) -> Path:
        return get_model_filepath("arpege", self.grid_name, self.mfwam_run_date)

    @property
    def save_path(self) -> Path:
        """Save path on the disk, in npz format.
        Saved in 'bin/5_prepare_dataset' script.
        """

        date_run = self.mfwam_run_date.strftime(FORMATSTR)
        hours = int(self.mfwam_leadtime.total_seconds() // 3600)
        name = f"{date_run}_{hours}.npz"
        folder = SCRATCH_PATH / f"dataset_{self.grid_name}_{self.downscaling_stride}"
        return folder / name

    def get_data(
        self, ds: xr.Dataset, params: list[WaveParamType] | list[WindParamType]
    ) -> Tensor:
        """Retrieve wave data (MFWAM or WW3) from a xarray Dataset.

        Args:
            ds (xr.Dataset): Dataset containing model data.
            params (list[WaveParamType]): List of parameters names.

        Returns:
            Tensor: Tensor with all wave data parameters stacked.
        """
        arrays = []
        for param in params:
            if param.startswith("cos"):
                arr = ds[param.split("_")[1]].values
                arr = np.cos(np.deg2rad(arr))
            elif param.startswith("sin"):
                arr = ds[param.split("_")[1]].values
                arr = np.sin(np.deg2rad(arr))
            else:
                arr = ds[param].values
            arr = arr[:: self.downscaling_stride, :: self.downscaling_stride]
            arrays.append(arr)
        return Tensor(np.stack(arrays, axis=0))

    def get_ww3_data(self) -> NamedTensor:
        """Returns a NamedTensor with the desired WW3 parameters data."""
        ds = xr.open_zarr(self.path_ww3, decode_timedelta=True)
        tensor = self.get_data(ds, self.wave_params)
        param_names = ["ww3_" + name for name in self.wave_params]
        nt = NamedTensor(
            tensor,
            names=["features", "lat", "lon"],
            feature_names=param_names,
        )
        return nt

    def get_mfwam_data(self) -> NamedTensor:
        """Returns a NamedTensor with the desired MFWAM parameters data."""
        ds = xr.open_zarr(self.path_mfwam, decode_timedelta=True)
        ds = ds.sel(step=self.mfwam_leadtime)  # keep only necessary leadtime
        tensor = self.get_data(ds, self.wave_params)
        param_names = ["mfwam_" + name for name in self.wave_params]
        nt = NamedTensor(
            tensor,
            names=["features", "lat", "lon"],
            feature_names=param_names,
        )
        return nt

    def get_arpege_data(self) -> NamedTensor:
        """Returns a NamedTensor with the desired ARPEGE parameters data."""
        # We use the same run date and leadtime for ARPEGE and MFWAM :
        ds = xr.open_zarr(self.path_arpege, decode_timedelta=True)
        ds = ds.sel(step=self.mfwam_leadtime)  # keep only necessary leadtime
        arp_params: list[WindParamType] = ["u10", "v10"]
        tensor = self.get_data(ds, arp_params)
        param_names = ["arpege_" + name for name in arp_params]
        nt = NamedTensor(
            tensor,
            names=["features", "lat", "lon"],
            feature_names=param_names,
        )
        return nt

    def get_forcings_data(self) -> NamedTensor:
        """Returns a NamedTensor with the forcings data: bathymetry and land mask."""
        path = SCRATCH_PATH / f"conf_{self.grid_name}.grib"
        ds = xr.open_dataset(path, decode_timedelta=True)
        bathymetry = ds.unknown.values
        log_bathy = np.log(bathymetry + 1)
        log_bathy = log_bathy[:: self.downscaling_stride, :: self.downscaling_stride]
        landsea_mask = ~np.isnan(log_bathy)  # 0 = land, 1 = sea
        log_bathy = np.nan_to_num(log_bathy)
        tensor = Tensor(np.stack([log_bathy, landsea_mask], axis=0))
        nt = NamedTensor(
            tensor,
            names=["features", "lat", "lon"],
            feature_names=["log_bathy", "landsea_mask"],
        )
        return nt

    def get_leadtime_data(self) -> Tensor:
        """Returns a Tensor with the leadtime."""
        leadtime = self.mfwam_leadtime.total_seconds() // 3600
        return Tensor([leadtime]).to(dtype=torch.long)

    def get_x_y(self) -> tuple[NamedTensor, NamedTensor]:
        """Returns the input and target data of the Sample for training."""
        inputs = [self.get_mfwam_data()]
        if self.include_arpege:
            inputs.append(self.get_arpege_data())
        if self.include_forcings:
            inputs.append(self.get_forcings_data())
        x = NamedTensor.concat(inputs)
        y = self.get_ww3_data()
        return x, y

    def is_valid(self) -> bool:
        return all(
            [
                self.path_mfwam.exists(),
                self.path_arpege.exists(),
                self.path_ww3.exists(),
            ]
        )

    def save(self) -> None:
        """Saves sample to the dataset folder on the disk"""
        x, y = self.get_x_y()
        nt = NamedTensor.concat([x, y])
        dict_array = {name: nt[name].numpy() for name in nt.feature_names}
        np.savez_compressed(self.save_path, **dict_array)

    def load(self) -> tuple[NamedTensor, NamedTensor]:
        """Loads the input and target data from the prepared dataset folder."""
        npz = np.load(self.save_path)
        x_tensors, y_tensors = [], []
        x_names, y_names = [], []
        for name in npz.files:
            if "ww3" in name:
                y_tensors.append(Tensor(npz[name]))
                y_names.append(name)
            else:
                x_tensors.append(Tensor(npz[name]))
                x_names.append(name)
        x = NamedTensor(
            torch.concat(x_tensors),
            names=["features", "lat", "lon"],
            feature_names=x_names,
        )
        y = NamedTensor(
            torch.concat(y_tensors),
            names=["features", "lat", "lon"],
            feature_names=y_names,
        )
        return x, y


if __name__ == "__main__":
    from ww3.plots import plot_sample

    s = Sample(
        mfwam_run_date=dt.datetime(2023, 12, 30, 6),
        mfwam_leadtime=dt.timedelta(hours=4),
        downscaling_stride=2,
        grid_name="BRETAGNE0002",
        wave_params=[
            "cos_mdps",
            "sin_mdps",
            "cos_mdww",
            "sin_mdww",
            "mpps",
            "mpww",
            "shps",
            "shww",
            "swh",
        ],
        include_arpege=True,
        include_forcings=True,
    )
    print(s)
    print("Is valid : ", s.is_valid())
    x, y = s.get_x_y()
    print(x)
    print(y)
    title = f"{s.mfwam_run_date} + {s.mfwam_leadtime}"
    fig = plot_sample(x, y, None, title, grid_name=s.grid_name)
    fig.savefig("test_sample.png")
