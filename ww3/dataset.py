import datetime as dt
from pathlib import Path
from typing import Literal, TypedDict

import torch
import xarray as xr
from mfai.pytorch.namedtensor import NamedTensor
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from ww3.sample import Sample
from ww3.settings import FORMATSTR, SCRATCH_PATH, GridType, WaveParamType


class Item(TypedDict):
    """Represents an input / target pair used during model training"""

    inputs: NamedTensor
    targets: NamedTensor
    leadtimes: torch.Tensor
    dates: dt.datetime | list[dt.datetime]


class WW3Dataset(Dataset):
    """Dataset used by the Ligthning DataModule to load train, valid and test data."""

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        pct_in_train: float = 0.7,
        transforms_list: list[torch.nn.Module] = [],
        max_leadtime: int = 48,  # in hours
        grids_list: list[GridType] = ["BRETAGNE0002"],
        wave_params: list[WaveParamType] = ["shww", "cos_mdps"],
        include_arpege: bool = False,
        include_forcings: bool = False,
        include_temporality: bool = False,
        downscaling_stride: int = 2,
        build_format: Literal["zarr", "npz"] = "npz",
    ):
        """
        Args:
            split (Literal["train", "val", "test"], optional): Split of the dataset. Defaults to "train".
            pct_in_train (float, optional): Percentage of the dataset in the train split. Defaults to 0.7.
            transforms_list (list[torch.nn.Module], optional): List of transforms to apply to the data. Defaults to [].
            max_leadtime (int, optional): Maximum leadtime of MFWAM forecasts used in the dataset. Defaults to 48.
            grids_list (list[GridType], optional): Domains used in the dataset. Defaults to ["BRETAGNE0002"].
            wave_params (list[WaveParamType], optional): Wave model parameters. Defaults to ["shww", "cos_mdps"].
            include_arpege (bool, optional): Whether to include Arpege data in the inputs. Defaults to False.
            include_forcings (bool, optional): Whether to include forcing data in the inputs. Defaults to False.
            include_temporality (bool, optional): Whether to include temporal data in the inputs. Defaults to False.
            downscaling_stride (int, optional): Inside [1;50]. Stride of pixel to keep between full resolution data (200m) and working resolution. Lower is a better resolution. Defaults to 2.
            build_format (Literal["zarr", "npz"]): Data format used to build the dataset. Defaults to "npz".
        """
        self.split = split
        self.pct_in_train = pct_in_train
        self.max_leadtime = max_leadtime
        self.transforms_list = transforms_list
        self.grids_list = grids_list
        self.wave_params = wave_params
        self.include_arpege = include_arpege
        self.include_forcings = include_forcings
        self.include_temporality = include_temporality
        self.downscaling_stride = downscaling_stride

        self.samples: list[Sample] = []
        for grid in self.grids_list:
            dataset_path = SCRATCH_PATH / f"dataset_{grid}_{self.downscaling_stride}"
            if (
                build_format == "npz"
                and dataset_path.exists()
                and len(list(dataset_path.glob("*.npz")))
            ):
                # Try to load the samples from the npz because it's quicker
                self.samples += self.build_from_npz(dataset_path)
            else:
                # Fall back on zarr files
                self.samples += self.build_from_zarr(grid)
        print(f"---> Loaded {len(self.samples)} samples for {self.split} split.")

    def build_from_npz(self, dataset_path: Path) -> list[Sample]:
        """Builds the list of samples from the npz dataset path.
        This is quicker than 'build_from_zarr'.
        """

        # Retrieve list of npz sample files
        npz_paths = sorted(list(dataset_path.glob("*.npz")))

        # Split dates between train and validation.
        # To avoid an overlap of valid times btw train and validation
        # we remove the first 4 days of samples from val set
        # -> 16 runs per day, and 102 leadtimes per run = 16 * 102
        nb_train_samples = int(len(npz_paths) * self.pct_in_train)
        nb_val_samples = len(npz_paths) - nb_train_samples - 16 * 102
        if self.split == "train":
            npz_paths = npz_paths[:nb_train_samples]
        else:
            npz_paths = npz_paths[-nb_val_samples:]
        print(f"First {self.split} sample : {npz_paths[0].stem}")
        print(f"Last {self.split} sample : {npz_paths[-1].stem}")

        # Builds samples from their npz files
        samples: list[Sample] = []
        for path in tqdm(npz_paths, desc="Building samples"):
            samples.append(
                Sample.from_npz_path(
                    path,
                    self.wave_params,
                    self.include_arpege,
                    self.include_forcings,
                )
            )
        return samples

    def build_from_zarr(self, grid: str) -> list[Sample]:
        """Builds the list of samples from the zarr files.
        This is slower than 'build_from_npz'.
        """

        # Retrieve list of MFWAM runs
        folder = SCRATCH_PATH / "mfwam" / grid / "zarr"
        dates_paths = sorted(list(folder.glob("*.zarr")))

        # Split dates btw train and validation:
        # To avoid an overlap of valid times btw train and validation
        # We remove the first 16 runs = 4 days of mfwam runs from val set
        nb_train_dates = int(len(dates_paths) * self.pct_in_train)
        nb_val_dates = len(dates_paths) - nb_train_dates - 16
        print(f"MFWAM run dates for grid {grid}:")
        print(f"""
+ {'Split':^10} | {'Dates':^9} +
+{'':-^12}+{'':-^11}+
| {'train':<10} | {nb_train_dates:>9} |
| {'val = test':<10} | {nb_val_dates:>9} |
+{'':-^12}+{'':-^11}+
""")
        if self.split == "train":
            dates_paths = dates_paths[:nb_train_dates]
        else:
            dates_paths = dates_paths[-nb_val_dates:]

        print(f"First MFWAM {self.split} run: {dates_paths[0].stem}")
        print(f"Last MFWAM {self.split} run: {dates_paths[-1].stem}")

        # Create samples by looping through MFWAM run dates and leadtimes
        samples = []
        for mfwam_path in tqdm(dates_paths, desc="Building samples"):
            mfwam_run_date = dt.datetime.strptime(mfwam_path.stem, FORMATSTR)
            ds = xr.open_zarr(mfwam_path, decode_timedelta=False)
            for leadtime in ds.step.values:
                if leadtime > self.max_leadtime:
                    continue
                sample = Sample(
                    mfwam_run_date=mfwam_run_date,
                    mfwam_leadtime=dt.timedelta(hours=int(leadtime)),
                    downscaling_stride=self.downscaling_stride,
                    grid_name=grid,
                    wave_params=self.wave_params,
                    include_arpege=self.include_arpege,
                    include_forcings=self.include_forcings,
                )
                if sample.is_valid():
                    samples.append(sample)
        return samples

    def transforms(
        self, x: NamedTensor, y: NamedTensor
    ) -> tuple[NamedTensor, NamedTensor]:
        """Applies transformations in sequence to inputs and targets"""
        for transform in self.transforms_list:
            x, y = transform(x, y)
        return x, y

    def undo_transforms(
        self, x: NamedTensor, y: NamedTensor
    ) -> tuple[NamedTensor, NamedTensor]:
        """Reverses transformations in sequence to inputs and targets"""
        for transform in self.transforms_list:
            x, y = transform.undo(x, y)
        return x, y

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Item:
        sample = self.samples[idx]
        if sample.save_path.exists():
            # if possible, we directly load the data at the right dowscaling_stride
            x, y = sample.load()
        else:
            x, y = sample.get_x_y()
        x, y = self.transforms(x, y)
        return {
            "inputs": x,
            "targets": y,
            "leadtimes": sample.get_leadtime_data(),
            "dates": sample.ww3_analysis_date,
        }


if __name__ == "__main__":
    from ww3.plots import plot_sample
    from ww3.transforms import NaNToNum, Normalize

    dataset = WW3Dataset(
        "train",
        downscaling_stride=25,
        transforms_list=[Normalize(), NaNToNum(downscaling_stride=25)],
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
        include_temporality=True,
    )
    x, y, date = dataset[1]
    print(x)
    print(y)
    print(date)
    for i in trange(10, desc="Plotting samples"):
        idx = i * 42
        s = dataset.samples[idx]
        x, y, date = dataset[idx]

        title = f"{s.mfwam_run_date} +{s.mfwam_leadtime}"
        fig = plot_sample(x, y, None, title, s.grid_name)
        fig.savefig(f"dataset_{idx}.png")
