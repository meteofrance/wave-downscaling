from typing import Literal

import torch
from lightning import LightningDataModule
from mfai.pytorch.namedtensor import NamedTensor
from torch.utils.data import DataLoader
from ww3.dataset import Item, WW3Dataset
from ww3.settings import GridType, WaveParamType


class WW3DataModule(LightningDataModule):
    """LightningDataModule for the WW3 dataset.
    Responsabilities:
    - instantiate fit/val/test datasets
    - package datasets into a DataLoarder class
    """

    train_dataset: WW3Dataset | None = None  # Set at setup
    val_dataset: WW3Dataset | None = None
    test_dataset: WW3Dataset | None = None

    def __init__(
        self,
        batch_size: int = 2,
        pct_in_train: float = 0.7,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        transforms_list: list[torch.nn.Module] = [],
        max_leadtime: int = 48,  # in hours
        grids_list: list[GridType] = ["BRETAGNE0002"],
        wave_params: list[WaveParamType] = ["shww", "cos_mdps"],
        include_arpege: bool = False,
        include_forcings: bool = False,
        include_temporality: bool = False,
        downscaling_stride: int = 2,
    ) -> None:
        """
        Args:
            batch_size: the batch size
            pct_in_train: percentage of the data included in the train dataset, test and valid dataset share the remainder. Defaults to 0.7.
            num_workers: number of processes used to load data from disk. Defaults to 1.
            prefetch_factor : Number of batches loaded in advance by each worker. Defaults to 2.
            transforms_list (list[torch.nn.Module], optional): List of transforms to apply to the data. Defaults to [].
            max_leadtime (int, optional): Maximum leadtime of MFWAM forecasts used in the dataset. Defaults to 48.
            grids_list (list[GridType], optional): Domains used in the dataset. Defaults to ["BRETAGNE0002"].
            wave_params (list[WaveParamType], optional): Wave model parameters. Defaults to ["shww", "cos_mdps"].
            include_arpege (bool, optional): Whether to include Arpege data in the inputs. Defaults to False.
            include_forcings (bool, optional): Whether to include forcing data in the inputs. Defaults to False.
            include_temporality (bool, optional): Whether to include temporality data in the inputs. Defaults to False.
            downscaling_stride (int, optional): Inside [1;50]. Stride of pixel to keep between full resolution data (200m) and working resolution. Lower is a better resolution. Defaults to 2.
        """
        super().__init__()
        self.batch_size = batch_size
        self.pct_in_train = pct_in_train
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.transforms_list = transforms_list
        self.max_leadtime = max_leadtime
        self.grids_list = grids_list
        self.wave_params = wave_params
        self.include_arpege = include_arpege
        self.include_forcings = include_forcings
        self.include_temporality = include_temporality
        self.downscaling_stride = downscaling_stride

        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": self.collate_batch,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "prefetch_factor": self.prefetch_factor,
        }

    def setup(self, stage: Literal["fit", "val", "validate", "test"]) -> None:  # type: ignore
        """Called by lightning, at the start of a stage.

        Args:
            stage: either 'fit', 'val', 'validate' or 'test'.
        """
        kwargs = {
            "pct_in_train": self.pct_in_train,
            "transforms_list": self.transforms_list,
            "max_leadtime": self.max_leadtime,
            "grids_list": self.grids_list,
            "wave_params": self.wave_params,
            "include_arpege": self.include_arpege,
            "include_forcings": self.include_forcings,
            "include_temporality": self.include_temporality,
            "downscaling_stride": self.downscaling_stride,
        }
        if stage == "fit":
            self.train_dataset = (
                WW3Dataset("train", **kwargs)
                if self.train_dataset is None
                else self.train_dataset
            )
        elif stage == "test":
            self.test_dataset = (
                WW3Dataset("test", **kwargs)
                if self.test_dataset is None
                else self.test_dataset
            )
        elif stage in ["fit", "val", "validate"]:
            self.val_dataset = (
                WW3Dataset("val", **kwargs)
                if self.val_dataset is None
                else self.val_dataset
            )
        else:
            raise ValueError(
                f"BMRDatamodule.setup():\n\tparameter stage should be either 'fit', 'val', 'validate' or 'test', got '{stage}'."
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("val")
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_kwargs)

    def collate_batch(
        self,
        batch: list[Item],
    ) -> Item:
        """Collate a batch of NamedTensor data."""

        inputs = NamedTensor.collate_fn([item["inputs"] for item in batch])
        targets = NamedTensor.collate_fn([item["targets"] for item in batch])
        leadtimes = torch.concat([item["leadtimes"] for item in batch])
        dates = [item["dates"] for item in batch]

        return {
            "inputs": inputs,
            "targets": targets,
            "leadtimes": leadtimes,
            "dates": dates,
        }


if __name__ == "__main__":
    dm = WW3DataModule()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    x, y = batch
    print(x)
    print(y)
