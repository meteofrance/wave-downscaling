from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
import torchmetrics as tm
from lightning import LightningModule
from mfai.pytorch.models.base import BaseModel
from mfai.pytorch.namedtensor import NamedTensor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW

from ww3.dataset import Item
from ww3.metrics import SSIM, NormalizedStdDiff, PerChannelMAE
from ww3.plots import plot_error_map, plot_sample
from ww3.settings import WaveParamType


class WW3BaseLightningModule(LightningModule, ABC):
    def __init__(
        self,
        model: BaseModel,
        loss: torch.nn.Module,
        wave_params: list[WaveParamType],
        learning_rate: float = 0.0001,
        min_learning_rate: float = 0.0,
        lr_scheduler_interval: Literal[False, "step", "epoch"] = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.model = torch.compile(self.model)
        self.loss = loss
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.lr_scheduler_interval = lr_scheduler_interval
        self.wave_params = wave_params

        # Setup metrics
        self.metrics = self.get_metrics()
        self.metric_ssim = SSIM(reduction="none")
        self.metric_scatter_index = NormalizedStdDiff()

        self.save_hyperparameters()

    ########################################################################################
    #                                      SETUP                                           #
    ########################################################################################

    def get_metrics(self) -> tm.MetricCollection:
        """Defines the metrics that will be computed during train and valid steps."""
        metrics_dict = {
            f"mae/{name}": PerChannelMAE(channel_idx=i)
            for i, name in enumerate(self.wave_params)
        }
        metrics_dict["MSE"] = tm.MeanSquaredError(squared=False)
        metrics_dict["MAE"] = tm.MeanAbsoluteError()
        metrics_dict["MeanAbsolutePercentageError"] = tm.MeanAbsolutePercentageError()

        return tm.MetricCollection(metrics_dict)

    def configure_optimizers(self) -> AdamW | dict[str, Any]:
        """Lightning method to define optimizers and learning-rate schedulers used for optimization.
        If we try to define the scheduler directly in the lightning config file,
        the learning rate is set at zero during the whole first epoch.
        For more details about this method, please see:
        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler_interval:
            warmup_epochs = 100 if self.lr_scheduler_interval == "step" else 1
            num_batches = len(self.trainer.datamodule.train_dataloader())
            if self.trainer.max_steps > 0:
                max_steps_or_epochs = self.trainer.max_steps
            elif self.trainer.max_epochs:
                max_steps_or_epochs = self.trainer.max_epochs
                if (
                    self.lr_scheduler_interval == "step"
                ):  # Multiply epochs by number of batches
                    max_steps_or_epochs *= num_batches
            else:
                raise ValueError(
                    "Please set 'trainer.max_steps' or 'trainer.max_epochs' to use an LRScheduler."
                )
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_steps_or_epochs,
                eta_min=self.min_learning_rate,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.lr_scheduler_interval,
                    "frequency": 1,
                    "name": "lr",
                },
            }
        else:
            return optimizer

    ########################################################################################
    #                                      SHARED STEPS                                    #
    ########################################################################################

    def _shared_plot_step(
        self,
        batch_idx: int,
        x: NamedTensor,
        y: NamedTensor,
        y_hat: NamedTensor,
        mode: str,
    ) -> None:
        """Plots images on some batches and log them in tensorboard."""
        if self.logger is None:
            return
        interesting_batches = [0, 6, 12, 42, 66]
        if batch_idx not in interesting_batches:
            return
        fig = plot_sample(
            x.select_dim("batch", 0),
            y.select_dim("batch", 0),
            y_hat.select_dim("batch", 0),
            f"Sample {batch_idx}",
        )
        tb = self.logger.experiment  # type: ignore[reporteAttributeAccessIssue]
        tb.add_figure(f"{mode}_plots/val_figure_{batch_idx}", fig, self.current_epoch)

    ########################################################################################
    #                                      TRAIN STEPS                                     #
    ########################################################################################
    def on_train_start(self) -> None:
        if self.logger and self.logger.log_dir:
            print(
                f"Logs will be saved in \033[96m{self.logger.log_dir}\033[0m"
            )  # bright cyan

    @abstractmethod
    def training_step(self, batch: Item, batch_idx: int) -> Any:
        pass

    ########################################################################################
    #                                      VALID STEPS                                     #
    ########################################################################################

    @abstractmethod
    def validation_step(self, batch: Item, batch_idx: int) -> Any:
        pass

    def on_validation_epoch_end(self) -> None:
        if self.logger is None:
            return
        self.log_dict(
            self.metrics, logger=True if self.logger else False, sync_dist=True
        )
        self.log_dict(self.metric_ssim.compute(), sync_dist=True)
        self.log_dict(self.metric_scatter_index.compute(), sync_dist=True)

    ########################################################################################
    #                                      TEST STEPS                                     #
    ########################################################################################

    def on_test_epoch_end(self) -> None:
        if self.logger is None:
            return
        self.log_dict(
            self.metrics, logger=True if self.logger else False, sync_dist=True
        )
        self.log_dict(self.metric_ssim.compute(), sync_dist=True)
        self.log_dict(self.metric_scatter_index.compute(), sync_dist=True)

        error_maps_tensor = [
            self.metrics[f"mae/{name}"].compute_maps() for name in self.wave_params
        ]
        error_maps = NamedTensor(
            torch.stack(error_maps_tensor, dim=0),
            names=["features", "lat", "lon"],
            feature_names=self.wave_params,
        )

        fig = plot_error_map(error_maps)

        tb = self.logger.experiment
        tb.add_figure("test_mae_map_plots", fig, self.current_epoch)
