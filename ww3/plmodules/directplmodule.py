from pathlib import Path
from typing import Any, Literal

import torch
from mfai.pytorch.losses.perceptual import PerceptualLoss
from mfai.pytorch.models.base import BaseModel
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor
from ww3.dataset import Item
from ww3.plmodules.baseplmodule import WW3BaseLightningModule
from ww3.settings import WaveParamType


class MSEPlusPerceptualLoss(torch.nn.Module):
    """Linear combination of MSE and Perceptual loss, with configurable coefficients.
    Loss = lambda_mse * MSE + lambda_perceptual * Perceptual.
    """

    def __init__(
        self,
        lambda_mse: float = 1,
        lambda_perceptual: float = 1,
        **perceptual_kwargs: Any,
    ):
        """Linear combination of MSE and Perceptual loss, with configurable coefficients.

        Args:
            lambda_mse (float, optional): Coefficient of the MSE loss. Defaults to 1.
            lambda_perceptual (float, optional): Coefficient of the Perceptual loss. Defaults to 1.
            **perceptual_kwargs (dict, optional): Optional arguments of the Perceptual loss.
        """
        super().__init__()
        self.perceptual_loss = PerceptualLoss(**perceptual_kwargs)
        self.mse_loss = torch.nn.MSELoss()
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(y_hat, y)
        perceptual = self.mse_loss(y_hat, y)
        return self.lambda_mse * mse + self.lambda_perceptual * perceptual


class WW3DirectLightningModule(WW3BaseLightningModule):
    def __init__(
        self,
        model: BaseModel,
        loss: torch.nn.Module,
        wave_params: list[WaveParamType],
        residual_prediction: bool = False,
        learning_rate: float = 0.0001,
        min_learning_rate: float = 0.0,
        lr_scheduler_interval: Literal[False, "step", "epoch"] = False,
    ) -> None:
        super().__init__(
            model,
            loss,
            wave_params,
            learning_rate,
            min_learning_rate,
            lr_scheduler_interval,
        )

        self.residual_prediction = residual_prediction
        self.save_hyperparameters()

        # Example input used to make the model summary that is printed to the console and logged to tensorboard.
        input_shape = (2, model.in_channels, model.input_shape[0], model.input_shape[1])
        feature_names = []
        for i in range(model.in_channels):
            name = f"mfwam_{i}" if i < model.out_channels else f"f_{i}"
            feature_names.append(name)
        self.example_input_array = NamedTensor(
            torch.zeros(input_shape),
            names=["batch", "features", "lat", "lon"],
            feature_names=feature_names,
        )

    ########################################################################################
    #                                      SHARED STEPS                                    #
    ########################################################################################

    def last_activation(self, y_hat: Tensor) -> Tensor:
        """Applies appropriate activation according to task."""
        if self.residual_prediction:
            return y_hat
        return torch.nn.functional.relu(y_hat)

    def make_residual_prediction(
        self, x: NamedTensor, output: Tensor, param_names: list[str]
    ) -> Tensor:
        """Makes a residual prediction from the output of the model.
        We want y_hat = x + model(x) = x + output, so here we sum x and output
        parameter by parameter.
        """
        list_tensors = []
        for i, param in enumerate(param_names):
            x_param_name = param.replace("ww3", "mfwam")
            list_tensors.append(x[x_param_name][:, 0] + output[:, i])
        return torch.stack(list_tensors, dim=1)

    def forward(self, inputs: NamedTensor) -> NamedTensor:
        """Runs data through the model. Separate from training step."""
        inputs_tensor = inputs.tensor
        feature_names = [
            name for name in inputs.feature_names if name.startswith("mfwam")
        ]
        feature_names = [name.replace("mfwam", "ww3") for name in feature_names]
        output = self.model(inputs_tensor)
        output = self.last_activation(output)
        if self.residual_prediction:
            # y_hat = x + model(x)
            y_hat_tensor = self.make_residual_prediction(inputs, output, feature_names)
        else:
            # y_hat = model(x)
            y_hat_tensor = output
        y_hat = NamedTensor(
            y_hat_tensor,
            names=inputs.names,
            feature_names=feature_names,
        )
        return y_hat

    def _shared_forward_step(
        self, x: NamedTensor, y: NamedTensor
    ) -> tuple[NamedTensor, Any]:
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps
        """
        output = self.model(x.tensor)
        output = self.last_activation(output)
        if self.residual_prediction:
            # y_hat = x + model(x)
            y_hat_tensor = self.make_residual_prediction(x, output, y.feature_names)
        else:
            # y_hat = model(x)
            y_hat_tensor = output
        if isinstance(self.loss, PerceptualLoss):
            y_hat_tensor = torch.clamp(y_hat_tensor, 0, 1)
        mask = x["landsea_mask"]
        loss = self.loss(y_hat_tensor * mask, y.tensor * mask)
        y_hat = NamedTensor(
            y_hat_tensor,
            names=y.names,
            feature_names=y.feature_names,
        )
        return y_hat, loss

    ########################################################################################
    #                                      TRAIN STEPS                                     #
    ########################################################################################

    def training_step(self, batch: Item, batch_idx: int) -> Any:
        x = batch["inputs"]
        y = batch["targets"]
        _, loss = self._shared_forward_step(x, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    ########################################################################################
    #                                      VALID STEPS                                     #
    ########################################################################################

    def validation_step(self, batch: Item, batch_idx: int) -> Any:
        x = batch["inputs"]
        y = batch["targets"]
        dates = batch["dates"]
        y_hat, loss = self._shared_forward_step(x, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        self.metric_ssim(y_hat, y, x["landsea_mask"])
        _, y = self.trainer.datamodule.val_dataset.undo_transforms(x, y)
        x, y_hat = self.trainer.datamodule.val_dataset.undo_transforms(x, y_hat)
        self.metrics.update(torch.nan_to_num(y_hat.tensor), torch.nan_to_num(y.tensor))
        self.metric_scatter_index(y_hat, dates)
        self._shared_plot_step(batch_idx, x, y, y_hat, mode="val")
        return loss

    ########################################################################################
    #                                      TEST STEPS                                     #
    ########################################################################################

    def test_step(self, batch: Item, batch_idx: int) -> Any:
        x = batch["inputs"]
        y = batch["targets"]
        dates = batch["dates"]
        y_hat, loss = self._shared_forward_step(x, y)

        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.metric_ssim(y_hat, y, x["landsea_mask"])
        _, y = self.trainer.datamodule.test_dataset.undo_transforms(x, y)
        x, y_hat = self.trainer.datamodule.test_dataset.undo_transforms(x, y_hat)
        self.metrics.update(torch.nan_to_num(y_hat.tensor), torch.nan_to_num(y.tensor))
        self.metric_scatter_index(y_hat, dates)
        self._shared_plot_step(batch_idx, x, y, y_hat, mode="test")


def load_model(last_ckpt: Path) -> WW3DirectLightningModule:
    model = WW3DirectLightningModule.load_from_checkpoint(last_ckpt)
    model.eval()  # disable randomness, dropout, etc...
    return model
