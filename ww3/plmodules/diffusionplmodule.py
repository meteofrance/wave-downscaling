from pathlib import Path
from typing import Any, Literal

import diffusers
import torch
from diffusers.training_utils import EMAModel
from mfai.pytorch.namedtensor import NamedTensor
from mfai.pytorch.padding import pad_batch, undo_padding
from torch import Tensor

from ww3.dataset import Item
from ww3.plmodules.baseplmodule import WW3BaseLightningModule
from ww3.settings import WaveParamType
from ww3.transforms import Normalize


class DiffusionLightningModule(WW3BaseLightningModule):
    """https://arxiv.org/pdf/2006.11239"""

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        wave_params: list[WaveParamType],
        learning_rate: float = 0.0001,
        min_learning_rate: float = 0.0,
        lr_scheduler_interval: Literal[False, "step", "epoch"] = False,
        use_ema: bool = True,
    ) -> None:
        super().__init__(
            model,
            loss,
            wave_params,
            learning_rate,
            min_learning_rate,
            lr_scheduler_interval,
        )

        # EMA model (exponential moving average)
        # The standard model searches for directions, the EMA model takes the average of directions
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(
                self.model.parameters(), decay=0.9999, use_ema_warmup=True, power=3 / 4
            )

        # a scheduler defines how to iteratively add noise to an image or how to update a sample based on a model’s output
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            prediction_type="v_prediction",
            beta_schedule="squaredcos_cap_v2",
            rescale_betas_zero_snr=True,
        )
        # Test scheduler: we're doing 20 steps instead of 1000 (faster)
        self.test_scheduler = diffusers.schedulers.DDPMScheduler.from_config(
            self.scheduler.config,
        )
        self.test_scheduler.set_timesteps(20)

        # Setup Normalizations
        self.normalization_01 = Normalize()
        self.normalization_11 = Normalize(interval=[-1, 1])

        self.save_hyperparameters()

        # Example input used to make the model summary that is printed to the console and logged to tensorboard.
        input_shape = (
            2,
            model.config.in_channels - model.config.out_channels,
            model.sample_size[0],
            model.sample_size[1],
        )
        feature_names = []
        for i in range(model.config.in_channels - model.config.out_channels):
            name = f"mfwam_{i}" if i < model.out_channels else f"f_{i}"
            feature_names.append(name)
        self.example_input_array = (
            NamedTensor(
                torch.zeros(input_shape),
                names=["batch", "features", "lat", "lon"],
                feature_names=feature_names,
            ),
            torch.zeros((2,), dtype=torch.long),
        )

    ########################################################################################
    #                                      SETUP                                           #
    ########################################################################################

    def on_fit_start(self):
        if self.use_ema:
            self.ema_model.to(self.device)

    def on_save_checkpoint(self, checkpoint):
        # add EMA model to load dictionary
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # load EMA model
        if self.use_ema and "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])

    ########################################################################################
    #                                      SHARED STEPS                                    #
    ########################################################################################

    def forward(self, inputs: NamedTensor, leadtime: Tensor) -> NamedTensor:
        """Runs data through the model. Separate from training step."""
        # The EMA weights are temporarily transferred to the main model
        # Save the weights of the current model and load the weights of the EMA
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())

        self.test_scheduler.set_timesteps(20)

        # Gaussien noise
        prev_sample = torch.randn_like(
            inputs.tensor[:, : self.model.config.out_channels]
        )

        # Denoising
        for t in self.test_scheduler.timesteps:
            # Ensures interchangeability with schedulers that need to scale the denoising model input depending on the current timestep
            prev_sample = self.test_scheduler.scale_model_input(prev_sample, t)

            # Noise and conditioning concatenation
            model_input = torch.cat([prev_sample, inputs.tensor], dim=1)
            # Pad the input
            pad_model_input = pad_batch(
                model_input,
                new_shape=(self.model.sample_size[0], self.model.sample_size[1]),
            )
            # Noise prediction
            pad_residual = self.model(pad_model_input, t, class_labels=leadtime).sample
            residual = undo_padding(pad_residual, inputs.tensor.shape[2:])

            # Predict the sample from the previous timestep (x_{t-1})
            prev_sample = self.test_scheduler.step(residual, t, prev_sample).prev_sample

        # We put the training weights back on
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())

        return NamedTensor(
            prev_sample,
            names=inputs.names,
            feature_names=["ww3_" + param for param in self.wave_params],
        )

    def _shared_forward_step(
        self, x: NamedTensor, y: NamedTensor, leadtime: Tensor, mode: str = "train"
    ) -> tuple[Tensor, Tensor, Tensor, Any]:
        """Computes forward pass and loss for a batch.
        Step shared by training, validation and test steps
        """
        # Gaussien noise
        noise = torch.randn_like(y.tensor)
        # Select a random step of denoising between 0 and num_train_timesteps
        if mode == "train":
            steps = torch.randint(
                self.scheduler.config.num_train_timesteps,
                (y.tensor.size(0),),
                device=self.device,
            )
        else:
            # For validation, it is required that the target be 90% noisy in order to evaluate the model's learning.
            steps = (
                torch.ones((y.tensor.size(0),), device=self.device, dtype=torch.long)
                * self.scheduler.timesteps[-1]
                * 0.9
            )
        # Add noise to the original samples according to the noise magnitude at each timestep
        noisy_y = self.scheduler.add_noise(y.tensor, noise, steps)

        # Concatenate noisy target and conditional features
        model_input = torch.cat([noisy_y, x.tensor], dim=1)

        # Predict the noise for the denoising step
        pad_model_input = pad_batch(
            model_input,
            new_shape=(self.model.sample_size[0], self.model.sample_size[1]),
        )
        pad_residual = self.model(pad_model_input, steps, class_labels=leadtime).sample
        residual = undo_padding(pad_residual, y.tensor.shape[2:])

        mask = x["landsea_mask"]

        target = self.scheduler.get_velocity(y.tensor, noise, steps)
        loss = self.loss(residual * mask, target * mask)

        return residual, steps, noisy_y, loss

    ########################################################################################
    #                                      TRAIN STEPS                                     #
    ########################################################################################

    def training_step(self, batch: Item, batch_idx: int) -> Any:
        x = batch["inputs"]
        y = batch["targets"]
        leadtime = batch["leadtimes"]

        _, _, _, loss = self._shared_forward_step(x, y, leadtime)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Item, batch_idx: int) -> None:
        # update EMA model
        if self.use_ema:
            self.ema_model.step(self.model.parameters())

    ########################################################################################
    #                                      VALID STEPS                                     #
    ########################################################################################

    def on_validation_epoch_start(self) -> None:
        # The EMA weights are temporarily transferred to the main model
        # Save the weights of the current model and load the weights of the EMA
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())

    def validation_step(self, batch: Item, batch_idx: int) -> Any:
        x = batch["inputs"]
        y = batch["targets"]
        dates = batch["dates"]
        leadtimes = batch["leadtimes"]

        residual, steps, noisy_y, loss = self._shared_forward_step(
            x, y, leadtimes, mode="val"
        )
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)

        # Predict the denoised sample (x_{0}) based on the model output from the current timestep.
        y_hat_tensor = self.scheduler.step(
            residual, int(steps[0]), noisy_y
        ).pred_original_sample

        y_hat = NamedTensor(
            y_hat_tensor,
            names=y.names,
            feature_names=y.feature_names,
        )

        # Undo the normalization to compute the metrics and the plots
        # Normalize to 0 1
        x_denormalized, y_denormalized = self.normalization_11.undo(x, y)
        _, y_hat_denormalized = self.normalization_11.undo(x, y_hat)

        self.metrics.update(y_hat_denormalized.tensor, y_denormalized.tensor)

        x_01, y_01 = self.normalization_01(x_denormalized, y_denormalized)
        _, y_hat_01 = self.normalization_01(x_denormalized, y_hat_denormalized)

        self.metric_ssim(y_hat_01, y_01, x_01["landsea_mask"])
        _, y_hat = self.trainer.datamodule.val_dataset.undo_transforms(x, y_hat)
        x, y = self.trainer.datamodule.val_dataset.undo_transforms(x, y)

        self.metric_scatter_index(y_hat, dates)
        self._shared_plot_step(batch_idx, x, y, y_hat, mode="val")

        return loss

    def on_validation_epoch_end(self) -> None:
        # We put the training weights back on
        self.ema_model.restore(self.model.parameters())
        return super().on_validation_epoch_end()

    ########################################################################################
    #                                      TEST STEPS                                     #
    ########################################################################################

    def test_step(self, batch: Item, batch_idx: int) -> Any:
        x = batch["inputs"]
        y = batch["targets"]
        dates = batch["dates"]
        leadtimes = batch["leadtimes"]

        y_hat = self(x, leadtimes)

        # Undo the normalization to compute the metrics and the plots
        # Normalize to 0 1
        x_denormalized, y_denormalized = self.normalization_11.undo(x, y)
        _, y_hat_denormalized = self.normalization_11.undo(x, y_hat)

        self.metrics.update(y_hat_denormalized.tensor, y_denormalized.tensor)

        x_01, y_01 = self.normalization_01(x_denormalized, y_denormalized)
        _, y_hat_01 = self.normalization_01(x_denormalized, y_hat_denormalized)

        self.metric_ssim(y_hat_01, y_01, x_01["landsea_mask"])

        _, y_hat = self.trainer.datamodule.test_dataset.undo_transforms(x, y_hat)
        x, y = self.trainer.datamodule.test_dataset.undo_transforms(x, y)

        self.metric_scatter_index(y_hat, dates)
        self._shared_plot_step(batch_idx, x, y, y_hat, mode="test")


def load_model(last_ckpt: Path) -> DiffusionLightningModule:
    model = DiffusionLightningModule.load_from_checkpoint(last_ckpt)
    model.eval()  # disable randomness, dropout, etc...
    return model
