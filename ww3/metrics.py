import datetime as dt
from pathlib import Path

import pandas as pd
import torch
from einops import rearrange
from mfai.pytorch.namedtensor import NamedTensor
from torchmetrics import Metric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from ww3.settings import GRIDS, OBS_PATH


class SSIM(StructuralSimilarityIndexMeasure):
    """Structural Similarity Index Measure
    https://arxiv.org/pdf/2006.13846
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feature_names = None

    def update(
        self,
        preds: NamedTensor,
        target: NamedTensor,
        mask: torch.Tensor,
    ) -> None:
        """Apply mask and change order of features before updating with StructuralSimilarityIndexMeasure"""

        if self.feature_names is None:
            self.feature_names = preds.feature_names

        preds = preds.tensor * mask
        target = target.tensor * mask

        B, F, _, _ = preds.shape

        # Features in third dimension
        preds = rearrange(preds, "b f h w -> (b f) 1 h w")
        target = rearrange(target, "b f h w -> (b f) 1 h w")
        super().update(preds, target)

        # if no reduction put the features back in the last dimension
        if self.reduction == "none":
            self.similarity[-1] = self.similarity[-1].reshape(B, F)

    def compute(self, prefix: str = "val") -> dict:
        """Compute the SSIM.
        return a dict with the ssim for each feature at each timestep if no reduction
        else return a dict with the ssim over all the feature and timestep.
        """
        ssim_result = super().compute()
        if self.reduction == "none":
            ssim_result = torch.mean(ssim_result, dim=0)

            metric_log_dict = {
                f"{prefix}_ssim/{name}": ssim_result[i]
                for i, name in enumerate(self.feature_names)
            }
        else:
            metric_log_dict = {f"{prefix}_ssim": ssim_result}
        return metric_log_dict


class NormalizedStdDiff(Metric):
    """Normalized Standard Deviation of Differences (Scatter Index).
    100.0*(((diff**2).mean() - bias**2))**0.5/obs.mean()
    """

    full_state_update = False

    def __init__(
        self, grid_name: str = "BRETAGNE0002", path_obs: Path = OBS_PATH
    ) -> None:
        super().__init__()
        # add torchmetrics state for lightning
        self.add_state(
            "sum_values", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

        self.add_state(
            "count", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )

        # load the csv with the observations
        df_obs = pd.read_csv(path_obs)
        obs_datetime = pd.to_datetime(df_obs["time"])
        df_obs["time"] = obs_datetime.apply(
            lambda x: x.replace(minute=0, second=0, microsecond=0)
            if x.minute < 30
            else (x + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        ).dt.to_pydatetime()
        self.df_obs = df_obs

        # Grid info
        H, W = GRIDS[grid_name]["downscale_size"]
        lat_max, lat_min, lon_min, lon_max = GRIDS[grid_name]["extent"]
        self.H, self.W = H, W
        self.lat_max, self.lat_min = lat_max, lat_min
        self.lon_max, self.lon_min = lon_max, lon_min

        # Compute lat/lon indices
        df_obs["idx_lat"] = (
            (lat_max - df_obs["latitude"]) * (H - 1) / (lat_max - lat_min)
        ).astype(int)
        df_obs["idx_lon"] = (
            (lon_min - df_obs["longitude"]) * (W - 1) / (lon_min - lon_max)
        ).astype(int)

    def update(self, preds: NamedTensor, dates: list) -> None:
        """
        Compute the Scatter Index for each predictions.
        preds.tensor shape : [B, T, H, W, 1]
        dates : dates of the predictions [B, T]
        """
        device = preds.tensor.device
        preds_tensor = preds["ww3_swh"]
        batch_size = preds_tensor.shape[0]

        for b in range(batch_size):
            subset_obs = self.df_obs[
                self.df_obs["time"] == dates[b].replace(tzinfo=dt.timezone.utc)
            ]
            if subset_obs.empty:
                continue

            idx_lat = torch.tensor(subset_obs["idx_lat"].values, device=device)
            idx_lon = torch.tensor(subset_obs["idx_lon"].values, device=device)
            obs_values = torch.tensor(subset_obs["value"].values, device=device)

            # Gather predictions vectorized
            pred_values = preds_tensor[b, 0, idx_lat, idx_lon].float()

            if obs_values.numel() == 0:
                continue

            diff = pred_values - obs_values
            mean_diff2 = torch.nanmean(diff**2)
            mean_diff = torch.nanmean(diff)
            std_diff = torch.sqrt(mean_diff2 - mean_diff**2)
            mean_obs = obs_values.mean()
            result = 100 * std_diff / mean_obs

            mask = ~torch.isnan(result)
            results = torch.nan_to_num(result, nan=0.0)
            self.sum_values += results
            self.count += mask.float()  # 0 or 1

    def compute(self, prefix: str = "val") -> dict:
        """
        Return a dict with the Scatter Index per timestep
        """
        # avoid divide-by-zero
        mean_si = self.sum_values / torch.clamp(self.count, min=1)

        # if no value replace by nan
        mean_si = torch.where(self.count > 0, mean_si, torch.tensor(float("nan")))

        metric_log_dict = {f"{prefix}_scatter_index/ww3_swh": mean_si}
        return metric_log_dict


class PerChannelMAE(Metric):
    """Compute MAE for a specific channel."""

    full_state_update = False

    def __init__(self, channel_idx: int) -> None:
        super().__init__()
        self.channel_idx = channel_idx

        # Sum of error map
        self.add_state(
            "sum_abs_error_map", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # Only keep the channel of interest
        p = preds[:, self.channel_idx, :, :]
        t = target[:, self.channel_idx, :, :]

        if self.sum_abs_error_map.ndim == 0:
            self.sum_abs_error_map = torch.zeros_like(p[0])

        self.sum_abs_error_map += torch.abs(p - t).sum(dim=0)
        self.total_images += p.shape[0]

    def compute(self) -> torch.Tensor:
        """Return scalar"""
        return self.sum_abs_error_map.sum() / (
            self.total_images * self.sum_abs_error_map.numel()
        )

    def compute_maps(self) -> torch.Tensor:
        """Return mean error map (H, W)"""
        return self.sum_abs_error_map / self.total_images
