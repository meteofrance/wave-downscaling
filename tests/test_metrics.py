import datetime as dt
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import torch
from mfai.pytorch.namedtensor import NamedTensor

from ww3.metrics import SSIM, NormalizedStdDiff
from ww3.settings import GRIDS

# ------------------ FIXTURES -------------------


@pytest.fixture
def fake_obs_csv(tmp_path: Path) -> Path:
    """Create a fake CSV of WW3 observations"""
    now = pd.Timestamp("2023-01-01T18:00:00Z")
    df = pd.DataFrame(
        {
            "time": [
                now,
                now,
                now + pd.Timedelta(hours=1),
                now + pd.Timedelta(hours=1),
            ],
            "latitude": [48.5, 48.0, 47.5, 48.0],
            "longitude": [-5.0, -4.5, -5.5, -4.5],
            "value": [2.0, 3.0, 4.0, 1.0],
        }
    )
    path = tmp_path / "obs.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def fake_preds() -> Callable:
    """Returns a function that creates a prediction tensor with or without error."""

    def make_fake_preds(is_error: bool = False) -> NamedTensor:
        names = ["batch", "features", "height", "width"]
        features_names = ["ww3_swh"]
        H, W = GRIDS["BRETAGNE0002"]["downscale_size"]
        preds_tensor = torch.zeros((2, 1, H, W))
        preds_tensor[0, 0, 7, 2] = 2.0 + (0.1 if is_error else 0.0)
        preds_tensor[0, 0, 9, 4] = 3.0 + (0.2 if is_error else 0.0)
        preds_tensor[1, 0, 12, 0] = 4.0 + (0.2 if is_error else 0.0)
        preds_tensor[1, 0, 9, 4] = 1.0 + (0.1 if is_error else 0.0)
        # Invert the latitude as in the dataset
        return NamedTensor(preds_tensor, names, features_names)

    return make_fake_preds


@pytest.fixture
def fake_dates() -> list[list[dt.datetime]]:
    now = dt.datetime(year=2023, month=1, day=1, hour=18)
    return [now, now + dt.timedelta(hours=1)]


# ------------------ TESTS ---------------------


def test_ssim() -> None:
    names = ["batch", "features", "height", "width"]
    features_names = [f"feature_{i}" for i in range(7)]
    mask = torch.ones(1, 7, 16, 16)

    # Same inputs
    metric = SSIM(data_range=1.0)
    preds = NamedTensor(torch.rand(1, 7, 16, 16), names, features_names)
    target = preds.clone()
    metric.update(preds, target, mask)
    value = metric.compute()
    assert torch.allclose(
        value["val_ssim"], torch.tensor(1.0)
    ), "SSIM should be 1 when preds == target"

    # Different inputs
    metric = SSIM(data_range=1.0)
    preds = NamedTensor(torch.ones(1, 7, 16, 16), names, features_names)
    target = NamedTensor(1 - preds.tensor, names, features_names)
    metric.update(preds, target, mask)
    value = metric.compute()
    assert torch.allclose(
        value["val_ssim"], torch.tensor(0.0), atol=0.1
    ), "SSIM should be close to 0"

    # Without reduction
    metric = SSIM(data_range=1.0, reduction="none")
    preds = NamedTensor(torch.rand(2, 7, 16, 16), names, features_names)
    target = preds.clone()
    metric.update(preds, target, mask)
    value = metric.compute()
    assert len(value) == 7


@pytest.mark.parametrize(
    "is_error, expected_value", [(False, [0.0, 0.0]), (True, [2.0, 2.0])]
)
def test_normalized_std_diff(
    monkeypatch: pytest.MonkeyPatch,
    fake_obs_csv: Path,
    fake_preds: Callable,
    fake_dates: list[dt.datetime],
    is_error: bool,
    expected_value: list[float],
) -> None:
    # Temporarily modify the downscale_size value in GRIDS
    monkeypatch.setitem(GRIDS["BRETAGNE0002"], "downscale_size", [16, 16])

    # Same inputs
    metric = NormalizedStdDiff(path_obs=fake_obs_csv)
    metric.update(fake_preds(is_error), fake_dates)
    result = metric.compute()

    assert len(result) > 0

    # Check the type
    assert isinstance(result["val_scatter_index/ww3_swh"], torch.Tensor)

    # Compute the result
    expected = torch.tensor(expected_value)
    assert torch.isclose(result["val_scatter_index/ww3_swh"], expected[0], rtol=1e-5)
