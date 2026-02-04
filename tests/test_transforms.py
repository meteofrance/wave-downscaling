from pathlib import Path

import numpy as np
import pytest
import torch
from mfai.pytorch.namedtensor import NamedTensor
from ww3.transforms import NaNToNum, Normalize


@pytest.fixture
def x_named_tensor() -> NamedTensor:
    """Fixture used by the transform tests that returns fake input data."""
    feature = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]])
    landsea_mask = torch.tensor([[0, 1], [1, 0]])
    tensor = torch.stack([feature, landsea_mask], 0)
    return NamedTensor(
        tensor=tensor,
        names=["features", "lat", "lon"],
        feature_names=["feature0", "landsea_mask"],
        feature_dim_name="features",
    )


@pytest.fixture
def y_named_tensor() -> NamedTensor:
    """Fixture used by the transform tests that returns fake target data."""
    feature1 = torch.tensor([[float("nan"), 1.0], [2.0, float("nan")]])
    feature2 = torch.tensor([[float("nan"), 4.0], [3.0, float("nan")]])
    tensor = torch.stack([feature1, feature2], 0)
    return NamedTensor(
        tensor=tensor,
        names=["features", "lat", "lon"],
        feature_names=["feature1", "feature2"],
        feature_dim_name="features",
    )


@pytest.fixture
def fake_mfwam_mask():
    block_size = 25
    mask = np.block(
        [
            [np.zeros((block_size, block_size)), np.ones((block_size, block_size))],
            [np.ones((block_size, block_size)), np.zeros((block_size, block_size))],
        ]
    )
    return mask


def test_nan_to_num(
    monkeypatch: pytest.MonkeyPatch, x_named_tensor, y_named_tensor, fake_mfwam_mask
):
    # remplace the value of mfwam_mask
    monkeypatch.setattr(
        "ww3.transforms.np.load", lambda path: {"arr_0": fake_mfwam_mask}
    )

    transform = NaNToNum()
    x_processed, y_processed = transform(x_named_tensor, y_named_tensor)
    expected_x = torch.tensor(
        [[[0.0, 1.0], [2.0, 0.0]], [[0, 1], [1, 0]]], dtype=torch.float32
    )
    expected_y = torch.tensor(
        [[[0.0, 1.0], [2.0, 0.0]], [[0.0, 4.0], [3.0, 0.0]]], dtype=torch.float32
    )
    assert torch.allclose(x_processed.tensor, expected_x)
    assert torch.allclose(y_processed.tensor, expected_y)

    x_reversed, y_reversed = transform.undo(x_processed, y_processed)
    assert torch.allclose(
        torch.isnan(x_reversed.tensor), torch.isnan(x_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(x_reversed.tensor), torch.nan_to_num(x_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.isnan(y_reversed.tensor), torch.isnan(y_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(y_reversed.tensor), torch.nan_to_num(y_named_tensor.tensor)
    )


@pytest.fixture
def stats_file_path(tmp_path: Path) -> Path:
    """Creates a fake file of WW3 data statistics"""
    stats_dict = {
        "feature0": {"min": 1.0, "max": 5.0},
        "feature1": {"min": 0.0, "max": 2.0},
        "feature2": {"min": -1.0, "max": 3.0},
        "landsea_mask": {"min": 0, "max": 1},
    }
    path_file = tmp_path / "stats.pt"
    torch.save(stats_dict, path_file)
    return path_file


expected_x_01 = torch.tensor(
    [[[float("nan"), 0], [0.25, float("nan")]], [[0, 1], [1, 0]]],
    dtype=torch.float32,
)
expected_y_01 = torch.tensor(
    [
        [[float("nan"), 0.5], [1.0, float("nan")]],
        [[float("nan"), 1.25], [1.0, float("nan")]],
    ],
    dtype=torch.float32,
)
expected_x_11 = torch.tensor(
    [[[float("nan"), -1.0], [-0.5, float("nan")]], [[-1, 1], [1, -1]]],
    dtype=torch.float32,
)
expected_y_11 = torch.tensor(
    [
        [[float("nan"), 0.0], [1.0, float("nan")]],
        [[float("nan"), 1.50], [1.0, float("nan")]],
    ],
    dtype=torch.float32,
)


@pytest.mark.parametrize(
    "interval, expected_x, expected_y",
    [([0, 1], expected_x_01, expected_y_01), ([-1, 1], expected_x_11, expected_y_11)],
)
def test_normalize(
    x_named_tensor, y_named_tensor, stats_file_path, interval, expected_x, expected_y
):
    transform = Normalize(stats_file_path=stats_file_path, interval=interval)
    x_processed, y_processed = transform(x_named_tensor, y_named_tensor)

    print(x_processed.tensor)
    print(y_processed.tensor)
    assert torch.allclose(
        torch.nan_to_num(x_processed.tensor), torch.nan_to_num(expected_x)
    )
    assert torch.allclose(
        torch.nan_to_num(y_processed.tensor), torch.nan_to_num(expected_y)
    )

    x_reversed, y_reversed = transform.undo(x_processed, y_processed)
    assert torch.allclose(
        torch.isnan(x_reversed.tensor), torch.isnan(x_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(x_reversed.tensor), torch.nan_to_num(x_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.isnan(y_reversed.tensor), torch.isnan(y_named_tensor.tensor)
    )
    assert torch.allclose(
        torch.nan_to_num(y_reversed.tensor), torch.nan_to_num(y_named_tensor.tensor)
    )
