"""Computes statistics (min/max) on the dataset.

Usage: 4_compute_stats.py [OPTIONS]

Args:
     force (bool, optional): Whether to overwrite an existing stats file. Defaults to False.
     num_workers (int, optional): Number of workers to parallelize the job. Defaults to 10.
"""

from collections import defaultdict

import joblib
import numpy as np
import torch
import tqdm
from mfai.pytorch.namedtensor import NamedTensor
from typer import Typer
from ww3.dataset import WW3Dataset
from ww3.sample import Sample
from ww3.settings import STATS_PATH

app = Typer()


def compute_stats(sample: Sample) -> dict[str, dict[str, float]]:
    """Computes min and max of all parameters for one sample."""
    x, y = sample.get_x_y()
    nt = NamedTensor.concat([x, y])
    stats = defaultdict(lambda: defaultdict(float))
    for param in nt.feature_names:
        tensor = nt[param]
        stats[param]["min"] = np.nanmin(tensor)
        stats[param]["max"] = np.nanmax(tensor)
    return stats


@app.command()
def run(force: bool = False, num_workers: int = 10):
    """Computes statistics (min/max) on the dataset.

    Args:
        force (bool, optional): Whether to overwrite an existing stats file. Defaults to False.
        num_workers (int, optional): Number of workers to parallelize the job. Defaults to 10.
    """

    if STATS_PATH.exists() and not force:
        raise FileExistsError(
            "Stats file already exists. To overwrite it, use '--force'."
        )

    train_ds = WW3Dataset(
        "train",
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
        downscaling_stride=1,
    )

    min_max_dicts = joblib.Parallel(n_jobs=num_workers)(
        joblib.delayed(compute_stats)(sample)
        for sample in tqdm.tqdm(train_ds.samples, desc="Computing stats")
    )

    stats = {
        param: {"min": torch.tensor(0), "max": torch.tensor(0)}
        for param in min_max_dicts[0].keys()
    }
    for param in tqdm.tqdm(stats.keys(), desc="Aggregating"):
        min_vals = [dico[param]["min"] for dico in min_max_dicts]
        max_vals = [dico[param]["max"] for dico in min_max_dicts]
        stats[param]["min"] = torch.tensor(np.nanmin(min_vals))
        stats[param]["max"] = torch.tensor(np.nanmax(max_vals))

    print("stats : ", stats)

    torch.save(stats, STATS_PATH)
    STATS_PATH.chmod(0o666)


if __name__ == "__main__":
    app()
