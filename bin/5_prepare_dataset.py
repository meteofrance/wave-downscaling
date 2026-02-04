"""Pre-saves the dataset at a given downscaling stride.

usage: 5_prepare_dataset.py [-h] [--downscaling_stride DOWNSCALING_STRIDE] [--workers WORKERS]

options:
  -h, --help            show this help message and exit
  --downscaling_stride DOWNSCALING_STRIDE
                        Inside [1;50]. Stride of pixel to keep between full resolution data (200m) and working resolution. Lower is a better resolution.
  --workers WORKERS     Number of workers to parallelize the job.
"""

import joblib
from tqdm import tqdm
from ww3.dataset import WW3Dataset
from ww3.sample import Sample
from ww3.settings import SCRATCH_PATH


def prepare_dataset(downscaling_stride: int, workers: int) -> None:
    """Pre-saves a dataset on disk.

    Args:
        downscaling_stride (int): Inside [1;50]. Stride of pixel to keep between full resolution data (200m) and working resolution. Lower is a better resolution.
        workers (int): Number of workers to parallelize the job.
    """
    # Creates a dataset with all the samples and weather params
    # at the given downscaling stride
    dataset = WW3Dataset(
        "train",
        pct_in_train=1,
        max_leadtime=102,
        downscaling_stride=downscaling_stride,
        transforms_list=[],
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
        build_format="zarr",
    )
    print(f"--> {len(dataset)} samples in dataset.")

    # Creates dataset folder
    folder = SCRATCH_PATH / f"dataset_{downscaling_stride}"
    if not folder.exists():
        folder.mkdir(exist_ok=True, parents=True)

    # Keep only samples that are not already written on disk
    samples = [s for s in dataset.samples if not s.save_path.exists()]
    print(f"--> {len(samples)} samples to prepare.")

    joblib.Parallel(n_jobs=workers)(
        joblib.delayed(Sample.save)(sample)
        for sample in tqdm(dataset.samples, desc="Saving dataset")
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--downscaling_stride",
        type=int,
        default=25,
        help="Inside [1;50]. Stride of pixel to keep between full resolution data (200m) and working resolution. Lower is a better resolution.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of workers to parallelize the job.",
    )
    args = parser.parse_args()

    if args.downscaling_stride < 1 or args.downscaling_stride > 50:
        raise ValueError(
            f"Downscaling stride must be inside [1,50], got {args.downscaling_stride}."
        )

    prepare_dataset(args.downscaling_stride, args.workers)
