from typing import cast

import cartopy
import matplotlib.pyplot as plt
import numpy as np
from cartopy.crs import PlateCarree
from matplotlib.figure import Figure
from mfai.pytorch.namedtensor import NamedTensor
from torch import Tensor

from ww3.settings import GRIDS, METADATA

# Setup cache dir for cartopy to avoid downloading data each time
cartopy.config["data_dir"] = "/scratch/shared/cartopy"


def compute_figsize(ref_array: Tensor, nrows: int = 3, ncols: int = 4) -> list[float]:
    """
    Compute the optimal figure size for displaying arrays
    in a grid of subplots while preserving geometric proportions.

    Parameters:
    - ref_array: Tensor of shape (X, Y)
    - nrows: number of rows in the subplot grid
    - ncols: number of columns in the subplot grid

    Returns:
    - figsize: tuple (width, height) for plt.subplots
    """
    y_size, x_size = ref_array.shape

    # Aspect ratio of the map
    aspect_ratio = x_size / y_size

    # Base size for one subplot (in inches)
    # Start with 4 inches width (typical value)
    subplot_width = 4
    subplot_height = subplot_width / aspect_ratio

    # Total figure size
    fig_width = ncols * subplot_width
    fig_height = nrows * subplot_height

    # Add some margin (10% on each side)
    fig_width *= 1.1
    fig_height *= 1.1
    return [fig_width, fig_height]


def plot_one_field(
    fig: Figure, field: Tensor, extent: tuple[float, float, float, float], title: str
) -> None:
    """Plots one field on a map, in a Figure."""
    ax = fig.subplots(nrows=1, ncols=1, subplot_kw={"projection": PlateCarree()})
    ax.imshow(field.cpu(), extent=extent, interpolation="none")
    ax.coastlines(resolution="10m", color="black")
    ax.set_title(title, size=15)


def plot_error_map(
    error_maps: NamedTensor,
    title: str = "",
    grid_name: str = "BRETAGNE0002",
) -> Figure:
    """Plots Error Maps."""

    feature_names = error_maps.feature_names
    n_features = len(feature_names)

    n_cols = 2
    n_rows = (n_features + 1) // 2

    example_data = error_maps[feature_names[0]][0]
    figsize = compute_figsize(example_data, n_rows, n_cols)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize,
        subplot_kw={"projection": PlateCarree()},
        dpi=200,
    )

    axes_flat = axes.flatten()

    grid = cast(tuple[float, ...], GRIDS[grid_name]["extent"])
    extent = (grid[2], grid[3], grid[1], grid[0])

    for i, param_name in enumerate(feature_names):
        ax = axes_flat[i]

        param_error_map = error_maps[param_name][0].cpu()

        vmax = np.nanmax(param_error_map)
        vmin = np.nanmin(param_error_map)

        kwargs = {
            "vmin": vmax,
            "vmax": vmin,
            "extent": extent,
            "cmap": "Reds",
            "interpolation": "nearest",
        }

        # Plot
        im = ax.imshow(param_error_map, **kwargs)
        ax.coastlines(resolution="10m", color="black", linewidth=1)
        ax.set_title(param_name)

        plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(title, size=20, y=1.02)
    plt.tight_layout()

    return fig


def plot_sample(
    x: NamedTensor,
    y: NamedTensor,
    y_hat: None | NamedTensor | list[NamedTensor],
    title: str = "",
    model_names: list[str] = ["AI"],
    grid_name: str = "BRETAGNE0002",
) -> Figure:
    """Plots a Sample and shows MFWAM and WW3 data."""
    if isinstance(y_hat, NamedTensor):
        y_hat = [y_hat]
    if isinstance(y_hat, list) and len(y_hat) != len(model_names):
        print(model_names)
        raise ValueError(
            f"model_names and y_hat must have the same length, currently {len(model_names)} != {len(y_hat)}"
        )

    grid = cast(tuple[float, ...], GRIDS[grid_name]["extent"])
    extent = (grid[2], grid[3], grid[1], grid[0])

    n_rows = max(4, len(y.feature_names))
    if y_hat is None:
        n_cols = 3
    else:
        n_cols = 3 + len(y_hat)
    example_data = y[y.feature_names[0]][0]
    figsize = compute_figsize(example_data, n_rows, n_cols)
    fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=200)
    subfigs = fig.subfigures(nrows=n_rows, ncols=2, width_ratios=[1, 2])

    if "arpege_u10" in x.feature_names and "arpege_v10" in x.feature_names:
        plot_one_field(subfigs[0, 0], x["arpege_u10"][0], extent, "ARPEGE U10")
        plot_one_field(subfigs[1, 0], x["arpege_v10"][0], extent, "ARPEGE V10")
    if "log_bathy" in x.feature_names:
        plot_one_field(
            subfigs[2, 0], x["log_bathy"][0], extent, "log( Bathymetry + 1 )"
        )
    if "landsea_mask" in x.feature_names:
        plot_one_field(subfigs[3, 0], x["landsea_mask"][0], extent, "Land / Sea mask")

    ncols_subf = n_cols - 1
    for i, ww3_param_name in enumerate(y.feature_names):
        axes = subfigs[i, 1].subplots(
            nrows=1, ncols=ncols_subf, subplot_kw={"projection": PlateCarree()}
        )
        param_name = ww3_param_name.replace("ww3_", "")
        mfwam_param_data = x["mfwam_" + param_name].cpu()
        ww3_param_data = y[ww3_param_name].cpu()
        if param_name.startswith(("cos", "sin")):
            vmin, vmax = -1, 1
        else:
            vmin = 0
            vmax = max(np.nanmax(vals) for vals in (mfwam_param_data, ww3_param_data))
        kwargs = {
            "extent": extent,
            "vmin": vmin,
            "vmax": vmax,
            "cmap": "Spectral" if param_name.startswith(("cos", "sin")) else "turbo",
            "interpolation": "none",
        }

        list_models = (
            ["MFWAM", "WW3"] if y_hat is None else ["MFWAM"] + model_names + ["WW3"]
        )

        for j, model_name in enumerate(list_models):
            if model_name == "MFWAM":
                data = mfwam_param_data
            elif model_name == "WW3":
                data = ww3_param_data
            else:
                data = y_hat[j - 1][ww3_param_name]
            model_param_name = model_name.lower() + "_" + param_name
            im = axes[j].imshow(data[0].cpu(), **kwargs)
            axes[j].coastlines(resolution="10m", color="black")
            if i == 0:
                axes[j].set_title(model_name, size=15)
            ylabel = param_name.replace("_", " ").upper()
            axes[0].text(
                -0.07,
                0.55,
                ylabel,
                va="bottom",
                ha="center",
                rotation="vertical",
                rotation_mode="anchor",
                transform=axes[0].transAxes,
                size=15,
            )
        long_name = METADATA["WEATHER_PARAMS"][model_param_name]["long_name"]
        subfigs[i, 1].colorbar(im, ax=axes, orientation="vertical", label=long_name)

    fig.suptitle(title, size=20)
    return fig
