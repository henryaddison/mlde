# Copyright 2023 Google LLC
# Modifications copyright 2025 Henry Addison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Significant modifications to the original work have been made by Henry Addison
# to work with their UK Met Office CPM emulator framework.
# Based on https://github.com/google-research/weatherbench2/blob/16e0131309a2b3916875a2cf8806190ffedea308/weatherbench2/utils.py#L127


import logging
import numpy as np
from pathlib import Path
from typing import Callable, Union
import typer
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()


@app.callback()
def callback():
    pass


@app.command()
def mean(
    input_path: Path,
    output_path: Path = typer.Argument(
        None, help="Path to save the mean climatology dataset."
    ),
    variable: str = typer.Option(...),
):
    r"""Script to compute the mean climatology of a dataset in the style of weatherbench2."""
    typer.echo("Running mean...")
    input_ds = xr.open_dataset(input_path)

    output_ds = compute_daily_stat(input_ds, window_size=3, stat_fn="mean")

    typer.echo(f"Saving mean to {output_path}...")
    output_ds[variable].encoding.update(zlib=True, complevel=5)
    output_ds.to_netcdf(output_path)


@app.command()
def std(
    input_path: Path,
    output_path: Path = typer.Argument(
        None, help="Path to save the std climatology dataset."
    ),
    variable: str = typer.Option(...),
):
    r"""Script to compute the std climatology of a dataset in the style of weatherbench2."""
    typer.echo("Running std...")
    input_ds = xr.open_dataset(input_path)

    output_ds = compute_daily_stat(input_ds, window_size=3, stat_fn="std")

    typer.echo(f"Saving std to {output_path}...")
    output_ds[variable].encoding.update(zlib=True, complevel=5)
    output_ds.to_netcdf(output_path)


def create_window_weights(window_size: int) -> xr.DataArray:
    """Create linearly decaying window weights."""
    assert window_size % 2 == 1, "Window size must be odd."
    half_window_size = window_size // 2
    window_weights = np.concatenate(
        [
            np.linspace(0, 1, half_window_size + 1),
            np.linspace(1, 0, half_window_size + 1)[1:],
        ]
    )
    window_weights = window_weights / window_weights.mean()
    window_weights = xr.DataArray(window_weights, dims=["window"])
    return window_weights


def replace_time_with_doy(ds: xr.Dataset) -> xr.Dataset:
    """Replace time coordinate with days of year."""
    return ds.assign_coords({"time": ds.time.dt.dayofyear}).rename(
        {"time": "dayofyear"}
    )


def compute_rolling_stat(
    ds: xr.Dataset,
    window_weights: xr.DataArray,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = "mean",
) -> xr.Dataset:
    """Compute rolling climatology."""
    window_size = len(window_weights)
    half_window_size = window_size // 2  # For padding
    # Stack years
    stacked = xr.concat(
        [
            replace_time_with_doy(ds.sel(time=str(y)))
            for y in np.unique(ds.time.dt.year)
        ],
        dim="year",
    )
    # Fill gap day (366) with values from previous day 365
    # stacked = stacked.fillna(stacked.sel(dayofyear=365))
    # Pad edges for perioding window
    stacked = stacked.pad(pad_width={"dayofyear": half_window_size}, mode="wrap")
    # Weighted rolling mean
    stacked = stacked.rolling(dayofyear=window_size, center=True).construct("window")
    if stat_fn == "mean":
        rolling_stat = stacked.weighted(window_weights).mean(dim=("window", "year"))
    elif stat_fn == "std":
        rolling_stat = stacked.weighted(window_weights).std(dim=("window", "year"))
    else:
        rolling_stat = stat_fn(stacked, weights=window_weights, dim=("window", "year"))
    # Remove edges
    rolling_stat = rolling_stat.isel(
        dayofyear=slice(half_window_size, -half_window_size)
    )
    return rolling_stat


def compute_daily_stat(
    ds: xr.Dataset,
    window_size: int,
    # clim_years: slice,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = "mean",
) -> xr.Dataset:
    """Compute daily average climatology with running window."""
    # NOTE: Loading seems to be necessary, otherwise computation takes forever
    # Will be converted to xarray-beam pipeline anyway
    ds = ds.load()

    window_weights = create_window_weights(window_size)
    daily_rolling_clim = compute_rolling_stat(ds, window_weights, stat_fn)
    return daily_rolling_clim
