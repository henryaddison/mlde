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
from pathlib import Path
import typer
import xarray as xr

from ml_downscaling_emulator.statistical_downscaling import compute_daily_stat

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
