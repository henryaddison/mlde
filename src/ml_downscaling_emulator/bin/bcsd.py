# Copyright 2025 The swirl_dynamics Authors.
# Modifications copyright 2025 Henry Addison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Significant modifications to the original work have been made by Henry Addison
# to work with their UK Met Office CPM emulator framework.
# Based on https://github.com/google-research/swirl-dynamics/blob/b79f03f78a64fb9259412ee62c591400484ecbff/swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/analysis/bcsd.py and https://github.com/google-research/swirl-dynamics/blob/b79f03f78a64fb9259412ee62c591400484ecbff/swirl_dynamics/projects/probabilistic_diffusion/downscaling/gcm_wrf/input_pipelines/filter_in_space.py

import logging
import os
from pathlib import Path
import shortuuid
import typer
import xarray as xr

from mlde_utils import samples_path, DEFAULT_ENSEMBLE_MEMBER, TIME_PERIODS
from mlde_utils.training.dataset import open_raw_dataset_split
from ml_downscaling_emulator.bin.sample import _np_samples_to_xr
from ml_downscaling_emulator.statistical_downscaling import bcsd_on_chunks, bcsd2

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
def bcsd(
    workdir: Path,
    dataset: str,
    train_dataset: str,
    coarse_train_dataset: str = typer.Argument(None),
    variable: str = "pr",
    split: str = "val",
    ensemble_member: str = DEFAULT_ENSEMBLE_MEMBER,
    window_size: int = 3,
    version="v1",
):
    r"""Script to perform downscaling with the time-aligned BCSD method.

    This script implements the Bias Correction and Spatial Disaggregation (BCSD)
    method for downscaling (Wood et al, 2002): https://doi.org/10.1029/2001JD000659.
    This implementation follows the time-aligned BCSD method of Thrasher et al (2012):
    https://doi.org/10.5194/hess-16-3309-2012. The methodology is modified to
    operate on a single domain discretization (the high-resolution grid).

    The method assumes we have a low-resolution dataset, its climatology, and the
    spatially filtered and unfiltered climatology of a high-resolution dataset. The
    input low-resolution dataset and all climatologies are assumed to have been
    interpolated to the high-resolution grid for convenience. Nevertheless, the
    low-resolution and filtered high-resolution fields are assumed to have a low
    effective --and similar-- spatial resolution, since interpolation does not yield
    energy at the interpolated lengthscales.

    The bias correction step is performed by computing the percentiles of the input
    dataset with respect to its climatology, and mapping them to the full field
    values corresponding to the same percentiles in the filtered high-resolution
    climatology.

    The spatial disaggregation consists, for most variables, of subtracting the
    climatological mean of the filtered high-resolution field from the
    bias-corrected low-resolution field, and then adding the climatological mean of
    the unfiltered high-resolution field. For some variables, such as precipitation,
    we use a multiplicative correction instead: we multiply the bias-corrected
    low-resolution field by the ratio of the filtered and unfiltered climatological
    means. The variables to be corrected multiplicatively are specified by the
    `multiplicative_vars` flag.

    Although the temporal disaggregation step of Wood et al. (2002) is omitted in
    our implementation (Thrasher et al., 2012), we describe it here for completeness.
    Let the low-resolution data correspond to time averages over periods `T`. In the
    temporal disaggregation step, for each sample resulting from the spatial
    disaggregation step, we sample a high-resolution data sequence over period `T`
    and the same time of the year as the spatially
    disaggregated sample. The high--resolution sample sequence is drawn at random
    from the dataset used to construct the high-resolution climatology. Once the
    sample sequence is drawn, we scale it such that the time average over `T`
    matches the spatially disaggregated sample. Finally, we use this scaled sequence
    as the final BCSD sample.

    Example usage:

    ```
    WORKDIR=<base dir for storing downscaled samples>

    DATASET=<name of dataset to downscale>
    SPLIT=<split of dataset to downscale, e.g. "test">
    ENSEMBLE_MEMBER=<ensemble member to downscale, e.g. "01">

    TRAIN_DATASET=<name of dataset to fit BCSD method>

    VARIABLE=<variable to downscale, e.g. "pr">

    WINDOW_SIZE=<size of the window used to compute the climatology statistics, e.g. 61>

    mlde bcsd bcsd \
    ${WORKDIR} ${DATASET} ${TRAIN_DATASET} \
    --split=${SPLIT} \
    --ensemble_member=${ENSEMBLE_MEMBER} \
    --variable=${VARIABLE} \
    --window_size=${WINDOW_SIZE}
    ```
    """

    if coarse_train_dataset is None:
        coarse_train_dataset = train_dataset

    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=f"epoch-0",
        dataset=dataset,
        input_xfm=f"{coarse_train_dataset}-{train_dataset}-bcsd",
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(output_dirpath, exist_ok=True)

    train_ds = open_raw_dataset_split(train_dataset, "train")

    coarse_train_ds = open_raw_dataset_split(coarse_train_dataset, "train")

    eval_ds = open_raw_dataset_split(dataset, split).sel(
        ensemble_member=[ensemble_member]
    )

    def tp_from_time(x):
        for tp_key, (tp_start, tp_end) in TIME_PERIODS.items():
            if (x >= tp_start) and (x <= tp_end):
                return tp_key
        raise RuntimeError(f"No time period for {x}")

    train_time_period_coord_values = xr.apply_ufunc(
        tp_from_time, train_ds["time"], input_core_dims=None, vectorize=True
    )
    train_ds = train_ds.assign_coords(
        time_period=("time", train_time_period_coord_values.data)
    )

    coarse_train_time_period_coord_values = xr.apply_ufunc(
        tp_from_time, coarse_train_ds["time"], input_core_dims=None, vectorize=True
    )
    coarse_train_ds = coarse_train_ds.assign_coords(
        time_period=("time", coarse_train_time_period_coord_values.data)
    )

    eval_time_period_coord_values = xr.apply_ufunc(
        tp_from_time, eval_ds["time"], input_core_dims=None, vectorize=True
    )
    eval_ds = eval_ds.assign_coords(
        time_period=("time", eval_time_period_coord_values.data)
    )

    xr_samples = []

    logger.info(f"Selecting low-res and target {variable}.")
    lr_da = coarse_train_ds[f"lin{variable}"]
    target_da = train_ds[f"target_{variable}"]

    for tp, tp_eval_ds in eval_ds.groupby("time_period"):
        logger.info(f"Selecting variables for time period: {tp}")
        tp_target_da = target_da.where(target_da["time_period"] == tp, drop=True)
        tp_lr_da = lr_da.where(lr_da["time_period"] == tp, drop=True)
        tp_source_da = tp_eval_ds[f"lin{variable}"]

        typer.echo(f"Running BCSD for {tp}...")

        if version == "v1":
            bcsd_func = bcsd_on_chunks
        elif version == "v2":
            bcsd_func = bcsd2
        else:
            raise ValueError(
                f"Unknown BCSD version: {version}. Supported versions: v1, v2."
            )

        bcsd_da = bcsd_func(
            source=tp_source_da,
            lr_da=tp_lr_da,
            target_da=tp_target_da,
            window_size=window_size,
        )

        cf_data_vars = {
            key: tp_eval_ds.data_vars[key]
            for key in [
                "rotated_latitude_longitude",
                "time_bnds",
                "grid_latitude_bnds",
                "grid_longitude_bnds",
            ]
            if key in tp_eval_ds.variables
        }
        coords = tp_eval_ds.coords

        xr_samples.append(
            _np_samples_to_xr(
                bcsd_da.values,
                coords=coords,
                target_transform=None,
                cf_data_vars=cf_data_vars,
            )
        )

    xr_samples = xr.concat(xr_samples, dim="time").sortby("time")

    output_filepath = os.path.join(output_dirpath, f"predictions-{shortuuid.uuid()}.nc")
    logger.info(f"Saving BCSD predictions to {output_filepath}")
    xr_samples[f"pred_{variable}"].encoding.update(zlib=True, complevel=5)
    xr_samples.to_netcdf(output_filepath)
