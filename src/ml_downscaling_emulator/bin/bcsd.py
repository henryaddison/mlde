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

import functools
import logging
import os
from pathlib import Path
from scipy import ndimage
import shortuuid
import typer
import xarray as xr

from mlde_utils import samples_path, DEFAULT_ENSEMBLE_MEMBER
from mlde_utils.training.dataset import open_raw_dataset_split
from ml_downscaling_emulator.bin.climatology import compute_daily_stat
from ml_downscaling_emulator.bin.sample import _np_samples_to_xr

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
    variable: str = "pr",
    split: str = "test",
    ensemble_member: str = DEFAULT_ENSEMBLE_MEMBER,
    window_size: int = 61,
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

    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=f"epoch-0",
        dataset=dataset,
        input_xfm=f"{train_dataset}-bcsd",
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(output_dirpath, exist_ok=True)

    train_ds = open_raw_dataset_split(train_dataset, "train")

    eval_ds = open_raw_dataset_split(dataset, split).sel(
        ensemble_member=[ensemble_member]
    )

    logger.info(f"Loading low-res and target {variable}.")
    lr_da = train_ds[f"lin{variable}"]
    target_da = train_ds[f"target_{variable}"]
    source_da = eval_ds[f"lin{variable}"]

    typer.echo("Running BCSD...")

    bcsd_da = _bcsd_on_chunks(
        source=source_da,
        lr_da=lr_da,
        target_da=target_da,
        window_size=window_size,
    )

    output_filepath = os.path.join(output_dirpath, f"predictions-{shortuuid.uuid()}.nc")

    cf_data_vars = {
        key: eval_ds.data_vars[key]
        for key in [
            "rotated_latitude_longitude",
            "time_bnds",
            "grid_latitude_bnds",
            "grid_longitude_bnds",
        ]
        if key in eval_ds.variables
    }
    coords = eval_ds.coords

    xr_samples = _np_samples_to_xr(
        bcsd_da.values, coords=coords, target_transform=None, cf_data_vars=cf_data_vars
    )

    logger.info(f"Saving BCSD predictions to {output_filepath}")
    xr_samples[f"pred_{variable}"].encoding.update(zlib=True, complevel=5)
    xr_samples.to_netcdf(output_filepath)


def _bcsd_on_chunks(
    source: xr.DataArray,
    lr_da: xr.DataArray,
    target_da: xr.DataArray,
    method: str = "gaussian",
    multiplicative: bool = True,
    window_size: int = 3,
) -> xr.DataArray:
    """Process an input data chunk with the BCSD method.

    Args:
      source: The source data to be processed with the BCSD method.
      lr_da: The low-resolution data used to fit the BCSD method.

      target_da: The unfiltered target data for fitting the BCSD method.
      method: The method to use for quantile mapping.
      multiplicative: Whether to use multiplicative correction (e.g. for precipitation variables).
      window_size: The size of the window used to compute the climatology statistics.
    Returns:
      The BCSD-downscaled data as an xarray DataArray.
    """

    logger.info(f"Low-pass filtering target data.")
    # Compute the spatially filtered target data for fitting the BCSD method.
    filtered_da = _low_pass_filter(
        target_da,
        scale=(60.0 / 8.8) / 2.0,
        spatial_dims=("grid_longitude", "grid_latitude"),
    )

    sel = dict(
        dayofyear=source["time"].dt.dayofyear,
        drop=True,
    )

    if method == "gaussian":
        # Compute the climatology of the low-resolution data.
        clim_mean = compute_daily_stat(
            lr_da, window_size=window_size, stat_fn="mean"
        ).sel(**sel)
        clim_std = compute_daily_stat(
            lr_da, window_size=window_size, stat_fn="std"
        ).sel(**sel)

        # Standardize with respect to the original climatology.
        source_standard = (source - clim_mean) / clim_std

        # Get value of the same quantile in the filtered climatology, keep anom.
        filtered_clim_std = compute_daily_stat(
            filtered_da, window_size=window_size, stat_fn="std"
        ).sel(**sel)
        source_bc_anom = source_standard * filtered_clim_std

        # Add anom to the mean of the unfiltered climatology.
        target_clim_mean = compute_daily_stat(
            target_da, window_size=window_size, stat_fn="mean"
        ).sel(**sel)

        source_bcsd = target_clim_mean + source_bc_anom

        # Use multiplicative correction for precipitation variables
        if multiplicative:
            filtered_clim_mean = compute_daily_stat(
                filtered_da, window_size=window_size, stat_fn="mean"
            ).sel(**sel)
            # Compute the full field corresponding to the filtered climatology
            # quantile. Then, keep the multiplicative anomaly.
            bc_mult_anom = (source_bc_anom + filtered_clim_mean) / (filtered_clim_mean)
            # Multiply the anomaly by the target climatology.
            var_bcsd = bc_mult_anom * target_clim_mean
            source_bcsd = var_bcsd

        return source_bcsd.drop_vars(["dayofyear"])
    else:
        raise ValueError(f"BCSD method {method} not yet implemented.")


def _low_pass_filter(
    source: xr.DataArray,
    *,
    scale: float = (60.0 / 8.8)
    / 2.0,  # default is half the scale factor from 60km input to 8.8km target
    mode: str = "nearest",
    spatial_dims: tuple[str, str] = ("grid_latitude", "grid_longitude"),
) -> xr.Dataset:
    """Filters a chunk in its spatial dimensions.

    Args:
        source: The source dataset chunk to interpolate to a new spatial grid.
        scale: Scale (in pixels) used to construct the Gaussian filter.
        mode: Determines how the input array is extended when the filter overlaps a
        border.
        spatial_dims: The dimensions of the source dataset to filter over.

    Returns:
        The spatially filtered chunk.
    """
    # Ordered dimensions of the source dataset can be fetched from a data_var.
    # Dataset dimensions are not guaranteed to be ordered.
    ordered_dims = source.dims
    # We only filter the spatial dimensions.
    scales = tuple(scale * float(dim in spatial_dims) for dim in ordered_dims)

    gaussian_filter = functools.partial(
        ndimage.gaussian_filter,
        sigma=scales,
    )
    return xr.apply_ufunc(gaussian_filter, source.load())
