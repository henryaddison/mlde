"""Generate samples"""

from collections import defaultdict
import hashlib
import itertools
import os
from pathlib import Path

from codetiming import Timer
from dotenv import load_dotenv
from ml_collections import config_dict
import shortuuid
import torch
import typer
from tqdm import tqdm
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr
import yaml

from ml_downscaling_emulator.data import get_dataloader, np_samples_to_xr
from mlde_utils import samples_path, DEFAULT_ENSEMBLE_MEMBER
from mlde_utils.training.dataset import get_variables

from ml_downscaling_emulator.losses import get_optimizer
from ml_downscaling_emulator.models.ema import (
    ExponentialMovingAverage,
)
from ml_downscaling_emulator.models.location_params import (
    LocationParams,
)

from ml_downscaling_emulator.utils import restore_checkpoint

import ml_downscaling_emulator.models as models  # noqa: F401
from ml_downscaling_emulator.models import utils as mutils

from ml_downscaling_emulator.models import cncsnpp  # noqa: F401
from ml_downscaling_emulator.models import cunet  # noqa: F401
from ml_downscaling_emulator.models import det_cunet  # noqa: F401

from ml_downscaling_emulator.models import (  # noqa: F401
    layerspp,  # noqa: F401
)  # noqa: F401
from ml_downscaling_emulator.models import layers  # noqa: F401
from ml_downscaling_emulator.models import (  # noqa: F401
    normalization,  # noqa: F401
)  # noqa: F401
import ml_downscaling_emulator.sampling as sampling

from ml_downscaling_emulator.sde_lib import (
    VESDE,
    VPSDE,
    subVPSDE,
)

load_dotenv()  # take environment variables from .env.

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()

app = typer.Typer()


def load_config(config_path):
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = config_dict.ConfigDict(yaml.unsafe_load(f))

    return config


def _init_state(config):
    score_model = mutils.create_model(config)
    location_params = LocationParams(
        config.model.loc_spec_channels, config.data.image_size
    )
    location_params = location_params.to(config.device)
    location_params = torch.nn.DataParallel(location_params)
    optimizer = get_optimizer(
        config, itertools.chain(score_model.parameters(), location_params.parameters())
    )
    ema = ExponentialMovingAverage(
        itertools.chain(score_model.parameters(), location_params.parameters()),
        decay=config.model.ema_rate,
    )
    state = dict(
        step=0,
        optimizer=optimizer,
        model=score_model,
        location_params=location_params,
        ema=ema,
    )

    return state


def load_model(config, ckpt_filename):
    deterministic = "deterministic" in config and config.deterministic
    if deterministic:
        sde = None
        sampling_eps = 0
    else:
        if config.training.sde == "vesde":
            sde = VESDE(
                sigma_min=config.model.sigma_min,
                sigma_max=config.model.sigma_max,
                N=config.sampling.num_scales,
            )
            sampling_eps = 1e-5
        elif config.training.sde == "vpsde":
            sde = VPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.sampling.num_scales,
            )
            sampling_eps = 1e-3
        elif config.training.sde == "subvpsde":
            sde = subVPSDE(
                beta_min=config.model.beta_min,
                beta_max=config.model.beta_max,
                N=config.sampling.num_scales,
            )
            sampling_eps = 1e-3
        else:
            raise RuntimeError(f"Unknown SDE {config.training.sde}")

    # sigmas = mutils.get_sigmas(config)  # noqa: F841
    state = _init_state(config)
    state, loaded = restore_checkpoint(ckpt_filename, state, config.device)
    assert loaded, "Did not load state from checkpoint"
    state["ema"].copy_to(state["model"].parameters())

    # Sampling
    input_variables, target_vars = get_variables(config.data.dataset_name)
    num_output_channels = len(target_vars)
    sampling_shape = (
        config.eval.batch_size,
        num_output_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    return state, sampling_fn, target_vars


def generate_np_sample_batch(sampling_fn, score_model, config, cond_batch):
    cond_batch = cond_batch.to(config.device)

    samples = sampling_fn(score_model, cond_batch)[0]

    # extract numpy array
    samples = samples.cpu().numpy()
    return samples


def sample(sampling_fn, state, config, eval_dl, target_transform, target_vars):
    score_model = state["model"]
    location_params = state["location_params"]

    cf_data_vars = {
        key: eval_dl.dataset.ds.data_vars[key]
        for key in [
            "rotated_latitude_longitude",
            "time_bnds",
            "grid_latitude_bnds",
            "grid_longitude_bnds",
        ]
    }

    xr_sample_batches = []
    with logging_redirect_tqdm():
        with tqdm(
            total=len(eval_dl.dataset),
            desc=f"Sampling",
            unit=" timesteps",
        ) as pbar:
            for cond_batch, _, time_batch in eval_dl:
                # append any location-specific parameters
                cond_batch = location_params(cond_batch)

                coords = eval_dl.dataset.ds.sel(time=time_batch).coords

                np_sample_batch = generate_np_sample_batch(
                    sampling_fn, score_model, config, cond_batch
                )

                xr_sample_batch = np_samples_to_xr(
                    np_sample_batch,
                    target_transform,
                    target_vars,
                    coords,
                    cf_data_vars,
                )

                xr_sample_batches.append(xr_sample_batch)

                pbar.update(cond_batch.shape[0])

    ds = xr.combine_by_coords(
        xr_sample_batches,
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )
    return ds


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logger.info)
def main(
    workdir: Path,
    dataset: str = typer.Option(...),
    split: str = "val",
    checkpoint: str = typer.Option(...),
    batch_size: int = None,
    num_samples: int = 3,
    input_transform_dataset: str = None,
    input_transform_key: str = None,
    ensemble_member: str = DEFAULT_ENSEMBLE_MEMBER,
    num_scales: int = None,
):
    config_path = os.path.join(workdir, "config.yml")
    config = load_config(config_path)
    if batch_size is not None:
        config.eval.batch_size = batch_size
    with config.unlocked():
        if input_transform_dataset is not None:
            config.data.input_transform_dataset = input_transform_dataset
        else:
            config.data.input_transform_dataset = dataset

        if "target_transform_overrides" not in config.data:
            config.data.target_transform_overrides = config_dict.ConfigDict()
        if "num_scales" not in config.sampling:
            config.sampling.num_scales = config.model.num_scales

    if input_transform_key is not None:
        config.data.input_transform_key = input_transform_key

    if num_scales is not None:
        config.sampling.num_scales = num_scales

    config_hash = hashlib.shake_256(
        bytes(repr(config.to_yaml(sort_keys=True)), "UTF-8")
    ).hexdigest(8)

    output_dirpath = (
        samples_path(
            workdir=workdir,
            checkpoint=checkpoint,
            dataset=dataset,
            input_xfm=f"{config.data.input_transform_dataset}-{config.data.input_transform_key}",
            split=split,
            ensemble_member=ensemble_member,
        )
        / config_hash
    )
    os.makedirs(output_dirpath, exist_ok=True)

    sampling_config_path = os.path.join(output_dirpath, "config.yml")
    with open(sampling_config_path, "w") as f:
        f.write(config.to_yaml())

    transform_dir = os.path.join(workdir, "transforms")

    target_xfm_keys = defaultdict(lambda: config.data.target_transform_key) | dict(
        config.data.target_transform_overrides
    )

    # Data
    eval_dl, _, target_transform = get_dataloader(
        dataset,
        config.data.dataset_name,
        config.data.input_transform_dataset,
        config.data.input_transform_key,
        target_xfm_keys,
        transform_dir,
        split=split,
        ensemble_members=[ensemble_member],
        include_time_inputs=config.data.time_inputs,
        evaluation=True,
        batch_size=config.eval.batch_size,
        shuffle=False,
    )

    ckpt_filename = os.path.join(workdir, "checkpoints", f"{checkpoint}.pth")
    logger.info(f"Loading model from {ckpt_filename}")
    state, sampling_fn, target_vars = load_model(config, ckpt_filename)

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        xr_samples = sample(
            sampling_fn, state, config, eval_dl, target_transform, target_vars
        )

        output_filepath = output_dirpath / f"predictions-{shortuuid.uuid()}.nc"

        logger.info(f"Saving samples to {output_filepath}...")
        for varname in target_vars:
            for prefix in ["pred_", "raw_pred_"]:
                xr_samples[varname.replace("target_", prefix)].encoding.update(
                    zlib=True, complevel=5
                )
        xr_samples.to_netcdf(output_filepath)


if __name__ == "__main__":
    app()
