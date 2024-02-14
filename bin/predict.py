import itertools
import os
from pathlib import Path

from codetiming import Timer
from knockknock import slack_sender
from ml_collections import config_dict
import numpy as np
import shortuuid
import torch
import typer
from tqdm import tqdm
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr
import yaml

from ml_downscaling_emulator.torch import get_dataloader
from mlde_utils import samples_path, DEFAULT_ENSEMBLE_MEMBER
from mlde_utils.training.dataset import get_variables

from ml_downscaling_emulator.score_sde_pytorch_hja22.losses import get_optimizer
from ml_downscaling_emulator.score_sde_pytorch_hja22.models.ema import (
    ExponentialMovingAverage,
)
from ml_downscaling_emulator.score_sde_pytorch_hja22.models.location_params import (
    LocationParams,
)

from ml_downscaling_emulator.score_sde_pytorch_hja22.utils import restore_checkpoint

import ml_downscaling_emulator.score_sde_pytorch_hja22.models as models  # noqa: F401
from ml_downscaling_emulator.score_sde_pytorch_hja22.models import utils as mutils

# from score_sde_pytorch_hja22.models import ncsnv2
# from score_sde_pytorch_hja22.models import ncsnpp
from ml_downscaling_emulator.score_sde_pytorch_hja22.models import cncsnpp  # noqa: F401
from ml_downscaling_emulator.score_sde_pytorch_hja22.models import cunet  # noqa: F401

# from score_sde_pytorch_hja22.models import ddpm as ddpm_model
from ml_downscaling_emulator.score_sde_pytorch_hja22.models import (  # noqa: F401
    layerspp,  # noqa: F401
)  # noqa: F401
from ml_downscaling_emulator.score_sde_pytorch_hja22.models import layers  # noqa: F401
from ml_downscaling_emulator.score_sde_pytorch_hja22.models import (  # noqa: F401
    normalization,  # noqa: F401
)  # noqa: F401
import ml_downscaling_emulator.score_sde_pytorch_hja22.sampling as sampling

# from likelihood import get_likelihood_fn
from ml_downscaling_emulator.score_sde_pytorch_hja22.sde_lib import (
    VESDE,
    VPSDE,
    subVPSDE,
)

# from score_sde_pytorch_hja22.sampling import (ReverseDiffusionPredictor,
#                       LangevinCorrector,
#                       EulerMaruyamaPredictor,
#                       AncestralSamplingPredictor,
#                       NoneCorrector,
#                       NonePredictor,
#                       AnnealedLangevinDynamics)


logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()


def load_model(config, ckpt_filename):
    if config.training.sde == "vesde":
        sde = VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    elif config.training.sde == "vpsde":
        sde = VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde == "subvpsde":
        sde = subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    else:
        raise RuntimeError(f"Unknown SDE {config.training.sde}")

    random_seed = 0  # @param {"type": "integer"}  # noqa: F841

    sigmas = mutils.get_sigmas(config)  # noqa: F841
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

    state, loaded = restore_checkpoint(ckpt_filename, state, config.device)
    assert loaded, "Did not load state from checkpoint"
    ema.copy_to(score_model.parameters())

    # Sampling
    num_output_channels = len(get_variables(config.data.dataset_name)[1])
    sampling_shape = (
        config.eval.batch_size,
        num_output_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    return state, sampling_fn


def generate_np_samples(sampling_fn, score_model, config, cond_batch):
    cond_batch = cond_batch.to(config.device)

    samples = sampling_fn(score_model, cond_batch)[0]
    # drop the feature channel dimension (only have target pr as output)
    samples = samples.squeeze(dim=1)
    # extract numpy array
    samples = samples.cpu().numpy()
    return samples


def np_samples_to_xr(np_samples, target_transform, coords, cf_data_vars):
    coords = {**dict(coords)}

    pred_pr_dims = ["ensemble_member", "time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {
        "grid_mapping": "rotated_latitude_longitude",
        "standard_name": "pred_pr",
        "units": "kg m-2 s-1",
    }
    # add ensemble member axis to np samples
    np_samples = np_samples[np.newaxis, :]
    pred_pr_var = (pred_pr_dims, np_samples, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    samples_ds = target_transform.invert(
        xr.Dataset(data_vars=data_vars, coords=coords, attrs={})
    )
    samples_ds = samples_ds.rename({"target_pr": "pred_pr"})
    samples_ds["pred_pr"] = samples_ds["pred_pr"].assign_attrs(pred_pr_attrs)
    return samples_ds


def load_config(config_path):
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = config_dict.ConfigDict(yaml.unsafe_load(f))

    return config


def sample(sampling_fn, state, config, eval_dl, target_transform):
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

    preds = []
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

                np_samples = generate_np_samples(
                    sampling_fn, score_model, config, cond_batch
                )

                xr_samples = np_samples_to_xr(
                    np_samples,
                    target_transform,
                    coords,
                    cf_data_vars,
                )

                preds.append(xr_samples)

                pbar.update(cond_batch.shape[0])

    ds = xr.combine_by_coords(
        preds,
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
        coords="all",
        join="inner",
        data_vars="all",
    )
    return ds


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logger.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def main(
    workdir: Path,
    dataset: str = typer.Option(...),
    split: str = "val",
    epoch: int = typer.Option(...),
    batch_size: int = None,
    num_samples: int = 3,
    input_transform_dataset: str = None,
    input_transform_key: str = None,
    ensemble_member: str = DEFAULT_ENSEMBLE_MEMBER,
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
    if input_transform_key is not None:
        config.data.input_transform_key = input_transform_key

    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=f"epoch-{epoch}",
        dataset=dataset,
        input_xfm=f"{config.data.input_transform_dataset}-{config.data.input_transform_key}",
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(output_dirpath, exist_ok=True)

    sampling_config_path = os.path.join(output_dirpath, "config.yml")
    with open(sampling_config_path, "w") as f:
        f.write(config.to_yaml())

    transform_dir = os.path.join(workdir, "transforms")

    # Data
    eval_dl, _, target_transform = get_dataloader(
        dataset,
        config.data.dataset_name,
        config.data.input_transform_dataset,
        config.data.input_transform_key,
        config.data.target_transform_key,
        transform_dir,
        split=split,
        ensemble_members=[ensemble_member],
        include_time_inputs=config.data.time_inputs,
        evaluation=True,
        batch_size=config.eval.batch_size,
        shuffle=False,
    )

    ckpt_filename = os.path.join(workdir, "checkpoints", f"epoch_{epoch}.pth")
    logger.info(f"Loading model from {ckpt_filename}")
    state, sampling_fn = load_model(config, ckpt_filename)

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        xr_samples = sample(sampling_fn, state, config, eval_dl, target_transform)

        output_filepath = output_dirpath / f"predictions-{shortuuid.uuid()}.nc"

        logger.info(f"Saving samples to {output_filepath}...")
        xr_samples.to_netcdf(output_filepath)


if __name__ == "__main__":
    app()
