import itertools
import os
from pathlib import Path

from codetiming import Timer
from knockknock import slack_sender
from ml_collections import config_dict
import shortuuid
import torch
import typer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr
import yaml

from ml_downscaling_emulator.torch import XRDataset
from mlde_utils.training.dataset import get_dataset, get_variables

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

import logging

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

    state = restore_checkpoint(ckpt_filename, state, config.device)
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


def generate_samples(sampling_fn, score_model, config, cond_batch):
    cond_batch = cond_batch.to(config.device)

    samples = sampling_fn(score_model, cond_batch)[0]
    # drop the feature channel dimension (only have target pr as output)
    samples = samples.squeeze(dim=1)
    # extract numpy array
    samples = samples.cpu().numpy()
    return samples


def generate_predictions(
    sampling_fn, score_model, config, cond_batch, target_transform, coords, cf_data_vars
):
    samples = generate_samples(sampling_fn, score_model, config, cond_batch)

    coords = {**dict(coords)}

    pred_pr_dims = ["time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {
        "grid_mapping": "rotated_latitude_longitude",
        "standard_name": "pred_pr",
        "units": "kg m-2 s-1",
    }
    pred_pr_var = (pred_pr_dims, samples, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    samples_ds = target_transform.invert(
        xr.Dataset(data_vars=data_vars, coords=coords, attrs={})
    )
    samples_ds = samples_ds.rename({"target_pr": "pred_pr"})
    return samples_ds


def load_config(config_path):
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = config_dict.ConfigDict(yaml.unsafe_load(f))

    return config


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
    input_transform_key: str = None,
):
    config_path = os.path.join(workdir, "config.yml")
    config = load_config(config_path)
    if batch_size is not None:
        config.eval.batch_size = batch_size
    if input_transform_key is not None:
        config.data.input_transform_key = input_transform_key

    output_dirpath = (
        workdir
        / "samples"
        / f"epoch-{epoch}"
        / dataset
        / config.data.input_transform_key
        / split
    )
    os.makedirs(output_dirpath, exist_ok=True)

    ckpt_filename = os.path.join(workdir, "checkpoints", f"epoch_{epoch}.pth")
    logger.info(f"Loading model from {ckpt_filename}")
    state, sampling_fn = load_model(config, ckpt_filename)
    score_model = state["model"]
    location_params = state["location_params"]
    transform_dir = os.path.join(workdir, "transforms")

    # Data
    xr_data_eval, _, target_transform = get_dataset(
        dataset,
        config.data.dataset_name,
        config.data.input_transform_key,
        config.data.target_transform_key,
        transform_dir,
        split=split,
        evaluation=True,
    )
    variables, _ = get_variables(config.data.dataset_name)

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        # TODO: EV/storm case add a way to just output numpy files for predictions without tying to netcdf and time and lat and long
        cf_data_vars = {
            key: xr_data_eval.data_vars[key]
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
                total=len(xr_data_eval["time"]), desc=f"Sampling", unit=" timesteps"
            ) as pbar:
                for i in range(
                    0, xr_data_eval["time"].shape[0], config.eval.batch_size
                ):
                    batch_times = xr_data_eval["time"][i : i + config.eval.batch_size]
                    batch_ds = xr_data_eval.sel(time=batch_times)

                    cond_batch = XRDataset.to_tensor(batch_ds, variables)
                    # append any location-specific parameters
                    cond_batch = location_params(cond_batch)

                    coords = batch_ds.coords

                    preds.append(
                        generate_predictions(
                            sampling_fn,
                            score_model,
                            config,
                            cond_batch,
                            target_transform,
                            coords,
                            cf_data_vars,
                        )
                    )

                    pbar.update(cond_batch.shape[0])

        ds = xr.combine_by_coords(
            preds,
            compat="no_conflicts",
            combine_attrs="drop_conflicts",
            coords="all",
            join="inner",
            data_vars="all",
        )

        output_filepath = output_dirpath / f"predictions-{shortuuid.uuid()}.nc"
        typer.echo(f"Saving samples to {output_filepath}...")
        ds.to_netcdf(output_filepath)


if __name__ == "__main__":
    app()
