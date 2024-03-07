from codetiming import Timer
import logging
from knockknock import slack_sender
from ml_collections import config_dict
import os
from pathlib import Path
import shortuuid
import torch
import typer
import yaml

from mlde_utils import samples_path, DEFAULT_ENSEMBLE_MEMBER
from mlde_utils.training.dataset import load_raw_dataset_split
from ..deterministic import sampling
from ..deterministic.utils import create_model, restore_checkpoint
from ..data import get_dataloader


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


def load_config(config_path):
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = config_dict.ConfigDict(yaml.unsafe_load(f))

    return config


def load_model(config, num_predictors, ckpt_filename):
    model = torch.nn.DataParallel(
        create_model(config, num_predictors).to(device=config.device)
    )
    optimizer = torch.optim.Adam(model.parameters())
    state = dict(step=0, epoch=0, optimizer=optimizer, model=model)
    state, loaded = restore_checkpoint(ckpt_filename, state, config.device)
    assert loaded, "Did not load state from checkpoint"

    return state


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logging.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def sample(
    workdir: Path,
    dataset: str = typer.Option(...),
    split: str = "val",
    epoch: int = typer.Option(...),
    batch_size: int = None,
    num_samples: int = 1,
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

    transform_dir = os.path.join(workdir, "transforms")

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
    num_predictors = eval_dl.dataset[0][0].shape[0]
    state = load_model(config, num_predictors, ckpt_filename)

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        xr_samples = sampling.sample(state["model"], eval_dl, target_transform)

        output_filepath = output_dirpath / f"predictions-{shortuuid.uuid()}.nc"

        logger.info(f"Saving predictions to {output_filepath}")
        xr_samples.to_netcdf(output_filepath)


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logging.info)
@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def sample_id(
    workdir: Path,
    dataset: str = typer.Option(...),
    variable: str = "pr",
    split: str = "val",
    ensemble_member: str = "01",
):

    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=f"epoch-0",
        dataset=dataset,
        input_xfm="none",
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(output_dirpath, exist_ok=True)

    eval_ds = load_raw_dataset_split(dataset, split).sel(
        ensemble_member=[ensemble_member]
    )
    xr_samples = sampling.sample_id(variable, eval_ds)

    output_filepath = os.path.join(output_dirpath, f"predictions-{shortuuid.uuid()}.nc")

    logger.info(f"Saving predictions to {output_filepath}")
    xr_samples.to_netcdf(output_filepath)
