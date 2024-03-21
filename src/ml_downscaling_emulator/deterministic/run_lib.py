import logging
import os

from absl import flags
from codetiming import Timer
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import yaml

from mlde_utils import DatasetMetadata

from ..training import log_epoch, track_run
from .utils import restore_checkpoint, save_checkpoint, create_model
from ..data import get_dataloader

FLAGS = flags.FLAGS
EXPERIMENT_NAME = os.getenv("WANDB_EXPERIMENT_NAME")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")


def val_loss(config, val_dl, eval_step_fn, state):
    val_set_loss = 0.0
    for val_cond_batch, val_x_batch, val_time_batch in val_dl:
        val_x_batch = val_x_batch.to(config.device)
        val_cond_batch = val_cond_batch.to(config.device)

        val_batch_loss = eval_step_fn(state, val_x_batch, val_cond_batch)

        # Progress
        val_set_loss += val_batch_loss.item()
        val_set_loss = val_set_loss / len(val_dl)

    return val_set_loss


@Timer(name="train", text="{name}: {minutes:.1f} minutes", logger=logging.info)
def train(config, workdir):
    os.makedirs(workdir, exist_ok=True)

    gfile_stream = open(os.path.join(workdir, "stdout.txt"), "w")
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Create transform saving directory
    transform_dir = os.path.join(workdir, "transforms")
    os.makedirs(transform_dir, exist_ok=True)

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    logging.info(f"Starting {os.path.basename(__file__)}")

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    dataset_meta = DatasetMetadata(config.data.dataset_name)

    # Build dataloaders
    train_dl, _, _ = get_dataloader(
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.input_transform_key,
        config.data.target_transform_key,
        transform_dir,
        batch_size=config.training.batch_size,
        split="train",
        ensemble_members=dataset_meta.ensemble_members(),
        include_time_inputs=config.data.time_inputs,
        evaluation=False,
    )
    val_dl, _, _ = get_dataloader(
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.dataset_name,
        config.data.input_transform_key,
        config.data.target_transform_key,
        transform_dir,
        batch_size=config.training.batch_size,
        split="val",
        ensemble_members=dataset_meta.ensemble_members(),
        include_time_inputs=config.data.time_inputs,
        evaluation=False,
    )

    # Setup model, loss and optimiser
    num_predictors = train_dl.dataset[0][0].shape[0]
    model = torch.nn.DataParallel(
        create_model(config, num_predictors).to(device=config.device)
    )

    if config.model.loss == "MSELoss":
        criterion = torch.nn.MSELoss().to(config.device)
    else:
        raise NotImplementedError(f"Loss {config.model.loss} not supported yet!")

    if config.optim.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    state = dict(optimizer=optimizer, model=model, step=0, epoch=0)
    # Resume training when intermediate checkpoints are detected
    state, _ = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_epoch = (
        int(state["epoch"]) + 1
    )  # start from the epoch after the one currently reached

    initial_epoch = (
        int(state["epoch"]) + 1
    )  # start from the epoch after the one currently reached
    # step = state["step"]

    def loss_fn(model, batch, cond):
        return criterion(model(cond), batch)

    def optimize_fn(optimizer, params, step, lr, warmup=5000, grad_clip=1.0):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    # Compute validation loss
    def eval_step_fn(state, batch, cond):
        """Running one step of training or evaluation.

        Args:
        state: A dictionary of training information, containing the score model, optimizer,
        EMA status, and number of optimization steps.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.

        Returns:
        loss: The average loss value of this state.
        """
        model = state["model"]
        with torch.no_grad():
            loss = loss_fn(model, batch, cond)

        return loss

    def train_step_fn(state, batch, cond):
        """Running one step of training or evaluation.

        Args:
        state: A dictionary of training information, containing the score model, optimizer,
        EMA status, and number of optimization steps.
        batch: A mini-batch of training/evaluation data to model.
        cond: A mini-batch of conditioning inputs.

        Returns:
        loss: The average loss value of this state.
        """
        model = state["model"]
        optimizer = state["optimizer"]
        optimizer.zero_grad()
        loss = loss_fn(model, batch, cond)
        loss.backward()
        optimize_fn(
            optimizer, model.parameters(), step=state["step"], lr=config.optim.lr
        )
        state["step"] += 1

        return loss

    # save the config
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_name = os.path.basename(workdir)
    run_config = dict(
        dataset=config.data.dataset_name,
        input_transform_key=config.data.input_transform_key,
        target_transform_key=config.data.target_transform_key,
        architecture=config.model.name,
        name=run_name,
        loss=config.model.loss,
        time_inputs=config.data.time_inputs,
    )

    with track_run(
        EXPERIMENT_NAME, run_name, run_config, [config.model.name, "baseline"], tb_dir
    ) as (wandb_run, tb_writer):
        # Fit model

        logging.info("Starting training loop at epoch %d." % (initial_epoch,))

        for epoch in range(initial_epoch, config.training.n_epochs + 1):
            state["epoch"] = epoch
            # Update model based on training data
            model.train()

            train_set_loss = 0.0
            with logging_redirect_tqdm():
                with tqdm(
                    total=len(train_dl.dataset),
                    desc=f"Epoch {state['epoch']}",
                    unit=" timesteps",
                ) as pbar:
                    for (cond_batch, x_batch, time_batch) in train_dl:
                        cond_batch = cond_batch.to(config.device)
                        x_batch = x_batch.to(config.device)

                        train_batch_loss = train_step_fn(state, x_batch, cond_batch)
                        train_set_loss += train_batch_loss.item()

                        # Log progress so far on epoch
                        pbar.update(cond_batch.shape[0])

            train_set_loss = train_set_loss / len(train_dl)

            # Save a temporary checkpoint to resume training after each epoch
            save_checkpoint(checkpoint_meta_dir, state)

            # Report the loss on an validation dataset each epoch
            model.eval()
            val_set_loss = val_loss(config, val_dl, eval_step_fn, state)
            epoch_metrics = {
                "epoch/train/loss": train_set_loss,
                "epoch/val/loss": val_set_loss,
            }
            log_epoch(state["epoch"], epoch_metrics, wandb_run, tb_writer)
            # Checkpoint model
            if (
                state["epoch"] != 0
                and state["epoch"] % config.training.snapshot_freq == 0
            ) or state["epoch"] == config.training.n_epochs:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"epoch_{state['epoch']}.pth"
                )
                save_checkpoint(checkpoint_path, state)
                logging.info(
                    f"epoch: {state['epoch']}, checkpoint saved to {checkpoint_path}"
                )

    logging.info(f"Finished {os.path.basename(__file__)}")
