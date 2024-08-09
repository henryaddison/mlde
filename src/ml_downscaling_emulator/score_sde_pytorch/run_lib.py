# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Modifications copyright 2024 Henry Addison
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
# to allow for conditional modelling, location-specific parameters,
# removal of tensorflow dependency, tracking for training via Weights and Biases
# and MLFlow, and iterating by epoch using PyTorch DataLoaders

# pylint: skip-file
"""Training for score-based generative models. """

import itertools
import os

from codetiming import Timer
import logging
# Keep the import below for registering all model definitions
from .models import det_cunet, cunet, cncsnpp
from . import losses
from .models.location_params import LocationParams
from . import sampling
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from . import likelihood
from . import sde_lib
from absl import flags
import torch
import torchvision
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import save_checkpoint, restore_checkpoint

from ml_downscaling_emulator.data import get_dataloader
from mlde_utils import DatasetMetadata
from ml_downscaling_emulator.training import log_epoch, track_run

FLAGS = flags.FLAGS

EXPERIMENT_NAME = os.getenv("WANDB_EXPERIMENT_NAME")

def val_loss(config, eval_dl, eval_step_fn, state):
  val_set_loss = 0.0
  # use a consistent generator for computing validation set loss
  # so value is not down to vagaries of random choice of initial noise samples or schedules
  g = torch.Generator(device=config.device)
  g.manual_seed(42)
  for eval_cond_batch, eval_target_batch, eval_time_batch in eval_dl:
    # eval_cond_batch, eval_target_batch = next(iter(eval_dl))
    eval_target_batch = eval_target_batch.to(config.device)
    eval_cond_batch = eval_cond_batch.to(config.device)
    # append any location-specific parameters
    eval_cond_batch = state['location_params'](eval_cond_batch)
    # eval_batch = eval_batch.permute(0, 3, 1, 2)
    eval_loss = eval_step_fn(state, eval_target_batch, eval_cond_batch, generator=g)

    # Progress
    val_set_loss += eval_loss.item()
    val_set_loss = val_set_loss/len(eval_dl)

  return val_set_loss


@Timer(name="train", text="{name}: {minutes:.1f} minutes", logger=logging.info)
def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # save the config
  config_path = os.path.join(workdir, "config.yml")
  with open(config_path, 'w') as f:
    f.write(config.to_yaml())

  # Create transform saving directory
  transform_dir = os.path.join(workdir, "transforms")
  os.makedirs(transform_dir, exist_ok=True)

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True)

  tb_dir = os.path.join(workdir, "tensorboard")
  os.makedirs(tb_dir, exist_ok=True)

  run_name = os.path.basename(workdir)
  run_config = dict(
        dataset=config.data.dataset_name,
        input_transform_key=config.data.input_transform_key,
        target_transform_key=config.data.target_transform_key,
        architecture=config.model.name,
        sde=config.training.sde,
        name=run_name,
    )

  with track_run(
        EXPERIMENT_NAME, run_name, run_config, ["score_sde"], tb_dir
    ) as (wandb_run, writer):
    # Build dataloaders
    dataset_meta = DatasetMetadata(config.data.dataset_name)
    train_dl, _, _ = get_dataloader(config.data.dataset_name, config.data.dataset_name, config.data.dataset_name, config.data.input_transform_key, config.data.target_transform_key, transform_dir, batch_size=config.training.batch_size, split="train", ensemble_members=dataset_meta.ensemble_members(), include_time_inputs=config.data.time_inputs, evaluation=False)
    eval_dl, _, _ = get_dataloader(config.data.dataset_name, config.data.dataset_name, config.data.dataset_name, config.data.input_transform_key, config.data.target_transform_key, transform_dir, batch_size=config.training.batch_size, split="val", ensemble_members=dataset_meta.ensemble_members(), include_time_inputs=config.data.time_inputs, evaluation=False, shuffle=False)

    # Initialize model.
    score_model = mutils.create_model(config)
    # include a learnable feature map
    location_params = LocationParams(config.model.loc_spec_channels, config.data.image_size)
    location_params = location_params.to(config.device)
    location_params = torch.nn.DataParallel(location_params)
    ema = ExponentialMovingAverage(itertools.chain(score_model.parameters(), location_params.parameters()), decay=config.model.ema_rate)

    optimizer = losses.get_optimizer(config, itertools.chain(score_model.parameters(), location_params.parameters()))
    state = dict(optimizer=optimizer, model=score_model, location_params=location_params, ema=ema, step=0, epoch=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state, _ = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_epoch = int(state['epoch'])+1 # start from the epoch after the one currently reached

    # Setup SDEs
    if config.deterministic:
      sde = None
    else:
      if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
      elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
      elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
      else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting,
                                      deterministic=config.deterministic,)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting,
                                      deterministic=config.deterministic,)

    num_train_epochs = config.training.n_epochs

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at epoch %d." % (initial_epoch,))

    if config.training.random_crop_size > 0:
      random_crop = torchvision.transforms.RandomCrop(config.training.random_crop_size)

    for epoch in range(initial_epoch, num_train_epochs + 1):
      state['epoch'] = epoch
      train_set_loss = 0.0
      with logging_redirect_tqdm():
        with tqdm(total=len(train_dl.dataset), desc=f"Epoch {state['epoch']}", unit=' timesteps') as pbar:
          for cond_batch, target_batch, time_batch in train_dl:

            target_batch = target_batch.to(config.device)
            cond_batch = cond_batch.to(config.device)
            # append any location-specific parameters
            cond_batch = state['location_params'](cond_batch)

            if config.training.random_crop_size > 0:
              x_ch = target_batch.shape[1]
              cropped = random_crop(torch.cat([target_batch, cond_batch], dim=1))
              target_batch = cropped[:,:x_ch]
              cond_batch = cropped[:,x_ch:]

            # Execute one training step
            loss = train_step_fn(state, target_batch, cond_batch)
            train_set_loss += loss.item()
            if state['step'] % config.training.log_freq == 0:
              logging.info("epoch: %d, step: %d, train_loss: %.5e" % (state['epoch'], state['step'], loss.item()))
              writer.add_scalar("step/train/loss", loss.cpu().detach(), global_step=state['step'])

            # Report the loss on an evaluation dataset periodically
            if state['step'] % config.training.eval_freq == 0:
              val_set_loss = val_loss(config, eval_dl, eval_step_fn, state)
              logging.info("epoch: %d, step: %d, val_loss: %.5e" % (state['epoch'], state['step'], val_set_loss))
              writer.add_scalar("step/val/loss", val_set_loss, global_step=state['step'])

            # Log progress so far on epoch
            pbar.update(cond_batch.shape[0])

      train_set_loss = train_set_loss / len(train_dl)
      # Save a temporary checkpoint to resume training after each epoch
      save_checkpoint(checkpoint_meta_dir, state)
      # Report the loss on an evaluation dataset each epoch
      val_set_loss = val_loss(config, eval_dl, eval_step_fn, state)
      epoch_metrics = {"epoch/train/loss": train_set_loss, "epoch/val/loss": val_set_loss}

      log_epoch(state['epoch'], epoch_metrics, wandb_run, writer)

      if (state['epoch'] != 0 and state['epoch'] % config.training.snapshot_freq == 0) or state['epoch'] == num_train_epochs:
        # Save the checkpoint.
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{state['epoch']}.pth")
        save_checkpoint(checkpoint_path, state)
        logging.info(f"epoch: {state['epoch']}, checkpoint saved to {checkpoint_path}")
