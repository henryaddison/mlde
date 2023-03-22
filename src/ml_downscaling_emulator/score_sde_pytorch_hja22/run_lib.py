# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import itertools
import os

from codetiming import Timer
# import tensorflow as tf
# import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
# from models import ddpm, ncsnv2, ncsnpp
from .models import cunet
from .models import cncsnpp
from . import losses
from .models.location_params import LocationParams
from . import sampling
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
# import .evaluation
from . import likelihood
from . import sde_lib
from absl import flags
import torch
import torchvision
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import save_checkpoint, restore_checkpoint

from ml_downscaling_emulator.torch import get_dataloader

FLAGS = flags.FLAGS

def val_loss(config, eval_ds, eval_step_fn, state):
  val_set_loss = 0.0
  for eval_cond_batch, eval_x_batch in eval_ds:
    # eval_cond_batch, eval_x_batch = next(iter(eval_ds))
    eval_x_batch = eval_x_batch.to(config.device)
    eval_cond_batch = eval_cond_batch.to(config.device)
    # append any location-specific parameters
    eval_cond_batch = state['location_params'](eval_cond_batch)
    # eval_batch = eval_batch.permute(0, 3, 1, 2)
    eval_loss = eval_step_fn(state, eval_x_batch, eval_cond_batch)

    # Progress
    val_set_loss += eval_loss.item()
    val_set_loss = val_set_loss/len(eval_ds)

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

  writer = SummaryWriter(tb_dir)

  # Build dataloaders
  # TODO: in EV/storm case need a  way to load data from numpy files rather than netcdf
  # then transform inputs and outputs with log(1+x)
  train_dl, _, _ = get_dataloader(config.data.dataset_name, config.data.dataset_name, config.data.input_transform_key, config.data.target_transform_key, transform_dir, batch_size=config.training.batch_size, split="train", evaluation=False)
  eval_dl, _, _ = get_dataloader(config.data.dataset_name, config.data.dataset_name, config.data.input_transform_key, config.data.target_transform_key, transform_dir, batch_size=config.training.batch_size, split="val", evaluation=False)

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
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_epoch = int(state['epoch'])

  # Setup SDEs
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
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  num_train_epochs = config.training.n_epochs

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at epoch %d." % (initial_epoch,))

  if config.training.random_crop_size > 0:
    random_crop = torchvision.transforms.RandomCrop(config.training.random_crop_size)

  step = state["step"]
  for epoch in range(initial_epoch, num_train_epochs + 1):
    state['epoch'] = epoch
    with logging_redirect_tqdm():
      with tqdm(total=len(train_dl.dataset), desc=f'Epoch {epoch}', unit=' timesteps') as pbar:
        for cond_batch, x_batch in train_dl:

          x_batch = x_batch.to(config.device)
          cond_batch = cond_batch.to(config.device)
          # append any location-specific parameters
          cond_batch = state['location_params'](cond_batch)

          if config.training.random_crop_size > 0:
            x_ch = x_batch.shape[1]
            cropped = random_crop(torch.cat([x_batch, cond_batch], dim=1))
            x_batch = cropped[:,:x_ch]
            cond_batch = cropped[:,x_ch:]

          # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
          # batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
          # batch = batch.permute(0, 3, 1, 2)
          # Execute one training step
          loss = train_step_fn(state, x_batch, cond_batch)
          if step % config.training.log_freq == 0:
            logging.info("epoch: %d, step: %d, training_loss: %.5e" % (epoch, step, loss.item()))
            writer.add_scalar("training_loss", loss.cpu().detach(), global_step=step)

          # Report the loss on an evaluation dataset periodically
          if step % config.training.eval_freq == 0:
            val_set_loss = val_loss(config, eval_dl, eval_step_fn, state)
            logging.info("epoch: %d, step: %d, eval_loss: %.5e" % (epoch, step, val_set_loss))
            writer.add_scalar("eval_loss", val_set_loss, global_step=step)

          # Log progress so far on epoch
          pbar.update(cond_batch.shape[0])

          step += 1

    # Save a temporary checkpoint to resume training after each epoch
    save_checkpoint(checkpoint_meta_dir, state)
    # Report the loss on an evaluation dataset each epoch
    val_set_loss = val_loss(config, eval_dl, eval_step_fn, state)
    logging.info("epoch: %d, eval_loss: %.5e" % (epoch, val_set_loss))
    writer.add_scalar("epoch_eval_loss", val_set_loss, global_step=epoch)

    if (epoch != 0 and epoch % config.training.snapshot_freq == 0) or epoch == num_train_epochs:
      # Save the checkpoint.
      checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
      save_checkpoint(checkpoint_path, state)
      logging.info(f"epoch: {epoch}, checkpoint saved to {checkpoint_path}")

  writer.flush()
  writer.close()
