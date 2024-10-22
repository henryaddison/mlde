import ml_collections
import torch

from ml_downscaling_emulator.configs.default_ukcp_local_pr_1em_configs import get_default_configs as get_base_configs


def get_default_configs():
  config = get_base_configs()

  # training
  training = config.training
  training.n_epochs = 20
  training.snapshot_freq = 5
  training.eval_freq = 5000

  # data
  data = config.data
  data.dataset_name = 'bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr'

  return config
