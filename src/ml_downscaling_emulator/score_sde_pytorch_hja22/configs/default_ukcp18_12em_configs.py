import ml_collections
import torch

from ml_downscaling_emulator.score_sde_pytorch_hja22.configs.default_ukcp18_configs import get_default_configs as get_base_configs


def get_default_configs():
  config = get_base_configs()

  # training
  training = config.training
  training.n_epochs = 20
  training.snapshot_freq = 5
  training.eval_freq = 5000

  # data
  data = config.data
  data.dataset_name = 'bham_gcmx-4x_12em_psl-temp4th-vort4th_eqvt_random-season'
  data.input_transform_key = "stan"
  data.target_transform_key = "sqrturrecen"

  return config
