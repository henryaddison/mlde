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

# Lint as: python3
"""Debug config for training a purely deterministic model.

This is opposed to using a model ready for score-based denoising
but training it in a deterministic fashion.
"""

from ml_downscaling_emulator.configs.deterministic.default_configs import get_default_configs

def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.n_epochs = 100
  training.snapshot_freq = 20
  training.batch_size = 64

  # data
  data = config.data
  data.dataset_name = 'bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr'
  data.input_transform_key = "stan"
  data.target_transform_key = "sqrturrecen"
  data.input_transform_dataset = None
  data.time_inputs = False

  # model
  model = config.model
  model.name = 'det_cunet'
  model.ema_rate = 1 # basically disables EMA

  # optimizer
  optim = config.optim
  optim.optimizer = "Adam"
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.weight_decay = 0
  optim.warmup = -1 # 5000
  optim.grad_clip = -1. # 1.
  return config
