# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Modifications copyright 2025 Henry Addison
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
"""Default config for simple U-Net on CORDEX-ML data used in a deterministic fashion.
"""

from ml_downscaling_emulator.configs.deterministic.default_configs import get_default_configs

def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.n_epochs = 200
  training.snapshot_freq = 20
  training.batch_size = 256

  # data
  data = config.data
  data.image_size = 128
  data.predictor_image_size = 16
  data.input_transform_key = "stan"
  data.target_transform_key = "sqrturrecen"
  data.input_transform_dataset = None
  data.time_inputs = False

  # model
  model = config.model
  model.name = 'det_cunet'

  # optimizer
  optim = config.optim
  optim.optimizer = "Adam"
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.weight_decay = 0
  optim.warmup = 5000
  optim.grad_clip = 1.
  return config
