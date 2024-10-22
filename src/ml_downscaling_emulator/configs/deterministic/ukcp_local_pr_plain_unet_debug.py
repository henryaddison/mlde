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

from ml_downscaling_emulator.configs.deterministic.ukcp_local_pr_12em_tuned_plain_unet import get_config as get_default_configs

def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.n_epochs = 2
  training.snapshot_freq = 5
  training.eval_freq = 100
  training.log_freq = 50
  training.batch_size = 2

  # data
  data = config.data
  data.dataset_name = 'debug-sample'
  data.time_inputs = True

  return config
