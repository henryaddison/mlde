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
"""
Config for simple U-Net on multivariate SA-domain ESD-experiment data used in a deterministic fashion.
"""

from ml_downscaling_emulator.configs.deterministic.cordex_ml_defaults import get_config as get_default_configs

def get_config():
  config = get_default_configs()

  # data
  data = config.data
  data.dataset_name = 'ALPS_domain-Emulator_hist_future-CNRMCM5-perfect'

  return config
