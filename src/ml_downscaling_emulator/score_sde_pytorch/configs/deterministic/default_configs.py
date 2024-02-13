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
"""Defaults for training in a deterministic fashion."""

import ml_collections
import torch

def get_default_configs():
  config = ml_collections.ConfigDict()

  config.deterministic = True

  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 16#128
  training.snapshot_freq = 25
  training.log_freq = 50
  training.eval_freq = 1000
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 1000
  ## produce samples at each snapshot.
  training.snapshot_sampling = False
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.random_crop_size = 0
  training.continuous = True
  training.reduce_mean = True
  training.n_epochs = 20
  training.snapshot_freq = 5
  training.eval_freq = 5000
  training.sde = ""

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 128

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'UKCP_Local'
  data.image_size = 64
  data.random_flip = False
  data.uniform_dequantization = False
  data.time_inputs = False
  data.centered = True
  data.input_transform_key = "stan"
  data.target_transform_key = "sqrturrecen"

  # model
  config.model = model = ml_collections.ConfigDict()
  model.loc_spec_channels = 0
  model.num_scales = 0
  model.ema_rate = 0.9999

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
