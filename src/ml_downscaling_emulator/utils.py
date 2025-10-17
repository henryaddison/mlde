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
# to allow for location-specific parameters and iterating by epoch using PyTorch
# DataLoaders and helpers for determining a model size.

import torch
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state, False
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["location_params"].load_state_dict(loaded_state["location_params"])
        state["step"] = loaded_state["step"]
        state["epoch"] = loaded_state["epoch"]
        logging.info(
            f"Checkpoint found at {ckpt_dir}. "
            f"Returned the state from {state['epoch']}/{state['step']}"
        )
        return state, True


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
        "epoch": state["epoch"],
        "location_params": state["location_params"].state_dict(),
    }
    torch.save(saved_state, ckpt_dir)


def param_count(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def model_size(model):
    """Compute size in memory of model in MB."""
    param_size = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in model.buffers()
    )

    return (param_size + buffer_size) / 1024**2
