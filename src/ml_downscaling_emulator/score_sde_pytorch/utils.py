import torch
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state, False
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['location_params'].load_state_dict(loaded_state['location_params'])
    state['step'] = loaded_state['step']
    state['epoch'] = loaded_state['epoch']
    logging.info(
        f"Checkpoint found at {ckpt_dir}. "
        f"Returned the state from {state['epoch']}/{state['step']}"
    )
    return state, True


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'epoch': state['epoch'],
    'location_params': state['location_params'].state_dict(),
  }
  torch.save(saved_state, ckpt_dir)
