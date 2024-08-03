"""Loading UKCP Local data into PyTorch"""

import cftime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from mlde_utils.training.dataset import get_dataset, get_variables

TIME_RANGE = (
    cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
    cftime.Datetime360Day(2080, 11, 30, 12, 0, 0, 0, has_year_zero=True),
)


class UKCPLocalDataset(Dataset):
    def __init__(self, ds, variables, target_variables, time_range):
        self.ds = ds
        self.variables = variables
        self.target_variables = target_variables
        self.time_range = time_range

    @classmethod
    def variables_to_tensor(cls, ds, variables):
        return torch.tensor(
            # stack features before lat-lon (HW)
            np.stack([ds[var].values for var in variables], axis=-3)
        ).float()

    @classmethod
    def time_to_tensor(cls, ds, shape, time_range):
        climate_time = np.array(ds["time"] - time_range[0]) / np.array(
            [time_range[1] - time_range[0]], dtype=np.dtype("timedelta64[ns]")
        )
        season_time = ds["time.dayofyear"].values / 360

        return (
            torch.stack(
                [
                    torch.tensor(climate_time).broadcast_to(
                        (climate_time.shape[0], *shape[-2:])
                    ),
                    torch.sin(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                    torch.cos(
                        2
                        * np.pi
                        * torch.tensor(season_time).broadcast_to(
                            (climate_time.shape[0], *shape[-2:])
                        )
                    ),
                ],
                dim=-3,
            )
            .squeeze()
            .float()
        )

    def __len__(self):
        return len(self.ds.time) * len(self.ds.ensemble_member)

    def __getitem__(self, idx):
        subds = self.sel(idx)

        cond = self.variables_to_tensor(subds, self.variables)
        if self.time_range is not None:
            cond_time = self.time_to_tensor(subds, cond.shape, self.time_range)
            cond = torch.cat([cond, cond_time])

        x = self.variables_to_tensor(subds, self.target_variables)

        time = subds["time"].values.reshape(-1)

        return cond, x, time

    def sel(self, idx):
        em_idx, time_idx = divmod(idx, len(self.ds.time))
        return self.ds.isel(time=time_idx, ensemble_member=em_idx)


def build_dataloader(
    xr_data, variables, target_variables, batch_size, shuffle, include_time_inputs
):
    def custom_collate(batch):
        from torch.utils.data import default_collate

        return *default_collate([(e[0], e[1]) for e in batch]), np.concatenate(
            [e[2] for e in batch]
        )

    time_range = None
    if include_time_inputs:
        time_range = TIME_RANGE
    xr_dataset = UKCPLocalDataset(xr_data, variables, target_variables, time_range)
    data_loader = DataLoader(
        xr_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate
    )
    return data_loader


def get_dataloader(
    active_dataset_name,
    model_src_dataset_name,
    input_transform_dataset_name,
    input_transform_key,
    target_transform_key,
    transform_dir,
    batch_size,
    split,
    ensemble_members,
    include_time_inputs,
    evaluation=False,
    shuffle=True,
):
    """Create data loaders for given split.

    Args:
      active_dataset_name: Name of dataset from which to load data splits
      model_src_dataset_name: Name of dataset used to train the diffusion model (may be the same)
      input_transform_dataset_name: Name of dataset to use for fitting input transform (may be the same as active_dataset_name or model_src_dataset_name)
      transform_dir: Path to where transforms should be stored
      input_transform_key: Name of input transform pipeline to use
      target_transform_key: Name of target transform pipeline to use
      batch_size: Size of batch to use for DataLoaders
      split: Split of the active dataset to load
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      data_loader, transform, target_transform.
    """
    xr_data, transform, target_transform = get_dataset(
        active_dataset_name,
        model_src_dataset_name,
        input_transform_dataset_name,
        input_transform_key,
        target_transform_key,
        transform_dir,
        split,
        ensemble_members,
        evaluation,
    )

    variables, target_variables = get_variables(model_src_dataset_name)

    data_loader = build_dataloader(
        xr_data,
        variables,
        target_variables,
        batch_size,
        shuffle,
        include_time_inputs,
    )

    return data_loader, transform, target_transform


def np_samples_to_xr(np_samples, target_transform, target_vars, coords, cf_data_vars):
    """
    Convert samples from a model in numpy format to an xarray Dataset, including inverting any transformation applied to the target variables before modelling.
    """
    coords = {**dict(coords)}

    pred_dims = ["ensemble_member", "time", "grid_latitude", "grid_longitude"]

    data_vars = {**cf_data_vars}
    for var_idx, var in enumerate(target_vars):
        # add ensemble member axis to np samples and get just values for current variable
        np_var_pred = np_samples[np.newaxis, :, var_idx, :]
        pred_attrs = {
            "grid_mapping": "rotated_latitude_longitude",
            "standard_name": var.replace("target_", "pred_"),
            # "units": "kg m-2 s-1",
        }
        pred_var = (pred_dims, np_var_pred, pred_attrs)
        raw_pred_var = (
            pred_dims,
            {"grid_mapping": "rotated_latitude_longitude"},
        )
        data_vars.update(
            {
                var.replace("target_", "pred_"): pred_var,
                var.replace("target_", "raw_pred_"): raw_pred_var,
            }
        )

    samples_ds = target_transform.invert(
        xr.Dataset(data_vars=data_vars, coords=coords, attrs={})
    )
    samples_ds = samples_ds.rename(
        {var: var.replace("target_", "pred_") for var in target_vars}
    )

    for var_idx, var in enumerate(target_vars):
        pred_attrs = {
            "grid_mapping": "rotated_latitude_longitude",
            "standard_name": var.replace("target_", "pred_"),
            # "units": "kg m-2 s-1",
        }
        samples_ds[var.replace("target_", "pred_")] = samples_ds[
            var.replace("target_", "pred_")
        ].assign_attrs(pred_attrs)
    return samples_ds
