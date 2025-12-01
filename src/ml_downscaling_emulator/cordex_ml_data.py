"""Loading CORDEX ML data into PyTorch"""

import cftime
import cf_xarray  # noqa: F401
import gc
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from mlde_utils.transforms import build_input_transform, build_target_transform

DATA_PATH = Path(os.getenv("DATA_PATH"))
DATASETS_PATH = DATA_PATH / "datasets"

TIME_RANGE = (
    cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
    cftime.Datetime360Day(2080, 11, 30, 12, 0, 0, 0, has_year_zero=True),
)

SPLIT_YEARS = {
    "ESD_pseudo_reality": {
        "train": list(range(1961, 1975)),
        "val": list(range(1975, 1980 + 1)),
    },
    "Emulator_hist_future": {
        "train": list(range(1961, 1980 + 1)) + list(range(2080, 2090)),
        "val": list(range(2090, 2099 + 1)),
    },
}


def get_variables(_dataset_name):
    predictor_variables = [
        f"{v}_{p}" for v in ["t", "u", "v", "z", "q"] for p in [500, 700, 850]
    ]
    target_variables = ["pr", "tasmax"]
    return predictor_variables, target_variables


def open_raw_dataset_split_predictors(
    dataset_name,
    split,
):
    domain, experiment, gcm_name, framework = dataset_name.split("-")
    split_dir = split
    if split == "val":
        split_dir = "train"

    experiment_path = DATASETS_PATH / dataset_name / split_dir

    predictor_filepath = experiment_path / "predictors" / "Variable_fields.nc"
    predictor_ds = xr.open_dataset(predictor_filepath)

    if split in ["train", "val"]:
        split_years = SPLIT_YEARS[experiment][split]

        split_mask = predictor_ds["time.year"].isin(split_years)
        predictor_ds = predictor_ds.sel(time=split_mask)

    return predictor_ds


def open_raw_dataset_split_predictands(
    dataset_name,
    split,
):
    domain, experiment, gcm_name, framework = dataset_name.split("-")
    split_dir = split
    if split == "val":
        split_dir = "train"

    experiment_path = DATASETS_PATH / dataset_name / split_dir

    predictand_filepath = experiment_path / "target" / "pr_tasmax.nc"
    predictand_ds = xr.open_dataset(predictand_filepath)

    if split in ["train", "val"]:
        split_years = SPLIT_YEARS[experiment][split]

        split_mask = predictand_ds["time.year"].isin(split_years)
        predictand_ds = predictand_ds.sel(time=split_mask)

    return predictand_ds


def get_predictor_transform(
    dataset_name,
    key,
    variables,
    transform_dir,
):
    ds = open_raw_dataset_split_predictors(
        dataset_name,
        "train",
    )

    input_transform = build_input_transform(variables, key)

    input_transform.fit(ds, ds)

    ds.close()
    del ds
    gc.collect()

    return input_transform


def get_target_transform(
    dataset_name,
    keys,
    variables,
    transform_dir,
):

    ds = open_raw_dataset_split_predictands(
        dataset_name,
        "train",
    )

    target_transform = build_target_transform(variables, keys)

    target_transform.fit(ds, ds)

    ds.close()
    del ds
    gc.collect()

    return target_transform


def get_dataloader(
    dataset_name,
    predictor_variables,
    target_variables,
    transform,
    target_transform,
    split,
    batch_size,
    shuffle=True,
    training=True,
):

    predictor_variables, target_variables = get_variables(dataset_name)

    predictor_ds = open_raw_dataset_split_predictors(
        dataset_name,
        split,
    )

    predictor_ds = transform.transform(predictor_ds)

    if training:
        predictand_ds = open_raw_dataset_split_predictands(
            dataset_name,
            split,
        )
        predictand_ds = target_transform.transform(predictand_ds)
        pt_dataset = CordexMLTrainingDataset(
            predictor_ds, predictand_ds, predictor_variables, target_variables
        )
    else:
        pt_dataset = CordexMLDataset(
            predictor_ds, predictor_variables, target_variables
        )

    def custom_collate(batch):
        from torch.utils.data import default_collate

        return *default_collate([tuple(e[:-1]) for e in batch]), np.concatenate(
            [e[-1] for e in batch]
        )

    data_loader = DataLoader(
        pt_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate
    )

    return data_loader


class CordexMLDataset(Dataset):
    def __init__(self, predictor_ds, variables, target_variables):
        self.predictor_da = predictor_ds[variables].cf.transpose("T", "Y", "X")

        self.variables = variables
        self.target_variables = target_variables

    def __len__(self):
        return len(self.predictor_da.time)

    def __getitem__(self, idx):
        predictors = torch.tensor(
            # stack features before lat-lon (HW)
            np.stack(
                [self.predictor_da[var].isel(time=idx) for var in self.variables],
                axis=-3,
            ),
            dtype=torch.float32,
        )

        time = self.predictor_da.isel(time=idx)["time"].values.reshape(-1)

        return predictors, time


class CordexMLTrainingDataset(CordexMLDataset):
    def __init__(self, predictor_ds, predictand_ds, variables, target_variables):
        super().__init__(predictor_ds, variables, target_variables)

        self.predictand_da = predictand_ds[target_variables].cf.transpose("T", "Y", "X")

    def __getitem__(self, idx):
        predictors, time = super().__getitem__(idx)

        predictands = torch.tensor(
            # stack features before lat-lon (HW)
            np.stack(
                [
                    self.predictand_da[var].isel(time=idx)
                    for var in self.target_variables
                ],
                axis=-3,
            ),
            dtype=torch.float32,
        )

        return predictors, predictands, time


def np_samples_to_xr(
    np_samples, target_transform, target_vars, dims, var_attrs, coords, cf_data_vars
):
    """
    Convert samples from a model in numpy format to an xarray Dataset, including inverting any transformation applied to the target variables before modelling.
    """
    coords = {**dict(coords)}

    data_vars = {**cf_data_vars}
    for var_idx, var in enumerate(target_vars):
        np_var_pred = np_samples[:, var_idx, :]
        pred_var = (dims, np_var_pred, var_attrs[var])
        raw_pred_var = (
            dims,
            np_var_pred,
            {},
        )
        data_vars.update(
            {
                var: pred_var,  # don't rename pred var until after inverting target transform
                f"raw_pred_{var}": raw_pred_var,
            }
        )

    samples_ds = target_transform.invert(
        xr.Dataset(data_vars=data_vars, coords=coords, attrs={})
    ).rename({var: f"pred_{var}" for var in target_vars})

    # Re-assign attributes as target_transform inversion removes them
    for var_idx, var in enumerate(target_vars):
        samples_ds[f"pred_{var}"] = samples_ds[f"pred_{var}"].assign_attrs(
            var_attrs[var]
        )
    return samples_ds
