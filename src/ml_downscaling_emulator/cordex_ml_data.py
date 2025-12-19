"""Loading CORDEX ML data into PyTorch"""

import logging
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

VAL_SPLIT_YEARS = [1967, 1975, 2087, 2095]

logger = logging.getLogger(__name__)


def get_variables(config):
    predictor_variables = [
        f"{v}_{p}" for v in ["t", "u", "v", "z", "q"] for p in [500, 700, 850]
    ]
    target_variables = config.data.target_variables
    static_variables = config.data.static_variables

    return predictor_variables, static_variables, target_variables


def _experiment_path(dataset_name, split):
    split_dir = split
    if split == "val":
        split_dir = "train"
    elif split == "train+val":
        split_dir = "train"

    return DATASETS_PATH / dataset_name / split_dir


def _open_raw_split(filepath, split):
    ds = xr.open_dataset(filepath)

    if split in ["train", "val"]:
        split_mask = ds["time.year"].isin(VAL_SPLIT_YEARS)
        if split == "train":
            split_mask = ~split_mask
        ds = ds.sel(time=split_mask)

    if "axis" not in ds["time"].attrs:
        ds["time"].attrs["axis"] = "T"
    return ds


def open_raw_dataset_split_predictands(
    dataset_name,
    split,
):
    experiment_path = _experiment_path(dataset_name, split)

    filepath = experiment_path / "target" / "pr_tasmax.nc"

    return _open_raw_split(filepath, split)


def open_raw_dataset_split_predictors(
    dataset_name,
    split,
):
    experiment_path = _experiment_path(dataset_name, split)

    filepath = experiment_path / "predictors" / "Variable_fields.nc"

    return _open_raw_split(filepath, split)


def open_raw_dataset_split_static_inputs(
    dataset_name,
    split,
):
    experiment_path = _experiment_path(dataset_name, split)

    filepath = experiment_path / "predictors" / "Static_fields.nc"

    return xr.open_dataset(filepath)


def get_predictor_transform(
    dataset_name,
    key,
    variables,
    transform_dir,
):
    logger.debug("Opening training predictor dataset for input transform fitting")
    ds = open_raw_dataset_split_predictors(
        dataset_name,
        "train",
    )

    logger.debug("Building input transform object")
    input_transform = build_input_transform(variables, key)

    logger.debug("Fitting input transform")
    input_transform.fit(ds, ds)

    logger.debug("Memory cleanup after input transform fitting")
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
    logger.debug("Opening training predictand dataset for target transform fitting")
    ds = open_raw_dataset_split_predictands(
        dataset_name,
        "train",
    )

    logger.debug("Building target transform object")
    target_transform = build_target_transform(variables, keys)

    logger.debug("Fitting target transform")
    target_transform.fit(ds, ds)

    logger.debug("Memory cleanup after target transform fitting")
    ds.close()
    del ds
    gc.collect()

    return target_transform


def get_dataloader(
    dataset_name,
    predictor_variables,
    static_variables,
    target_variables,
    transform,
    target_transform,
    split,
    batch_size,
    shuffle=True,
    training=True,
):
    predictor_ds = open_raw_dataset_split_predictors(
        dataset_name,
        split,
    )

    static_ds = open_raw_dataset_split_static_inputs(
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
            predictor_ds,
            static_ds,
            predictand_ds,
            predictor_variables,
            static_variables,
            target_variables,
        )
    else:
        pt_dataset = CordexMLDataset(
            predictor_ds, static_ds, predictor_variables, static_variables
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
    def __init__(self, predictor_ds, static_ds, variables, static_variables):
        self.variables = list(variables)
        self.static_variables = list(static_variables)

        self.predictor_da = predictor_ds[self.variables].cf.transpose("T", "Y", "X")
        self.static_da = static_ds.cf.transpose("Y", "X")[self.static_variables]

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

        if len(self.static_variables) > 0:
            statics = torch.tensor(
                # stack features before lat-lon (HW)
                np.stack(
                    [self.static_da[var] for var in self.static_variables],
                    axis=-3,
                ),
                dtype=torch.float32,
            )
        else:
            statics = torch.tensor([])

        time = self.predictor_da.isel(time=idx)["time"].values.reshape(-1)

        return predictors, statics, time


class CordexMLTrainingDataset(CordexMLDataset):
    def __init__(
        self,
        predictor_ds,
        static_ds,
        predictand_ds,
        variables,
        static_variables,
        target_variables,
    ):
        super().__init__(predictor_ds, static_ds, variables, static_variables)

        self.target_variables = list(target_variables)

        self.predictand_da = predictand_ds[self.target_variables].cf.transpose(
            "T", "Y", "X"
        )

    def __getitem__(self, idx):
        predictors, statics, time = super().__getitem__(idx)

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

        return predictors, statics, predictands, time


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
