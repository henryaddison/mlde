from codetiming import Timer
import logging
import os
from pathlib import Path
import shortuuid
import typer
import xarray as xr

from mlde_utils import samples_path, DEFAULT_ENSEMBLE_MEMBER
from mlde_utils.training.dataset import load_raw_dataset_split

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")

app = typer.Typer()


@app.callback()
def callback():
    pass


def _np_samples_to_xr(np_samples, coords, target_transform, cf_data_vars):
    coords = {**dict(coords)}

    pred_pr_dims = ["ensemble_member", "time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {
        "grid_mapping": "rotated_latitude_longitude",
        "standard_name": "pred_pr",
        "units": "kg m-2 s-1",
    }
    pred_pr_var = (pred_pr_dims, np_samples, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    pred_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={})

    if target_transform is not None:
        pred_ds = target_transform.invert(pred_ds)

    pred_ds = pred_ds.rename({"target_pr": "pred_pr"})

    return pred_ds


def _sample_id(variable: str, eval_ds: xr.Dataset) -> xr.Dataset:
    """Create a Dataset of pr samples set to the values the given variable from the dataset."""
    cf_data_vars = {
        key: eval_ds.data_vars[key]
        for key in [
            "rotated_latitude_longitude",
            "time_bnds",
            "grid_latitude_bnds",
            "grid_longitude_bnds",
        ]
        if key in eval_ds.variables
    }
    coords = eval_ds.coords
    np_samples = eval_ds[variable].data
    xr_samples = _np_samples_to_xr(
        np_samples, coords=coords, target_transform=None, cf_data_vars=cf_data_vars
    )

    return xr_samples


@app.command()
@Timer(name="sample", text="{name}: {minutes:.1f} minutes", logger=logging.info)
def as_input(
    workdir: Path,
    dataset: str = typer.Option(...),
    variable: str = "pr",
    split: str = "val",
    ensemble_member: str = DEFAULT_ENSEMBLE_MEMBER,
):
    """
    Use a given variable from the dataset to create a file of prediction samples.

    Commonly used to create samples based on an already processed variable like using a bilinearly interpolated coarse resolution variable as the predicted "high-resolution" value directly.
    """
    output_dirpath = samples_path(
        workdir=workdir,
        checkpoint=f"epoch-0",
        dataset=dataset,
        input_xfm="none",
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(output_dirpath, exist_ok=True)

    eval_ds = load_raw_dataset_split(dataset, split).sel(
        ensemble_member=[ensemble_member]
    )
    xr_samples = _sample_id(variable, eval_ds)

    output_filepath = os.path.join(output_dirpath, f"predictions-{shortuuid.uuid()}.nc")

    logger.info(f"Saving predictions to {output_filepath}")
    xr_samples.to_netcdf(output_filepath)
