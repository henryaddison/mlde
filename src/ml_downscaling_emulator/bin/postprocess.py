import glob
import logging
import os
from pathlib import Path
from typing import Callable
import typer
import xarray as xr

from mlde_utils import samples_path, samples_glob, TIME_PERIODS
from mlde_utils.training.dataset import open_raw_dataset_split

from ml_downscaling_emulator.postprocess import xrqm, to_gcm_domain

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


def process_each_sample(
    workdir: Path,
    checkpoint: str,
    dataset: str,
    ensemble_member: str,
    input_xfm: str,
    split: str,
    processing_func: Callable,
    new_workdir: Path,
):
    samples_dirpath = samples_path(
        workdir,
        checkpoint=checkpoint,
        input_xfm=input_xfm,
        dataset=dataset,
        split=split,
        ensemble_member=ensemble_member,
    )
    logger.info(f"Iterating on samples in {samples_dirpath}")
    for sample_filepath in samples_glob(samples_dirpath):
        logger.info(f"Working on {sample_filepath}")
        # open the samples
        samples_ds = xr.open_dataset(sample_filepath)

        processed_samples_ds = processing_func(samples_ds)

        # save output
        processed_sample_filepath = (
            samples_path(
                new_workdir,
                checkpoint=checkpoint,
                input_xfm=input_xfm,
                dataset=dataset,
                split=split,
                ensemble_member=ensemble_member,
            )
            / sample_filepath.name
        )

        logger.info(f"Saving to {processed_sample_filepath}")
        processed_sample_filepath.parent.mkdir(parents=True, exist_ok=True)
        processed_samples_ds.to_netcdf(processed_sample_filepath)


@app.command()
def filter(
    workdir: Path,
    dataset: str = typer.Option(...),
    time_period: str = typer.Option(...),
    checkpoint: str = typer.Option(...),
    input_xfm: str = "stan",
    split: str = "val",
    ensemble_member: str = typer.Option(...),
):
    """Filter a set of samples based on time period."""

    new_dataset = f"{dataset}-{time_period}"
    filtered_samples_dirpath = samples_path(
        workdir,
        checkpoint=checkpoint,
        input_xfm=input_xfm,
        dataset=new_dataset,
        split=split,
        ensemble_member=ensemble_member,
    )
    os.makedirs(filtered_samples_dirpath, exist_ok=False)

    samples_filepaths_to_filter = samples_path(
        workdir,
        checkpoint=checkpoint,
        input_xfm=input_xfm,
        dataset=dataset,
        split=split,
        ensemble_member=ensemble_member,
    )

    logger.info(f"Found for filtering: {samples_filepaths_to_filter}")
    for sample_filepath in samples_glob(samples_filepaths_to_filter):
        logger.info(f"Working on {sample_filepath}")
        samples_ds = xr.open_dataset(sample_filepath)

        filtered_samples_filepath = filtered_samples_dirpath / sample_filepath.name

        logger.info(f"Saving to {filtered_samples_filepath}")
        samples_ds.sel(time=slice(*TIME_PERIODS[time_period])).to_netcdf(
            filtered_samples_filepath
        )


@app.command()
def qm(
    workdir: Path,
    checkpoint: str = typer.Option(...),
    train_dataset: str = typer.Option(...),
    train_input_xfm: str = "stan",
    eval_dataset: str = typer.Option(...),
    eval_input_xfm: str = "stan",
    split: str = "val",
    ensemble_member: str = typer.Option(...),
):
    # to compute the mapping, use train split data
    # open train split of dataset for the target_pr
    sim_train_da = open_raw_dataset_split(train_dataset, "train").sel(
        ensemble_member=ensemble_member
    )["target_pr"]

    # open sample of model from train split
    ml_train_da = xr.open_dataset(
        list(
            samples_glob(
                samples_path(
                    workdir,
                    checkpoint=checkpoint,
                    input_xfm=train_input_xfm,
                    dataset=train_dataset,
                    split="train",
                    ensemble_member=ensemble_member,
                )
            )
        )[0]
    )["pred_pr"]

    def process_samples(ds):
        # do the qmapping
        qmapped_eval_da = xrqm(sim_train_da, ml_train_da, ds["pred_pr"])

        processed_samples_ds = ds.copy()
        processed_samples_ds["pred_pr"] = qmapped_eval_da

        return processed_samples_ds

    process_each_sample(
        workdir,
        checkpoint,
        eval_dataset,
        ensemble_member,
        eval_input_xfm,
        split,
        process_samples,
        new_workdir=workdir / "postprocess" / "qm",
    )


@app.command()
def gcmify(
    workdir: Path,
    checkpoint: str = typer.Option(...),
    dataset: str = typer.Option(...),
    input_xfm: str = typer.Option(...),
    split: str = typer.Option(...),
    ensemble_member: str = typer.Option(...),
):
    def process_samples(ds):
        ds = to_gcm_domain(ds.sel(ensemble_member=ensemble_member))
        ds = ds.expand_dims({"ensemble_member": ensemble_member})
        return ds

    process_each_sample(
        workdir,
        checkpoint,
        dataset,
        ensemble_member,
        input_xfm,
        split,
        process_samples,
        new_workdir=workdir / "postprocess" / "gcm-grid",
    )


@app.command()
def merge(
    input_dirs: list[Path],
    output_dir: Path,
):
    pred_file_globs = [
        sorted(glob.glob(os.path.join(samples_dir, "*.nc")))
        for samples_dir in input_dirs
    ]
    # there should be the same number of samples in each input dir
    assert 1 == len(set(map(len, pred_file_globs)))

    for pred_file_group in zip(*pred_file_globs):
        typer.echo(f"Concat {pred_file_group}")

        # take a bit of the random id in each sample file's name
        random_ids = [fn[-25:] for fn in pred_file_group]
        if len(set(random_ids)) == 1:
            # if all the random ids are the same (they are from the same sampling run), just use one of them for the output filepath
            output_filepath = os.path.join(output_dir, f"predictions-{random_ids[0]}")
        else:
            # join those partial random ids together for the output filepath in the train directory (rather than one of the subset train dirs)
            random_ids = [rid[:5] for rid in random_ids]
            output_filepath = os.path.join(
                output_dir, f"predictions-{'-'.join(random_ids)}.nc"
            )

        typer.echo(f"save to {output_filepath}")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        xr.concat([xr.open_dataset(f) for f in pred_file_group], dim="time").to_netcdf(
            output_filepath
        )
