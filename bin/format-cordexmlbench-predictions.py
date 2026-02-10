import shutil
import tempfile
import dotenv
import logging
import os
from pathlib import Path
import typer
import xarray as xr

dotenv.load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEST_YEARS = {
    "historical": "1981-2000",
    "mid_century": "2041-2060",
    "end_century": "2081-2100",
}

TRAINING_GCMS = {
    "ALPS": "CNRMCM5",
    "NZ": "ACCESSCM2",
    "SA": "ACCESSCM2",
}

DATASETS = {
    "CORDEXMLBench": {
        "NZ": [
            "NZ_domain-historical-ACCESSCM2-perfect",
            "NZ_domain-historical-ECEarth3-perfect",
            "NZ_domain-historical-ACCESSCM2-imperfect",
            "NZ_domain-historical-ECEarth3-imperfect",
            "NZ_domain-mid_century-ACCESSCM2-perfect",
            "NZ_domain-mid_century-ECEarth3-perfect",
            "NZ_domain-mid_century-ACCESSCM2-imperfect",
            "NZ_domain-mid_century-ECEarth3-imperfect",
            "NZ_domain-end_century-ACCESSCM2-perfect",
            "NZ_domain-end_century-ECEarth3-perfect",
            "NZ_domain-end_century-ACCESSCM2-imperfect",
            "NZ_domain-end_century-ECEarth3-imperfect",
        ],
        "SA": [
            "SA_domain-historical-ACCESSCM2-perfect",
            "SA_domain-historical-NorESM2MM-perfect",
            "SA_domain-historical-ACCESSCM2-imperfect",
            "SA_domain-historical-NorESM2MM-imperfect",
            "SA_domain-mid_century-ACCESSCM2-perfect",
            "SA_domain-mid_century-NorESM2MM-perfect",
            "SA_domain-mid_century-ACCESSCM2-imperfect",
            "SA_domain-mid_century-NorESM2MM-imperfect",
            "SA_domain-end_century-ACCESSCM2-perfect",
            "SA_domain-end_century-NorESM2MM-perfect",
            "SA_domain-end_century-ACCESSCM2-imperfect",
            "SA_domain-end_century-NorESM2MM-imperfect",
        ],
        "ALPS": [
            "ALPS_domain-historical-CNRMCM5-perfect",
            "ALPS_domain-historical-MPIESMLR-perfect",
            "ALPS_domain-historical-CNRMCM5-imperfect",
            "ALPS_domain-historical-MPIESMLR-imperfect",
            "ALPS_domain-mid_century-CNRMCM5-perfect",
            "ALPS_domain-mid_century-MPIESMLR-perfect",
            "ALPS_domain-mid_century-CNRMCM5-imperfect",
            "ALPS_domain-mid_century-MPIESMLR-imperfect",
            "ALPS_domain-end_century-CNRMCM5-perfect",
            "ALPS_domain-end_century-MPIESMLR-perfect",
            "ALPS_domain-end_century-CNRMCM5-imperfect",
            "ALPS_domain-end_century-MPIESMLR-imperfect",
        ],
    },
    "C3S2_384": {
        "ALPS": [
            "ALPS_domain-mid_century-CNRMCM5-perfect",
            "ALPS_domain-end_century-CNRMCM5-perfect",
            "ALPS_domain-historical-CNRMCM5-perfect",
            "ALPS_domain-historical-CNRMCM5-imperfect",
        ],
        "NZ": [
            "NZ_domain-mid_century-ACCESSCM2-perfect",
            "NZ_domain-end_century-ACCESSCM2-perfect",
            "NZ_domain-historical-ACCESSCM2-perfect",
            "NZ_domain-historical-ACCESSCM2-imperfect",
        ],
        "SA": [
            "SA_domain-mid_century-ACCESSCM2-perfect",
            "SA_domain-end_century-ACCESSCM2-perfect",
            "SA_domain-historical-ACCESSCM2-perfect",
            "SA_domain-historical-ACCESSCM2-imperfect",
        ],
    },
}

TRAINING_MODES = {
    "CORDEXMLBench": ["Emulator_hist_future", "ESD_pseudo_reality"],
    "C3S2_384": ["Emulator_hist_future"],
}

EMULATORS = {
    "Emulator_hist_future": {
        "ALPS": {
            "checkpoint": "epoch_540",
        },
        "NZ": {
            "checkpoint": "epoch_260",
        },
        "SA": {
            "checkpoint": "epoch_560",
        },
    },
    "ESD_pseudo_reality": {
        "ALPS": {
            "checkpoint": "epoch_1000",
        },
        "NZ": {
            "checkpoint": "epoch_300",
        },
        "SA": {
            "checkpoint": "epoch_2000",
        },
    },
}

GCM_RENAME = {
    "ACCESSCM2": "ACCESS-CM2",
    "CNRMCM5": "CNRM-CM5",
    "ECEarth3": "EC-Earth3",
    "MPIESMLR": "MPI-ESM-LR",
    "NorESM2MM": "NorESM2-MM",
}


def save_netcdf_atomic(ds: xr.Dataset, fp: Path):
    """Save xarray Dataset to NetCDF file atomically."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmpf:
            tmp_path = tmpf.name

        # Save to the temporary file
        ds.to_netcdf(tmp_path)

        # Move the completed file into the final location. Use shutil.move
        # to handle cross-filesystem renames (e.g., local /tmp -> NFS).
        shutil.move(tmp_path, fp)
        tmp_path = None
    finally:
        # Cleanup any leftover temp file on error
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)


def format_samples(samples_filepaths, domain):
    # combine samples along new "member" dimension
    # rename variables
    logger.info(f"Combining samples and renaming variables...")
    ds = xr.concat(
        [xr.open_dataset(f) for f in samples_filepaths], dim="member"
    ).rename({f"pred_{var}": var for var in ["pr", "tasmax"]})

    for var in ["pr", "tasmax"]:
        template_ds = xr.open_dataset(
            f"vendor/ml-benchmark/format_predictions/templates/{var}_{domain}.nc"
        )
        logger.info(f"Formatting time attrs...")
        # copy attributes from template for time coordinate
        ds["time"].attrs = template_ds["time"].attrs

        logger.info(f"Validating dimensions and coordinate attributes for {var} ...")
        assert (
            ds[var].dims == ("member",) + template_ds[var].dims
        ), f"Variable {var} has different dims in samples ({ds[var].dims}) and template ({template_ds[var].dims})"
        for c in template_ds.coords:
            assert c in ds.coords
            assert (
                ds[c].attrs == template_ds[c].attrs
            ), f"Coordinate {c} has different attributes in samples ({ds[c].attrs}) and template ({template_ds[c].attrs})"

    return ds


app = typer.Typer()


@app.command()
def main(
    workdir_root: Path,
    project: str = "CORDEXMLBench",
    domain: str = typer.Option(
        None, help="Domain to process (e.g. 'ALPS', 'NZ', 'SA')"
    ),
    training_mode: str = typer.Option(
        None,
        help="Training mode to process (e.g. 'Emulator_hist_future', 'ESD_pseudo_reality')",
    ),
    nsamples_required: int = 5,
):
    assert (
        training_mode in TRAINING_MODES[project]
    ), f"Invalid training mode {training_mode} for project {project}. Must be one of {TRAINING_MODES[project]}"
    emu_config = EMULATORS[training_mode][domain]
    for dataset in DATASETS[project][domain]:
        _, period, gcm, src = dataset.split("-")
        samples_path = Path(
            workdir_root,
            f"mlde/score-sde/subvpsde/cordex_ml_mv_hist_fut_{domain.lower()}_cncsnpp_continuous/w_static_rcmgem",
            "samples",
            emu_config["checkpoint"],
            dataset,
            f"{domain}_domain-{training_mode}-{TRAINING_GCMS[domain]}-perfect-stan",
            "test",
            "01",
        )
        logger.info(f"Looking for samples in {samples_path}")
        samples_filepaths = list(samples_path.glob("*/predictions-*.nc"))[
            :nsamples_required
        ]
        assert (
            len(samples_filepaths) == nsamples_required
        ), f"Expected {nsamples_required} sample files in {samples_path}, found {len(samples_filepaths)}"

        ds = format_samples(samples_filepaths, domain)
        output_base = Path(
            os.getenv("DATA_PATH"),
            "formatted_predictions",
            project,
        )

        output_path = (
            output_base
            / f"{domain}_domain"
            / training_mode
            / period
            / src
            / f"Predictions_pr_tasmax_{GCM_RENAME[gcm]}_{TEST_YEARS[period]}.nc"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving formatted predictions to {output_path}")
        save_netcdf_atomic(ds, output_path)
        logger.info(f"DONE")

    logger.info(f"DONE")


if __name__ == "__main__":
    app()
