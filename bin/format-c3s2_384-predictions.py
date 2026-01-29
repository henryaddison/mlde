import dotenv
import logging
import os
from pathlib import Path
import typer
import xarray as xr
import zipfile

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

PROJECT = "C3S2_384"
EMULATOR_ID = "emulator_id"

NSAMPLES_REQUIRED = 5


def format_samples(samples_filepaths, domain):
    # combine samples along new "member" dimension
    # rename variables
    ds = xr.concat(
        [xr.open_dataset(f) for f in samples_filepaths], dim="member"
    ).rename({f"pred_{var}": var for var in ["pr", "tasmax"]})

    for var in ["pr", "tasmax"]:
        template_ds = xr.open_dataset(
            f"vendor/ml-benchmark/format_predictions/templates/{var}_{domain}.nc"
        )

        # copy attributes from template for time coordinate
        ds["time"].attrs = template_ds["time"].attrs

        assert (
            ds[var].dims == ("member",) + template_ds[var].dims
        ), f"Variable {var} has different dims in samples and template"
        for c in template_ds.coords:
            assert c in ds.coords
            assert (
                ds[c].attrs == template_ds[c].attrs
            ), f"Coordinate {c} has different attributes in samples and template"

    return ds


def zip_submission(output_base: Path):
    # ZIP the submission using the "emulator_id" code specified in the registration
    zip_filename = f"{EMULATOR_ID}.zip"
    output_base = Path(
        os.getenv("DATA_PATH"),
        "formatted_predictions",
        PROJECT,
    )
    zip_path = os.path.join(output_base, zip_filename)

    logger.info(f"Creating submission package: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_base):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_base)
                zipf.write(abs_path, rel_path)


app = typer.Typer()


@app.command()
def main(workdir_root: Path):
    for training_mode in TRAINING_MODES[PROJECT]:
        for domain, emu_config in EMULATORS[training_mode].items():
            for dataset in DATASETS[PROJECT][domain]:
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
                    :NSAMPLES_REQUIRED
                ]
                assert (
                    len(samples_filepaths) == NSAMPLES_REQUIRED
                ), f"Expected {NSAMPLES_REQUIRED} sample files in {samples_path}, found {len(samples_filepaths)}"

                _ = format_samples(samples_filepaths, domain)
                output_base = Path(
                    os.getenv("DATA_PATH"),
                    "formatted_predictions",
                    PROJECT,
                )
                output_path = output_base.join(
                    f"{domain}_domain",
                    training_mode,
                    period,
                    src,
                    f"Predictions_pr_tasmax_{gcm}_{TEST_YEARS[period]}.nc",
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving formatted predictions to {output_path}")
                # ds.to_netcdf(output_path)
                logger.info(f"DONE")

    # zip_submission(output_base)
    logger.info(f"DONE")


if __name__ == "__main__":
    app()
