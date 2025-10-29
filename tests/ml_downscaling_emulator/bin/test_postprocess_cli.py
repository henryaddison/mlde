import pytest
import shortuuid
from typer.testing import CliRunner

from mlde_utils import samples_path

from ml_downscaling_emulator.bin import app

runner = CliRunner()


def test_filter(tmp_path, samples_file):
    time_period = "historic"
    workdir = tmp_path / "test-model"
    checkpoint = "epoch-1"
    ensemble_member = "01"
    dataset = "test-dataset"
    input_xfm = "stan"

    result = runner.invoke(
        app,
        [
            "postprocess",
            "filter",
            str(workdir),
            "--dataset",
            dataset,
            "--time-period",
            time_period,
            "--checkpoint",
            checkpoint,
            "--input-xfm",
            input_xfm,
            "--ensemble-member",
            ensemble_member,
        ],
    )

    assert result.exit_code == 0


@pytest.fixture
def samples_file(tmp_path, samples_set):
    workdir = tmp_path / "test-model"
    checkpoint = "epoch-1"
    input_xfm = "stan"
    split = "val"
    ensemble_member = "01"
    dataset = "test-dataset"

    dirpath = samples_path(
        workdir=workdir,
        checkpoint=checkpoint,
        input_xfm=input_xfm,
        dataset=dataset,
        split=split,
        ensemble_member=ensemble_member,
    )
    filepath = dirpath / f"predictions-{shortuuid.uuid()}.nc"

    dirpath.mkdir(parents=True, exist_ok=True)

    samples_set.to_netcdf(filepath)
    return filepath
