import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from ml_downscaling_emulator.data import UKCPLocalDataset


def test_UKCPLocalDataset_item_cond_var(xr_dataset, time_range):
    pt_dataset = UKCPLocalDataset(xr_dataset, ["var1", "var2"], ["target"], time_range)
    cond = pt_dataset[0][0][:2]
    expected_cond = torch.stack(
        [
            torch.Tensor(np.arange(5 * 7).reshape(5, 7)),
            torch.Tensor(np.arange(5 * 7).reshape(5, 7)),
        ]
    )

    assert torch.all(cond == expected_cond)


def test_UKCPLocalDataset_item_cond_time(
    xr_dataset, time_range, earliest_time, latest_time
):
    pt_dataset = UKCPLocalDataset(xr_dataset, ["var1", "var2"], ["target"], time_range)

    test_date = cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True)
    expected_climate_time = (test_date - earliest_time) / (latest_time - earliest_time)
    time_cond = pt_dataset[0][0][2:]
    assert torch.all(
        time_cond[0] == torch.Tensor([expected_climate_time]).broadcast_to(5, 7)
    )
    assert torch.all(
        time_cond[1]
        == torch.Tensor([np.sin(2 * np.pi * 331.0 / 360.0)]).broadcast_to(5, 7)
    )
    assert torch.all(
        time_cond[2]
        == torch.Tensor([np.cos(2 * np.pi * 331.0 / 360.0)]).broadcast_to(5, 7)
    )

    test_date = cftime.Datetime360Day(1981, 6, 1, 12, 0, 0, 0, has_year_zero=True)
    expected_climate_time = (test_date - earliest_time) / (latest_time - earliest_time)
    time_cond = pt_dataset[180][0][2:]
    assert torch.all(
        time_cond[0] == torch.Tensor([expected_climate_time]).broadcast_to(5, 7)
    )
    assert torch.all(
        time_cond[1]
        == torch.Tensor([np.sin(2 * np.pi * 151 / 360.0)]).broadcast_to(5, 7)
    )
    assert torch.all(
        time_cond[2]
        == torch.Tensor([np.cos(2 * np.pi * 151 / 360.0)]).broadcast_to(5, 7)
    )


def test_UKCPLocalDataset_item_target(xr_dataset, time_range):
    pt_dataset = UKCPLocalDataset(xr_dataset, ["var1", "var2"], ["target"], time_range)
    target = pt_dataset[0][1]
    expected_target = torch.Tensor(np.arange(5 * 7).reshape(1, 5, 7))

    assert torch.all(target == expected_target)


def test_UKCPLocalDataset_item_time(xr_dataset, time_range):
    pt_dataset = UKCPLocalDataset(xr_dataset, ["var1", "var2"], ["target"], time_range)

    time = pt_dataset[0][2]
    expected_time = cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True)
    assert time == expected_time

    time = pt_dataset[175][2]
    expected_time = cftime.Datetime360Day(1981, 5, 26, 12, 0, 0, 0, has_year_zero=True)
    assert time == expected_time


@pytest.fixture
def earliest_time():
    return cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True)


@pytest.fixture
def latest_time():
    return cftime.Datetime360Day(2080, 11, 30, 12, 0, 0, 0, has_year_zero=True)


@pytest.fixture
def time_range(earliest_time, latest_time):
    return (earliest_time, latest_time)


@pytest.fixture
def lat_coords():
    return np.linspace(-3, 3, 7)


@pytest.fixture
def lon_coords():
    return np.linspace(-2, 2, 5)


@pytest.fixture
def time_coords():
    return xr.cftime_range(
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        periods=360 * 2,
        freq="D",
    )


@pytest.fixture
def em_coords():
    return ["01", "02"]


def values(shape):
    return np.arange(np.prod(shape)).reshape(*shape)


@pytest.fixture
def xr_dataset(em_coords, time_coords, lat_coords, lon_coords):
    ds = xr.Dataset(
        data_vars={
            "var1": (
                ["ensemble_member", "time", "grid_longitude", "grid_latitude"],
                values(
                    (len(em_coords), len(time_coords), len(lon_coords), len(lat_coords))
                ),
            ),
            "var2": (
                ["ensemble_member", "time", "grid_longitude", "grid_latitude"],
                values(
                    (len(em_coords), len(time_coords), len(lon_coords), len(lat_coords))
                ),
            ),
            "target": (
                ["ensemble_member", "time", "grid_longitude", "grid_latitude"],
                values(
                    (len(em_coords), len(time_coords), len(lon_coords), len(lat_coords))
                ),
            ),
        },
        coords=dict(
            ensemble_member=(["ensemble_member"], em_coords),
            time=(["time"], time_coords),
            grid_longitude=(["grid_longitude"], lon_coords),
            grid_latitude=(["grid_latitude"], lat_coords),
        ),
    )

    return ds
