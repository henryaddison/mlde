from importlib.resources import files
from typing import Callable
import numpy as np
import xarray as xr

from mlde_utils.data.remapcon import Remapcon
from mlde_utils.data.shift_lon_break import ShiftLonBreak
from mlde_utils.data.select_gcm_domain import SelectGCMDomain


def _get_cdf(x, xbins):
    pdf, _ = np.histogram(x, xbins)
    return np.insert(np.cumsum(pdf), 0, 0.0)


def qm_1d_dom_aware(
    obs: np.ndarray,
    simh: np.ndarray,
    simp: np.ndarray,
    n_quantiles: int = 250,
    kind: str = "+",
):
    """
    A 1D quantile mapping function replacement for CMethods.quantile_mapping

    Unlike the CMethods version it takes into account that the obs and simh may have different max and mins (i.e. their CDFs have different supported domains). The CMethods version just uses a domain between the min and max of both obs and simh.
    """
    obs, simh, simp = np.array(obs), np.array(simh), np.array(simp)

    obs_min = np.amin(obs)
    obs_max = np.amax(obs)
    wide = abs(obs_max - obs_min) / n_quantiles
    xbins_obs = np.arange(obs_min, obs_max + wide, wide)

    simh_min = np.amin(simh)
    simh_max = np.amax(simh)
    wide = abs(simh_max - simh_min) / n_quantiles
    xbins_simh = np.arange(simh_min, simh_max + wide, wide)

    cdf_obs = _get_cdf(obs, xbins_obs)
    cdf_simh = _get_cdf(simh, xbins_simh)

    epsilon = np.interp(simp, xbins_simh, cdf_simh)

    return np.interp(epsilon, cdf_obs, xbins_obs)


def xrqm(
    sim_train_da: xr.DataArray,
    ml_train_da: xr.DataArray,
    ml_eval_da: xr.DataArray,
    qm_func: Callable = qm_1d_dom_aware,
):
    """Apply a 1D quantile mapping function point-by-point to a multi-dimensional xarray DataArray."""
    return (
        xr.apply_ufunc(
            qm_func,  # first the function
            sim_train_da,  # now arguments in the order expected by the function
            ml_train_da,
            ml_eval_da,
            kwargs=dict(n_quantiles=250, kind="+"),
            input_core_dims=[
                ["time"],
                ["time"],
                ["time"],
            ],  # list with one entry per arg
            output_core_dims=[["time"]],
            exclude_dims=set(
                ("time",)
            ),  # dimensions allowed to change size. Must be set!
            vectorize=True,
        )
        .transpose("ensemble_member", "time", "grid_latitude", "grid_longitude")
        .assign_coords(time=ml_eval_da["time"])
    )


def to_gcm_domain(ds: xr.Dataset):
    target_grid_filepath = files("mlde_utils.data").joinpath(
        "target_grids/60km/global/pr/moose_grid.nc"
    )
    ds = Remapcon(target_grid_filepath).run(ds)
    ds = ShiftLonBreak().run(ds)
    ds = SelectGCMDomain(subdomain="birmingham", size=9).run(ds)
    nan_count = ds["pred_pr"].isnull().sum().values.item()
    assert 0 == nan_count, f"nan count: {nan_count}"
    ds = ds.drop_vars(["rotated_latitude_longitude"], errors="ignore")
    return ds
