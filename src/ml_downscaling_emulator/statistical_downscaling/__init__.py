import functools
import logging
import numpy as np
import scipy
from typing import Callable, Union
import xarray as xr


def bcsd_on_chunks(
    source: xr.DataArray,
    lr_da: xr.DataArray,
    target_da: xr.DataArray,
    method: str = "gaussian",
    multiplicative: bool = True,
    window_size: int = 3,
) -> xr.DataArray:
    """Process an input data chunk with the BCSD method.

    Args:
      source: The source data to be processed with the BCSD method.
      lr_da: The low-resolution data used to fit the BCSD method.

      target_da: The unfiltered target data for fitting the BCSD method.
      method: The method to use for quantile mapping.
      multiplicative: Whether to use multiplicative correction (e.g. for precipitation variables).
      window_size: The size of the window used to compute the climatology statistics.
    Returns:
      The BCSD-downscaled data as an xarray DataArray.
    """

    logging.info(f"Low-pass filtering target data.")
    # Compute the spatially filtered target data for fitting the BCSD method.
    filtered_da = low_pass_filter(
        target_da,
        scale=(60.0 / 8.8) / 2.0,
        spatial_dims=("grid_longitude", "grid_latitude"),
    )

    sel = dict(
        dayofyear=source["time"].dt.dayofyear,
        drop=True,
    )

    if method == "gaussian":
        # Compute the climatology of the low-resolution data.
        logging.info(f"Computing low-res climatology")
        clim_mean = compute_daily_stat(
            lr_da, window_size=window_size, stat_fn="mean"
        ).sel(**sel)
        clim_std = compute_daily_stat(
            lr_da, window_size=window_size, stat_fn="std"
        ).sel(**sel)

        # Standardize with respect to the original climatology.
        source_standard = (source - clim_mean) / clim_std

        # Get value of the same quantile in the filtered climatology, keep anom.
        logging.info(f"Computing filtered hi-res climatology")
        filtered_clim_std = compute_daily_stat(
            filtered_da, window_size=window_size, stat_fn="std"
        ).sel(**sel)
        source_bc_anom = source_standard * filtered_clim_std

        # Add anom to the mean of the unfiltered climatology.
        logging.info(f"Computing hi-res climatology")
        target_clim_mean = compute_daily_stat(
            target_da, window_size=window_size, stat_fn="mean"
        ).sel(**sel)

        source_bcsd = target_clim_mean + source_bc_anom

        # Use multiplicative correction for precipitation variables
        if multiplicative:
            filtered_clim_mean = compute_daily_stat(
                filtered_da, window_size=window_size, stat_fn="mean"
            ).sel(**sel)
            # Compute the full field corresponding to the filtered climatology
            # quantile. Then, keep the multiplicative anomaly.
            bc_mult_anom = (source_bc_anom + filtered_clim_mean) / (filtered_clim_mean)
            # Multiply the anomaly by the target climatology.
            var_bcsd = bc_mult_anom * target_clim_mean
            source_bcsd = var_bcsd

        # Ensure no negative precip values in the output.
        source_bcsd = source_bcsd.clip(min=0)

        return source_bcsd.drop_vars(["dayofyear"])
    else:
        raise ValueError(f"BCSD method {method} not yet implemented.")


def low_pass_filter(
    source: xr.DataArray,
    *,
    scale: float = (60.0 / 8.8)
    / 2.0,  # default is half the scale factor from 60km input to 8.8km target
    mode: str = "nearest",
    spatial_dims: tuple[str, str] = ("grid_latitude", "grid_longitude"),
) -> xr.Dataset:
    """Filters a chunk in its spatial dimensions.

    Args:
        source: The source dataset chunk to interpolate to a new spatial grid.
        scale: Scale (in pixels) used to construct the Gaussian filter.
        mode: Determines how the input array is extended when the filter overlaps a
        border.
        spatial_dims: The dimensions of the source dataset to filter over.

    Returns:
        The spatially filtered chunk.
    """
    # Ordered dimensions of the source dataset can be fetched from a data_var.
    # Dataset dimensions are not guaranteed to be ordered.
    ordered_dims = source.dims
    # We only filter the spatial dimensions.
    scales = tuple(scale * float(dim in spatial_dims) for dim in ordered_dims)

    gaussian_filter = functools.partial(
        scipy.ndimage.gaussian_filter,
        sigma=scales,
    )
    return xr.apply_ufunc(gaussian_filter, source.load())


def bcsd2(
    source: xr.DataArray,
    lr_da: xr.DataArray,
    target_da: xr.DataArray,
    method: str = "gaussian",
    multiplicative: bool = True,
    window_size: int = 3,
) -> xr.DataArray:
    """Process an input data chunk with the BCSD method.

    Args:
      source: The source data to be processed with the BCSD method.
      lr_da: The low-resolution data used to fit the BCSD method.

      target_da: The unfiltered target data for fitting the BCSD method.
      method: The method to use for quantile mapping.
      multiplicative: Whether to use multiplicative correction (e.g. for precipitation variables).
      window_size: The size of the window used to compute the climatology statistics.
    Returns:
      The BCSD-downscaled data as an xarray DataArray.
    """
    # compute the quantile corresponding to the threshold in the high-res data for each location
    target_threshold = 0.1 / (
        24 * 60 * 60
    )  # 0.1 mm/day wet threshold for precipitation

    # thresh_q = (hr_pr <= threshold).mean(dim=["time", "ensemble_member"]) # simpler but incorrect? guess at quantiles for each location
    thresh_q = (
        xr.apply_ufunc(
            scipy.stats.percentileofscore,
            target_da.stack(ex=["time", "ensemble_member"]),
            target_threshold,
            input_core_dims=[["ex"], []],
            vectorize=True,
        )
        / 100
    )

    # convert quantiles to the corresponding threshold in the low-res data
    bcthresh = xr.apply_ufunc(
        np.quantile,
        lr_da,
        thresh_q,
        input_core_dims=[["time", "ensemble_member"], []],
        vectorize=True,
    )

    # exclude dry days based on bias corrected threshold for the low-res data
    # to ensure same proportion of days are dry for each location in high-res and low-res data
    source_wet_dry_mask = source >= bcthresh
    source = source.where(source_wet_dry_mask)
    lr_da = lr_da.where(lr_da >= bcthresh)
    target_da = target_da.where(target_da >= target_threshold)

    # square root transform the wet-day data to make it more Gaussian-like
    source = np.power(source, 1 / 2)
    lr_da = np.power(lr_da, 1 / 2)
    target_da = np.power(target_da, 1 / 2)

    bcsd_da = bcsd_on_chunks(
        source=source,
        lr_da=lr_da,
        target_da=target_da,
        window_size=window_size,
    )
    # square the data to reverse the square root transform.
    bcsd_da = np.power(bcsd_da, 2)

    # re-add any dry days as zeros
    bcsd_da = bcsd_da.where(source_wet_dry_mask, other=0)

    return bcsd_da


def create_window_weights(window_size: int) -> xr.DataArray:
    """Create linearly decaying window weights."""
    assert window_size % 2 == 1, "Window size must be odd."
    half_window_size = window_size // 2
    window_weights = np.concatenate(
        [
            np.linspace(0, 1, half_window_size + 1),
            np.linspace(1, 0, half_window_size + 1)[1:],
        ]
    )
    window_weights = window_weights / window_weights.mean()
    window_weights = xr.DataArray(window_weights, dims=["window"])
    return window_weights


def replace_time_with_doy(ds: xr.Dataset) -> xr.Dataset:
    """Replace time coordinate with days of year."""
    return ds.assign_coords({"time": ds.time.dt.dayofyear}).rename(
        {"time": "dayofyear"}
    )


def compute_rolling_stat(
    ds: xr.Dataset,
    window_weights: xr.DataArray,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = "mean",
) -> xr.Dataset:
    """Compute rolling climatology."""
    window_size = len(window_weights)
    half_window_size = window_size // 2  # For padding
    # Stack years
    stacked = xr.concat(
        [
            replace_time_with_doy(ds.sel(time=str(y)))
            for y in np.unique(ds.time.dt.year)
        ],
        dim="year",
    )
    # Fill gap day (366) with values from previous day 365
    # stacked = stacked.fillna(stacked.sel(dayofyear=365))
    # Pad edges for perioding window
    stacked = stacked.pad(pad_width={"dayofyear": half_window_size}, mode="wrap")
    # Weighted rolling mean
    stacked = stacked.rolling(dayofyear=window_size, center=True).construct("window")
    if stat_fn == "mean":
        rolling_stat = stacked.weighted(window_weights).mean(dim=("window", "year"))
    elif stat_fn == "std":
        rolling_stat = stacked.weighted(window_weights).std(dim=("window", "year"))
    else:
        rolling_stat = stat_fn(stacked, weights=window_weights, dim=("window", "year"))
    # Remove edges
    rolling_stat = rolling_stat.isel(
        dayofyear=slice(half_window_size, -half_window_size)
    )
    return rolling_stat


def compute_daily_stat(
    ds: xr.Dataset,
    window_size: int,
    # clim_years: slice,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = "mean",
) -> xr.Dataset:
    """Compute daily average climatology with running window."""
    # NOTE: Loading seems to be necessary, otherwise computation takes forever
    # Will be converted to xarray-beam pipeline anyway
    ds = ds.load()

    window_weights = create_window_weights(window_size)
    daily_rolling_clim = compute_rolling_stat(ds, window_weights, stat_fn)
    return daily_rolling_clim
