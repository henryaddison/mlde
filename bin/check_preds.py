import sys

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr

fpaths = sys.argv[1:]

PRED_VAR = "pred_pr"
META_VARS = (
    "rotated_latitude_longitude",
    "time_bnds",
    "grid_latitude_bnds",
    "grid_longitude_bnds",
)

PRED_PR_ATTRS = {
    "grid_mapping": "rotated_latitude_longitude",
    "standard_name": "pred_pr",
    "units": "kg m-2 s-1",
}


def fix(pred_ds):
    pred_ds[PRED_VAR] = pred_ds[PRED_VAR].assign_attrs(PRED_PR_ATTRS)
    for var in META_VARS:
        if "ensemble_member" in pred_ds[var].dims:
            pred_ds[var] = pred_ds[var].isel(ensemble_member=0)
        if "time" in pred_ds[var].dims:
            pred_ds[var] = pred_ds[var].isel(time=0)

    return pred_ds


def check(pred_ds):
    errors = []

    try:
        assert (
            pred_ds[PRED_VAR].attrs == PRED_PR_ATTRS
        ), f"Bad attrs on {PRED_VAR}: {pred_ds[PRED_VAR].attrs}"
    except AssertionError as e:
        errors.append(e)

    for var in META_VARS:
        try:
            assert ("ensemble_member" not in pred_ds[var].dims) and (
                "time" not in pred_ds[var].dims
            ), f"Bad dims on {var}: {pred_ds[var].dims}"
        except AssertionError as e:
            errors.append(e)

    return errors


with logging_redirect_tqdm():
    with tqdm(
        total=len(fpaths),
        desc=f"Checking prediction files",
        unit=" files",
    ) as pbar:
        for fpath in fpaths:
            pred_ds = xr.open_dataset(fpath)
            # import pdb; pdb.set_trace()
            errors = check(pred_ds)

            if len(errors) != 5:
                print(f"Errors in {fpath}:")
                for e in errors:
                    print(e)
            pbar.update(1)

with logging_redirect_tqdm():
    with tqdm(
        total=len(fpaths),
        desc=f"Fixing prediction files",
        unit=" files",
    ) as pbar:
        for fpath in fpaths:
            pred_ds = xr.load_dataset(fpath)
            # import pdb; pdb.set_trace()
            pred_ds = fix(pred_ds)
            errors = check(pred_ds)

            if len(errors) > 0:
                print(f"Errors in {fpath}:")
                for e in errors:
                    print(e)
            # else:
            # pred_ds.to_netcdf(fpath)
            pbar.update(1)
