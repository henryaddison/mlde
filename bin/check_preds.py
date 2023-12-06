import sys

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr

fpaths = sys.argv[1:]


def check(pred_ds):

    pred_var = "pred_pr"
    meta_vars = (
        "rotated_latitude_longitude",
        "time_bnds",
        "grid_latitude_bnds",
        "grid_longitude_bnds",
    )

    errors = []

    pred_pr_attrs = {
        "grid_mapping": "rotated_latitude_longitude",
        "standard_name": "pred_pr",
        "units": "kg m-2 s-1",
    }

    try:
        assert (
            pred_ds[pred_var].attrs == pred_pr_attrs
        ), f"Bad attrs on {pred_var}: {pred_ds[pred_var].attrs}"
    except AssertionError as e:
        errors.append(e)

    # pred_ds[var] = pred_ds[var].assign_attrs(pred_pr_attrs)

    for var in meta_vars:
        # print(var, pred_ds[var].dims)
        try:
            assert ("ensemble_member" not in pred_ds[var].dims) and (
                "time" not in pred_ds[var].dims
            ), f"Bad dims on {var}: {pred_ds[var].dims}"
        except AssertionError as e:
            errors.append(e)
        # pred_ds[var] = pred_ds[var].isel(ensemble_member=0, time=0)
        # print(var, pred_ds[var].dims)

    return errors


with logging_redirect_tqdm():
    with tqdm(
        total=len(fpaths),
        desc=f"CHecking prediction files",
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

    # print(pred_ds)
