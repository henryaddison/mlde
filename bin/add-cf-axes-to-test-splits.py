import xarray as xr
import cf_xarray  # noqa: F401
import glob
import os
import shutil
import tempfile

for fp in glob.glob(
    "/gws/nopw/j04/furflex/henrya/projects/cordex-ml-bench/data/datasets/*/test/predictors/Variable_fields.nc"
):
    ds = xr.load_dataset(fp)
    if ds.cf.axes == {}:
        ds["lon"] = ds["lon"].assign_attrs({"axis": "X"})
        ds["lat"] = ds["lat"].assign_attrs({"axis": "Y"})
        ds["time"] = ds["time"].assign_attrs({"axis": "T"})
        print(f"fixing {fp}")

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
