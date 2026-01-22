import xarray as xr
import cf_xarray  # noqa: F401
import glob

for fp in glob.glob(
    "/gws/nopw/j04/furflex/henrya/projects/cordex-ml-bench/data/datasets/*/test/predictors/Variable_fields.nc"
):
    ds = xr.load_dataset(fp)
    if ds.cf.axes == {}:
        ds["lon"] = ds["lon"].assign_attrs({"axis": "X"})
        ds["lat"] = ds["lat"].assign_attrs({"axis": "Y"})
        ds["time"] = ds["time"].assign_attrs({"axis": "T"})
        print(f"fixing {fp}")
        # ds.to_netcdf(fp)
