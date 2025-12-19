import xarray as xr

filepaths = [
    "/gws/nopw/j04/furflex/henrya/projects/cordex-ml-bench/data/datasets/ALPS_domain-ESD_pseudo_reality-CNRMCM5-perfect/train/target/pr_tasmax.nc",
    "/gws/nopw/j04/furflex/henrya/projects/cordex-ml-bench/data/datasets/ALPS_domain-Emulator_hist_future-CNRMCM5-perfect/train/target/pr_tasmax.nc",
]

for filepath in filepaths:
    print(f"Rechunking {filepath}")

    ds = xr.load_dataset(filepath)

    chunks = (1, 128, 128)
    encoding = {dv: {"chunksizes": chunks} for dv in ds.data_vars}
    ds.data_vars
    ds.to_netcdf(filepath, encoding=encoding)
