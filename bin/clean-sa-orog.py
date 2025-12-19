import xarray as xr
import os
import dotenv

dotenv.load_dotenv()


satarg = xr.open_dataset(os.path.join(os.getenv("DATA_PATH"), "datasets/SA_domain-Emulator_hist_future-ACCESSCM2-perfect/train/target/pr_tasmax.nc"))

stat_filepath=os.path.join(os.getenv("DATA_PATH"), "datasets/SA_domain-Emulator_hist_future-ACCESSCM2-perfect/train/predictors/Static_fields.nc")
# xr.load_dataset(stat_filepath).drop_dims("time").sel(lat=satarg.lat, lon=satarg.lon).to_netcdf(stat_filepath)
xr.load_dataset(stat_filepath).drop_dims("bnds").to_netcdf(stat_filepath)
