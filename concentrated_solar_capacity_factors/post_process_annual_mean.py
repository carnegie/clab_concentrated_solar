import xarray as xr
import dask
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Process annual mean capacity factors")
parser.add_argument("--cf_type", "-c", type=str, help="Capacity factor type (wind, solar or csp)")

def process_annual_mean(cf_type):
    # Calculate mean capacity factors for 2023
    for month in range(1, 13):
        print(f"Processing month {month}...")
        monthly_file = f'monthly_cf_files/world_{cf_type}_CF_timeseries_2023_{month}.nc'
        ds = xr.open_dataset(monthly_file, chunks='auto')

        monthly_mean = ds.mean(dim='time')  # Lazy operation

        if month == 1:
            mean_cf = monthly_mean
        else:
            mean_cf += monthly_mean

        ds.close()

    print("Computing mean capacity factor...")
    mean_cf /= 12  # Divide by months

    # Compute before saving & plotting
    mean_cf = dask.compute(mean_cf)[0]

    # Drop edge y-coordinate to make coarsening work
    mean_cf = mean_cf.isel(y=slice(0, -7))
    # Apply spatial downsampling
    mean_cf = mean_cf.coarsen(x=10, y=10, boundary="trim").mean()

    print("Saving mean capacity factor...")
    mean_cf.to_netcdf(
        f'world_{cf_type}_CF_mean_2023.nc',
        engine="h5netcdf",
        mode="w",
        format="NETCDF4",
        encoding={"capacity factor": {"zlib": True, "complevel": 5}}
    )


if __name__ == "__main__":
    args = parser.parse_args()
    process_annual_mean(args.cf_type)