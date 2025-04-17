import xarray as xr
import argparse
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="Process annual mean capacity factors")
parser.add_argument("--cf_type", "-c", type=str, help="Capacity factor type (wind, solar or csp)")


def process_time_series(cf_type):
    """
    Processes and merges monthly NetCDF files into a single file with hourly resolution.
    Reduces spatial resolution slightly using coarsening. Handles missing coordinates.
    
    Parameters:
    cf_type (str): Capacity factor type identifier for file naming.
    
    Saves:
    A NetCDF file with the full time series for the year.
    """
    
    # Define output file
    merged_file = f'world_{cf_type}_CF_timeseries_2023.nc'

    # Define spatial downsampling factors (adjust as needed)
    factor_x, factor_y = 10, 10  # Modify to balance resolution and file size

    merged_ds = None  # Initialize merged dataset as None

    # Process each month sequentially
    for month in range(1, 13):
        print(f'Processing month {month}...')
        
        monthly_file = f'monthly_cf_files/world_{cf_type}_CF_timeseries_2023_{month}.nc'
        ds = xr.open_dataset(monthly_file, chunks='auto')

        # Drop edge y-coordinate to make coarsening work
        ds = ds.isel(y=slice(0, -7))
        # Apply spatial downsampling
        ds = ds.coarsen(x=factor_x, y=factor_y, boundary="trim").mean()

        # Initialize merged_ds on first iteration
        if merged_ds is None:
            merged_ds = ds
        else:
            merged_ds = xr.concat([merged_ds, ds], dim="time")

        ds.close()

    print('Saving full time series...')

    # Use optimized compression
    encoding = {"capacity factor": {"zlib": True, "complevel": 5}}

    # Write to NetCDF efficiently
    merged_ds.to_netcdf(merged_file, engine="h5netcdf", format="NETCDF4", encoding=encoding)

    print(f"Finished saving full time series to {merged_file}")



if __name__ == "__main__":
    args = parser.parse_args()
    process_time_series(args.cf_type)