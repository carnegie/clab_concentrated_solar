import atlite
import cartopy.io.shapereader as shpreader
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import logging
from argparse import ArgumentParser

def main(year, month):
    
    # Load the world shapefile
    world = gpd.read_file('../../input_files/ne_110m_admin_0_countries.shp')
    # Drop Antarctica by excluding everything below -60 latitude
    world = world[world.geometry.centroid.y > -60]

    region = world
    region_name = "world"

    # Loop over the years
    logging.info(f"Processing {year}-{month} for {region_name}")

    # Define the cutout; this will not yet trigger any major operations
    cutout = atlite.Cutout(
        path=f"{region_name}-{year}-{month}_timeseries", module="era5", 
        bounds=region.geometry.union_all().bounds, 
        time=f"{year}-{month}",
        chunks={"time": 100,},)
    # This is where all the work happens (this can take some time).
    cutout.prepare(
        compression={"zlib": True, "complevel": 9},
        monthly_requests=True,
        concurrent_requests=True,)

    
    # Extract the concenctrated solar power generation capacity factors
    csp_power_generation = cutout.csp(
        installation="Glasspoint_parabolic_trough", 
        capacity_factor_timeseries=True,)

    # Save gridded data as netCDF files
    csp_power_generation.to_netcdf(f"{region_name}_csp_CF_timeseries_{year}_{month}.nc")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--year", type=int, help="Get data for this year", required=True)
    parser.add_argument("--month", type=int, help="Get data for this month", required=True)
    args = parser.parse_args()
    main(args.year, args.month)
