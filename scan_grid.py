import pandas as pd
import os
from table_pypsa.run_pypsa import build_network, run_pypsa
from table_pypsa.utilities.load_costs import load_costs
import argparse
import copy
import xarray as xr
from dask import delayed, compute

# Get file name from command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-f', help='Name of the base case file', required=True)


def update_capacity_factors(n, comp_list, cf_type, cfs_lat_lon):
    """
    Update capacity factors of the solar PV and concentrated solar for grid
    """
    # Run over all components that have solar or wind in their name
    for tech_component in [comp['name'] for comp in comp_list if cf_type in comp['name']]:

        # Make pandas dataframe with time and capacity factors, and drop the rest
        cfs_lat_lon = cfs_lat_lon.to_dataframe()
        cfs_lat_lon = cfs_lat_lon[["capacity factor"]]
        n.snapshots = cfs_lat_lon.index
        
        # Replace p_max_pu with the new capacity factors in network
        n.generators_t['p_max_pu'][tech_component] = cfs_lat_lon

        # Replace p_max_pu with the new capacity factors in component_list
        for comp in comp_list:
            if comp['name'] == tech_component:
                comp['p_max_pu'] = cfs_lat_lon

    return n, comp_list


def main():
    args = parser.parse_args()
    file_name = args.file_name

    name_suffix = file_name.split('/')[-1].split('.')[0].replace('_case', '')

    network, case_dict, component_list, comp_attributes = build_network(file_name)

    capacity_factors_csp = xr.open_dataset('input_files/world_csp_CF_timeseries_2023_coarse.nc')
    capacity_factors_pv = xr.open_dataset('input_files/world_solar_CF_timeseries_2023_coarse.nc')

    # Create an empty DataArray to store results without time dimension
    result_array = xr.DataArray(
        data=None,
        dims=['x', 'y'],
        coords={
            'x': capacity_factors_csp.x,
            'y': capacity_factors_csp.y},
        name='csp fraction')

    @delayed
    def process_grid_cell(lon, lat):

        # Create deep copies of network and component_list
        network_copy = copy.deepcopy(network)
        component_list_copy = copy.deepcopy(component_list)

        network_copy, component_list_copy = update_capacity_factors(network_copy, component_list_copy, "csp", capacity_factors_csp.sel(x=lon, y=lat))
        network_copy, component_list_copy = update_capacity_factors(network_copy, component_list_copy, "solar", capacity_factors_pv.sel(x=lon, y=lat))

        # Run PyPSA with new costs
        run_pypsa(network_copy, file_name, case_dict, component_list_copy, outfile_suffix=f'_{lon.values}_{lat.values}')

        # Extract fraction of supply from CSP
        result = network_copy.statistics.supply().loc[('Generator', 'concentrated solar')] / network_copy.statistics.supply().sum()

        # Extract relevant results and store them in the result_array
        return lon, lat, result

    tasks = [process_grid_cell(lon, lat) for lon in capacity_factors_csp.x for lat in capacity_factors_csp.y]
    results = compute(*tasks)

    for lon, lat, objective in results:
        result_array.loc[dict(x=lon, y=lat)] = objective

    # Save results to .nc file
    result_array.to_netcdf(f'output_data/results_{name_suffix}.nc')


if __name__ == "__main__":
    main()