import copy
import os
import numpy as np
import pandas as pd
import xarray as xr
import argparse
from table_pypsa.run_pypsa import build_network, run_pypsa, write_result
from table_pypsa.utilities.load_costs import load_costs

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


def update_fuel_cost(cell, fuel_cost, n, comp_list, base_costs, n_years):
    """
    Update fuel cost of gas with a binary search to find breakeven cost
    """
    # Update cost in inputs
    base_costs.loc[('gas', 'fuel'), 'value'] = fuel_cost
    # Write costs to temporary file
    base_costs.to_csv(f'temp_costs_{cell}_{fuel_cost}.csv')
    # Load new costs
    costs = load_costs(f'temp_costs_{cell}_{fuel_cost}.csv', 'table_pypsa/utilities/cost_config.yaml', Nyears=n_years)
    # Remove temporary file
    os.remove(f'temp_costs_{cell}_{fuel_cost}.csv')
    # Replace new costs
    n.generators.loc['gas boiler steam', 'marginal_cost'] = costs.at[('gas boiler steam', 'marginal_cost')]

    tech_indeces = [i for i in range(len(comp_list)) if 'gas boiler steam' in comp_list[i]['name']]
    for tech_index in tech_indeces:
        comp_list[tech_index]['marginal_cost'] = n.generators.loc['gas boiler steam', 'marginal_cost']
    return n, comp_list


def process_grid_cell(lon, lat, file_name, gas_cost):


    name_suffix = file_name.split('/')[-1].split('.')[0].replace('_case', '')

    network, case_dict, component_list, comp_attributes = build_network(file_name)
    base_costs = pd.read_csv(case_dict['costs_path'],index_col=[0, 1]).sort_index()


    capacity_factors_csp = xr.open_dataset('concentrated_solar_capacity_factors/world_csp_CF_timeseries_2023.nc')
    # capacity_factors_pv = xr.open_dataset('input_files/world_solar_CF_timeseries_2023_coarse.nc')
   
    # Create deep copies of network and component_list
    network_copy = copy.deepcopy(network)
    component_list_copy = copy.deepcopy(component_list)

    network_copy, component_list_copy = update_capacity_factors(network_copy, component_list_copy, "csp", capacity_factors_csp.sel(x=lon, y=lat))
    # network_copy, component_list_copy = update_capacity_factors(network_copy, component_list_copy, "solar", capacity_factors_pv.sel(x=lon, y=lat))

    network_copy, component_list_copy = update_fuel_cost(
        str(lon) + str(lat), gas_cost, network_copy, component_list_copy, base_costs, case_dict['nyears']
    )
    run_pypsa(network_copy, case_dict)
    write_result(network_copy, case_dict, component_list_copy, file_name, outfile_suffix=f'_gas{gas_cost}_{str(lon)}_{str(lat)}')

    # # Binary search in log space to find the breakeven cost
    # low_cost, high_cost = 1., 1000000.
    # low_log, high_log = np.log(low_cost), np.log(high_cost)  # Convert bounds to log space

    # while (np.exp(high_log) - np.exp(low_log)) > 0.001:  # Terminate when the difference is less than 10
    #     mid_log = (low_log + high_log) / 2  # Midpoint in log space
    #     mid_cost = np.exp(mid_log)  # Convert back to linear space
    #     print(f'Mid cost: {mid_cost}')
    #     # Run the optimization model with the current mid_cost
    #     network_copy, component_list_copy = update_fuel_cost(
    #         str(lon) + str(lat), mid_cost, network_copy, component_list_copy, base_costs, case_dict['nyears']
    #     )

    #     # Run PyPSA with new costs
    #     run_pypsa(network_copy, case_dict)
    #     write_result(network_copy, case_dict, component_list_copy, file_name, outfile_suffix=f'_{str(lon)}_{str(lat)}')

    #     # Include component statistics also if 0 after optimization
    #     network_copy.statistics.set_parameters(drop_zero=False)
    #     print(f'Concentrated solar: {network_copy.statistics.supply().loc[("Generator", "concentrated solar")]}')
    #     print(f'Load: {network_copy.statistics.withdrawal().loc[("Load", "load")]}')
    #     # Check if the technology is deployed or dominates
    #     if condition == 'deployed':
    #         passed_condition = network_copy.statistics.supply().loc[('Generator', 'concentrated solar')] > 0
    #         print(f'Passed deployed condition: {passed_condition}')
    #     elif condition == 'dominating':
    #         passed_condition = (
    #             network_copy.statistics.supply().loc[('Generator', 'concentrated solar')] /
    #             network_copy.statistics.withdrawal().loc[('Load', 'load')]
    #         ) > 0.5
    #         print(f'Passed dominating condition: {passed_condition}')
    #         print(f'Ratio: {network_copy.statistics.supply().loc[("Generator", "concentrated solar")] / network_copy.statistics.withdrawal().loc[("Load", "load")]}')
    #     else:
    #         raise ValueError('Invalid condition')

    #     if passed_condition:
    #         # If the technology is deployed, lower the upper bound in log space
    #         high_log = mid_log
    #     else:
    #         # If the technology is not deployed, raise the lower bound in log space
    #         low_log = mid_log

    # # Return the breakeven cost
    # breakeven_cost = np.exp((low_log + high_log) / 2)

    # Calculate concentrated solar supply as sum of what is dispatched by the generator and storage
    cs_supply = network_copy.statistics.supply().loc[('Generator', 'concentrated solar')] + network_copy.statistics.supply().loc[('Store', 'molten salt storage')] - network_copy.statistics.withdrawal().loc[('Store', 'molten salt storage')]
    # Get concentrated solar dispatch fraction
    cs_fraction = cs_supply / network_copy.statistics.withdrawal().loc[('Load', 'load')]

    # Write results to a csv file
    if not os.path.exists(f'output_data/cs_fraction_{name_suffix}_gas{gas_cost}'):
        os.makedirs(f'output_data/cs_fraction_{name_suffix}_gas{gas_cost}')
    with open(f'output_data/cs_fraction_{name_suffix}_gas{gas_cost}/results_{name_suffix}_gas{gas_cost}_{str(lon)}_{str(lat)}.csv', 'w') as f:
        f.write(f'{lon},{lat},{cs_fraction}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-f', help='Name of the base case file', required=True)
    parser.add_argument('--lon', '-x', help='Longitude of the grid cell', required=True)
    parser.add_argument('--lat', '-y', help='Latitude of the grid cell', required=True)
    parser.add_argument('--gas_cost', '-g', help='Gas cost for the grid cell', required=True)
    args = parser.parse_args()

    process_grid_cell(float(args.lon), float(args.lat), args.file_name, args.gas_cost)