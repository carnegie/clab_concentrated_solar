import os
import glob
import pickle
import numpy as np
import geopandas as gpd
import xarray as xr
import pandas as pd
import shapely.vectorized as sv
from shapely.geometry import box

def get_world():
    """
    Get world map from shapefile.
    """
    # Load the world shapefile
    world = gpd.read_file('input_files/ne_110m_admin_0_countries.shp')

    # Drop Antarctica by excluding everything below -60 latitude
    world = world[world.geometry.centroid.y > -60]

    return world

def get_country(country_name):
    """
    Get country map from shapefile.
    """
    # Load the world shapefile
    world = gpd.read_file('input_files/ne_110m_admin_0_countries.shp')

    # Filter the world GeoDataFrame to get the specific country
    country = world[world['NAME'] == country_name]

    # For the US, remove Alaska and Hawaii
    if country_name == 'United States of America':
        # Only keep contiguous US
        contiguous_48_bbox = box(minx=-125, miny=24.396308, maxx=-66.93457, maxy=49.384358)
        # Clip the US geometry to the bounding box
        country = country.geometry.intersection(contiguous_48_bbox)

    return country

def mask_data_region(data, region):
    """
    Mask data with world shapefile.
    """
    # Extract the geometry from the GeoSeries (use union_all if there are multiple polygons)
    region_geom = region.geometry.union_all() if len(region) > 1 else region.geometry.iloc[0]

    # Get the grid of coordinates (lon, lat) from the xarray DataArray
    lon, lat = np.meshgrid(data['x'], data['y'], indexing='ij')

    # Use shapely's vectorized.contains to create a mask for points within the region
    mask = sv.contains(region_geom, lon, lat)

    # Apply the mask to the dataset DataArray
    masked_data = data.where(mask)

    return masked_data


def get_cs_fraction(result_data):
    """
    Get the cs fraction from the result data
    """
        # Calculate concentrated solar supply as sum of what is dispatched by the generator and storage
    cs_supply = result_data["component results"]["Supply [MW]"].loc[("Generator", "cst glasspoint")] + result_data["component results"]["Supply [MW]"].loc[('Store', 'molten-salt-store glasspoint')] - result_data["component results"]["Withdrawal [MW]"].loc[('Store', 'molten-salt-store glasspoint')]
    # Get concentrated solar dispatch fraction
    cs_fraction = cs_supply / result_data["component results"]["Withdrawal [MW]"].loc[('Load', 'load')]

    return cs_fraction


def get_storage_ratio(result_data):
    """
    Get the storage ratio from the result data
    """
    storage_capacity = result_data["component results"]["Optimal Capacity [MW]"].loc[("Store", "molten-salt-store glasspoint")]

    return storage_capacity

def get_capacity(result_data, var):
    """
    Get the capacity of the given variable from the result data
    """
    if var == 'natgas':
        capacity = result_data["component results"]["Optimal Capacity [MW]"].loc[("Generator", "gas boiler steam")]
    elif var == 'cst':
        capacity = result_data["component results"]["Optimal Capacity [MW]"].loc[("Generator", "cst glasspoint")]
    else:
        raise ValueError(f"Variable {var} not supported")
    
    return capacity

def fill_results_from_pickle(result_array, case, cond, var, csfrac_threshold=0.5):
    """
    Fill the results array with the results from the pickle files
    """
    pickle_path = f'output_data/{case.lower()}/{case.lower()}_gas{cond}_*.pickle'
    print(f"Looking for pickle files at: {pickle_path}")
    for file in glob.glob(pickle_path):
        print(file)

        with open(file, 'rb') as f:
            result_data = pickle.load(f)
        f.close()


        # Extract the x and y coordinates from the file name
        x = float(file.split('_')[-2])
        y = float(file.split('_')[-1].replace('.pickle', ''))


        # Get the value of the variable
        if var == 'cs_fraction':
            value = get_cs_fraction(result_data)
            print(value)
        elif var == 'storage_ratio':
            value = get_storage_ratio(result_data)
        elif var == 'gas_price_min_frac':
            cs_fraction = get_cs_fraction(result_data)
            if cs_fraction > csfrac_threshold:
                value = result_array.loc[{'x': result_array.x.sel(x=x, method="nearest"),
                                        'y': result_array.y.sel(y=y, method="nearest")}].item()
                if pd.isna(value) or float(cond) < value:
                    value = float(cond)
            else:
                value = np.nan
        elif var == 'system_cost':
            value = result_data["component results"]["Capital Expenditure [$]"].sum() + result_data["component results"]["Operational Expenditure [$]"].sum()
        elif 'capacity_' in var:
            value = get_capacity(result_data, var.split('capacity_')[-1])
        else:
            raise ValueError(f"Variable {var} not supported")
        
        print(value)

        result_array.loc[{'x': result_array.x.sel(x=x, method="nearest"), 
                        'y': result_array.y.sel(y=y, method="nearest")}] = value
            
    return result_array


def store_results_map(out_path, case, cond, variable, result_array=None, csfrac_threshold=0.5):
    """
    Store the results of the breakeven cost analysis in a netcdf file.
    """
    filename = f'{out_path}/{variable}_{case}_gas{cond}.nc' if variable != 'gas_price_min_frac' else f'{out_path}/{variable}_{case}_threshold{str(csfrac_threshold).replace(".", "p")}.nc'
    if not os.path.exists(filename):

        # Create directory if it doesn't exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        if result_array is None:
            cap_fac = xr.open_dataset('concentrated_solar_capacity_factors/world_cst_CF_timeseries_2023.nc')
            # Create xarray to store results
            result_array = xr.DataArray(
                data=None,
                dims=['x', 'y'],
                coords={
                    'x': cap_fac.x,
                    'y': cap_fac.y},
                name='value')
        
        result_array = fill_results_from_pickle(result_array, case, cond, variable, csfrac_threshold)

    return result_array


def calculate_country_median(df, dataset, var, countries, gas_cost):
    cs_frac_countries = [mask_data_region(dataset, get_country(country)) for country in countries]

    new_df = pd.DataFrame({
        'country': countries,
        'gas cost': gas_cost,
        f'median {var}': [np.nanmedian(cs['value'].values) for cs in cs_frac_countries],
        f'lower error {var}': [
            np.nanmedian(cs['value'].values) - np.nanpercentile(cs['value'], 10) for cs in cs_frac_countries
        ],
        f'upper error {var}': [
            np.nanpercentile(cs['value'], 90) - np.nanmedian(cs['value'].values) for cs in cs_frac_countries
        ]
    })

    if df.empty:
        return new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)
        return df
    

def read_output_file(file_path):
    """
    Read the pickle file and return the data.
    """
    with open(file_path, 'rb') as f:
        result_data = pickle.load(f)
    f.close()

    return result_data

def get_gas_case_cost(system_cost, gas_cost):
    """
    Get the cost of the gas only case for each gas cost.
    """
    # Calculate the cost relative to 100% gas
    # TODO grab these values from the input file rather than hardcoding
    capital_cost = 28659.73903
    efficiency = 0.95
    VOM = 6
    fuel_cost = (system_cost - capital_cost - VOM*8760) * efficiency
    marginal_cost = VOM*8760 + (fuel_cost * (gas_cost)) / efficiency

    ref_cost = capital_cost + marginal_cost
    return ref_cost

def get_cost_contributions(filepath):
    """
    Get the cost contributions of the different technologies from the pickle file.
    """

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

        component_data = data['component results']
        # Add by carrier
        component_data_carrier = component_data.groupby('carrier').sum()
        cost = component_data_carrier['Capital Expenditure [$]'] + component_data_carrier['Operational Expenditure [$]']
        # Divide by total met demand i.e. withdrawal of load
        total_met_demand = component_data['Withdrawal [MW]']['Load'].sum()
        # Drop load
        cost = cost[['concentrated solar', 'gas boiler', 'molten salt storage']]
        
    # Close the file
    f.close()
    return cost, total_met_demand



def calculate_co2_abatement_cost(country, results_df,  gas_cost_dict, total_system_cost_dict, gas_cost=50):
    """
    Calculate the CO2 abatement cost for a given country and gas cost.
    """

    # Get CST fractions
    cst_fraction = results_df[(results_df['country'] == country) & (results_df['gas cost'] == gas_cost)]['median cs_fraction'].values[0]
    ng_fraction = 1 - cst_fraction

    ng_emissions = 0.181  # ton CO2/MWh for natural gas

    # Calculate emissions
    emissions = ng_fraction * ng_emissions  # ton CO2/MWh

    # Calculate abatement cost

    abatement_cost = (total_system_cost_dict[gas_cost] - gas_cost_dict[gas_cost]) / (0.181 - emissions)  # $/ton CO2

    print(f"Country: {country}, Gas cost: {gas_cost}, CST fraction: {cst_fraction:.2f}, Emissions: {emissions:.3f} ton CO2/MWh, Abatement cost: ${abatement_cost:.2f}/ton CO2")