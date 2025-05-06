import os
import glob
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import shapely.vectorized as sv


def mask_data_world(data, world):
    """
    Mask data with world shapefile.
    """
    # Extract the geometry from the GeoSeries (use union_all if there are multiple polygons)
    region_geom = world.geometry.union_all() if len(world) > 1 else world.geometry.iloc[0]

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
    cs_supply = result_data["component results"]["Supply [MW]"].loc[("Generator", "csp glasspoint")] + result_data["component results"]["Supply [MW]"].loc[('Store', 'molten-salt-store glasspoint')] - result_data["component results"]["Withdrawal [MW]"].loc[('Store', 'molten-salt-store glasspoint')]
    # Get concentrated solar dispatch fraction
    cs_fraction = cs_supply / result_data["component results"]["Withdrawal [MW]"].loc[('Load', 'load')]

    return cs_fraction


def get_storage_ratio(result_data):
    """
    Get the storage ratio from the result data
    """
    storage_capacity = result_data["component results"]["Optimal Capacity [MW]"].loc[("Store", "molten-salt-store glasspoint")]
    generator_capacity = result_data["component results"]["Optimal Capacity [MW]"].loc[("Generator", "csp glasspoint")]

    if not generator_capacity == 0:
        storage_ratio = storage_capacity / generator_capacity
    else:
        storage_ratio = np.nan

    return storage_ratio


def fill_results_from_pickle(result_array, case, cond, var, csfrac_threshold=0.5):
    """
    Fill the results array with the results from the pickle files
    """
    pickle_path = f'output_data/{case.lower()}/{case.lower()}_gas{cond}_*.pickle'
    for file in glob.glob(pickle_path):

        with open(file, 'rb') as f:
            result_data = pickle.load(f)
        f.close()


        # Extract the x and y coordinates from the file name
        x = float(file.split('_')[-2])
        y = float(file.split('_')[-1].replace('.pickle', ''))


        # Get the value of the variable
        if var == 'cs_fraction':
            value = get_cs_fraction(result_data)
        elif var == 'storage_ratio':
            value = get_storage_ratio(result_data)
        elif var == 'gas_price_min_frac':
            cs_fraction = get_cs_fraction(result_data)
            if cs_fraction >= csfrac_threshold:
                value = result_array.loc[{'x': result_array.x.sel(x=x, method="nearest"),
                                        'y': result_array.y.sel(y=y, method="nearest")}].item()
                if pd.isna(value) or float(cond) < value:
                    value = float(cond)

            else:
                value = np.nan
        else:
            raise ValueError(f"Variable {var} not supported")

        result_array.loc[{'x': result_array.x.sel(x=x, method="nearest"), 
                        'y': result_array.y.sel(y=y, method="nearest")}] = value
            
    return result_array



def store_results_map(case, cond, variable, result_array=None):
    """
    Store the results of the breakeven cost analysis in a netcdf file.
    """
    filename = f'output_data/{variable}_{case}_gas{cond}.nc' if variable != 'gas_price_min_frac' else f'output_data/{variable}_{case}_threshold0p5.nc'
    if not os.path.exists(filename):
        
        if result_array is None:
            cap_fac = xr.open_dataset('concentrated_solar_capacity_factors/world_csp_CF_timeseries_2023.nc')
            # Create xarray to store results
            result_array = xr.DataArray(
                data=None,
                dims=['x', 'y'],
                coords={
                    'x': cap_fac.x,
                    'y': cap_fac.y},
                name='value')
        
        result_array = fill_results_from_pickle(result_array, case, cond, variable)

    return result_array


def compute_equal_area_bands(num_bands=10):
    """
    Compute latitudinal bands that divide the Earth's total surface area (land + ocean) equally.
    """    
    # Define latitude bins (1-degree resolution for accuracy)
    lat_bins = np.arange(-60, 90, 1)  # Cover full latitude range
    
    # Earth's total surface area in square km
    R = 6371  # Earth radius in km

    # Compute total area per latitude band using the spherical cap formula
    total_areas = []
    for i in range(len(lat_bins) - 1):
        lat_min, lat_max = np.radians(lat_bins[i]), np.radians(lat_bins[i + 1])
        strip_area = 2 * np.pi * R**2 * (np.sin(lat_max) - np.sin(lat_min))
        total_areas.append((lat_bins[i], lat_bins[i + 1], strip_area))

    # Convert to NumPy array
    total_areas = np.array(total_areas, dtype=[('lat_min', 'f4'), ('lat_max', 'f4'), ('area', 'f4')])

    # Compute cumulative total area
    total_area = total_areas['area'].sum()
    cumulative_area = np.cumsum(total_areas['area'])

    # Identify breakpoints for equal-area division
    band_edges = np.interp(
        np.linspace(0, total_area, num_bands + 1),
        cumulative_area, 
        total_areas['lat_max']
    )

    return band_edges


def calculate_lat_band_mean(lat_bands_df, masked_dataset, lat_bands, gas_cost):
        """
        Calculate the mean concentrated solar fraction for each latitudinal band and store the results in a dictionary.
        """

        # Calculate the mean cs fraction for each latitudinal band
        cs_frac_bands = [masked_dataset.sel(y=slice(lat_min, lat_max)) for lat_min, lat_max in lat_bands]

        # Concatenate north and south bands by combining the first and last band, the second and second last band, etc.
        half_n = len(cs_frac_bands) // 2
        labels = ['high-latitudes', 'mid-latitudes', 'subtropical', 'tropical', 'equatorial']
        ns_merged_bands = {}

        for i in range(half_n):
            ns_merged_bands[labels[i]] = cs_frac_bands[i].merge(cs_frac_bands[len(cs_frac_bands)-1-i])

        # Store the results in a dataframe
        add_to_df = pd.DataFrame()
        add_to_df['latitude band'] = labels
        add_to_df['mean cs fraction'] = [np.nanmean(ns_merged_bands[label]['value'].values) for label in labels]
        add_to_df['std cs fraction'] = [np.nanstd(ns_merged_bands[label]['value'].values) for label in labels]
        add_to_df['median cs fraction'] = [np.nanmedian(ns_merged_bands[label]['value'].values) for label in labels]

        # Compute interquartile range (IQR) while ignoring NaNs
        q1_values = [np.nanpercentile(ns_merged_bands[label]['value'], 25) for label in labels]
        q3_values = [np.nanpercentile(ns_merged_bands[label]['value'], 75) for label in labels]

        # Error bars
        add_to_df['lower_error'] = add_to_df['median cs fraction'] - q1_values
        add_to_df['upper_error'] = q3_values - add_to_df['median cs fraction']

        add_to_df['gas cost'] = gas_cost

        # Add the results to the dataframe
        lat_bands_df = pd.concat([lat_bands_df, add_to_df])

        return lat_bands_df