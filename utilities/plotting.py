import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xarray as xr
from utilities.utils import mask_data_world, compute_equal_area_bands


def get_world():
    """
    Get world map from shapefile.
    """
    # Load the world shapefile
    world = gpd.read_file('input_files/ne_110m_admin_0_countries.shp')
    # Drop Antarctica by excluding everyhting below -60 latitude
    world = world[world.geometry.centroid.y > -60]

    return world


def plot_result_map(file_path, case_name, title, cmap_label):
    """
    Plot the world map with data.
    """

    dataset = xr.open_dataset(file_path)
    dataset.close()
    # Reindex the dataset to 0.25 degree resolution
    dataset = dataset.reindex(x=np.arange(-180, 180.25, 0.25), y=np.arange(-57, 85.25, 0.25), method='nearest')

    # Get world map
    world = get_world()

    # Mask dataset with world map
    masked_dataset = mask_data_world(dataset, world)

    # Plot the result with 1 step on x axis is the same as 1 step on y axis
    # use log scale for colorbar
    aspect_ratio = (dataset['x'].max()-dataset['x'].min())/(dataset['y'].max()-dataset['y'].min())
    if 'fraction' in cmap_label:
        cmap = 'viridis'
        label = 'fraction'
    else:
        cmap = 'inferno'
        label = 'time'
    masked_dataset[f'breakeven_cost'].plot(x='x', y='y', aspect=aspect_ratio, size=5, cmap=cmap, cbar_kwargs={'label': cmap_label}, vmin=0)#, vmax=1)

    # Remove x and y labels and ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel('')
    # Add title
    plt.gca().set_title(title)
    # Remove box around the plot
    for edge in ['top', 'right', 'bottom', 'left']:
        plt.gca().spines[edge].set_visible(False)

    # Add world map outline in light grey
    world.boundary.plot(ax=plt.gca(), color='lightgrey', linewidth=0.5)

    # Save the plot
    if not os.path.exists('figures'):
        os.makedirs('figures')
    gas_cost = title.replace("Gas cost = $", "").replace("/MWh", "")
    plt.savefig(f'figures/csp_fraction_map_{case_name}_gas{gas_cost}_{label}.pdf', bbox_inches='tight', dpi=200)


def plot_lat_bands(world):
    """
    Plot the world map with latitude bands.
    """
    # Compute latitude bands based on equal landmass
    band_edges = compute_equal_area_bands()
    # Group results based on computed latitude bands
    lat_bands = [(band_edges[i], band_edges[i+1]) for i in range(len(band_edges)-1)]

    # Plot the world map with latitude bands
    fig, ax = plt.subplots(figsize=(12, 6))
    world.plot(ax=ax, color='lightgrey')

    # Overlay latitudinal bands
    colors = sns.color_palette("rocket", as_cmap=True)(np.linspace(0, 1, int(len(lat_bands)/2)))
    # Set the last color to light blue with 4 numbers
    colors[-1] = (0.5, 0.5, 1, 1)
    # Mirror the colors to get the full range
    colors = [tuple(c) for c in colors]
    colors = colors[::-1] + colors

    for i, (lat_min, lat_max) in enumerate(lat_bands):
            # Clip the land polygons to the latitude band
            land_band = world.clip_by_rect(-180, lat_min, 180, lat_max)
            # Plot the landmass with the band color
            land_band.plot(ax=ax, color=colors[i])

    # Save figure
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/lat_bands.pdf', bbox_inches='tight')
    
    return lat_bands, colors



def plot_line(lat_df, colors):
    """
    Plot the mean cs fraction for each latitudinal band vs the gas cost.
    """
    plt.figure()
    # Plot the mean cs fraction for each latitudinal band vs the gas cost with error bars
    for i, lat in enumerate(lat_df['latitude band'].unique()):
        lat_df_sub = lat_df.loc[lat_df['latitude band'] == lat]
        plt.plot(lat_df_sub['gas cost'], lat_df_sub['median cs fraction'], color=colors[i])
        plt.errorbar(lat_df_sub['gas cost'], lat_df_sub['median cs fraction'], 
                     # Errors as interquartile range
                     yerr=[lat_df_sub['lower_error'], lat_df_sub['upper_error']],
                      fmt='o', color=colors[i], markersize=4, capsize=4,label=lat)
                

    # x-axis log scale
    plt.xscale('log')

    plt.xlabel('Gas cost ($/MWh)')
    plt.ylabel('Concentrated solar fraction')
    plt.legend()

    plt.title('Median concentrated solar fraction with 10th and 90th percentile')
    plt.savefig('figures/median_cs_fraction_vs_gas_cost.pdf', bbox_inches='tight')