import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xarray as xr
from utilities.utils import mask_data_world, compute_equal_area_bands
from matplotlib.colors import LogNorm

def get_world():
    """
    Get world map from shapefile.
    """
    # Load the world shapefile
    world = gpd.read_file('input_files/ne_110m_admin_0_countries.shp')
    # Drop Antarctica by excluding everyhting below -60 latitude
    world = world[world.geometry.centroid.y > -60]

    return world

def overlay_gas_infrastructure(ax):
    """
    Overlay the gas infrastructure on the world map.
    """
    gas_infrastructure_path = 'input_files/GEM-GGIT-Gas-Pipelines-2024-12/GEM-GGIT-Gas-Pipelines-2024-12.geojson'
    gas_infrastructure = gpd.read_file(gas_infrastructure_path)
    gas_infrastructure.plot(ax=ax, color='darkred', linewidth=0.5)


def plot_result_map(file_path, case_name, title, cmap_label, gas_infrastructure=False):
    """
    Plot the world map with data.
    """
    dataset = xr.open_dataset(file_path)
    dataset.close()

    dataset = dataset.reindex(x=np.arange(-180, 180.25, 0.25), y=np.arange(-57, 85.25, 0.25), method='nearest')
    world = get_world()
    masked_dataset = mask_data_world(dataset, world)

    gas_cost = title.replace("Gas cost = $", "").replace("/MWh", "")
    if 'fraction' in cmap_label:
        cmap = 'viridis'
        norm = None
        label = f'gas{gas_cost}_fraction'
        cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    elif 'fuel' in cmap_label:
        cmap = 'cividis'
        norm = LogNorm(vmin=10, vmax=10000)
        label = 'threshold'
        cbar_ticks = [10, 100, 1000, 10000]
    else:
        cmap = 'inferno'
        norm = plt.Normalize(vmin=0, vmax=10)
        label = f'gas{gas_cost}_time'
        cbar_ticks = [0, 5, 10]

    aspect_ratio = (dataset['x'].max() - dataset['x'].min()) / (dataset['y'].max() - dataset['y'].min())

    p = masked_dataset['value'].plot(
        x='x', y='y', aspect=aspect_ratio, size=5,
        cmap=cmap, norm=norm,
        cbar_kwargs={'label': cmap_label, 'ticks': cbar_ticks},
    )

    p.set_rasterized(True)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)

    for edge in ['top', 'right', 'bottom', 'left']:
        ax.spines[edge].set_visible(False)

    world.boundary.plot(ax=ax, color='lightgrey', linewidth=0.5)

    if gas_infrastructure:
        overlay_gas_infrastructure(ax)
        label += '_gas_infrastructure'

    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/csp_map_{case_name}_{label}.pdf', bbox_inches='tight', dpi=300)




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



def plot_line(lat_df, colors, var):
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
    if var == 'cs_fraction':
        plt.ylabel('Concentrated solar fraction')
        # Replace y-axis tick label 0.1 with 10% etc
        ax = plt.gca()
        ax.yaxis.tick_right()         # Move ticks to the right
        ax.yaxis.set_label_position('right')  # Move label to the right
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x*100)}%'))
        
    else:
        plt.ylabel('Charging time (h)')
    plt.legend()    

    plt.title('Median concentrated solar fraction with 10th and 90th percentile')
    plt.savefig(f'figures/median_{var}_vs_gas_cost.pdf', bbox_inches='tight')