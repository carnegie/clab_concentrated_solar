import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xarray as xr
from utilities.utils import get_world, mask_data_region, compute_equal_area_bands
from matplotlib.colors import LogNorm



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
    masked_dataset = mask_data_region(dataset, world)
    masked_dataset = masked_dataset.where(masked_dataset['value'] > 0)


    gas_cost = title.replace("Gas cost = $", "").replace("/MWh", "")
    if 'fraction' in cmap_label:
        cmap = 'viridis'
        norm = None
        label = f'gas{gas_cost}_fraction'
        cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    elif 'fuel' in cmap_label:
        cmap = 'inferno'
        norm = LogNorm(vmin=10, vmax=10000)
        label = file_path.split('_')[-1].split('.')[0]
        cbar_ticks = [10, 100, 1000, 10000]
    else:
        cmap = 'cividis'
        norm = plt.Normalize(vmin=0, vmax=10)
        label = f'gas{gas_cost}_time'
        cbar_ticks = [0, 5, 10]

    aspect_ratio = (dataset['x'].max() - dataset['x'].min()) / (dataset['y'].max() - dataset['y'].min())

    fig, ax = plt.subplots(figsize=(8, 5))  # explicitly create ax

    p = masked_dataset['value'].plot(
    x='x', y='y', ax=ax, cmap=cmap, norm=norm,
    cbar_kwargs={'label': cmap_label, 'ticks': cbar_ticks, 'shrink': 0.8, 'pad': 0.02},)

    p.set_rasterized(True)

    ax.set_aspect(aspect_ratio)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)


    for edge in ['top', 'right', 'bottom', 'left']:
        ax.spines[edge].set_visible(False)

    world["geometry"] = world["geometry"].simplify(0.5)
    world.boundary.plot(ax=ax, color='lightgrey', linewidth=0.5)

    if gas_infrastructure:
        overlay_gas_infrastructure(ax)
        label += '_gas_infrastructure'

    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    fig.canvas.draw()  # Forces all artists to be fully rendered before saving
    plt.savefig(f'figures/csp_map_{case_name}_{label}.pdf', bbox_inches='tight', dpi=300, rasterized=True)
    plt.savefig(f'figures/csp_map_{case_name}_{label}.png', bbox_inches='tight', dpi=300)




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



def plot_line(lat_df, country, color, var='cs_fraction'):
    """
    Plot the mean cs fraction for each latitudinal band vs the gas cost.
    """
    plt.figure()
    # Plot the mean cs fraction for each latitudinal band vs the gas cost with error bars
    lat_df_sub = lat_df.loc[lat_df['country'] == country]
    plt.plot(lat_df_sub['gas cost'], lat_df_sub['median cs fraction'], color=color)
    plt.plot(lat_df_sub['gas cost'], lat_df_sub['median cs fraction'], color=color, marker='o', markersize=5, label=country)
    # Draw band instead of error bars
    plt.fill_between(lat_df_sub['gas cost'], lat_df_sub['median cs fraction']-lat_df_sub['lower_error'], 
                     lat_df_sub['median cs fraction']+lat_df_sub['upper_error'], alpha=0.2, color=color,
                     label=f'{country} error band')
                
    # x-axis log scale
    plt.xscale('log')

    plt.xlabel('Gas cost ($/MWh)', fontsize=14)
    if var == 'cs_fraction':
        plt.ylabel('Concentrated solar fraction', fontsize=14)
        # Replace y-axis tick label 0.1 with 10% etc
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x*100)}%'))
        
    else:
        plt.ylabel('Charging time (h)', fontsize=14)
        # y-axis range
        plt.ylim(0, 20)
    
    # Make all labels larger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.legend()    
    plt.savefig(f'figures/median_{var}_vs_gas_cost_{country}.pdf', bbox_inches='tight')

def plot_marginal_abatement_cost(country_df, country, color):
    """
    Plot the marginal abatement cost.
    """
    plt.figure()
    country_df_sub = country_df.loc[country_df['country'] == country]
    # Calculate Marginal Abatement Cost (MAC)
    mac = (country_df_sub['system_cost'] - country_df_sub['gas cost']) / (1 - country_df_sub['cs_fraction'])
    plt.plot(country_df_sub['median cs fraction'], mac, color=color)

