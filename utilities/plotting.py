import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xarray as xr
import pickle
from utilities.utils import get_world, mask_data_region, read_output_file, get_cost_contributions, get_gas_case_cost
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors


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
    # masked_dataset = masked_dataset.where(masked_dataset['value'] > 0)


    gas_cost = title.replace("Gas fuel cost = $", "").replace("/MWh", "")
    if 'Fraction' in cmap_label:
        from matplotlib.colors import LinearSegmentedColormap
        def truncate_colormap(cmap, minval=0.25, maxval=1.0, n=256):
            new_cmap = LinearSegmentedColormap.from_list(
                f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                cmap(np.linspace(minval, maxval, n))
            )
            return new_cmap

        # Truncated viridis (lighter blue-green start)
        trunc_viridis = truncate_colormap(plt.cm.viridis)
        cmap = trunc_viridis
        # linear norm with 0-1 range
        norm = mcolors.Normalize(vmin=0, vmax=1)
        label = f'gas{gas_cost}_fraction'
        cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    else:
        cmap = 'inferno'
        norm = LogNorm(vmin=10, vmax=500)
        # norm = mcolors.Normalize(vmin=10, vmax=250)
        label = file_path.split('_')[-1].split('.')[0]
        cbar_ticks = [10, 100, 200]

    aspect_ratio = (dataset['x'].max() - dataset['x'].min()) / (dataset['y'].max() - dataset['y'].min())

    fig, ax = plt.subplots(figsize=(8, 5))  # explicitly create ax

    p = masked_dataset['value'].plot(
    x='x', y='y', ax=ax, cmap=cmap, norm=norm,
    cbar_kwargs={'label': cmap_label,  'shrink': 0.8, 'pad': 0.02, 'ticks': cbar_ticks})

    # p.set_rasterized(True)

    ax.set_aspect(aspect_ratio)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)
    ax.set_rasterization_zorder(0)


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
    plt.savefig(f'figures/cst_map_{case_name}_{label}.pdf', bbox_inches='tight', dpi=300, rasterized=True)
    plt.savefig(f'figures/cst_map_{case_name}_{label}.png', bbox_inches='tight', dpi=300)



def plot_line(lat_df, country, color, var='cs_fraction'):
    """
    Plot the median cs fraction, charging time, or system cost (depending on var) for each country 
    vs the gas cost with error bars.
    """
    plt.figure()
    lat_df_sub = lat_df.loc[lat_df['country'] == country]
    plt.plot(lat_df_sub['gas cost'], lat_df_sub[f'median {var}'], color=color)
    plt.plot(lat_df_sub['gas cost'], lat_df_sub[f'median {var}'], color=color, marker='o', markersize=5, label=country)
    # Draw band instead of error bars
    plt.fill_between(lat_df_sub['gas cost'], lat_df_sub[f'median {var}']-lat_df_sub[f'lower error {var}'], 
                     lat_df_sub[f'median {var}']+lat_df_sub[f'upper error {var}'], alpha=0.2, color=color,
                     label=f'{country} error band')
                
    # x-axis log scale
    # plt.xscale('log')
    plt.xlim(10,500)


    plt.xlabel('Gas fuel cost ($/MWh)', fontsize=14)
    if var == 'cs_fraction':
        plt.ylabel('Fraction of demand met with concentrated solar', fontsize=14)
        # Replace y-axis tick label 0.1 with 10% etc
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x*100)}%'))
        # y-axis range
        plt.ylim(0, 1.01)
        
    else:
        plt.ylabel('Storage capacity per mean demand (h)', fontsize=14)
        # y-axis range
        plt.ylim(0, 100)
    
    # Make all labels larger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # plt.legend()    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/median_{var}_vs_gas_cost_{country}.pdf', bbox_inches='tight')


def plot_cost_emissions_change(res_df, country, color, var):
    """
    Plot the marginal CO2 abatement cost. 
    """

    # Get country data with index country and gas cost
    country_df_sub = res_df[res_df['country'] == country]

    # Calculate the cost relative to 100% gas
    # TODO grab these values from the input file rather than hardcoding
    capital_cost = 28659.73903
    efficiency = 0.95
    VOM = 6
    fuel_cost = (country_df_sub.loc[country_df_sub['gas cost'] == 10, f'median {var}'].values[0] - capital_cost - VOM*8760) * efficiency
    marginal_cost = VOM*8760 + (fuel_cost * (country_df_sub['gas cost'] / 10)) / efficiency

    ref_cost = capital_cost + marginal_cost
    relative_cost = (country_df_sub[f'median {var}'] / ref_cost)
    relative_cost_lower = (country_df_sub[f'lower error {var}'] / ref_cost)
    relative_cost_upper = (country_df_sub[f'upper error {var}'] / ref_cost)

    relative_emissions = (1. - country_df_sub['median cs_fraction'])

    plt.figure(figsize=(6, 4))
    # Plot relative cost vs gas cost
    # plt.plot(country_df_sub['gas cost'], relative_cost, color=color, label='Median Cost with 10th&90th Percentile')
    # plt.fill_between(country_df_sub['gas cost'], relative_cost - relative_cost_lower,
    #                  relative_cost + relative_cost_upper, alpha=0.2, color=color)
    
    # Plot relative emissions vs gas cost
    plt.plot(country_df_sub['gas cost'], relative_emissions, color=color, linestyle='--', 
             label='Median Emissions with 10th&90th Percentile')
    plt.fill_between(country_df_sub['gas cost'], relative_emissions + country_df_sub['lower error cs_fraction'],
                     relative_emissions - country_df_sub['upper error cs_fraction'], alpha=0.2, color=color)

    # plt.xscale('log')
    plt.xticks(fontsize=14)
    plt.xlim(10, 500)
    # Add % sign to y-axis labels
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{int(y*100)}%'))
    plt.xlabel('Gas fuel cost ($/MWh)', fontsize=14)
    plt.ylabel(f'Relative emissions\n(% of gas only case)', fontsize=14)
    # y-axis range
    plt.ylim(0, 1.01)
    plt.yticks(fontsize=14)

    # plt.title(f'{country}')
    # plt.legend()
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/change_emissions_{country.replace(" ","_")}.pdf', bbox_inches='tight')



def plot_dispatch_curve(country, cell, gas_cost):
    """
    Plot the dispatch curve for a given grid cell.
    """
    # Load the data
    file_path = f'output_data/cst_storage_new/cst_storage_new_gas{gas_cost}_{cell}.pickle'
    results_data = read_output_file(file_path)

    # Extract the dispatch data
    dispatch_all = results_data['time results'].filter(like='dispatch').rename(columns=lambda x: x.replace(' dispatch', '').replace(' link', ''))
    # Order should be gas boiler steam, cst glasspoint, heat storage link out
    dispatch = dispatch_all[['gas boiler steam', 'cst glasspoint', 'heat storage out']]
    dispatch_out = dispatch_all[['heat storage in']]

    # Plot the dispatch data in a stacked plot
    plt.figure(figsize=(10, 6))
    dispatch.plot.area(stacked=True, alpha=0.7, color=["#4b4844", "#dd9a34", "#da2b3a"], linewidth=0.5)
    dispatch_out['heat storage in'] = -1 * dispatch_out['heat storage in']
    dispatch_out['heat storage in'].plot.area(color="#da2b3a", alpha=0.4, linewidth=0.5)
    plt.title(f'Dispatch Curve for {country} - Cell ({cell}) - Gas Cost ${gas_cost}/MWh', fontsize=16)
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel('Dispatch (MW)', fontsize=14)
    plt.ylim(-1, 2)
    # Legend outside plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    # Plot a line at y=1 to indicate the constant demand
    plt.axhline(y=1, color='black', linestyle='--', label='Constant Demand')
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/dispatch_curve_cell_{cell}_{country}_gas_cost_{gas_cost}.pdf', bbox_inches='tight')

def plot_system_cost_share(country, cell, gas_costs):
    """
    Plot the system cost share for a given grid cell.
    """
    gas_case_costs = []
    stack_data = {}
    for gas_cost in gas_costs:
        filepath = f"output_data/csp_storage/csp_storage_gas{gas_cost}_{cell}.pickle"
        cost_contr, tot_demand = get_cost_contributions(filepath)
        if gas_cost == 10:
            # For the gas only case, we need to normalize the cost contributions by the total met demand
            gas10_system_cost = cost_contr.sum()
        # Normalize the cost contributions by the total met demand
        cost_contr_norm = cost_contr / tot_demand

        for name, cost_value in cost_contr_norm.items():
            if name not in stack_data:
                stack_data[name] = []
            stack_data[name].append(cost_value)

        gas_case_cost = get_gas_case_cost(gas10_system_cost, gas_cost)
        gas_case_costs.append(gas_case_cost/tot_demand)
        # print(f"Gas cost: {gas_cost}, Gas case cost: {gas_case_cost/tot_demand}, Normal cost: {cost_contr_norm.sum()}")


    fig, ax = plt.subplots(figsize=(6, 4))

    ax.stackplot(gas_costs,
                 [stack_data[label] for label in stack_data],
                    labels=stack_data.keys(),
                    colors=["#dd9a34", "#4b4844", "#da2b3a"],)
    ax.plot(gas_costs, gas_case_costs, color='black', linestyle='--', label='Gas Case Cost', linewidth=2)
    ax.set_title(f'{country} - Cell ({cell})')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_xlabel('Gas fuel cost ($/MWh)', fontsize=14)
    ax.set_ylabel(f'System Cost\n($/MWh met demand)', fontsize=14)
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Save figure
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/system_cost_share_{country}.pdf', bbox_inches='tight')

