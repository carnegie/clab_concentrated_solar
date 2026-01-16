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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm


def truncate_colormap(cmap, minval=0.25, maxval=1.0, n=256):
    """ 
    Truncate a colormap to a specified range.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

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

    gas_cost = title.replace("Gas fuel cost = $", "").replace("/MWh", "")
    if 'Fraction' in cmap_label:
        # Truncated viridis (lighter blue-green start)
        trunc_viridis = truncate_colormap(plt.cm.viridis)
        cmap = trunc_viridis
        # linear norm with 0-1 range
        norm = mcolors.Normalize(vmin=0, vmax=1)
        label = f'gas{gas_cost}_fraction'
        cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    else:
        trunc_inferno = truncate_colormap(plt.cm.inferno, minval=0.1, maxval=1.0)
        bounds = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500])
        n_bins = len(bounds) - 1
        log_centers = (np.log10(bounds[:-1]) + np.log10(bounds[1:])) / 2
        normalized = (log_centers - log_centers[0]) / (log_centers[-1] - log_centers[0])
        colors = trunc_inferno(normalized)
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, ncolors=n_bins+1, extend='max')

        label = file_path.split('_')[-1].split('.')[0]
        cbar_ticks = [10, 50, 100, 500]

    aspect_ratio = (dataset['x'].max() - dataset['x'].min()) / (dataset['y'].max() - dataset['y'].min())

    fig, ax = plt.subplots(figsize=(8, 5))  # explicitly create ax

    p = masked_dataset['value'].plot(
    x='x', y='y', ax=ax, cmap=cmap, norm=norm,
    cbar_kwargs={'label': cmap_label,  'shrink': 0.8, 'pad': 0.02, 'ticks': cbar_ticks, 'extend': "neither" if "Fraction" in cmap_label else "max"})

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



def plot_line(plotting_df, country, color, var='cs_fraction'):
    """
    Plot the median cs fraction, charging time, or system cost (depending on var) for each country 
    vs the gas cost with error bars.
    """
    plt.figure()
    plotting_df_sub = plotting_df.loc[plotting_df['country'] == country]
    if var == 'capacities':
        plot_vars = ['capacity_natgas', 'capacity_cst']
    else:
        if var == 'co2_emissions':
            plotting_df_sub[f'median co2_emissions'] = 1 - plotting_df_sub['median cs_fraction']
            plotting_df_sub.loc[:, f'lower error {var}'] = -1 * plotting_df_sub[f'lower error cs_fraction']
            plotting_df_sub.loc[:, f'upper error {var}'] = -1 * plotting_df_sub[f'upper error cs_fraction']
            plotting_df_sub[f'cell used co2_emissions'] = 1 - plotting_df_sub['cell used cs_fraction']


        plot_vars = [var]

    for plot_var in plot_vars:
        if plot_var == 'capacity_natgas' and len(plot_vars) > 1:
            plot_color = 'grey'

        else:
            plot_color = color
        plt.plot(plotting_df_sub['gas cost'], plotting_df_sub[f'median {plot_var}'], color=plot_color, marker='o', 
                 markersize=5, label=country, linestyle='-')
        # if not plot_var == 'capacity_natgas':
        plt.plot(plotting_df_sub['gas cost'], plotting_df_sub[f'cell used {plot_var}'], color=plot_color, linestyle=':')        
        # Draw band instead of error bars
        plt.fill_between(plotting_df_sub['gas cost'], plotting_df_sub[f'median {plot_var}']-plotting_df_sub[f'lower error {plot_var}'],
                     plotting_df_sub[f'median {plot_var}']+plotting_df_sub[f'upper error {plot_var}'], alpha=0.2, color=plot_color,
                     label=f'{country} error band')
                
    plt.xlim(10,500)
    plt.xlabel('Gas fuel cost ($/MWh)', fontsize=14)

    if var == 'cs_fraction':
        plt.ylabel('Fraction of demand met with concentrated solar', fontsize=14)
        # Replace y-axis tick label 0.1 with 10% etc
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x*100)}%'))
        # y-axis range
        plt.ylim(0, 1.01)
        
    elif var == 'storage_ratio':
        plt.ylabel(f'Storage capacity\n(hours of mean demand)', fontsize=14)
        # y-axis range
        plt.ylim(0, 100)

    elif var == 'co2_emissions':
        plt.ylabel(f'Relative CO$_2$ emissions\n(% of gas only case)', fontsize=14)
        # Replace y-axis tick label 0.1 with 10% etc
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x*100)}%'))
        # y-axis range
        plt.ylim(0, 1.01)

    elif var == 'capacities':
        plt.ylabel('Nameplate capacity (MW/MW)', fontsize=14)
        # y-axis range
        plt.ylim(0, 20)
    
    elif var == 'capacity_natgas':
        plt.ylim(0, 1.02)
        plt.ylabel('Nameplate capacity (MW/MW)', fontsize=14)

    elif var == 'natgas_fuel_use':
        plt.ylim(0, 1.02)
        # Show y-labels as %, so times 100 and add % sign
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: f'{int(y*100)}%'))
        plt.ylabel('Natural gas fuel use (fraction of total supply)', fontsize=14)

    else:
        print(f"Variable {var} not recognized. Please use 'cs_fraction', 'storage_ratio', or 'co2_emissions'.")

    if (var == 'capacities' or var == 'co2_emissions'):
        # Plot vertical dashed black line at gas cost = 48 and gas cost = 350
        plt.axvline(x=48, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(x=350, color='black', linestyle='--', linewidth=0.8)
    
    # Make all labels larger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/median_{var}_vs_gas_cost_{country}.pdf', bbox_inches='tight')


def plot_cost_emissions_change(res_df, country, color, var):
    """
    Plot the marginal CO2 abatement cost. 
    """

    # Get country data with index country and gas cost
    country_df_sub = res_df[res_df['country'] == country]

    relative_emissions = (1. - country_df_sub['median cs_fraction'])

    plt.figure(figsize=(6, 4))
    
    # Plot relative emissions vs gas cost
    plt.plot(country_df_sub['gas cost'], relative_emissions, color=color, linestyle='--', 
             label='Median Emissions with 10th&90th Percentile')
    plt.fill_between(country_df_sub['gas cost'], relative_emissions + country_df_sub['lower error cs_fraction'],
                     relative_emissions - country_df_sub['upper error cs_fraction'], alpha=0.2, color=color)

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

    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/change_emissions_{country.replace(" ","_")}.pdf', bbox_inches='tight')


def plot_dispatch_curve(outpath, country, cell, gas_cost, month='07'):
    # Load data
    file_path = f'{outpath}/cst_storage_gas{gas_cost}_{cell}.pickle'
    results_data = read_output_file(file_path)

    # Build dispatch
    dispatch_all = results_data['time results'].filter(like='dispatch').rename(
        columns=lambda x: x.replace(' dispatch', '').replace(' link', '')
    )

    # Add anything with discharge to dispatch_all
    dispatch_all['heat storage out'] = results_data['time results'].loc[:, 'molten-salt-store glasspoint discharged']
    dispatch_all['heat storage in'] = results_data['time results'].loc[:, 'molten-salt-store glasspoint charged']

    # Select window
    dispatch_all = dispatch_all.loc[f'2023-{month}-17':f'2023-{month}-24']

    dispatch = dispatch_all[['gas boiler steam', 'heat storage out', 'cst glasspoint']]
    dispatch_out_neg = -1 * dispatch_all['heat storage in']

    # Create figure with 2 panels stacked vertically
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': [4, 1]}
    )

    # --- Panel 1: dispatch ---
    dispatch.plot.area(
        ax=ax1,
        stacked=True,
        color=["#4b4844", "#da2b3a", "#dd9a34", "#250a9a"],
        linewidth=0
    )
    dispatch_out_neg.plot(
        ax=ax1, kind='area', color="#da2b3a", linewidth=0, alpha=0.7
    )

    ax1.set_title(f'Dispatch Curve for {country}\nCell ({cell}) - Gas Cost ${gas_cost}/MWh', fontsize=16)
    ax1.set_ylabel('Dispatch\nper mean demand (MW/MW)', fontsize=18)
    ax1.tick_params(axis='both', which='both', labelsize=24)
    ax1.set_ylim(-3, 4)
    ax1.axhline(y=1, color='black', linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-')
    # Vertical grey line at each day boundary
    for day in range(18, 25):
        ax1.axvline(x=f'2023-{month}-{day} 00:00:00', color='grey', linestyle='-', linewidth=0.8)
    if ax1.get_legend():
        ax1.get_legend().remove()

    # --- Panel 2: storage SOC ---
    storage_soc = results_data['time results']['molten-salt-store glasspoint state of charge'].loc[f'2023-{month}-17':f'2023-{month}-24']
    storage_soc.plot(ax=ax2, color="#da2b3a", linewidth=2)
    ax2.set_ylabel('Storage SOC\n(h of mean demand)', fontsize=18)
    ax2.set_ylim(0, 15)
    ax2.tick_params(axis='both', which='both', labelsize=24)
    # Vertical grey line at each day boundary
    for day in range(18, 25):
        ax2.axvline(x=f'2023-{month}-{day} 00:00:00', color='grey', linestyle='-', linewidth=0.8)

    # X-axis label only on bottom panel
    ax2.set_xlabel('Date', fontsize=14)

    # Layout and save
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig(
        f'figures/dispatch_curve_cell_{cell}_{country}_gas_cost_{gas_cost}_{month}.pdf',
        bbox_inches='tight'
    )



def plot_system_cost_share(outpath, country, cell, gas_costs):
    """
    Plot the system cost share for a given grid cell.
    """
    gas_case_costs = []
    stack_data = {}
    for gas_cost in gas_costs:
        
        filepath = f"{outpath}/cst_storage_gas{gas_cost}_{cell}.pickle"
        cost_contr, tot_demand = get_cost_contributions(filepath)

        if gas_cost == gas_costs[0]:
            # For the gas only case, we need to normalize the cost contributions by the total met demand
            gas1_system_cost = cost_contr.sum()
        # Normalize the cost contributions by the total met demand
        cost_contr_norm = cost_contr / tot_demand

        for name, cost_value in cost_contr_norm.items():
            if name not in stack_data:
                stack_data[name] = []
            stack_data[name].append(cost_value)

        gas_case_cost = get_gas_case_cost(gas1_system_cost, gas_cost)
        gas_case_costs.append(gas_case_cost/tot_demand)

    # Create a dictionary of the gas costs and the gas case costs
    gas_case_costs = np.array(gas_case_costs)
    gas_cost_dict = dict(zip(gas_costs, gas_case_costs))

    # Create a dictionary for the total system cost at each gas cost
    total_system_cost = {}
    for i, gas_cost in enumerate(gas_costs):
        total_system_cost[gas_cost] = sum(stack_data[name][i] for name in stack_data)
    total_system_cost = np.array([total_system_cost[gc] for gc in gas_costs])
    total_system_cost_dict = dict(zip(gas_costs, total_system_cost))

    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots() 
    ax.stackplot(gas_costs,
                 [stack_data[label] for label in stack_data],
                    labels=stack_data.keys(),
                    colors=["#dd9a34", "#4b4844", "#da2b3a"],)
    ax.plot(gas_costs, gas_case_costs, color='black', linestyle='--', label='Gas Case Cost', linewidth=2)
    ax.set_xlim(0, 500)
    # fontsize for x and y ticks
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_ylim(0, 550)
    ax.set_xlabel('Gas fuel cost ($/MWh)', fontsize=14)
    ax.set_ylabel(f'System Cost\n($/MWh met demand)', fontsize=14)
    
    # Save figure
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig(f'figures/system_cost_share_{country}.pdf', bbox_inches='tight')

    return gas_cost_dict, total_system_cost_dict

