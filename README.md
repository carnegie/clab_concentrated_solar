# Repository for the publication "Supplying process heat using concentrated solar thermal energy with molten salt storage" by Alicia Wongel, Jacqueline A. Dowling, Gerhard Weinrebe, Steven J. Davis,  Ken Caldeira

Run a global cost optimization for generating constant heat supply with concentrated solar with molten salt storage vs natural gas on gridded solar capacity factors.

## Getting started

### Clone the repository

Clone this repository recursively (this ensures that the submodules are also cloned):

```git clone https://github.com/carnegie/clab_concentrated_solar.git --recursive```

### Set up the environment

The setup described below uses [conda](https://docs.conda.io/en/latest/miniconda.html) for easy package management.

   ```cd clab_concentrated_solar```

   ```conda env create -f table_pypsa/env.yaml```

   ```conda activate table_pypsa_env```

The environment needs to be activated every time a new session is started.

### Install a solver

Per default, the solver [Gurobi](https://www.gurobi.com/) will be installed. It requires a license to be used ([free for academics](https://www.gurobi.com/academia/academic-program-and-licenses/)). See instructions on the [PyPSA website](https://docs.pypsa.org/latest/home/installation/#solvers) for open source and other commercial solvers.

If you use a different solver, make sure to change the field "solver" in you case input file as well!

## Run the model

### Case input file

The network is defined in a case input file. The input file for this analysis can be found in

```input_files/CST_storage_case.xlsx```

### Data input files

Wind and solar capacity factors are obtained with [Atlite](https://github.com/PyPSA/atlite). These are obtained for each grid cell and inserted into the inputs by running the script scan_grid.py, where each optimization is run individually for each grid cell, see below. 

The gridded capacity factors can be found in 
```concentrated_solar_capacity_factors/```

and further details on how to obtain these in the README of that directory ```concentrated_solar_capacity_factors/README.md```.

Costs of the different technologies assumed in this analysis can be found in
```input_files/costs_concentrated_solar.csv```.

### Run the optimization

As the optimization is run on a global grid and for 31 gas fuel cost values, we recommend running the optimizations on an HPC cluster. Open the script ```scan_grid.py``` to update the paths and then submit the jobs that run the optimizations with the following command:

```python scan_grid.py -f input_files/CST_storage_case.xlsx```

which reads the global gridded capacity factors from the file ```concentrated_solar_capacity_factors/world_cst_CF_timeseries_2023.nc``` and runs PyPSA on each grid cell and for each gas fuel cost value to find the least-cost solution to supply the constant heat demand.

Store the results in maps in .nc files to prepare for the plotting step with
```python store_results.py -v <variable>```

replacing ```<variable>``` with the different plotting variables ```cs_fraction storage_ratio system_cost capacity_natgas capacity_cst natgas_fuel_use gas_price_min_frac```.

This can also be done on an HPC cluster, with the bash script
```run_store_results_job.sh```, make sure to open that script and update the paths.

The resulting maps are stored in ```output_data/cst_storage/maps/```.

Create the plots for the publication with the interactive jupyter script
```plot_result_map.ipynb```, which stores the resulting plots in ```figures/```.

