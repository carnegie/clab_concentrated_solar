import pandas as pd
import os
import numpy as np
from table_pypsa.run_pypsa import build_network, run_pypsa, write_result
from table_pypsa.utilities.load_costs import load_costs
import argparse
import copy
import xarray as xr
from dask import delayed, compute
import subprocess

# Get file name from command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-f', help='Name of the base case file', required=True)
parser.add_argument('--condition', '-c', help='Condition to optimize for', required=True, choices=['deployed', 'dominating'])

def main():
    args = parser.parse_args()
    file_name = args.file_name
    condition = args.condition

    capacity_factors_csp = xr.open_dataset('input_files/world_csp_CF_timeseries_2023_coarse.nc')
    capacity_factors_pv = xr.open_dataset('input_files/world_solar_CF_timeseries_2023_coarse.nc')


    # Submit a job for each grid cell
    for lon in capacity_factors_csp.x.values:
        for lat in capacity_factors_csp.y.values:

            # Construct the command for the job
            command = [
            'sbatch',
            '--job-name', f'grid_cell_{lon}_{lat}',
            '--output', f'logs/grid_cell_{lon}_{lat}.out',
            '--time', '24:00:00',
            '--error', f'logs/grid_cell_{lon}_{lat}.err',
            '--wrap',
            f'cd /groups/carnegie_poc/awongel/clab_concentrated_solar && '
            'source /home/awongel/miniconda3/etc/profile.d/conda.sh && '
            'export GRB_LICENSE_FILE=/central/software/gurobi/951/linux64/license_files/gurobi.lic && '
            'conda activate table_pypsa_env && '
            f'python process_grid_cell.py --lon {lon} --lat {lat} -f {file_name} --condition {condition}'
            ]


            # Submit the job
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Job submitted for grid cell ({lon}, {lat}). Job ID: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit job for grid cell ({lon}, {lat}). Error: {e.stderr.strip()}")

if __name__ == "__main__":
    main()