import argparse
import xarray as xr
import subprocess

# Get file name from command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-f', help='Name of the base case file', required=True)

def main():
    args = parser.parse_args()
    file_name = args.file_name

    capacity_factors_csp = xr.open_dataset('concentrated_solar_capacity_factors/world_csp_CF_timeseries_2023.nc')

    # Print number of grid cells
    print(f"Number of grid cells: {capacity_factors_csp.x.size * capacity_factors_csp.y.size}")
    # Print x and y size
    print(f"x values: {capacity_factors_csp.x.size}")
    print(f"y values: {capacity_factors_csp.y.size}")
    # Submit a job for each grid cell
    for lon in capacity_factors_csp.x.values:
        for lat in capacity_factors_csp.y.values:
            for gas_cost in [10000]:#[10, 50, 100, 500, 1000]:

                # Construct the command for the job
                command = [
                'sbatch',
                '--job-name', f'grid_cell_{lon}_{lat}_{gas_cost}',
                '--output', f'logs/grid_cell_{lon}_{lat}_{gas_cost}.out',
                '--time', '24:00:00',
                '--error', f'logs/grid_cell_{lon}_{lat}_{gas_cost}.err',
                '--wrap',
                f'cd /groups/carnegie_poc/awongel/clab_concentrated_solar && '
                'source /home/awongel/miniconda3/etc/profile.d/conda.sh && '
                'export GRB_LICENSE_FILE=/central/software/gurobi/gurobi1000/linux64/license_files/gurobi.lic && '
                'conda activate table_pypsa_env && '
                f'python process_grid_cell.py --lon {lon} --lat {lat} -f {file_name} --gas_cost {gas_cost}'
                ]


                # Submit the job
                try:
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"Job submitted for grid cell ({lon}, {lat} and gas cost {gas_cost}). Job ID: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to submit job for grid cell ({lon}, {lat} and gas cost {gas_cost}). Error: {e.stderr.strip()}")

if __name__ == "__main__":
    main()