import argparse
import xarray as xr
import subprocess
import itertools


# Get file name from command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', '-f', help='Name of the base case file', required=True)

def main():
    args = parser.parse_args()
    file_name = args.file_name

    capacity_factors_csp = xr.open_dataset('concentrated_solar_capacity_factors/world_cst_CF_timeseries_2023.nc')

    # Print number of grid cells
    print(f"Number of grid cells: {capacity_factors_csp.x.size * capacity_factors_csp.y.size}")
    # Print x and y size
    print(f"x values: {capacity_factors_csp.x.size}")
    print(f"y values: {capacity_factors_csp.y.size}")
    
    # Submit a job for each grid cell
    # All job configurations
    job_args = list(itertools.product(
        capacity_factors_csp.x.values,
        capacity_factors_csp.y.values,
        [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
         120, 140, 160, 180, 200, 
         220, 240, 260, 280, 300,
         320, 340, 360, 380, 400,
         420, 440, 460, 480, 500]  # Gas costs in $/MWh
        ))

    # Batch size: how many jobs per SLURM submission
    batch_size = 100

    # Break into batches
    for i in range(0, len(job_args), batch_size):
        batch = job_args[i:i+batch_size]

        job_name = f'grid_batch_{i}'
        out_log = f'logs/grid_batch_{i}.out'
        err_log = f'logs/grid_batch_{i}.err'

        # Build inner command: one python call per (lon, lat, gas_cost)
        inner_commands = []
        for lon, lat, gas_cost in batch:
            inner_commands.append(
                f'python process_grid_cell.py --lon {lon} --lat {lat} -f {file_name} --gas_cost {gas_cost}'
            )
        inner_script = ' && '.join(inner_commands)

        # Full sbatch command
        command = [
            'sbatch',
            '--job-name', job_name,
            '--output', out_log,
            '--time', '24:00:00',
            '--cpus-per-task', '8',
            '--error', err_log,
            '--wrap',
            # UPDATE REPOSITORY PATH HERE
            f'cd REPOSITORY_PATH && '
            # UPDATE PATH TO CONDA HERE
            'source conda.sh && '
            # UPDATE PATH TO GUROBI LICENSE HERE
            'export GRB_LICENSE_FILE=gurobi.lic && '
            'conda activate table_pypsa_env && '
            + inner_script
        ]

        # Submit the job
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f'Submitted job {job_name}')


if __name__ == "__main__":
    main()