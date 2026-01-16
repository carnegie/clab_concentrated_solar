#!/bin/bash
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --job-name=store_results
#SBATCH --output=logs/store_results_%A_%a.out

# UPDATE PATH TO REPOSITORY HERE
cd clab_concentrated_solar/
# UPDATE PATH TO CONDA HERE
source conda.sh
conda activate table_pypsa_env

vars=(cs_fraction storage_ratio system_cost capacity_natgas capacity_cst natgas_fuel_use gas_price_min_frac)
var=${vars[$SLURM_ARRAY_TASK_ID]}

echo "Running variable: $var"
python store_results.py -v $var
