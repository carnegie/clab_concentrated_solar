#!/bin/bash

cd /groups/carnegie_poc/awongel/clab_concentrated_solar/concentrated_solar_capacity_factors/monthly_cf_files

region="world"

# Loop over years
for year in 2023; do
  for month in 1 2 3 4 5 6 7 8 9 10 11 12; do
    # Define the expected output file
    output_csv="${region}_csp_CF_timeseries_${year}_${month}.nc"
    
    # Check if the file already exists
    if [ ! -f "$output_csv" ]; then
      # Create a unique job name and output file for each job
      job_name="run_atlite_${region}_${year}_${month}"
      # Create directory for output files if it doesn't exist
      mkdir -p out
      output_file="out/run_atlite_${region}_${year}_${month}.out"

      # Generate the sbatch script
      cat <<EOT > submit_${region}_${year}_${month}.sh
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --mem=100G
#SBATCH --time=96:00:00
#SBATCH --job-name=$job_name
#SBATCH --output=$output_file

cd /groups/carnegie_poc/awongel/clab_concentrated_solar/concentrated_solar_capacity_factors/monthly_cf_files

source /home/awongel/miniconda3/etc/profile.d/conda.sh

conda activate atlite

python get_global_CFs.py --year $year --month $month
EOT

      # Submit the job
      echo "Submitting job for $year $month"
      sbatch submit_${region}_${year}_${month}.sh

      # Optional: remove the temporary script file after submission
      rm submit_${region}_${year}_${month}.sh
    else
      echo "File $output_csv already exists, skipping job submission for $year $month"
    fi
  done  
done