# Concentrated solar thermal generation capacity factors

Install the Atlite version in this repository with

```
cd atlite
pip install -e .
```

Obtain monthly time series of global capacity factors with

```python monthly_cf_files/get_global_CFs.py```

See ```monthly_cf_files/run_atlite_global.sh``` for an example shell script to run it for all months. 

Postprocess the monthly files to merge them into one annual time series file and reduce grid by factor 10 in both dimensions with

```python post_process_time_series.py```

The resulting .nc file will be used in the optimization as input (see clab_concentrated_solar/README.md).

Calculate the annual mean with 

```python post_process_annual_mean.py```

and plot the resulting mean capacity factors with

```postprocess_cfs.ipynb```

