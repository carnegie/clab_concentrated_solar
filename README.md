# Concentrated solar for heat

Run a global cost optimization for generating constant heat supply with concentrated solar vs natural gas (and vs solar PV) with and without storage on gridded solar capacity factors.

Run the optimation with

```python scan_grid.py -f <input_file>```

where input_file is the case file, e.g.

```python scan_grid.py -f input_files/CSP_case.xlsx```

which reads the global capacity factors from an input .nc file () and runs PyPSA on each grid cell to find the least-cost solution to supply the constant heat demand.

Store the results in maps in .nc file to prepare for the plotting step with
```python store_results.py```


Plot the resulting .nc file with the interactive jupyter script
```plot_result_map.ipynb```

