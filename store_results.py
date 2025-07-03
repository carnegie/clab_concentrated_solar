import os
from utilities.utils import store_results_map


def main():
    """
    Store the fraction of CSP storage and breakeven costs in a netcdf file
    """
    gas_costs=[10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 250, 300]
    case = 'CSP_storage'
    frac_results = {'0p0': None, '0p25': None, '0p5': None, '0p75': None}
    # get thresholds from frac_results keys and replace 'p' with '.'
    cs_thresholds = [float(key.replace('p', '.')) for key in frac_results.keys()]

    for var in ['cs_fraction','storage_ratio', 'system_cost', 'gas_price_min_frac']:
        print(var)
        results = None
        for gas_cost in gas_costs:
            print(gas_cost)
            if not var == 'gas_price_min_frac':
                results = store_results_map(case, gas_cost, var)

                # Save the results to a netcdf file
                if not os.path.exists(f'output_data/{var}_{case}_gas{gas_cost}.nc'):
                    results.to_netcdf(f'output_data/{var}_{case}_gas{gas_cost}.nc')

            else:
                for cs_thresh in cs_thresholds:
                    print(cs_thresh)
                    frac_results[str(cs_thresh).replace(".", "p")] = store_results_map(case, gas_cost, var, result_array=frac_results[str(cs_thresh).replace(".", "p")], csfrac_threshold=cs_thresh)
                    print("results dict", frac_results)
                    
        if var == 'gas_price_min_frac':
            for cs_threshold in cs_thresholds:
                # Save the results to a netcdf file
                output_file = f'output_data/{var}_{case}_threshold{str(cs_threshold).replace(".", "p")}.nc'
                if not os.path.exists(output_file):
                    frac_results[str(cs_threshold).replace(".", "p")].to_netcdf(output_file)

if __name__ == "__main__":
    main()
