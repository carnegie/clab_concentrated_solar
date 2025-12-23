import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

import atlite

from shapely.geometry import box

# Define your target lat/lon (e.g., Ras Al Khair, Saudi Arabia)
lat = 27.625
lon = 49.125

# Define a small bounding box around the point (ERA5 resolution is ~0.25°)
delta = 0.125  # half a grid cell
# Create bounding box and extract bounds
bounds_polygon = box(lon - delta, lat - delta, lon + delta, lat + delta)
bounds = bounds_polygon.bounds

# Create the cutout
cutout = atlite.Cutout(
    path="ras_al_khair_2023.nc",
    module="era5",
    bounds=bounds,
    time="2023",
)

# # Prepare the cutout (downloads data and processes it)
# cutout.prepare()


# world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
# saudi_arabia_ = world.query('name == "Saudi Arabia"')


# cutout = atlite.Cutout(
#     path="ras_al_khair_2023_era5.nc",
#     module="era5",
#     bounds=saudi_arabia_.iloc[0].geometry.bounds,
#     time="2023",
# )

cutout.prepare(["influx"], monthly_requests=True)


# Calculation of Capacity Factor and Specific Generation for: SAM_solar_tower installation
pt = {
    "capacity factor": cutout.csp(
        installation="SAM_parabolic_trough", capacity_factor=True
    ).rename("SAM_parabolic_trough CF"),
    "specific generation": cutout.csp(installation="SAM_parabolic_trough").rename(
        "SAM_parabolic_trough SG"
    ),
}

# Calculation of Capacity Factor and Specific Generation for: lossless solar tower installation
ll = {
    "capacity factor": cutout.csp(
        installation="lossless_installation",
        technology="parabolic trough",
        capacity_factor=True,
    ).rename("lossless_installation CF"),
    "specific generation": cutout.csp(
        installation="lossless_installation",
        technology="parabolic trough",
    ).rename("lossless_installation SG"),
}

# # Plot results side by side
# fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# pt["capacity factor"].plot(ax=axes[0][0], cmap="inferno")
# pt["specific generation"].plot(ax=axes[1][0], cmap="inferno")
# axes[0][0].set_title("SAM_parabolic_trough")

# ll["capacity factor"].plot(ax=axes[0][1], cmap="inferno")
# ll["specific generation"].plot(ax=axes[1][1], cmap="inferno")
# axes[0][1].set_title("lossless_installation")

# # Overlay Spainish borders
# for ax in axes.ravel():
#     spain.plot(ax=ax, fc="none", ec="black")

# fig.tight_layout()

# # Save figure
# plt.savefig("csp_capacity_factor_Atlite_default.png", dpi=300)

# Extract the concenctrated solar power generation capacity factors
csp_power_generation = cutout.csp(
    installation="SAM_parabolic_trough", 
    capacity_factor_timeseries=True,)

# Save gridded data as netCDF files
csp_power_generation.to_netcdf(f"rasalkhair_csp_CF_timeseries_2023.nc")