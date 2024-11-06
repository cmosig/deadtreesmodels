import torch
import os
import rasterio
import numpy as np
from doitall import save_poly

from doitall import inference_deadwood

data_path = "/mnt/gsdata/projects/projects/deadtrees/drone_campaigns/naturpark_schwarzwald/blackforestnationalparktimeseries/TDOP_mosaic/"
file_name = "TDOP_2023_123_jpg85_btif_mosaic.tif"

# Use device directly instead of set_device
energy_map, polygons = inference_deadwood(data_path + file_name, write_energy_map=True)
path = data_path + file_name.replace(".tif", "_deadwood_pred.shp")

save_poly(path, polygons, "epsg:4326")

# Write the energy map to a GeoTIFF file
energy_map_path = data_path + file_name.replace(".tif", "_deadwood_energy_map.tif")

# Set up the metadata for the new file
kwargs = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'count': 1,
    'width': energy_map.shape[1],
    'height': energy_map.shape[0],
    'crs': 'epsg:4326',  # Assuming same CRS as input
    'compress': 'lzw'    # Optional compression
}

# Write the file
with rasterio.open(energy_map_path, "w", **kwargs) as dst:
    dst.write(energy_map.astype(np.float32), 1)
