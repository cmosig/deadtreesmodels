from doitall import *
import rasterio
import sys

filename = sys.argv[1]

poly = inference_deadwood(filename)
save_poly(filename.split("/")[-1].replace(".tif", ".gpkg"), poly, rasterio.open(filename).crs) 
