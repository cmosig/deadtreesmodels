from doitall import *
import os
import rasterio
import sys

filename = sys.argv[1]

model = load_deadwood_model()

for f in open(filename).readlines():
    f = f.strip()
    outpath = f.split("/")[-1].replace(".tif", ".gpkg")

    # skip if exists
    if os.path.exists(outpath):
        continue

    try:
        poly = inference_deadwood(f, model)
    except Exception as e:
        print(f, e)
        continue
    save_poly(outpath, poly, rasterio.open(f).crs) 
