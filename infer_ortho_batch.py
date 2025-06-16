from deadwood.deadwood_inference import DeadwoodInference
import rasterio
from common.common import *
import sys
import os

pathlist = sys.argv[1]

deadwodinference = DeadwoodInference(
	config_path='/net/home/cmosig/projects/deadtreesmodels/deadwood_inference_config.json'
)

with open(pathlist) as f:
	files = f.readlines()

	for filename in files:
		print('processing', filename)
		f = filename.strip()

		outpath = f.split('/')[-1].replace('.tif', '_prediction.gpkg')

		if os.path.exists(outpath):
			print('skipping', filename)
			continue

		polygons = deadwodinference.inference_deadwood(f)
		save_poly(outpath, polygons, crs=rasterio.open(f).crs)
