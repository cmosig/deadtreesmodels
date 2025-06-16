import sys
import segmentation_models_pytorch as smp
from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
from unet_model import UNet
from torchvision.transforms.functional import crop
import os

import geopandas as gpd
import numpy as np
import rasterio
import utm
from safetensors.torch import load_model
from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon, MultiPolygon

# DEADWOOD_MODEL_PATH = "/net/scratch/cmosig/model.safetensors"
# DEADWOOD_MODEL_PATH = "/net/scratch/jmoehring/checkpoints/3d_18k_100e_vanilla_tversky_a01b09g2/fold_0_epoch_99/model.safetensors"
DEADWOOD_MODEL_PATH = '/net/scratch/cmosig/experiment_dir_deadwood_segmentation/segformer_b5_oversample_newdata/fold_0_epoch_74/model.safetensors'

# pathtotile = sys.argv[1]
pathtotile = '/net/scratch/jmoehring/tiles/spain_20_09_2023_south_tejeda_1_ortho/0.08/1588_4484.tif'
# pathtotile = "/net/scratch/jmoehring/tiles/spain_20_09_2023_south_tejeda_1_ortho/0.08/1588_900.tif"

# preferably use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model with three input channels (RGB)
# model = UNet(
#     n_channels=3,
#     n_classes=1,
# ).to(memory_format=torch.channels_last)

model = smp.Unet(
	encoder_name='mit_b5',
	encoder_weights='imagenet',
	in_channels=3,
	classes=1,
).to(memory_format=torch.channels_last)
model = torch.compile(model)

load_model(model, DEADWOOD_MODEL_PATH)
model = model.to(memory_format=torch.channels_last, device=device)
model.eval()


# load image patch
image = None
with rasterio.open(pathtotile) as src:
	image = src.read()

	# Reshape the image tensor to have 3 channels
	image = image.transpose(1, 2, 0)

	image_transforms = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)
	image = image_transforms(image).float().contiguous()

	# convert to torch tensor
	image = torch.tensor(image)

	# add batch dimension
	image = image.unsqueeze(0)

	# to device
	image = image.to(device=device, memory_format=torch.channels_last)

output = model(image)
output = torch.sigmoid(output)
output = output > 0.5

# save output to disk as geotiff with same projection as input
output = output.cpu().detach().numpy()
output = output.astype(np.uint8)
output = output.squeeze()

# get affine transformation from input
with rasterio.open(pathtotile) as src:
	transform = src.transform
	crs = src.crs
	nodata = src.nodata

# outpath is filename replaced with _deadwood.tif, but only filename
output_path = pathtotile.split('/')[-1].replace('.tif', '_deadwood_heavy.tif')

# save output to disk
with rasterio.open(
	output_path,
	'w',
	driver='GTiff',
	height=output.shape[0],
	width=output.shape[1],
	count=1,
	dtype=output.dtype,
	crs=crs,
	transform=transform,
	nodata=nodata,
) as dst:
	dst.write(output, 1)
