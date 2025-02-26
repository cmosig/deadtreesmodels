import torch
import rasterio
from torchvision.transforms.functional import crop
import json
import safetensors.torch
import segmentation_models_pytorch as smp
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .InferenceDataset import InferenceDataset
from ..common import mask_to_polygons, filter_polygons_by_area, reproject_polygons, image_reprojector


class DeadwoodInference:
	def __init__(self, config_path: str, model_path: str):
		torch.set_float32_matmul_precision('high')

		with open(config_path, 'r') as f:
			self.config = json.load(f)

		self.model_path = model_path
		self.model = None
		# set float32 matmul precision for higher performance

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.load_model()

	def get_cache_path(self):
		model_path = Path(self.model_path)
		return model_path.parent / f'{self.config["model_name"]}_pretrained.pt'

	def load_model(self):
		if 'segformer_b5' not in self.config['model_name']:
			print('Invalid model name: ', self.config['model_name'], 'Exiting...')
			exit()

		cache_path = self.get_cache_path()

		# Try to load from cache first
		if cache_path.exists():
			model = smp.Unet(
				encoder_name='mit_b5',
				encoder_weights=None,  # Don't load pretrained weights
				in_channels=3,
				classes=1,
			).to(memory_format=torch.channels_last)

			model.load_state_dict(torch.load(str(cache_path)))
		else:
			# Load with pretrained weights and cache
			model = smp.Unet(
				encoder_name='mit_b5',
				encoder_weights='imagenet',
				in_channels=3,
				classes=1,
			).to(memory_format=torch.channels_last)

			torch.save(model.state_dict(), str(cache_path))

		# Apply final model preparations
		model = torch.compile(model)
		safetensors.torch.load_model(model, self.model_path)
		model = model.to(memory_format=torch.channels_last, device=self.device)
		model.eval()

		self.model = model

	def inference_deadwood(self, input_tif):
		"""
		gets path to tif file and returns polygons of deadwood in the CRS of the tif
		"""

		# will always return a memory file, also when not reprojecting
		reprojected_mem_file = image_reprojector(
			input_tif, min_res=self.config['deadwood_minimum_inference_resolution']
		)

		dataset = InferenceDataset(image_path=reprojected_mem_file, tile_size=1024, padding=256)

		loader_args = {
			'batch_size': self.config['batch_size'],
			'num_workers': self.config['num_dataloader_workers'],
			'pin_memory': True,
			'shuffle': False,
		}
		inference_loader = DataLoader(dataset, **loader_args)

		outimage = np.zeros((dataset.height, dataset.width), dtype=np.float32)
		for images, cropped_windows in tqdm(inference_loader, desc='inference'):
			images = images.to(device=self.device, memory_format=torch.channels_last)

			output = None
			with torch.no_grad():
				# if the the batch size is smaller than the configured batch size, apply padding
				if images.shape[0] < self.config['batch_size']:
					pad = torch.zeros((self.config['batch_size'], 3, 1024, 1024), dtype=torch.float32)
					pad[: images.shape[0]] = images
					# move to device
					pad = pad.to(device=self.device, memory_format=torch.channels_last)
					output = self.model(pad)
					output = output[: images.shape[0]]
				else:
					output = self.model(images)

				output = torch.sigmoid(output)

			# go through batch and save to output
			for i in range(output.shape[0]):
				output_tile = output[i].cpu()

				# crop tensor by dataset padding
				output_tile = crop(
					output_tile,
					top=dataset.padding,
					left=dataset.padding,
					height=dataset.tile_size - (2 * dataset.padding),
					width=dataset.tile_size - (2 * dataset.padding),
				)

				# derive min/max from cropped window
				minx = cropped_windows['col_off'][i]
				maxx = minx + cropped_windows['width'][i]
				miny = cropped_windows['row_off'][i]
				maxy = miny + cropped_windows['width'][i]

				# clip to positive values when writing and also to the image size
				diff_minx = 0
				if minx < 0:
					diff_minx = abs(minx)
					minx = 0

				diff_miny = 0
				if miny < 0:
					diff_miny = abs(miny)
					miny = 0

				diff_maxx = 0
				if maxx > outimage.shape[1]:
					diff_maxx = maxx - outimage.shape[1]
					maxx = outimage.shape[1]

				diff_maxy = 0
				if maxy > outimage.shape[0]:
					diff_maxy = maxy - outimage.shape[0]
					maxy = outimage.shape[0]

				# crop output tile to the correct size
				output_tile = output_tile[
					:, diff_miny : output_tile.shape[1] - diff_maxy, diff_minx : output_tile.shape[2] - diff_maxx
				]

				# save tile to output array
				outimage[miny:maxy, minx:maxx] = output_tile[0].numpy()

		print('Postprocessing mask into polygons and filtering....')

		# threshold the output image
		outimage = (outimage > self.config['probabilty_threshold']).astype(np.uint8)

		# get nodata mask
		nodata_mask = dataset.image_src.read_masks()[0] == dataset.image_src.nodata

		# mask out nodata in predictions
		outimage[~nodata_mask] = 0

		# get polygons from mask
		polygons = mask_to_polygons(outimage, dataset.image_src)

		# close the vrt
		dataset.image_src.close()

		polygons = filter_polygons_by_area(polygons, self.config['minimum_polygon_area'])

		# reproject the polygons back into the crs of the input tif
		polygons = reproject_polygons(polygons, dataset.image_src.crs, rasterio.open(input_tif).crs)

		print('done')

		return polygons
