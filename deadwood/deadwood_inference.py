import json
from os.path import join
from pathlib import Path

import numpy as np
import safetensors.torch
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
from tqdm import tqdm

from common import *

from .InferenceDataset import InferenceDataset


class DeadwoodInference:
    def __init__(self, config_path):
        # set float32 matmul precision for higher performance
        torch.set_float32_matmul_precision("high")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()

    def load_model(self):
        if "segformer_b5" in self.config["model_name"]:
            model = smp.Unet(
                encoder_name="mit_b5",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            ).to(memory_format=torch.channels_last)

            model = torch.compile(model)
            safetensors.torch.load_model(
                model,
                join(
                    str(Path(__file__).parent.parent),
                    "data",
                    self.config["model_name"] + ".safetensors",
                ),
            )
            # model = nn.DataParallel(model)
            model = model.to(memory_format=torch.channels_last, device=self.device)

            model.eval()

            self.model = model

        else:
            print("Invalid model name: ", self.config["model_name"], "Exiting...")
            exit()

    def inference_deadwood(self, input_tif):
        """
        gets path to tif file and returns polygons of deadwood in the CRS of the tif
        """

        # will always return a vrt, even when not reprojecting
        vrt_src = image_reprojector(
            input_tif, min_res=self.config["deadwood_minimum_inference_resolution"]
        )

        dataset = InferenceDataset(image_src=vrt_src, tile_size=1024, padding=256)

        loader_args = {
            "batch_size": self.config["batch_size"],
            "num_workers": self.config["num_dataloader_workers"],
            "pin_memory": True,
            "shuffle": False,
        }
        inference_loader = DataLoader(dataset, **loader_args)

        outimage = np.zeros((dataset.height, dataset.width), dtype=bool)
        for nodata_mask, images, cropped_windows in tqdm(
            inference_loader, desc="inference"
        ):
            images = images.to(device=self.device, memory_format=torch.channels_last)

            output = None
            with torch.no_grad():
                # if the the batch size is smaller than the configured batch size, apply padding
                if images.shape[0] < self.config["batch_size"]:
                    pad = torch.zeros(
                        (self.config["batch_size"], 3, 1024, 1024), dtype=torch.float32
                    )
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
                nodata_mask_tile = nodata_mask[i]

                # crop tensor by dataset padding
                output_tile = crop(
                    output_tile,
                    top=dataset.padding,
                    left=dataset.padding,
                    height=dataset.tile_size - (2 * dataset.padding),
                    width=dataset.tile_size - (2 * dataset.padding),
                )

                nodata_mask_tile = crop(
                    nodata_mask_tile,
                    top=dataset.padding,
                    left=dataset.padding,
                    height=dataset.tile_size - (2 * dataset.padding),
                    width=dataset.tile_size - (2 * dataset.padding),
                )

                # derive min/max from cropped window
                minx = cropped_windows["col_off"][i]
                maxx = minx + cropped_windows["width"][i]
                miny = cropped_windows["row_off"][i]
                maxy = miny + cropped_windows["width"][i]

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
                    :,
                    diff_miny : output_tile.shape[1] - diff_maxy,
                    diff_minx : output_tile.shape[2] - diff_maxx,
                ]

                # crop nodata mask to correct size
                nodata_mask_tile = nodata_mask_tile[
                    diff_miny : nodata_mask_tile.shape[0] - diff_maxy,
                    diff_minx : nodata_mask_tile.shape[1] - diff_maxx,
                ]

                output_tile = output_tile[0].numpy()

                # threshold the output image
                output_tile = (
                    output_tile > self.config["probabilty_threshold"]
                ).astype(bool)

                # apply nodata mask
                output_tile[~nodata_mask_tile] = 0

                # save tile to output array
                outimage[miny:maxy, minx:maxx] = output_tile

        print("Postprocessing mask into polygons and filtering....")
        if outimage.shape[0] * outimage.shape[1] > self.config["mask_tiling_threshold"]:
            # derive tile size based on threshold
            tile_size = int(np.sqrt(self.config["mask_tiling_threshold"]))

            print(
                "Mask size exceeds threshold, processing in tiles of size ", tile_size
            )

            pbar_tiles = tqdm(
                total=(
                    (outimage.shape[0] // tile_size + 1)
                    * (outimage.shape[1] // tile_size + 1)
                ),
                desc="vectorizing mask",
            )
            for y in range(0, outimage.shape[0], tile_size):
                for x in range(0, outimage.shape[1], tile_size):
                    outimage_tile = outimage[
                        y : min(y + tile_size, outimage.shape[0]),
                        x : min(x + tile_size, outimage.shape[1]),
                    ]

                    # get polygons from mask tile
                    tile_polygons = mask_to_polygons(
                        outimage_tile, dataset.image_src, offset_x=x, offset_y=y
                    )

                    if x == 0 and y == 0:
                        polygons = tile_polygons
                    else:
                        polygons.extend(tile_polygons)

                    pbar_tiles.update(1)

        else:
            # get polygons from mask
            polygons = mask_to_polygons(
                outimage,
                dataset.image_src,
            )

        # close the vrt
        vrt_src.close()

        polygons = filter_polygons_by_area(
            polygons, self.config["minimum_polygon_area"]
        )

        # reproject the polygons back into the crs of the input tif
        polygons = reproject_polygons(
            polygons, dataset.image_src.crs, rasterio.open(input_tif).crs
        )

        return polygons
