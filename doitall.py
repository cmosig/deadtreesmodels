import os

from json import loads, dumps
import geopandas as gpd
import numpy as np
import rasterio
from tqdm import tqdm
import utm
import torch
import torch.nn as nn
import cv2
from safetensors.torch import load_model
from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon
from shapely.geometry import mapping

# from tcd.tcd_pipeline.pipeline import Pipeline
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
from shapely.geometry import MultiPolygon
from deadwood.InferenceDataset import InferenceDataset
from deadwood.unet_model import UNet

TCD_RESOLUTION = 0.1  # m -> tree crown detection only works as 10cm
TCD_THRESHOLD = 200
DEADWOOD_THRESHOLD = 0.8
DEADWOOD_MODEL_PATH = "./model/model.safetensors"

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def reproject_to_10cm(input_tif, output_tif):
    """takes an input tif file and reprojects it to 10cm resolution and writes it to output_tif"""

    with rasterio.open(input_tif) as src:
        # figure out centroid in epsg 4326
        centroid = src.lnglat()

        # dst crs is native utm zone for max precision
        dst_crs = utm.from_latlon(centroid[1], centroid[0])

        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=TCD_RESOLUTION,
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(output_tif, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def mask_to_polygons(mask, dataset_reader):
    """
    this function takes a numpy mask as input and returns a list of polygons
    that are in the crs of the passed dataset reader
    """

    padding = 10

    # add padding for cv to work properly
    mask_padded = np.pad(mask[0], padding)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8).copy(),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )
    poly = []
    for p in contours:
        if len(p) == 1:
            continue

        p = p.squeeze()
        p = np.concatenate([p, p[:1]], axis=0)
        p = Polygon(p)

        # reverse padding effect
        p = translate(p, xoff=-padding, yoff=-padding)

        poly.append(p)

    # affine transform from pixel to world coordinates
    transform = dataset_reader.transform
    transform_matrix = (
        transform.a,
        transform.b,
        transform.d,
        transform.e,
        transform.c,
        transform.f,
    )
    poly = [affine_transform(p, transform_matrix) for p in poly]

    return poly


def get_utm_string_from_latlon(lat, lon):
    zone = utm.from_latlon(lat, lon)
    utm_code = 32600 + zone[2]
    if lat < 0:
        utm_code -= 100
    return f"EPSG:{utm_code}"


def inference_deadwood(input_tif: str, return_energy_map: bool = False):
    """
    gets path to tif file and returns polygons of deadwood in the CRS of the tif
    """

    dataset = InferenceDataset(image_path=input_tif, tile_size=1024, padding=256)

    loader_args = {
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": False,
    }
    inference_loader = DataLoader(dataset, **loader_args)

    # preferably use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use only one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # model with three input channels (RGB)
    model = UNet(
        n_channels=3,
        n_classes=1,
    ).to(memory_format=torch.channels_last)

    load_model(model, DEADWOOD_MODEL_PATH)
    model = nn.DataParallel(model)
    model = model.to(memory_format=torch.channels_last, device=device)

    model.eval()

    outimage = np.zeros((dataset.height, dataset.width))
    for images, cropped_windows in tqdm(inference_loader):
        images = images.to(device=device, memory_format=torch.channels_last)
        with torch.no_grad():
            output = model(images)
            output = torch.sigmoid(output)
            output = (output > 0.3).float()

            # crop tensor by dataset padding
            output = crop(
                output,
                top=dataset.padding,
                left=dataset.padding,
                height=dataset.tile_size - (2 * dataset.padding),
                width=dataset.tile_size - (2 * dataset.padding),
            )

            # derive min/max from cropped window
            minx = cropped_windows["col_off"]
            maxx = minx + cropped_windows["width"]
            miny = cropped_windows["row_off"]
            maxy = miny + cropped_windows["width"]

            # save tile to output array
            outimage[miny:maxy, minx:maxx] = output[0][0].cpu().numpy()

    # threshold the output image
    # store the output as enegy_map.tif
    # print(f"outimage shape: {outimage.shape}")
    pred_mask = (outimage > DEADWOOD_THRESHOLD).astype(np.uint8)

    # get polygons from mask
    polygons = mask_to_polygons(pred_mask, dataset.image_src)
    if return_energy_map:
        return outimage, polygons
    else:
        return polygons


def inference_forestcover(input_tif: str):
    # reproject tif to 10cm
    temp_reproject_path = os.path.join(TEMP_DIR, input_tif.str.split("/")[-1])
    reproject_to_10cm(input_tif, temp_reproject_path)

    pipeline = Pipeline(model_or_config="restor/tcd-segformer-mit-b5")

    res = pipeline.predict(temp_reproject_path)

    dataset_reader_result = res.confidence_map

    # threshold the output image
    outimage = (res.confidence_map > TCD_THRESHOLD).astype(np.uint8)

    # convert to polygons
    polygons = mask_to_polygons(outimage, dataset_reader_result)

    # TODO need to cleanup temp file and prediction that was generated by the pipeline

    return polygons


def save_poly(filename, poly, crs):
    """
    Save polygons to a GeoJSON file.

    Args:
        filename (str): Output filename (should end in .geojson)
        poly (list): List of shapely polygons
        crs: Coordinate reference system
    """
    # Create GeoDataFrame from the polygons
    gdf = gpd.GeoDataFrame(geometry=poly, crs=crs)
    # Save to GeoJSON format
    gdf.to_file(filename)


def transform_mask(polygons, image_path):
    """
    Transform polygons to the CRS of the image
    """
    with rasterio.open(image_path) as src:
        deadwood_gdf = gpd.GeoDataFrame(
            geometry=polygons,
            crs=src.crs,
        )
        deadwood_gdf = deadwood_gdf.to_crs("EPSG:4326")
        geojson = loads(deadwood_gdf.geometry.to_json())

        labels = {
            "type": "MultiPolygon",
            "coordinates": [
                feature["geometry"]["coordinates"] for feature in geojson["features"]
            ],
        }
    return labels


def extract_bbox(image_path: str) -> dict:
    """
    Extract bounding box from raster file and return as GeoJSON MultiPolygon in EPSG:4326

    Args:
        image_path (str): Path to the raster file

    Returns:
        dict: GeoJSON MultiPolygon of the bounding box in WGS84 projection
    """
    with rasterio.open(image_path) as src:
        bounds = src.bounds

        # Create polygon from bounds
        bbox_poly = Polygon(
            [
                [bounds.left, bounds.top],  # top-left
                [bounds.right, bounds.top],  # top-right
                [bounds.right, bounds.bottom],  # bottom-right
                [bounds.left, bounds.bottom],  # bottom-left
                [bounds.left, bounds.top],  # close the polygon
            ]
        )

        # Convert to GeoDataFrame with source CRS
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs=src.crs)

        # Reproject to WGS84 if needed
        if src.crs != "EPSG:4326":
            bbox_gdf = bbox_gdf.to_crs("EPSG:4326")

        # Convert coordinates to lists instead of tuples
        coords = [list(coord) for coord in bbox_gdf.geometry.iloc[0].exterior.coords[:]]

        # Convert to required MultiPolygon GeoJSON format
        bbox_geojson = {
            "type": "MultiPolygon",
            "coordinates": [[coords]],
        }
    return bbox_geojson
