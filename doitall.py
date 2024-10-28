import rasterio
from torch.utils.data import DataLoader
from InferenceDataset import InferenceDataset

import utm
from safetensors.torch import load_model
from shapely.affinity import affine_transform
from unet.unet_model import UNet
from torchvision.transforms.functional import crop

TCD_RESOLUTION = 0.1  # m -> tree crown detection only works as 10cm
TCD_THRESHOLD = 200

DEADWOOD_THRESHOLD = 0.9

DEADWOOD_MODEL_PATH = "deadwood_model.pth"
TCD_MODEL_PATH = "tcd_model.pth"


def reproject_to_10cm(input_tif, output_tif):
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
            resolution=TCD_RESOLUTION)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i),
                          destination=rasterio.band(dst, i),
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=transform,
                          dst_crs=dst_crs,
                          resampling=Resampling.nearest)


def mask_to_polygons(mask, dataset_reader):
    """
    this function takes a numpy mask as input and returns a list of polygons
    that are in the crs of the passed dataset reader
    """

    contours, _ = cv2.findContours(mask.astype(np.uint8).copy(),
                                   mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    poly = []
    for p in contours:
        if len(p) == 1:
            continue

        p = p.squeeze()
        p = np.concatenate([p, p[:1]], axis=0)
        p = Polygon(p)
        poly.append(p)

    # affine transform from pixel to world coordinates
    transform = dataset_reader.transform
    transform_matrix = (transform.a, transform.b, transform.d, transform.e,
                        transform.c, transform.f)
    poly = [affine_transform(p, transform_matrix) for p in poly]

    return poly


def get_utm_string_from_latlon(lat, lon):
    zone = utm.from_latlon(lat, lon)
    utm_code = 32600 + zone[2]
    if lat < 0:
        utm_code -= 100
    return f"EPSG:{utm_code}"


def inference_deadwood(input_tif: str) -> List[Polygon]:
    """
    gets path to tif file and returns polygons of deadwood in the CRS of the tif
    """

    dataset = InferenceDataset(image_path=input_tif,
                               tile_size=1024,
                               padding=256)

    loader_args = {
        "batch_size": 1,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
    }
    inference_loader = DataLoader(dataset, **loader_args)

    # preferably use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    outimage = (outimage > DEADWOOD_THRESHOLD).astype(np.uint8)

    # get polygons from mask
    polygons = mask_to_polygons(outimage, dataset.image_src)

    return polygons


def inference_forestcover():
    pass
