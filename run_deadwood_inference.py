import fire
from supabase_service import upload_to_supabase
from doitall import inference_deadwood, transform_mask, extract_bbox
import logging
import torch
import gc
import sys
import rasterio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# data_path = "/mnt/gsdata/projects/deadtrees/deadwood_segmentation/test/"
# file_name = "uavforsat_2017_CFB030_ortho.tif"

file_path = "/mnt/gsdata/projects/deadtrees/deadwood_segmentation/test/uavforsat_2017_CFB030_ortho.tif"


def run_deadwood_inference(dataset_id, file_path):
    # try:
    logging.info("Running deadwood inference")
    polygons = inference_deadwood(file_path)

    logging.info("Transforming polygons")
    transformed_polygons = transform_mask(polygons, file_path)

    logging.info("Extracting bbox")
    bbox_geojson = extract_bbox(file_path)

    # logging.info("Uploading to supabase")
    res = upload_to_supabase(
        dataset_id,
        transformed_polygons,
        bbox_geojson,
        "segmentation",
        "model_prediction",
        3,
    )
    print(res)
    logging.info("Inference deadwood Done")
    # sys.exit()


# finally:
#     # Clean up GPU resources
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         gc.collect()

if __name__ == "__main__":
    try:
        fire.Fire(run_deadwood_inference)
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        raise
    finally:
        # Ensure all file handles are closed
        for obj in gc.get_objects():
            if isinstance(obj, rasterio.io.DatasetReader):
                obj.close()


# path = data_path + file_name.replace(".tif", "_deadwood_pred_4.shp")
# save_poly(path, polygons, "epsg:4326")
# # Write the energy map to a GeoTIFF file
# energy_map_path = data_path + file_name.replace(".tif", "_deadwood_energy_map_4.tif")

# # Set up the metadata for the new file
# kwargs = {
#     'driver': 'GTiff',
#     'dtype': 'float32',
#     'count': 1,
#     'width': energy_map.shape[1],
#     'height': energy_map.shape[0],
#     'crs': 'epsg:4326',  # Assuming same CRS as input
#     'compress': 'lzw'    # Optional compression
# }

# # Write the file
# with rasterio.open(energy_map_path, "w", **kwargs) as dst:
#     dst.write(energy_map.astype(np.float32), 1)
