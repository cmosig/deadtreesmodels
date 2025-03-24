import tempfile
import os
import numpy as np
import rasterio
from tcd_pipeline.pipeline import Pipeline
from tcd_pipeline.util import convert_to_projected
from shapely.geometry import Polygon, MultiPolygon

# Import from common module
from ..common import mask_to_polygons

# Define threshold for tree cover detection
TCD_THRESHOLD = 200

def inference_forestcover(input_tif: str):
    """
    Run tree cover detection on an orthophoto and return the polygons.
    
    Args:
        input_tif (str): Path to the input GeoTIFF file
        
    Returns:
        list: List of polygons representing tree cover
    """
    with tempfile.TemporaryDirectory() as tempdir:
        # Reproject tif to 10cm resolution in a projected CRS
        temp_reproject_path = os.path.join(tempdir, os.path.basename(input_tif))
        
        # Use the TCD utility to convert and resample the image
        convert_to_projected(
            input_tif, 
            temp_reproject_path,
            resample=True,
            target_gsd_m=0.1,  # 10cm resolution
            dst_crs="EPSG:3395"  # Mercator projection
        )

        # Initialize the TCD pipeline with the segformer model
        pipeline = Pipeline(model_or_config="restor/tcd-segformer-mit-b5")

        # Run prediction
        result = pipeline.predict(temp_reproject_path)

        # The confidence map is a numpy array in the result object
        # Check the type of confidence_map to handle it properly
        print(f"Type of confidence_map: {type(result.confidence_map)}")
        
        # If it's a DatasetReader, we need to read the data
        if hasattr(result.confidence_map, 'read'):
            # Read the first band as a numpy array
            confidence_map = result.confidence_map.read(1)
        elif isinstance(result.confidence_map, np.ndarray):
            # It's already a numpy array
            confidence_map = result.confidence_map
        else:
            # Try to convert to numpy array
            try:
                confidence_map = np.array(result.confidence_map)
            except Exception as e:
                raise TypeError(f"Cannot convert confidence_map to numpy array: {e}")
        
        
        # Threshold the output image to get binary mask
        outimage = (confidence_map > TCD_THRESHOLD).astype(np.uint8)
        
        # Open the dataset to get the transform for mask_to_polygons
        with rasterio.open(temp_reproject_path) as dataset:
            polygons = mask_to_polygons(outimage, dataset)

        return polygons


