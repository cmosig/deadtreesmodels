import tempfile
import os

def inference_forestcover(input_tif: str):
        
    with tempfile.TemporaryDirectory() as tempdir:
        # reproject tif to 10cm
        temp_reproject_path = os.path.join(temp, input_tif.str.split('/')[-1])
        reproject_to_10cm(input_tif, temp_reproject_path)

        pipeline = Pipeline(model_or_config="restor/tcd-segformer-mit-b5")

        res = pipeline.predict(temp_reproject_path)

        dataset_reader_result = res.confidence_map

        # threshold the output image
        outimage = (res.confidence_map > TCD_THRESHOLD).astype(np.uint8)

        # convert to polygons
        polygons = mask_to_polygons(outimage, dataset_reader_result)

        return polygons


