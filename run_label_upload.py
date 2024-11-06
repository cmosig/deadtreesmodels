from upload_labels import upload_labels

aoi_path = "/mnt/gsdata/projects/deadtrees/deadwood_segmentation/test/TDOP_2014_123_jpg85_btif_mosaic.geojson"
labels_path = "/mnt/gsdata/projects/deadtrees/deadwood_segmentation/test/TDOP_2014_123_jpg85_btif_mosaic_deadwood_pred.geojson"
dataset_id = "123"
label_type = "segmentation"
label_source = "model_prediction"
label_quality = 3

upload_labels(labels_path, aoi_path, dataset_id, label_type, label_source, label_quality)