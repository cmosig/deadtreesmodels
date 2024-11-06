import requests
from supabase import create_client
from settings import API_ENDPOINT, SUPABASE_USER, SUPABASE_USER_PASSWORD, SUPABASE_URL, SUPABASE_KEY
import json
import geopandas as gpd

def upload_labels(labels_path, aoi_path, dataset_id, label_type, label_source, label_quality):

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    token = supabase.auth.sign_in_with_password(SUPABASE_USER, SUPABASE_USER_PASSWORD)
    
    # Read files with GeoPandas and extract only the geometry
    if aoi_path:
        aoi_gdf = gpd.read_file(aoi_path)
        aoi_geojson = json.loads(aoi_gdf.geometry.to_json())
    else:
        aoi_geojson = None
    
    labels_gdf = gpd.read_file(labels_path)
    labels_geojson = json.loads(labels_gdf.geometry.to_json())

    api_endpoint = API_ENDPOINT + dataset_id + '/labels'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
    data = {
        'aoi': aoi_geojson,
        'label': labels_geojson,
        'label_type': label_type,
        'label_source': label_source,
        'label_quality': label_quality,
    }

    response = requests.post(api_endpoint, headers=headers, json=data)
    print(response.json())
