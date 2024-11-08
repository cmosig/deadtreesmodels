
import requests
from supabase import create_client
from settings import API_ENDPOINT, SUPABASE_USER, SUPABASE_USER_PASSWORD, SUPABASE_URL, SUPABASE_KEY

def upload_to_supabase(dataset_id, label, aoi, label_type, label_source, label_quality):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    token = supabase.auth.sign_in_with_password({'email': SUPABASE_USER, 'password': SUPABASE_USER_PASSWORD})

    api_endpoint = API_ENDPOINT + str(dataset_id) + '/labels'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token.session.access_token}'}
    data = {
        'dataset_id': dataset_id,
        'label': label,
        'aoi': aoi,
        'label_type': label_type,
        'label_source': label_source,
        'label_quality': label_quality
        }
    response = requests.post(api_endpoint, headers=headers, json=data)
    return response

