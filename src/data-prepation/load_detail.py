from googleapiclient.discovery import build
from google.cloud import api_keys_v2

# ตั้งค่า API Key
api_key = "AIzaSyAZIPvz-QNBQI2YxGAvWu-qfi0OLGUO6DI"
service = build('places', 'v1', credentials=None, discoveryServiceUrl=f"https://places.googleapis.com/v1/places:searchNearby?key={api_key}")

# ค้นหาสถานที่ในเชียงใหม่
request_body = {
    'location': {'latitude': 18.7883, 'longitude': 98.9853},  # พิกัดเชียงใหม่
    'radius': 5000,  # รัศมี 5 กม.
    'type': 'restaurant'
}
response = service.places().searchNearby(body=request_body).execute()

# บันทึกข้อมูล
restaurants = []
for place in response.get('places', []):
    restaurants.append({
        'name': place.get('displayName', {}).get('text', ''),
        'rating': place.get('rating', 0),
        'address': place.get('formattedAddress', '')
    })

# บันทึกเป็น CSV
import pandas as pd
df = pd.DataFrame(restaurants)
df.to_csv('chiangmai_places.csv', index=False)
print("Data_fetched_and_saved_to_chiangmai_places.csv")