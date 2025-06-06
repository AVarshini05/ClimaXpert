import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# Take latitude and longitude from user
latitude = input("Enter latitude: ").strip()
longitude = input("Enter longitude: ").strip()
location = f"{latitude},{longitude}"


# API details
api_key = 'APCZ275HP2UHSLPLHREHGTDGH'  # Replace with your actual API key
unit_group = 'metric'
content_type = 'json'
# Construct the API URL for future data (next 10 days)
future_url = (
    f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    f'{location}?unitGroup={unit_group}&include=days&key={api_key}&contentType={content_type}'
)
# Make the API request for future data
future_response = requests.get(future_url)
# Check if the future data request was successful
if future_response.status_code == 200:
    future_data = future_response.json()
    days_future = future_data.get('days', [])

    # Collect future data
    futureData = []
    for day in days_future:
        precip_type = day.get('preciptype')
        if isinstance(precip_type, list):
            precip_type = ', '.join(precip_type)
        elif precip_type is None:
            precip_type = 'None'

        futureData.append({
            'Date': day.get('datetime'),     
            'Temperature (°C)': day.get('temp'),
            'Precipitation (mm)': day.get('precip'),                                                                  
            'Humidity (%)': day.get('humidity'),
            'Pressure (hPa)': day.get('pressure'),
            'Sunrise Time': day.get('sunrise'),
            'Sunset Time': day.get('sunset'),
            'Wind Speed (km/h)': day.get('windspeed')
        })
        # Create a DataFrame for future data
    weather_future_df = pd.DataFrame(futureData)
    
    # Save future data to CSV
    weather_future_df.to_csv('futureData_api.csv', index=False)
    print('✅ Future data successfully saved to futureData_api.csv')
else:
    print(f'❌ Error fetching future data: {future_response.status_code} - {future_response.text}')