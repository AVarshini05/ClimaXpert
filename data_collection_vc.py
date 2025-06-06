import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# Take latitude and longitude from user
latitude = input("Enter latitude: ").strip()
longitude = input("Enter longitude: ").strip()
location = f"{latitude},{longitude}"

# Calculate date range: past 365 days from today
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365)

# Format dates as strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# API details
api_key = 'TFYPCBKAN62TDLMYJPS95S3AM'  # Replace with your actual API key
unit_group = 'metric'
content_type = 'json'

# Construct the API URL for past data
past_url = (
    f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    f'{location}/{start_date_str}/{end_date_str}?unitGroup={unit_group}&include=days&key={api_key}&contentType={content_type}'
)

# Construct the API URL for future data (next 10 days)
future_url = (
    f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    f'{location}?unitGroup={unit_group}&include=days&key={api_key}&contentType={content_type}'
)

# Make the API request for past data
past_response = requests.get(past_url)

# Make the API request for future data
future_response = requests.get(future_url)

# Check if the past data request was successful
if past_response.status_code == 200:
    past_data = past_response.json()
    days_past = past_data.get('days', [])
    elevation = past_data.get('elevation', None)

    # Collect past data
    cleaned_past_data = []
    for day in days_past:
        precip_type = day.get('preciptype')
        if isinstance(precip_type, list):
            precip_type = ', '.join(precip_type)
        elif precip_type is None:
            precip_type = 'None'

        cleaned_past_data.append({
            'Date': day.get('datetime'),
            'Temperature (°C)': day.get('temp'),
            'Humidity (%)': day.get('humidity'),
            'Precipitation (mm)': day.get('precip'),
            'Precipitation Type': precip_type,
            'Pressure (hPa)': day.get('pressure'),
            'Sunrise Time': day.get('sunrise'),
            'Sunset Time': day.get('sunset'),
            'Wind Speed (km/h)': day.get('windspeed'),
            'Wind Direction (°)': day.get('winddir'),
            'Dew Point (°C)': day.get('dew'),
            'Feels Like (°C)': day.get('feelslike'),
            'Cloud Cover (%)': day.get('cloudcover'),
            'UV Index': day.get('uvindex'),
            'Visibility (km)': day.get('visibility'),
            'Solar Energy (kWh/m²)': day.get('solarenergy'),
            'Solar Radiation (W/m²)': day.get('solarradiation'),
            'Wind Gust (km/h)': day.get('windgust'),
        })

    # Create a DataFrame for past data
    weather_past_df = pd.DataFrame(cleaned_past_data)

    # Save past data to CSV
    weather_past_df.to_csv('weather_past_data.csv', index=False)
    print('Past data successfully saved to weather_past_data.csv')
else:
    print(f' Error fetching past data: {past_response.status_code} - {past_response.text}')

# Check if the future data request was successful
if future_response.status_code == 200:
    future_data = future_response.json()
    days_future = future_data.get('days', [])

    # Collect future data
    cleaned_future_data = []
    for day in days_future:
        precip_type = day.get('preciptype')
        if isinstance(precip_type, list):
            precip_type = ', '.join(precip_type)
        elif precip_type is None:
            precip_type = 'None'

        cleaned_future_data.append({
            'Date': day.get('datetime'),
            'Humidity (%)': day.get('humidity'),
            'Pressure (hPa)': day.get('pressure'),
            'Sunrise Time': day.get('sunrise'),
            'Sunset Time': day.get('sunset'),
            'Wind Speed (km/h)': day.get('windspeed'),
            'Wind Direction (°)': day.get('winddir'),
            'Dew Point (°C)': day.get('dew'),
            'Feels Like (°C)': day.get('feelslike'),
            'Cloud Cover (%)': day.get('cloudcover'),
            'UV Index': day.get('uvindex'),
            'Visibility (km)': day.get('visibility'),
            'Solar Energy (kWh/m²)': day.get('solarenergy'),
            'Solar Radiation (W/m²)': day.get('solarradiation'),
            'Wind Gust (km/h)': day.get('windgust'),
        })

    # Create a DataFrame for future data
    weather_future_df = pd.DataFrame(cleaned_future_data)

    # Save future data to CSV
    weather_future_df.to_csv('weather_future_data.csv', index=False)
    print('Future data successfully saved to weather_future_data.csv')
else:
    print(f'Error fetching future data: {future_response.status_code} - {future_response.text}')
