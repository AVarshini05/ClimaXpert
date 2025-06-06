import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('weather_past_data.csv')

# Drop rows with missing or invalid pressure
df = df[df['Pressure (hPa)'] != -9999]

#  Convert Sunrise and Sunset to datetime
df['Sunrise Time'] = pd.to_datetime(df['Date'] + ' ' + df['Sunrise Time'])
df['Sunset Time'] = pd.to_datetime(df['Date'] + ' ' + df['Sunset Time'])

#  Calculate daylight duration in minutes
df['Daylight Duration (min)'] = (df['Sunset Time'] - df['Sunrise Time']).dt.total_seconds() / 60.0

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract seasonality features from date
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday = 0, Sunday = 6

# Convert wind direction to x and y components
df['Wind_X'] = np.cos(np.radians(df['Wind Direction (°)']))
df['Wind_Y'] = np.sin(np.radians(df['Wind Direction (°)']))

# Drop raw sunrise, sunset, winddir columns (we've converted them)
df.drop(['Sunrise Time', 'Sunset Time', 'Wind Direction (°)'], axis=1, inplace=True)

#Target columns
y_temp = df['Temperature (°C)']
y_rain = df['Precipitation (mm)']

# Drop non-feature columns
X = df.drop(['Date', 'Temperature (°C)', 'Precipitation (mm)', 'Precipitation Type'], axis=1)

# Generate humidity levels
humidity_level = pd.cut(df['Humidity (%)'], bins=[0, 30, 60, 100], labels=['Low', 'Medium', 'High'])
humidity_dummies = pd.get_dummies(humidity_level, prefix='Humidity', drop_first=True)

# Append dummies to features
X = pd.concat([X, humidity_dummies], axis=1)

# Save processed data
X.to_csv('processed_features.csv', index=False)
y_temp.to_csv('target_temperature.csv', index=False)
y_rain.to_csv('target_rainfall.csv', index=False)

print("Preprocessing complete. Files saved:")
print("- processed_features.csv")
print("- target_temperature.csv")
print("- target_rainfall.csv")

# Load the future dataset
df_future = pd.read_csv('weather_future_data.csv')

# Drop rows with missing or invalid pressure
df_future = df_future[df_future['Pressure (hPa)'] != -9999]

# Convert Sunrise and Sunset to datetime
df_future['Sunrise Time'] = pd.to_datetime(df_future['Date'] + ' ' + df_future['Sunrise Time'])
df_future['Sunset Time'] = pd.to_datetime(df_future['Date'] + ' ' + df_future['Sunset Time'])

# Calculate daylight duration in minutes
df_future['Daylight Duration (min)'] = (df_future['Sunset Time'] - df_future['Sunrise Time']).dt.total_seconds() / 60.0

# Convert 'Date' to datetime format
df_future['Date'] = pd.to_datetime(df_future['Date'])
df_future['Month'] = df_future['Date'].dt.month
df_future['DayOfWeek'] = df_future['Date'].dt.dayofweek

# Convert wind direction to vector components
df_future['Wind_X'] = np.cos(np.radians(df_future['Wind Direction (°)']))
df_future['Wind_Y'] = np.sin(np.radians(df_future['Wind Direction (°)']))

# Drop raw columns
df_future.drop(['Sunrise Time', 'Sunset Time', 'Wind Direction (°)'], axis=1, inplace=True)

# Humidity level bins and dummies
humidity_level_future = pd.cut(df_future['Humidity (%)'], bins=[0, 30, 60, 100], labels=['Low', 'Medium', 'High'])
humidity_dummies_future = pd.get_dummies(humidity_level_future, prefix='Humidity', drop_first=True)

# Drop non-feature columns and combine
X_future = df_future.drop(['Date', 'Precipitation (mm)', 'Temperature (°C)', 'Precipitation Type'], axis=1, errors='ignore')
X_future = pd.concat([X_future, humidity_dummies_future], axis=1)

# Save
X_future.to_csv('future_data.csv', index=False)
print(" Future data preprocessing complete. File saved: future_data.csv")
