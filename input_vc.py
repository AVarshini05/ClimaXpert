import pandas as pd
import numpy as np
import joblib

# ==========  Load Models ==========
temp_model = joblib.load('temperature_model.pkl')
rain_clf = joblib.load('rain_classifier_model.pkl')
rain_reg = joblib.load('rainfall_regressor_model.pkl')

# ==========  Load Full Raw Future Data ==========
df_raw = pd.read_csv('weather_future_data.csv')
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw['Day'] = df_raw['Date'].dt.strftime('%A')

# ==========  Load Processed Features and Predict ==========
X_future = pd.read_csv('future_data.csv')
temp_preds = temp_model.predict(X_future)

rain_labels = rain_clf.predict(X_future)
rain_preds = np.zeros(X_future.shape[0])
X_rain = X_future[rain_labels == 1]
if not X_rain.empty:
    rain_preds_log = rain_reg.predict(X_rain)
    rain_preds[rain_labels == 1] = np.expm1(rain_preds_log)
else:
    print("No rain predicted for any of the future dates.")


# ========== 🧾 Combine All Data ==========

# Add predictions to raw data
df_raw['Predicted Temperature (°C)'] = temp_preds
df_raw['Predicted Rainfall (mm)'] = rain_preds
df_raw['Source'] = 'Future'

# Select only required columns
columns_order = [
    'Date', 'Day',
    'Predicted Temperature (°C)',
    'Humidity (%)',
    'Predicted Rainfall (mm)',
    'Pressure (hPa)',
    'Wind Speed (km/h)'
]


future_df = df_raw[columns_order].copy()

# ==========  Show First 5 Days ==========
print("\n Weather Forecast: Next 5 Days (With Parameters)\n")
print(future_df.head(5).to_string(index=False))

# ==========  Save ONLY FUTURE predictions ==========
future_df.to_csv('futureData_prediction.csv', index=False)
print("\n Saved: 'futureData_prediction.csv'")

# ========== 🔍 Date-specific Query ==========
date_query = input("\n Enter a date (YYYY-MM-DD) to view full forecast (or press Enter to skip): ").strip()

if date_query:
    try:
        query_date = pd.to_datetime(date_query).date()
        result = future_df[future_df['Date'].dt.date == query_date]
        if not result.empty:
            print(f"\n Forecast for {query_date}:\n")
            print(result.to_string(index=False))
        else:
            print(f" No forecast found for {query_date}.")
    except Exception as e:
        print(f"Invalid format. Please use YYYY-MM-DD. Error: {e}")
