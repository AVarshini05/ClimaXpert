import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import os
import random
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="ClimaXpert - ML Weather Forecasting", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            scroll-behavior: smooth;
        }

        .navbar {
            position: fixed;
            top: 60px;  /* Adjust this based on title height */
            width: 90%;
            background-color: #003366;
            padding: 10px 10px;
            display: flex;
            justify-content: center;
            gap: 40px;
            z-index: 999;
            border-bottom: 3px solid #0055A5;
        }

        .navbar a {
            color: #ffffff;
            font-size: 18px;
            text-decoration: none;
            font-weight: bold;
            padding: 6px 12px;
        }

        .navbar a:hover {
            background-color: #0055A5;
            border-radius: 5px;
        }

        .title {
            text-align: center;
            font-size: 36px;
            color: #003366;
            margin-top: 60px;
            margin-bottom: 10px;
        }
        .section {
            padding: 60px 20px 60px 20px; 
        }
        .compact-section {
            padding: 30px 20px 60px 20px; 
        }
        .section h2 {
            color: #0055A5;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
    <div class="navbar">
        <a href="#home">Home</a>
        <a href="#forecast">Forecast</a>
        <a href="#about">About</a>
        <a href="#contact">Contact</a>
    </div>
""", unsafe_allow_html=True)

# Logo and Title Section
st.markdown('<div class="compact-section" id="home">', unsafe_allow_html=True)
top_col1, top_col2,top_col3 = st.columns([1,10,1])  # Adjust ratio as needed
with top_col1:
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.image("logo.png", width=90)
with top_col2:
    st.markdown('<div class="title">ClimaXpert— Smart Weather Forecasting by ML</div>', unsafe_allow_html=True)
with top_col3:
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.image("vignan_logo.png", width=90)


# Random weather facts
weather_quotes = [
    "🌧️ Did you know? Raindrops can fall at speeds of about 22 miles per hour!",
    "❄️ The largest snowflake ever recorded was 15 inches wide — bigger than a pizza!",
    "🌪️ Tornadoes can spin faster than a Formula 1 car — over 300 mph!",
    "☀️ The Sun is 400 times farther away from us than the Moon — but looks the same size!",
    "🌈 You can never get to the end of a rainbow — it's actually a full circle!",
    "🌩️ Lightning is hotter than the surface of the sun — about 30,000°C!",
    "🌦️ Weather forecasters have a job where you can be wrong half the time and still get paid!",
    "💨 Wind is just air that got pushed around — basically, weather's version of mood swings!",
    "🌫️ Fog is basically a cloud that decided to chill on the ground — lazy weather!",
    "🌤️ The sky isn’t really blue — it just scatters that way!",
    "🌬️ A hurricane can release more energy in a day than all the world’s power plants combined!",
    "🌨️ Snowflakes always have six sides — Mother Nature’s obsession with symmetry!",
    "☔ Ever smell rain? That fresh scent is called petrichor, and it’s basically Earth’s perfume!"
]

# Display scrolling banner + fact box
st.markdown(f"""
<style>
.scroll-line {{
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    box-sizing: border-box;
    background: #D3D3D3;
    padding: 10px 0;
    font-size: 18px;
    color: #003366;
    font-weight: bold;
}}
.scroll-line span {{
    display: inline-block;
    padding-left: 100%;
    animation: scroll-text 25s linear infinite;
}}
@keyframes scroll-text {{
    0% {{ transform: translateX(0%); }}
    100% {{ transform: translateX(-100%); }}
}}
</style>

<div class="scroll-line">
   <span>🌤️ Welcome to <strong>ClimaXpert</strong> — Get your 15-day weather forecast with AI! Enter coordinates and see trends!</span>
</div>

<div style='background-color:#f7faff;padding:10px;border-radius:8px;margin-top:10px;text-align:center;'>
    <strong>☁️ Fun Weather Fact:</strong> {random.choice(weather_quotes)}
</div>
""", unsafe_allow_html=True)

# --- Sections ---

# Forcast Section
st.markdown('<div class="compact-section" id="forecast">', unsafe_allow_html=True)
# ========== 🔄 Get City Name from Lat/Lon ==========
def get_city_name(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        response = requests.get(url, headers={"User-Agent": "weather-app"})
        if response.status_code == 200:
            data = response.json()
            return data.get("address", {}).get("city") or data.get("address", {}).get("town") or data.get("address", {}).get("village") or "Unknown location"
        else:
            return "Location not found"
    except:
        return "Error fetching location"



# ========== 📍 Map and 📈 Trend Analysis ==========
try:
    latitude = "17.6868"
    longitude = "83.2185"
    
    col_trend, col_map = st.columns(2)

    with col_trend:
        st.subheader("📈 Weather Trends")  # Title inside the column
        # ========== 🌎 Location Input ==========
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.text_input("Latitude", "17.6868")
        with col2:
            longitude = st.text_input("Longitude", "83.2185")

        lat_float = float(latitude)
        lon_float = float(longitude)

        city_name = get_city_name(lat_float, lon_float)
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Location: {city_name}")

        def fetch_weather_data(lat, lon):
            api_key = 'TFYPCBKAN62TDLMYJPS95S3AM'
            location = f"{lat},{lon}"
            url = (
                f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
                f'{location}?unitGroup=metric&include=days&key={api_key}&contentType=json'
            )
            response = requests.get(url)
            if response.status_code == 200:
                weather_data = response.json()
                days = weather_data.get('days', [])
                cleaned = []
                for day in days:
                    precip_type = day.get('preciptype')
                    if isinstance(precip_type, list):
                        precip_type = ', '.join(precip_type)
                    elif precip_type is None:
                        precip_type = 'None'
                    cleaned.append({
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
                        'Temperature (°C)': day.get('temp'),
                        'Precipitation (mm)': day.get('precip'),
                        'Precipitation Type': precip_type,
                    })
                df = pd.DataFrame(cleaned)
                df.to_csv('weather_future_data.csv', index=False)
                return True
            else:
                return False

        def preprocess_future_data():
            df = pd.read_csv('weather_future_data.csv')
            df = df[df['Pressure (hPa)'] != -9999]
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Sunrise Time'] = pd.to_datetime(df['Date'].dt.date.astype(str) + ' ' + df['Sunrise Time'], errors='coerce')
            df['Sunset Time'] = pd.to_datetime(df['Date'].dt.date.astype(str) + ' ' + df['Sunset Time'], errors='coerce')
            df['Daylight Duration (min)'] = (df['Sunset Time'] - df['Sunrise Time']).dt.total_seconds() / 60.0
            df['Month'] = df['Date'].dt.month
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Wind_X'] = np.cos(np.radians(df['Wind Direction (°)']))
            df['Wind_Y'] = np.sin(np.radians(df['Wind Direction (°)']))
            df.drop(['Sunrise Time', 'Sunset Time', 'Wind Direction (°)'], axis=1, inplace=True)
            humidity_level = pd.cut(df['Humidity (%)'], bins=[0, 30, 60, 100], labels=['Low', 'Medium', 'High'])
            humidity_dummies = pd.get_dummies(humidity_level, prefix='Humidity', drop_first=True)
            X = df.drop(['Date', 'Temperature (°C)', 'Precipitation (mm)', 'Precipitation Type'], axis=1, errors='ignore')
            X = pd.concat([X, humidity_dummies], axis=1)
            X.to_csv('future_data.csv', index=False)
            df.to_csv('weather_future_data.csv', index=False)
            return df
        
        with col2:
            if st.button("🔄 Refresh Forecast"):
                with st.spinner("Fetching and preprocessing fresh weather data..."):
                    if fetch_weather_data(latitude, longitude):
                        preprocess_future_data()
                        st.success("Data updated for next 15 days!")
                    else:
                        st.error("Failed to fetch new data from API.")

        temp_model = joblib.load('temperature_model.pkl')
        rain_clf = joblib.load('rain_classifier_model.pkl')
        rain_reg = joblib.load('rainfall_regressor_model.pkl')

        df_raw = pd.read_csv('weather_future_data.csv')
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
        df_raw['Day'] = df_raw['Date'].dt.strftime('%A')
        X_future = pd.read_csv('future_data.csv')

        temp_preds = temp_model.predict(X_future)
        rain_probs = rain_clf.predict_proba(X_future)[:, 1]
        rain_labels = (rain_probs > 0.5).astype(int)
        rain_preds = np.zeros(X_future.shape[0])
        X_rain = X_future[rain_labels == 1]
        if not X_rain.empty:
            rain_preds_log = rain_reg.predict(X_rain)
            rain_preds[rain_labels == 1] = np.expm1(rain_preds_log)

        def classify_rainfall(mm):
            if mm == 0:
                return "☀️ No Rain"
            elif mm < 2.5:
                return "🌦️ Light Rain"
            elif mm < 10:
                return "🌧️ Moderate Rain"
            else:
                return "⛈️ Heavy Rain"

        df_raw['Predicted Temperature (°C)'] = temp_preds
        df_raw['Predicted Rainfall (mm)'] = rain_preds
        df_raw['Rainfall Severity'] = df_raw['Predicted Rainfall (mm)'].apply(classify_rainfall)

        future_df = df_raw[[
            'Date', 'Day', 'Predicted Temperature (°C)', 
            'Humidity (%)', 'Predicted Rainfall (mm)', 'Rainfall Severity',
            'Pressure (hPa)', 'Wind Speed (km/h)'
        ]]
        future_df.to_csv('futureData_prediction.csv', index=False)

        chart_data = future_df[['Date', 'Predicted Temperature (°C)', 'Predicted Rainfall (mm)']]
        chart_data.set_index('Date', inplace=True)
       # ========== 🔄 Animated Combined Weather Trends (Temperature & Rainfall) ==========

        # Preparing data for Plotly
        chart_data = future_df[['Date', 'Predicted Temperature (°C)', 'Predicted Rainfall (mm)']]
        chart_data.set_index('Date', inplace=True)

        # Create the figure
        fig_combined = go.Figure()

        # Add temperature trace
        fig_combined.add_trace(go.Scatter(x=future_df['Date'],
                                        y=future_df['Predicted Temperature (°C)'],
                                        mode='lines+markers',
                                        name='Temperature (°C)',
                                        line=dict(color='blue')))

        # Add rainfall trace
        fig_combined.add_trace(go.Scatter(x=future_df['Date'],
                                        y=future_df['Predicted Rainfall (mm)'],
                                        mode='lines+markers',
                                        name='Rainfall (mm)',
                                        line=dict(color='green')))

        # Add animation frames
        frames = []
        for i in range(1, len(future_df) + 1):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=future_df['Date'][:i], y=future_df['Predicted Temperature (°C)'][:i], mode='lines+markers'),
                    go.Scatter(x=future_df['Date'][:i], y=future_df['Predicted Rainfall (mm)'][:i], mode='lines+markers')
                ],
                name=str(i)
            ))

        # Update layout for the animation
        fig_combined.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Temperature (°C) and Rainfall (mm)', range=[0, max(future_df['Predicted Temperature (°C)'].max(), future_df['Predicted Rainfall (mm)'].max()) + 5]),
            updatemenus=[dict(type='buttons',
                            showactive=False,
                            buttons=[dict(label='Play',
                                            method='animate',
                                            args=[None, {'frame': {'duration': 300, 'redraw': True},
                                                        'fromcurrent': True}])])]
        )

        fig_combined.frames = frames

        # Display the animated graph in Streamlit
        st.plotly_chart(fig_combined)



    with col_map:
        st.subheader("🗺️ Location")  # Title inside the column
        st.map(pd.DataFrame({'lat': [lat_float], 'lon': [lon_float]}))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("5-Day Forecast Preview")
    st.dataframe(future_df.head(5), use_container_width=True)

    st.subheader("🔍 Query Forecast by Date")
    col1,col2 = st.columns(2)
    result=None
    with col1:
        query_date = st.date_input("Choose the Date:")
        result = future_df[future_df['Date'].dt.date == query_date]
    with col2:
        st.caption("Need Full 15-Day Forecast:")
        st.download_button(
            label="Download",
            data=future_df.to_csv(index=False).encode('utf-8'),
            file_name='15_day_forecast.csv',
            mime='text/csv'
        )
    if not result.empty:
        st.dataframe(result, use_container_width=True)
    else:
        st.warning("No data found for the selected date.")



except Exception as e:
    st.error(f"Error loading models or data: {e}")
st.markdown("---")
st.markdown('</div>', unsafe_allow_html=True)

# About Section
st.markdown('<div class="section" id="about">', unsafe_allow_html=True)
st.header("About the Project")

st.markdown("### Key Features")
features = {
    "📍": "Map view to visualize your selected location.",
    "📈": "15-day temperature & rainfall trends with interactive charts.",
    "📅": "Explore daily forecasts using the date picker.",
    "⬇️": "Download forecast data as a CSV file for offline use.",
    "🔄": "Refresh button to get the most recent data anytime."
}

for icon, description in features.items():
    st.markdown(f"- {icon} {description}")

# How To Use
st.markdown("### How to Use")
st.markdown("""
1. Enter your latitude and longitude under the Home section.  
2. Click "🔄Refresh Forecast" to fetch and display weather predictions.  
3. Explore your 📍location on the map and 📊 trend chart.  
4. View the 5-day forecast table or select a specific date from the calendar.  
5. Optionally, click Download to save your forecast.

Enjoy smarter planning, rain or shine! ☀️🌧️
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Contact Section
st.markdown('<div class="section" id="contact">', unsafe_allow_html=True)
st.header("Meet the Team")

# Styling still applies
st.markdown("""
<style>
.team-card {
    justify-content: center;
    text-align: center;
    padding: 20px;
    background-color: #ffffff;
    margin-bottom: 20px;
}
.team-card h4 {
    justify-content: center;
    margin-bottom: 4px;
    color: #003366;
    text-align: center;
}
.team-card p {
    justify-content: center;
    text-align: center;        
    margin: 2px 0;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# Top row - 3 members
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.image("41.jpg", width=120)
    st.markdown("""
    <h4>Tejasree</h4>
    <p>geddadabhanu57@gmail.com</p>
    <p>23L31A4641</p>
    </div>
    """, unsafe_allow_html=True)
    

with col2:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.image("16.jpg", width=120)
    st.markdown("""
    <h4>Jyothi</h4>
    <p>lenibodapati@gmail.com</p>
    <p>23L31A4616</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.image("5.jpg", width=120)
    st.markdown("""
    <h4>Varshini</h4>
    <p>varshinianupolu2005@gmail.com</p>
    <p>23L31A4605</p>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.image("31.jpg", width=120)
    st.markdown("""
    <h4>Lavanya</h4>
    <p>dlavanya2234@gmail.com</p>
    <p>23L31A4631</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.image("53.jpg", width=120)
    st.markdown("""
    <h4>Akshaya</h4>
    <p>sriakshaya908@gmail.com</p>
    <p>23L31A4653</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Footer Section ---
st.markdown("""
    <style>
        .footer {
            background-color: #002244;
            color: #ffffff;
            text-align: center;
            padding: 20px 10px;
            margin-top: 50px;
            font-size: 16px;
            border-top: 2px solid #0055A5;
        }
    </style>

    <div class="footer">
        © 2025 <strong>DT&I Project</strong>.Designed & developed for innovation.
    </div>
""", unsafe_allow_html=True)



