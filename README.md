# ClimaXpert
# 🌦️ Weather Prediction System using Machine Learning

This project is an end-to-end weather forecasting system built with Python and machine learning. It predicts **maximum temperature** and **rainfall** using historical weather data collected from the **Visual Crossing Weather API**.

## 🚀 Features

- 📊 **Data Collection**  
  - Fetches up to 1000 days of historical and 15 days of future weather data using the Visual Crossing API.
  - Location-based data collection using **latitude and longitude**.

- 🧹 **Data Preprocessing & Feature Engineering**  
  - Cleaned and structured data with features like:
    - `Date`, `Latitude`, `Longitude`
    - `Max Temperature`, `Rainfall`
    - `Wind Direction`, `Sunrise`, `Sunset`
  - Scaled and transformed features for model readiness.

- 🤖 **Model Training**  
  - Separate models for:
    - **Temperature Prediction**
    - **Rainfall Prediction**
  - Algorithms used:
    - Random Forest Regressor
    - LightGBM Regressor
  - Hyperparameter tuning with **GridSearchCV** for optimal performance.

- 📈 **Prediction & Visualization**  
  - Predicts weather for **new/unseen data**
  - Includes trend plots for:
    - Maximum Temperature
    - Rainfall Levels

- 🌐 **Planned Web App (Coming Soon)**  
  - Built using **Streamlit**
  - Features:
    - Interactive location input
    - Real-time forecast updates
    - Visualization dashboard

## 🛠️ Tech Stack

- **Programming:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib, Seaborn
- **API:** Visual Crossing Weather API
- **Frontend (Planned):** Streamlit

