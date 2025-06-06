import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
X = pd.read_csv('processed_features.csv')
y_rain = pd.read_csv('target_rainfall.csv').squeeze()
y_temp = pd.read_csv('target_temperature.csv').squeeze() 

# Split for temperature prediction
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)

# Temperature model
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_temp.fit(X_train_temp, y_train_temp)

y_pred_temp = rf_temp.predict(X_test_temp)

# Evaluation
mse_temp = mean_squared_error(y_test_temp, y_pred_temp)
r2_temp = r2_score(y_test_temp, y_pred_temp)
print(f'🌡️ Temperature - MSE: {mse_temp:.2f}, R²: {r2_temp:.2f}')

# Step 1: Binary Classification (Rain or No Rain)
y_binary = (y_rain > 0).astype(int)

# Train-test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_rain, test_size=0.2, random_state=42)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Classifier: Predict whether it will rain
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train_bin, y_train_bin)

# Predict rain probabilities and labels
rain_probs = clf.predict_proba(X_test_bin)[:, 1]  # Probability of rain
rain_probs = clf.predict_proba(X_test_bin)[:, 1]
rain_labels = (rain_probs > 0.6).astype(int)  # Increased threshold

# Step 2: Regressor for rainfall amount (only if rain is predicted)
# We only use the samples where rain is expected (rain_labels == 1)
rain_indices = y_train_r > 0  # Actual rain-based filter
X_train_reg = X_train_r[rain_indices]
y_train_reg = y_train_r[rain_indices]
X_test_reg = X_test_r[rain_labels == 1]

# Train regressor on the rainfall data (using LGBMRegressor)
rain_model = LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, min_child_samples=10, min_split_gain=0.1, random_state=42)
rain_model.fit(X_train_reg, np.log1p(y_train_reg))

# Predict on test data where rain is expected
y_pred_log_rain = rain_model.predict(X_test_reg)
y_pred_rain = np.expm1(y_pred_log_rain)

# Combine with zeros where no rain is predicted
final_rain_preds = np.zeros_like(y_test_r)
final_rain_preds[rain_labels == 1] = y_pred_rain

# Evaluate performance
mse_rain = mean_squared_error(y_test_r, final_rain_preds)
r2_rain = r2_score(y_test_r, final_rain_preds)

print(f" Rainfall R² Score (Hybrid Model): {r2_rain:.4f}")
print(f" Rainfall MSE (Hybrid Model): {mse_rain:.2f}")

# Temperature
plt.figure(figsize=(10, 5))
plt.plot(y_test_temp.values, label="Actual", color="orange")
plt.plot(y_pred_temp, label="Predicted", color="green")
plt.title("Temperature Prediction: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()

#  Rainfall Distribution
plt.figure(figsize=(8, 5))
sns.kdeplot(y_test_r, label='Actual', fill=True, color='blue')
sns.kdeplot(final_rain_preds, label='Predicted', fill=True, color='purple')
plt.title(" Rainfall Distribution: Actual vs Predicted")
plt.xlabel("Rainfall (mm)")
plt.legend()
plt.tight_layout()
plt.show()

# Precipitation
plt.figure(figsize=(10, 5))
plt.plot(y_test_r.values, label="Actual", color="blue")
plt.plot(final_rain_preds, label="Predicted", color="purple")
plt.title("Precipitation Prediction: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.tight_layout()
plt.show()
# Save the classifier, regressor, and scaler
joblib.dump(rf_temp, 'temperature_model.pkl')
print(" Temperature model saved")
joblib.dump(clf, 'rain_classifier_model.pkl')
joblib.dump(rain_model, 'rainfall_regressor_model.pkl')
print(" Rainfall classifier and regressor models saved!")

# ==========  Export Predictions ==========

results_df = pd.DataFrame({
    'Actual Max Temperature (°C)': y_test_temp.values,
    'Predicted Max Temperature (°C)': y_pred_temp,
    'Actual Rainfall (mm)': y_test_r.values,
    'Predicted Rainfall (mm)': final_rain_preds
})
results_df.to_csv('weather_predictions_testing.csv', index=False)
print(" Predictions saved to 'weather_predictions_testing.csv'")
