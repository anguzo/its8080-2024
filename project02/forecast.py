# %% Import Libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# %% Load Cleaned Data
train_data = pd.read_csv("data/train_data_cleaned.csv")
test_data = pd.read_csv("data/test_data_cleaned.csv")

train_data["time"] = pd.to_datetime(train_data["time"])
test_data["time"] = pd.to_datetime(test_data["time"])

train_data.set_index("time", inplace=True)
test_data.set_index("time", inplace=True)

# Target Variable and Exogenous Inputs
y_train = train_data["demand"]
y_test = test_data["demand"]
exog_train = train_data[["temp"]]
exog_test = test_data[["temp"]]

# %% Rolling Forecast Setup
horizon = 24  # 24-hour horizon
steps = len(y_test) // horizon  # Number of rolling windows
print(f"Number of rolling forecast steps: {steps}")

# Initialize lists for forecasts
sarima_forecast = []
sarimax_forecast = []
drift_forecast_values = []

# %% Rolling Out-of-Sample Forecasting
for i in range(0, len(y_test) - horizon + 1, horizon):
    rolling_y_train = pd.concat([y_train, y_test.iloc[:i]])
    rolling_exog_train = pd.concat([exog_train, exog_test.iloc[:i]])
    rolling_exog_test = exog_test.iloc[i : i + horizon]
    rolling_y_test = y_test.iloc[i : i + horizon]

    # --- 1. SARIMA Model ---
    arima_model = ARIMA(order=(1, 1, 1), seasonal_order=(0, 1, 1, 24))
    arima_results = arima_model.fit(rolling_y_train)
    sarima_forecast.extend(arima_results.predict(n_periods=horizon))

    # --- 2. SARIMAX Model ---
    arima_exog_model = ARIMA(order=(1, 1, 1), seasonal_order=(0, 1, 1, 24))
    arima_exog_results = arima_exog_model.fit(rolling_y_train, X=rolling_exog_train)
    sarimax_forecast.extend(
        arima_exog_results.predict(n_periods=horizon, X=rolling_exog_test)
    )

    # --- 3. Drift Forecast ---
    slope = (rolling_y_train.iloc[-1] - rolling_y_train.iloc[0]) / len(rolling_y_train)
    drift_forecast_values.extend(
        [rolling_y_train.iloc[-1] + slope * (j + 1) for j in range(horizon)]
    )

# Convert forecasts to Series
sarima_forecast = pd.Series(sarima_forecast, index=y_test.index[: len(sarima_forecast)])
sarimax_forecast = pd.Series(
    sarimax_forecast, index=y_test.index[: len(sarimax_forecast)]
)
drift_forecast_series = pd.Series(
    drift_forecast_values, index=y_test.index[: len(drift_forecast_values)]
)

# %% Naive Forecasting Technique
# Naive forecast: replicate the last observed value for each step
naive_forecast = y_test.shift(1).fillna(method="bfill")

# Evaluate naive forecast
naive_mae = mean_absolute_error(y_test, naive_forecast)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_forecast))

print(f"Naive Forecast MAE: {naive_mae:.4f}")
print(f"Naive Forecast RMSE: {naive_rmse:.4f}")

# %% Evaluate Forecast Performance
mae_sarima = mean_absolute_error(y_test[: len(sarima_forecast)], sarima_forecast)
mae_sarimax = mean_absolute_error(y_test[: len(sarimax_forecast)], sarimax_forecast)
mae_drift = mean_absolute_error(
    y_test[: len(drift_forecast_series)], drift_forecast_series
)

print("\n--- Forecast Performance ---")
print(f"MAE (SARIMA): {mae_sarima:.4f}")
print(f"MAE (SARIMAX): {mae_sarimax:.4f}")
print(f"MAE (Drift Forecast): {mae_drift:.4f}")
print(f"MAE (Naive Forecast): {naive_mae:.4f}")

# %% Visualize Forecasts
plt.figure(figsize=(15, 8))
plt.plot(y_test, label="Actual Demand", color="blue")
plt.plot(sarima_forecast, label="SARIMA Forecast", linestyle="--", color="red")
plt.plot(
    sarimax_forecast,
    label="SARIMAX Forecast",
    linestyle="--",
    color="green",
)
plt.plot(drift_forecast_series, label="Drift Forecast", linestyle="--", color="purple")
plt.plot(naive_forecast, label="Naive Forecast", linestyle="--", color="orange")

plt.title("Rolling Out-of-Sample Forecast: Actual vs Forecasted Demand")
plt.xlabel("Time")
plt.ylabel("Demand")
plt.legend()
plt.grid()
plt.show()

# %% Final Comparison of All Models
print("\n--- Final Model Comparison ---")
print(f"MAE (SARIMA): {mae_sarima:.4f}")
print(f"MAE (SARIMAX): {mae_sarimax:.4f}")
print(f"MAE (Drift Forecast): {mae_drift:.4f}")
print(f"MAE (Naive Forecast): {naive_mae:.4f}")

# Determine the best-performing model
mae_dict = {
    "SARIMA": mae_sarima,
    "SARIMAX": mae_sarimax,
    "Drift Forecast": mae_drift,
    "Naive Forecast": naive_mae,
}

best_model = min(mae_dict, key=mae_dict.get)
print(f"\nThe best-performing model is: {best_model}")

# %%
