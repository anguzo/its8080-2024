# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# %%
train_data = pd.read_csv("data/train_data_cleaned.csv")
test_data = pd.read_csv("data/test_data_cleaned.csv")

train_data["time"] = pd.to_datetime(train_data["time"])
test_data["time"] = pd.to_datetime(test_data["time"])

train_data.set_index("time", inplace=True)
test_data.set_index("time", inplace=True)

# Target variable
y_train = train_data["demand"]
y_test = test_data["demand"]

# Exogenous inputs: temp, dwpt, and price
exog_train = train_data[["temp"]]
exog_test = test_data[["temp"]]

# %%
auto_arima_model = auto_arima(
    y_train, seasonal=True, m=24, stepwise=True, trace=True, D=1, max_P=1, max_Q=1, d=1
)

# %%

exo_auto_arima_model = auto_arima(
    y_train, seasonal=True, m=24, stepwise=True, trace=True, D=1, max_P=1, max_Q=1, d=1
)


# %% Compare the two models using MSE
auto_arima_forecast = auto_arima_model.predict(n_periods=len(y_test))
exo_auto_arima_forecast = exo_auto_arima_model.predict(
    n_periods=len(y_test), exogenous=exog_test
)

auto_arima_mae = mean_absolute_error(y_test, auto_arima_forecast)
exo_auto_arima_mae = mean_absolute_error(y_test, exo_auto_arima_forecast)

print(f"Auto ARIMA MAE: {auto_arima_mae}")
print(f"Exogenous Auto ARIMA MAE: {exo_auto_arima_mae}")

# %% Plot the forecasts
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual")
plt.plot(auto_arima_forecast, label="Auto ARIMA Forecast")
plt.plot(exo_auto_arima_forecast, label="Exogenous Auto ARIMA Forecast")
plt.legend()
plt.title("Auto ARIMA vs. Exogenous Auto ARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("Demand [kWh]")
plt.show()

# %%
auto_arima_model.summary()
# %%
exo_auto_arima_model.summary()
