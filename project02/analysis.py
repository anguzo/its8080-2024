# %%
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox, yeojohnson
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

train_data["time"] = pd.to_datetime(train_data["time"], utc=True)
test_data["time"] = pd.to_datetime(test_data["time"], utc=True)

# %%
train_data

# %%
time_period_start = train_data["time"].min()
time_period_end = train_data["time"].max()

time_period_start, time_period_end, train_data["time"].dt.year.value_counts()

# %%
test_data

# %%
time_period_start = test_data["time"].min()
time_period_end = test_data["time"].max()

time_period_start, time_period_end, test_data["time"].dt.year.value_counts()

# %%
train_data.set_index("time", inplace=True)
train_data

# %%
test_data.set_index("time", inplace=True)
test_data

# %%
train_data.describe()

# %%
test_data.describe()

# %%
msno.matrix(train_data)
plt.show()

# %%
msno.matrix(test_data)
plt.show()

# %%
missing_values = train_data.isnull().sum()
missing_values

# %%
missing_values = test_data.isnull().sum()
missing_values

# %%
# Considering snow misses 8305 of 8424 values in train_data, drop it
train_data.drop(columns=["snow"], inplace=True)
test_data.drop(columns=["snow"], inplace=True)

train_data.describe()

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(train_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %%
# Considering demand misses 86 of 8424 values in train_data
# and there is no straighforward correlation with other features
# possibly MCAR
# evaluate different simple imputation methods

train_data_with_missing = train_data.copy()

# Create a random mask for 10% missing values
np.random.seed(42)
mask = np.random.rand(len(train_data)) < 0.1
true_demand = train_data.loc[mask, "demand"]  # Save true values
train_data_with_missing.loc[mask, "demand"] = np.nan  # Introduce missingness


def impute_and_evaluate(train_data_with_missing, true_demand, method):
    imputed_data = train_data_with_missing.copy()

    if method == "linear":
        imputed_data["demand"] = imputed_data["demand"].interpolate(method=method)
    elif method == "spline":
        imputed_data["demand"] = imputed_data["demand"].interpolate(
            method=method, order=2
        )
    elif method == "ffill_bfill":
        imputed_data["demand"] = imputed_data["demand"].ffill().bfill()
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Extract imputed values and compare with true values
    imputed_values = imputed_data.loc[mask, "demand"]
    combined = pd.DataFrame({"true": true_demand, "imputed": imputed_values}).dropna()

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(combined["true"], combined["imputed"]))
    mae = mean_absolute_error(combined["true"], combined["imputed"])
    return f"{method.capitalize()} - RMSE: {rmse:.4f}, MAE: {mae:.4f}"


methods = ["linear", "spline", "ffill_bfill"]

for method in methods:
    result = impute_and_evaluate(train_data_with_missing, true_demand, method)
    print(result)


# %%
# Considering linear imputation performs best, apply it to train_data
# Linear - RMSE: 0.6320, MAE: 0.4218
# Spline - RMSE: 0.8105, MAE: 0.5650
# Ffill_bfill - RMSE: 0.8657, MAE: 0.5174
train_data["demand"] = train_data["demand"].interpolate(method="linear")

# %%
train_data

# %%
# Save cleaned train and test data
train_data.to_csv("data/train_data_cleaned.csv", index=True)
test_data.to_csv("data/test_data_cleaned.csv", index=True)
train_data.describe()

# %%
# Visualization 1: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %%
# Visualization 2: Seasonal decomposition
# Testing decomposition with different periods to find the optimal one
# 24 hours (daily), 168 hours (weekly)

periods = [6, 12, 24, 7 * 24]

for period in periods:
    decomposition = seasonal_decompose(
        train_data["demand"][: 2 * period], model="additive", period=period
    )
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    plt.suptitle(
        f"Seasonal Decomposition (period = {period} hours)",
        fontsize=16,
    )
    plt.show()


# %%
# Analyze distributions of numerical features
numerical_features = train_data.select_dtypes(include=np.number).columns.drop("demand")

for feature in numerical_features:
    plt.figure(figsize=(10, 8))
    sns.histplot(train_data[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

for feature in numerical_features:
    print(f"Skewness of {feature}: {train_data[feature].skew():.4f}")

# Skewness of temp: 0.0588
# Skewness of dwpt: -0.1420
# Skewness of rhum: -0.9020
# Skewness of wdir: -0.5887
# Skewness of wspd: 0.6829
# Skewness of wpgt: 0.7630
# Skewness of pres: -0.4476
# Skewness of price: 6.1581
# Seems like price is skewed and wspd is multimodal
# Also wspd can be sin-cos transformed

# %% Transformation and Feature Engineering

# %% Binning wspd

train_data_copy = train_data.copy()


def bin_feature(data, feature, bins, labels):
    return pd.cut(data[feature], bins=bins, labels=labels, include_lowest=True)


wspd_bins = [0, 3, 6, 9, 12, 15, 18, 21, np.inf]
wspd_labels = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24" "24+"]

train_data_copy["wspd_binned"] = bin_feature(train_data, "wspd", wspd_bins, wspd_labels)

plt.figure(figsize=(10, 8))
train_data_copy["wspd_binned"].value_counts().sort_index().plot(kind="bar")
plt.title("Binned Wind Speed (wspd)")
plt.xlabel("Wind Speed Bins")
plt.ylabel("Frequency")
plt.show()

# %% Correlation Matrix of Transformed Features
plt.figure(figsize=(10, 8))
sns.heatmap(
    pd.get_dummies(train_data_copy, drop_first=True).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation Matrix")
plt.show()

# From examining correlation matrix, it seems that binned wspd is not reasonable to use

# %% Compare transformations for price
# Ensure price is positive for Box-Cox transformation
train_data_copy = train_data.copy()

price_positive = train_data_copy["price"] + 1e-6

train_data_copy["price_log"] = np.log1p(train_data_copy["price"])  # Log Transformation
train_data_copy["price_boxcox"], _ = boxcox(price_positive)  # Box-Cox Transformation
train_data_copy["price_yeojohnson"], _ = yeojohnson(
    train_data_copy["price"]
)  # Yeo-Johnson Transformation
scaler = MinMaxScaler()
train_data_copy["price_minmax"] = scaler.fit_transform(
    train_data_copy[["price"]]
)  # Min-Max Scaling

fig, axes = plt.subplots(1, 5, figsize=(20, 6))

sns.histplot(train_data_copy["price"], kde=True, ax=axes[0])
axes[0].set_title("Original Price Distribution")

sns.histplot(train_data_copy["price_log"], kde=True, ax=axes[1])
axes[1].set_title("Log-Transformed Price")

sns.histplot(train_data_copy["price_boxcox"], kde=True, ax=axes[2])
axes[2].set_title("Box-Cox Transformed Price")

sns.histplot(train_data_copy["price_yeojohnson"], kde=True, ax=axes[3])
axes[3].set_title("Yeo-Johnson Transformed Price")

sns.histplot(train_data_copy["price_minmax"], kde=True, ax=axes[4])
axes[4].set_title("Min-Max Scaled Price")

plt.tight_layout()
plt.show()

# %% Compare skewness
original_skew = train_data_copy["price"].skew()
log_skew = train_data_copy["price_log"].skew()
boxcox_skew = pd.Series(train_data_copy["price_boxcox"]).skew()
yeojohnson_skew = pd.Series(train_data_copy["price_yeojohnson"]).skew()
minmax_skew = pd.Series(train_data_copy["price_minmax"]).skew()

print(f"Original Price Skewness: {original_skew:.4f}")
print(f"Log-Transformed Price Skewness: {log_skew:.4f}")
print(f"Box-Cox Transformed Price Skewness: {boxcox_skew:.4f}")
print(f"Yeo-Johnson Transformed Skewness: {yeojohnson_skew:.4f}")
print(f"Min-Max Scaled Price Skewness: {minmax_skew:.4f}")

# Original Price Skewness: 6.1581
# Log-Transformed Price Skewness: 1.8920
# Box-Cox Transformed Price Skewness: 0.1347
# Yeo-Johnson Transformed Skewness: 0.0165
# Min-Max Scaled Price Skewness: 6.1581
# Seems like Yeo-Johnson transformation is the best for price, Box-Cox also introduces negative values

# %% Correlation Matrix of Transformed Features
plt.figure(figsize=(10, 8))
sns.heatmap(train_data_copy.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %% Sin-Cos Transformation for wdir
train_data_copy = train_data.copy()

train_data_copy["wdir_sin"] = np.sin(np.deg2rad(train_data_copy["wdir"]))
train_data_copy["wdir_cos"] = np.cos(np.deg2rad(train_data_copy["wdir"]))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(train_data_copy["wdir_sin"], kde=True, ax=ax[0])
ax[0].set_title("wdir Sin Transformation")
sns.histplot(train_data_copy["wdir_cos"], kde=True, ax=ax[1])
ax[1].set_title("wdir Cos Transformation")

plt.tight_layout()
plt.show()

# %% Correlation Matrix of Transformed Features
plt.figure(figsize=(10, 8))
sns.heatmap(train_data_copy.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %% Design New Features
train_data_copy = train_data.copy()

# Interaction Features
train_data_copy["temp_rhum_interaction"] = (
    train_data_copy["temp"] * train_data_copy["rhum"]
)
train_data_copy["wspd_wpgt_interaction"] = (
    train_data_copy["wspd"] * train_data_copy["wpgt"]
)
train_data_copy["wspd_temp_interaction"] = (
    train_data_copy["wspd"] * train_data_copy["temp"]
)
train_data_copy["temp_pres_ratio"] = train_data_copy["temp"] / (
    train_data_copy["pres"] + 1e-6
)

# Extract time-based features
train_data_copy["hour"] = train_data_copy.index.hour
train_data_copy["dayofweek"] = train_data_copy.index.dayofweek
train_data_copy["month"] = train_data_copy.index.month

train_data_copy["hourly_price_interaction"] = (
    train_data_copy["hour"] * train_data_copy["price"]
)

# Drop NaN Rows Created by Lags and Rolling Features
train_data_copy = train_data_copy.dropna()

# %% Display updated train_data_copy with new features
train_data_copy.head()

# %% Display ranking
feature_ranking = (
    train_data_copy.corr()["demand"].drop("demand").sort_values(ascending=False)
)

print("Feature Correlation Ranking with Demand:")
print(feature_ranking)

# Top 2 new negatively correlated features:
# 1. temp_rhum_interaction: Interaction of temperature and humidity (-0.2754)
# 2. temp_pres_ratio: Temperature-to-pressure ratio (-0.2669)

# %% Correlation Matrix of New Features
plt.figure(figsize=(20, 12))
sns.heatmap(train_data_copy.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %% Best Feature Engineering Steps
train_data_w_feat = train_data.copy()

# Yeo-Johnson Transformation for price
train_data_w_feat["price_yeojohnson"], _ = yeojohnson(train_data_w_feat["price"])

# Sin-Cos transformation for wind direction (wdir)
train_data_w_feat["wdir_sin"] = np.sin(np.deg2rad(train_data_w_feat["wdir"]))
train_data_w_feat["wdir_cos"] = np.cos(np.deg2rad(train_data_w_feat["wdir"]))

# Interaction Features
train_data_w_feat["temp_rhum_interaction"] = (
    train_data_w_feat["temp"] * train_data_w_feat["rhum"]
)
train_data_w_feat["temp_pres_ratio"] = train_data_w_feat["temp"] / (
    train_data_w_feat["pres"] + 1e-6
)

# Drop NaN Rows Created by Lag and Rolling Features
train_data_w_feat = train_data_w_feat.dropna()

# Save the final data with the best new features
# train_data_w_feat.to_csv("data/train_data_w_feat.csv", index=True)
