# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# %% [markdown]
# **1. Data Loading and Cleaning**

# %%
# Load the dataset
data = pd.read_csv('data/solar_232215.csv', delimiter=',', decimal='.')
# Delete unnecessary columns if they exist
data.drop(['modified1', 'modified2', 'modified3'],
          axis=1, inplace=True, errors='ignore')
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Display the first few rows of the data
data.head()

# Extract the solar output column
raw = data['raw']

data

# %% [markdown]
# **2. Additive Classical Decomposition with Multiple Periods**

# %%
# Testing decomposition with different periods to find the optimal one for solar generation
# 24 hours (daily), 168 hours (weekly), and 720 hours (monthly)
periods = [24, 168, 720]
decomposition_results = {}

for period in periods:
    print(f"\nDecomposition with period: {period} hours")
    decomposition = seasonal_decompose(
        raw, model='additive', period=period)
    decomposition_results[period] = decomposition

    # Plot the decomposition components
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle(
        f'Additive Decomposition of Solar Output (Period={period} hours)', fontsize=16)
    plt.show()

# %% [markdown]
# **3. Comparison of Decomposition Periods**

# %%
# Analyze each period's effectiveness in capturing solar generation patterns

print("Comparing decomposition periods for solar generation insights:")

for period, decomposition in decomposition_results.items():
    # Calculate variance explained by the seasonal component for each period
    seasonal_variance = decomposition.seasonal.var()
    residual_variance = decomposition.resid.var()
    explained_ratio = seasonal_variance / \
        (seasonal_variance + residual_variance)

    print(f"Period {period} hours:")
    print(f" - Seasonal Variance: {seasonal_variance:.2f}")
    print(f" - Residual Variance: {residual_variance:.2f}")
    print(f" - Seasonal Variance Explained Ratio: {explained_ratio:.2%}\n")

# The period with the highest seasonal variance explained ratio is considered optimal
optimal_period = max(decomposition_results, key=lambda p: decomposition_results[p].seasonal.var(
) / (decomposition_results[p].seasonal.var() + decomposition_results[p].resid.var()))

print(
    f"The optimal decomposition period for solar generation analysis is {optimal_period} hours.")

# %% [markdown]
# **4. Description of Decomposition Components for Optimal Period**

# %%
# Display components for the optimal period

optimal_decomposition = decomposition_results[optimal_period]

print("Observed Component (Optimal Period):")
display(optimal_decomposition.observed.head())

print("\nTrend Component (Optimal Period):")
display(optimal_decomposition.trend.head())

print("\nSeasonal Component (Optimal Period):")
# Displaying a full day's seasonality if period=24
display(optimal_decomposition.seasonal.head(25))

print("\nResidual Component (Optimal Period):")
display(optimal_decomposition.resid.head())

# %% [markdown]
# **5. Creation of Typical Generation Profiles**

# %%
# Extract hour, month, and season for profiling
data['hour'] = data.index.hour
data['month'] = data.index.month

# Define function to assign seasons based on month


def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


data['season'] = data['month'].apply(get_season)

# Ensure the raw column is correctly named
solar_column = 'raw'  # Replace with actual column name if different

# Calculate and plot the typical daily profile
hourly_profile = data.groupby('hour')[solar_column].mean()
plt.figure(figsize=(10, 6))
hourly_profile.plot()
plt.title('Typical Daily Solar Generation Profile')
plt.xlabel('Hour of Day')
plt.ylabel('Average Solar Output')
plt.grid(True)
plt.show()

# %% [markdown]
# **6. Monthly Profile of Solar Output**

# %%
# Calculate average solar output for each month and plot
monthly_profile = data.groupby('month')[solar_column].mean()
plt.figure(figsize=(10, 6))
monthly_profile.plot(kind='bar')
plt.title('Average Monthly Solar Generation Profile')
plt.xlabel('Month')
plt.ylabel('Average Solar Output')
plt.grid(True)
plt.show()

# %% [markdown]
# **7. Seasonal Hourly Profiles**

# %%
# Calculate and plot average hourly solar output for each season
seasonal_profiles = data.groupby(['season', 'hour'])[
    solar_column].mean().unstack('season')
seasonal_profiles.plot(figsize=(12, 8))
plt.title('Average Hourly Solar Generation by Season')
plt.xlabel('Hour of Day')
plt.ylabel('Average Solar Output')
plt.legend(title='Season')
plt.grid(True)
plt.show()

# %% [markdown]
# **8. Methodology Explanation**

# %% [markdown]
# **Methodology Explanation:**
#
# *Additive Classical Decomposition with Multiple Periods:*
# - Tested decomposition with daily (24 hours), weekly (168 hours), and monthly (720 hours) periods.
# - Optimal period selected based on the highest ratio of variance explained by the seasonal component.
#
# *Typical Generation Profiles:*
# - **Hourly Profile:** Averages solar output at each hour across days for typical daily pattern.
# - **Monthly Profile:** Averages solar output by month to observe annual variation.
# - **Seasonal Profiles:** Uses seasonal grouping for average hourly output per season.
#
# **Rationale:**
# - Understanding these patterns aids in energy production forecasting and optimization.
# - Facilitates planning for storage and grid management.
