# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

# %%
# Load and display the data
solar_232215 = pd.read_csv('data/solar_232215.csv', delimiter=',', decimal='.')
# Display the total number of entries in the dataset
print(f"Total number of entries: {len(solar_232215)}")
solar_232215.head(10)

# %% [markdown]
# The dataset consists of 17,016 entries with six columns:

# 1. **timestamp**: Unix timestamp (in milliseconds).
# 2. **raw**: Electricity production from the PV panel corresponding to each timestamp.
# 3. **temperature**: Temperature readings corresponding to each timestamp.
# 4. **modified1, modified2, modified3**: Modified versions of the raw production data. These columns have some missing values.

#
# Overall with the provided data it is possible to analyze the electricity production of the PV panel and its relationship with temperature.

#
# To analyze the PV panel's production effectively, the following potential external data sources might be useful:

# 1. **Solar Irradiance Data**: Information on solar irradiance levels (sunlight intensity) would help assess the PV production in relation to the available sunlight.
# 2. **Weather Conditions**: Data on cloud cover, humidity, or precipitation could help explain variations in electricity production that temperature alone might not cover.
# 3. **Panel Specifications**: Knowing the panel's efficiency, orientation, and tilt would allow for a more accurate interpretation of the production data.

# %%
# Convert timestamp from milliseconds to a readable datetime format and analyze the time range
data = solar_232215
data['timestamp_unix'] = solar_232215['timestamp']
data['timestamp'] = pd.to_datetime(data['timestamp_unix'], unit='ms')

time_period_start = data['timestamp'].min()
time_period_end = data['timestamp'].max()

time_period_start, time_period_end, data['timestamp'].dt.year.value_counts()

# %% [markdown]
# The dataset spans from February 2, 2021, to February 1, 2023. Data coverage by year:
# 1. 2021: Partial data, covering from February 2 onward, with 7,752 hours.
# 2. 2022: Full year, with 8,760 hours.
# 3. 2023: Partial data, covering until February 1, with 504 hours

# %%
# Explore missing values
missing_values = data.isnull().sum()
missing_values

# %%
# Visualize the missing values as a heatmap
msno.heatmap(data)
plt.show()

# %%
# Visualize the missing values as a matrix
msno.matrix(data)
plt.show()


# %% [markdown]
# The timestamp, raw, temperature, and timestamp_unix columns have no missing values.
#
# The modified1, modified2, and modified3 columns show substantial missing data:
# 1. modified1 has 1191 missing values.
# 2. modified2 has 1375 missing values.
# 3. modified3 has 1137 missing values.
#
# The heatmap reveals weak correlations in missing data patterns between modified1, modified2, and modified3, indicating they most likely have unrelated missing data trends.
# %%

# %%
# Remove rows where raw is 0
data_no_rows_zero = data[data['raw'] != 0]
print(f"Total number of entries: {len(data_no_rows_zero)}")
data_no_rows_zero.head(10)
# %%
missing_values = data.isnull().sum()
missing_values
# %%
# Visualize the missing values as a heatmap
msno.heatmap(data)
plt.show()

# %%
# Visualize the missing values as a matrix
msno.matrix(data)
plt.show()
