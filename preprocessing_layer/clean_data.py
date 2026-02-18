import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

# Load the massive national dataset
file_path = 'national_rice_data_2022_2024.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

print(f"Original Shape: {df.shape}")

# Features to process
features = ['NDVI', 'VV', 'VH']
window_length = 5
poly_order = 2

# We will store the cleaned groups here
cleaned_groups = []

# Group by State and Year so we don't mix different locations/seasons!
grouped = df.groupby(['State', 'Year'])

for (state, year), group in grouped:
    # Sort chronologically
    group = group.sort_values('date').set_index('date')

    # Gap Filling (Interpolation)
    group_filled = group[features].interpolate(method='time')
    group_filled = group_filled.bfill().ffill()

    # Temporal Smoothing (Savitzky-Golay Filter)
    group_smooth = group_filled.copy()

    # Only smooth if we have enough data points in this specific season
    if len(group_filled) >= window_length:
        for col in features:
            group_smooth[col] = savgol_filter(group_filled[col], window_length, poly_order)

    # Put State and Year back in
    group_smooth['State'] = state
    group_smooth['Year'] = year
    group_smooth = group_smooth.reset_index()

    cleaned_groups.append(group_smooth)

# Recombine into the final Master Dataset
df_final = pd.concat(cleaned_groups, ignore_index=True)

# Save the Final ML-Ready Dataset
output_file = 'national_processed_final.csv'
df_final.to_csv(output_file, index=False)

print(f"\nSuccess! Processed data saved to '{output_file}'")
print("Data breakdown by State:")
print(df_final['State'].value_counts())
