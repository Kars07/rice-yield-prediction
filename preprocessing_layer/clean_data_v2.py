import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

print("Loading V2 National Climate & Satellite Database...")
file_path = 'national_climate_rice_data_v2.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

print(f"Original Shape: {df.shape}")

# Define what to find
expected_smooth_features = ['NDVI', 'EVI', 'NDWI', 'VV', 'VH', 'Lai']
expected_climate_features = ['precipitation', 'temperature_2m']

# Dynamically filter for what ACTUALLY exists in the CSV
smooth_features = [col for col in expected_smooth_features if col in df.columns]
climate_features = [col for col in expected_climate_features if col in df.columns]

print(f"Detected phenology features for smoothing: {smooth_features}")
print(f"Detected climate features for gap-filling: {climate_features}")

window_length = 5
poly_order = 2

cleaned_groups = []
grouped = df.groupby(['State', 'Year'])

print("Applying interpolation and Savitzky-Golay filtering...")
for (state, year), group in grouped:
    group = group.sort_values('date').set_index('date')

    # 1. Gap Filling for available features (Interpolation for cloudy days)
    all_features = smooth_features + climate_features
    group_filled = group[all_features].interpolate(method='time')
    group_filled = group_filled.bfill().ffill()

    group_smooth = group_filled.copy()

    # 2. Temporal Smoothing ONLY for vegetation/radar indices (Phenology)
    if len(group_filled) >= window_length:
        for col in smooth_features:
            group_smooth[col] = savgol_filter(group_filled[col], window_length, poly_order)

    # Put State and Year back in
    group_smooth['State'] = state
    group_smooth['Year'] = year
    group_smooth = group_smooth.reset_index()

    cleaned_groups.append(group_smooth)

# Recombine and Save
df_final = pd.concat(cleaned_groups, ignore_index=True)
output_file = 'national_processed_v2.csv'
df_final.to_csv(output_file, index=False)

print(f"\nSuccess! Processed data saved to '{output_file}'")
print("\nData breakdown by State:")
print(df_final['State'].value_counts())
