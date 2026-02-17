import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Load the merged data
file_path = 'kebbi_rice_data_2024.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date')

print(f"Original Shape: {df.shape}")

# Select Key Features for the Model
# We focus on the indices mentioned in your proposal: NDVI (Optical) and VV/VH (Radar)
# We also keep 'TCI_G' (Green) if available, but let's stick to the core physics.
features = ['NDVI', 'VV', 'VH']
df_model = df[features].copy()

# Gap Filling (Interpolation)
# We use 'time' interpolation because the gaps between dates aren't equal.
# If May 1 and May 30 are known, May 15 should be right in the middle.
df_filled = df_model.interpolate(method='time')

# Backfill/Forwardfill any remaining edges (e.g., if the first row is NaN)
df_filled = df_filled.bfill().ffill()

print("\n--- Data after Interpolation (First 5 rows) ---")
print(df_filled.head())

# 4. Temporal Smoothing (Savitzky-Golay Filter)
# "smoothing (Savitzky-Golay filter)"
# window_length must be odd and less than the size of the dataset.
window_length = 5
poly_order = 2

df_smooth = df_filled.copy()

try:
    for col in features:
        df_smooth[col] = savgol_filter(df_filled[col], window_length, poly_order)
    print("\n--- Savitzky-Golay Smoothing Applied ---")
except ValueError as e:
    print(f"\nSkipping Smoothing (Dataset too small for window={window_length}): {e}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df_filled.index, df_filled['NDVI'], 'r--', label='Raw/Interpolated NDVI', alpha=0.5)
plt.plot(df_smooth.index, df_smooth['NDVI'], 'g-', label='Smoothed NDVI (Model Ready)', linewidth=2)
plt.title('Kebbi Rice Phenology: Raw vs Smoothed')
plt.legend()
plt.grid(True)
plt.savefig('preprocessing_layer/smoothing_check.png')
print("Saved plot to 'preprocessing_layer/smoothing_check.png'")

# Save the Final ML-Ready Dataset
output_file = 'kebbi_processed_final.csv'
df_smooth.to_csv(output_file)
print(f"\nSuccess! Processed data saved to '{output_file}'")
