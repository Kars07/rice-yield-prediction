import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

# 1. Create a simulated timeline (150 days of the growing season)
# e.g., May to November
days = np.arange(0, 150, 3)
dates = pd.date_range(start='2024-05-01', periods=len(days), freq='3D')

# 2. Create the "True" Biological Curve (A perfect Bell Curve)
# Starts around 0.15, peaks at 0.85 around day 75, drops to 0.25 at harvest
base_ndvi = 0.15 + 0.7 * np.exp(-0.5 * ((days - 75) / 25) ** 2)

# 3. Add "Jagged" Noise to simulate Cloud Cover and Atmospheric Scattering
np.random.seed(42) # For reproducibility
# Add general sensor noise
general_noise = np.random.normal(0, 0.03, len(days))
# Add harsh negative drops to simulate opaque clouds blocking the satellite
cloud_drops = np.random.choice([0, 1], size=len(days), p=[0.85, 0.15]) * np.random.uniform(0.15, 0.4, len(days))

# Final Raw Data (Jagged)
raw_ndvi = base_ndvi + general_noise - cloud_drops
raw_ndvi = np.clip(raw_ndvi, 0.05, 1.0) # Ensure realistic NDVI boundaries

# 4. Apply the Savitzky-Golay Smoothing Filter
# window_length determines how many points to look at (must be odd), polyorder is the polynomial degree
smoothed_ndvi = savgol_filter(raw_ndvi, window_length=15, polyorder=3)

# 5. Plotting for the Thesis
plt.figure(figsize=(10, 6), dpi=300) # High resolution for academic paper

# Plot Raw Jagged Data
plt.plot(dates, raw_ndvi, marker='o', markersize=5, linestyle='-', color='#e74c3c',
         alpha=0.6, label='Raw NDVI (Optical Cloud/Sensor Noise)')

# Plot Smoothed Data
plt.plot(dates, smoothed_ndvi, linewidth=3.5, color='#27ae60',
         label='Smoothed NDVI (Savitzky-Golay Filter)')

# Formatting the Chart
plt.title('Phenological Curve Construction: Raw vs. Smoothed NDVI Data', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Growing Season Timeline', fontsize=12, fontweight='bold')
plt.ylabel('NDVI Value', fontsize=12, fontweight='bold')

# Set y-axis limits to standard NDVI range
plt.ylim(0.0, 1.0)

# Add gridlines for readability
plt.grid(True, linestyle='--', alpha=0.5)

# Add a stylish legend
plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, borderpad=1)

# Annotate key growth stages to make it look highly academic
plt.axvline(pd.to_datetime('2024-05-15'), color='gray', linestyle=':', alpha=0.5)
plt.text(pd.to_datetime('2024-05-17'), 0.05, 'Transplanting', color='gray', fontsize=10, rotation=90)

plt.axvline(pd.to_datetime('2024-07-15'), color='gray', linestyle=':', alpha=0.5)
plt.text(pd.to_datetime('2024-07-17'), 0.05, 'Peak Vegetative', color='gray', fontsize=10, rotation=90)

plt.axvline(pd.to_datetime('2024-09-15'), color='gray', linestyle=':', alpha=0.5)
plt.text(pd.to_datetime('2024-09-17'), 0.05, 'Harvest Phase', color='gray', fontsize=10, rotation=90)

# Tight layout and save
plt.tight_layout()
plt.savefig('ndvi_smoothing_comparison.png')
print("Graph successfully generated and saved as 'ndvi_smoothing_comparison.png'")

# Display the plot if running in an interactive environment
plt.show()
