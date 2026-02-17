import ee
import geemap
import pandas as pd

ee.Initialize(project='rice-yield-prediction-487711')

# Define Region of Interest (ROI) - Example: A rice-growing area in Kebbi State
roi = ee.Geometry.Point([4.1953, 11.4836]).buffer(1000) # Coordinates for Kebbi

# Access Sentinel-2 Level-2A (Surface Reflectance)
# This handles the "Atmospheric Correction" box
s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate('2024-05-01', '2024-11-30') # Rice growing season
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) # Cloud Masking
)

# Function to Calculate NDVI (Feature Engineering Layer)
# Formula: (NIR - Red) / (NIR + Red)
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Apply the function to the collection
s2_with_ndvi = s2_collection.map(add_ndvi)

# Extract Data for Analysis
# We pull the mean NDVI and Date for the region to create a time-series.
def extract_data(image):
    mean_dict = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    # Add a date property
    return ee.Feature(None, mean_dict).set('date', image.date().format('YYYY-MM-dd'))

# Apply the function
mapped_data = s2_with_ndvi.map(extract_data)

data_series = ee.FeatureCollection(mapped_data)

# Filter out nulls (important for clean data)
data_series = data_series.filter(ee.Filter.notNull(['NDVI']))

# Convert to Pandas DataFrame
df = geemap.ee_to_df(data_series)

# Sort by date for proper time-series analysis
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

print(f"Retrieved {len(df)} data points.")
print(df[['date', 'NDVI']].head())

print("\n--- Fetching Sentinel-1 (Radar) Data ---")

# 5. Access Sentinel-1 SAR (Ground Range Detected)
s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate('2024-05-01', '2024-11-30')
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
)

# Preprocessing: Speckle Filtering
def preprocess_sar(image):
    smooth = image.focal_median(radius=50, units='meters')
    return smooth.select(['VV', 'VH']).rename(['VV', 'VH']).copyProperties(image, ['system:time_start'])

s1_clean = s1_collection.map(preprocess_sar)

# Extract SAR Data
def extract_sar_data(image):
    mean_dict = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    return ee.Feature(None, mean_dict).set('date', image.date().format('YYYY-MM-dd'))

# Map the function
s1_mapped = s1_clean.map(extract_sar_data)

# Cast to FeatureCollection
s1_series = ee.FeatureCollection(s1_mapped)

# Filter nulls
s1_series = s1_series.filter(ee.Filter.notNull(['VV', 'VH']))

# Convert to DataFrame
df_sar = geemap.ee_to_df(s1_series)

# Sort and Format
df_sar['date'] = pd.to_datetime(df_sar['date'])
df_sar = df_sar.sort_values(by='date')

print(f"Retrieved {len(df_sar)} SAR data points (Cloud-Free!).")
print(df_sar.head())

# Merge the Datasets (Sensor Harmonization)
# We use an 'outer' join to keep all dates.
df_combined = pd.merge(df_sar, df, on='date', how='outer').sort_values('date')

# Save to CSV for the next phase
df_combined.to_csv('kebbi_rice_data_2024.csv', index=False)
print("\n--- Merged Data Head ---")
print(df_combined.head(10))
print(f"\nSaved combined data to 'kebbi_rice_data_2024.csv'")
