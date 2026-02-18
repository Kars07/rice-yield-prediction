import ee
import geemap
import pandas as pd
import time

# Initialize GEE
ee.Initialize(project='rice-yield-prediction-487711')

# Define the 6 target states and their approximate rice-growing coordinates
regions = {
    "Kebbi": [4.1953, 11.4836],
    "Niger": [6.5569, 9.9309],
    "Kano": [8.5920, 12.0022],
    "Jigawa": [9.5616, 12.2280],
    "Ebonyi": [8.1137, 6.2649],
    "Taraba": [10.8198, 8.8937]
}

years = ['2022', '2023', '2024']
all_data = []

# --- Helper Functions for GEE ---
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def preprocess_sar(image):
    smooth = image.focal_median(radius=50, units='meters')
    return smooth.select(['VV', 'VH']).rename(['VV', 'VH']).copyProperties(image, ['system:time_start'])

print("Starting National Data Pipeline Extraction...")

# Main Extraction Loop
for state, coords in regions.items():
    roi = ee.Geometry.Point(coords).buffer(1000)

    # Reducers specific to this ROI
    def extract_s2(image):
        mean_dict = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e9)
        return ee.Feature(None, mean_dict).set('date', image.date().format('YYYY-MM-dd'))

    def extract_s1(image):
        mean_dict = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e9)
        return ee.Feature(None, mean_dict).set('date', image.date().format('YYYY-MM-dd'))

    for year in years:
        start_date = f'{year}-05-01'
        end_date = f'{year}-11-30'

        print(f"Fetching {state} for {year}...")

        # Fetch Sentinel-2
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

        s2_mapped = s2_collection.map(add_ndvi).map(extract_s2)
        s2_fc = ee.FeatureCollection(s2_mapped).filter(ee.Filter.notNull(['NDVI']))

        try:
            df_s2 = geemap.ee_to_df(s2_fc)
            if not df_s2.empty:
                df_s2['date'] = pd.to_datetime(df_s2['date'])
            else:
                df_s2 = pd.DataFrame(columns=['date', 'NDVI'])
        except Exception:
            df_s2 = pd.DataFrame(columns=['date', 'NDVI'])

        # Fetch Sentinel-1 (Radar) ---
        s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW')))

        s1_mapped = s1_collection.map(preprocess_sar).map(extract_s1)
        s1_fc = ee.FeatureCollection(s1_mapped).filter(ee.Filter.notNull(['VV', 'VH']))

        try:
            df_s1 = geemap.ee_to_df(s1_fc)
            if not df_s1.empty:
                df_s1['date'] = pd.to_datetime(df_s1['date'])
            else:
                df_s1 = pd.DataFrame(columns=['date', 'VV', 'VH'])
        except Exception:
            df_s1 = pd.DataFrame(columns=['date', 'VV', 'VH'])

        # Sensor Harmonization & Merging
        if not df_s1.empty or not df_s2.empty:
            if df_s1.empty:
                df_combined = df_s2.copy()
            elif df_s2.empty:
                df_combined = df_s1.copy()
            else:
                df_combined = pd.merge(df_s1, df_s2, on='date', how='outer')

            # Clean up and add identifiers
            df_combined = df_combined.sort_values('date')
            df_combined['State'] = state
            df_combined['Year'] = year

            # Drop unnecessary columns if they carried over
            cols_to_keep = ['date', 'NDVI', 'VV', 'VH', 'State', 'Year']
            existing_cols = [col for col in cols_to_keep if col in df_combined.columns]
            df_combined = df_combined[existing_cols]

            all_data.append(df_combined)
            print(f"  -> Successfully retrieved {len(df_combined)} data points.")
        else:
            print("  -> Warning: No valid data found for this period.")

        time.sleep(1) # Be polite to Google's servers

# Final Aggregation
print("\nCompiling national dataset...")
if len(all_data) > 0:
    final_national_df = pd.concat(all_data, ignore_index=True)
    final_national_df = final_national_df[['State', 'Year', 'date', 'NDVI', 'VV', 'VH']]
    final_national_df.to_csv('national_rice_data_2022_2024.csv', index=False)
    print(f"Massive extraction complete! Saved {len(final_national_df)} rows to 'national_rice_data_2022_2024.csv'")
else:
    print("Extraction failed: No data was appended.")
