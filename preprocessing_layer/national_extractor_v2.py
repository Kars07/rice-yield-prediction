import ee
import geemap
import pandas as pd
import time

# Initialize GEE
ee.Initialize(project='rice-yield-prediction-487711')

regions = {
    "Kebbi": [4.1953, 11.4836], "Niger": [6.5569, 9.9309],
    "Kano": [8.5920, 12.0022], "Jigawa": [9.5616, 12.2280],
    "Ebonyi": [8.1137, 6.2649], "Taraba": [10.8198, 8.8937]
}
years = ['2022', '2023', '2024']
all_data = []

# --- 1. SENSOR INDEX FUNCTIONS ---
def add_indices(image):
    # NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    # NDWI (Water Index for Rice Flooding)
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    # EVI (Enhanced Vegetation Index)
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 10000))', {
            'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')
        }).rename('EVI')
    return image.addBands([ndvi, ndwi, evi])

def preprocess_sar(image):
    smooth = image.focal_median(radius=50, units='meters')
    return smooth.select(['VV', 'VH']).rename(['VV', 'VH']).copyProperties(image, ['system:time_start'])

print("Starting V2 Multi-Sensor & Climate Pipeline Extraction...")

# --- 2. MAIN EXTRACTION LOOP ---
for state, coords in regions.items():
    roi = ee.Geometry.Point(coords).buffer(1000)

    def extract_mean(image):
        mean_dict = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e9)
        return ee.Feature(None, mean_dict).set('date', image.date().format('YYYY-MM-dd'))

    for year in years:
        start_date = f'{year}-05-01'
        end_date = f'{year}-11-30'
        print(f"\nFetching {state} for {year}...")

        try:
            # A. Sentinel-2 (Optical: NDVI, EVI, NDWI)
            s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(roi).filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(add_indices).map(extract_mean))
            df_s2 = geemap.ee_to_df(ee.FeatureCollection(s2))
            if not df_s2.empty: df_s2['date'] = pd.to_datetime(df_s2['date'])
            else: df_s2 = pd.DataFrame(columns=['date'])

            # B. Sentinel-1 (Radar: VV, VH)
            s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi).filterDate(start_date, end_date)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .map(preprocess_sar).map(extract_mean))
            df_s1 = geemap.ee_to_df(ee.FeatureCollection(s1))
            if not df_s1.empty: df_s1['date'] = pd.to_datetime(df_s1['date'])
            else: df_s1 = pd.DataFrame(columns=['date'])

            # C. MODIS (Leaf Area Index - LAI)
            modis = (ee.ImageCollection('MODIS/061/MCD15A3H')
                     .filterBounds(roi).filterDate(start_date, end_date)
                     .select('Lai').map(extract_mean)) # Changed band name to 'Lai'
            df_modis = geemap.ee_to_df(ee.FeatureCollection(modis))
            if not df_modis.empty:
                df_modis['date'] = pd.to_datetime(df_modis['date'])
                df_modis['Lai'] = df_modis['Lai'] * 0.1 # Apply MODIS scale factor
            else: df_modis = pd.DataFrame(columns=['date'])

            # D. CHIRPS (Precipitation / Rainfall)
            chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                      .filterBounds(roi).filterDate(start_date, end_date)
                      .select('precipitation').map(extract_mean))
            df_chirps = geemap.ee_to_df(ee.FeatureCollection(chirps))
            if not df_chirps.empty: df_chirps['date'] = pd.to_datetime(df_chirps['date'])
            else: df_chirps = pd.DataFrame(columns=['date'])

            # E. ERA5-Land (Temperature)
            era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .select('temperature_2m').map(extract_mean))
            df_era5 = geemap.ee_to_df(ee.FeatureCollection(era5))
            if not df_era5.empty:
                df_era5['date'] = pd.to_datetime(df_era5['date'])
                df_era5['temperature_2m'] = df_era5['temperature_2m'] - 273.15 # Kelvin to Celsius
            else: df_era5 = pd.DataFrame(columns=['date'])

            # --- MERGE ALL DATASETS ---
            dfs = [df_s2, df_s1, df_modis, df_chirps, df_era5]
            df_combined = dfs[0]
            for df_temp in dfs[1:]:
                if not df_temp.empty:
                    # Merge on date, sort, and backfill/forward fill to align the different satellite schedules
                    df_combined = pd.merge(df_combined, df_temp, on='date', how='outer').sort_values('date')

            df_combined['State'] = state
            df_combined['Year'] = year

            # Clean columns
            available_cols = ['date', 'NDVI', 'EVI', 'NDWI', 'VV', 'VH', 'Lai_500m', 'precipitation', 'temperature_2m', 'State', 'Year']
            final_cols = [c for c in available_cols if c in df_combined.columns]

            all_data.append(df_combined[final_cols])
            print(f"  -> Merged {len(df_combined)} multispectral & climate records.")

        except Exception as e:
            print(f"  -> Error processing {state} {year}: {e}")

        time.sleep(1)

# --- 3. SAVE MASTER DATASET ---
print("\nCompiling National V2 Database...")
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv('national_climate_rice_data_v2.csv', index=False)
    print(f"Extraction complete! Saved to 'national_climate_rice_data_v2.csv'")
else:
    print("Extraction failed.")
