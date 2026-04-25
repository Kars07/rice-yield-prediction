# AgroSense Intelligence: Integrated Rice Yield Prediction System

## Project Overview
This repository contains the data pipelines, pre-trained machine learning models, and processed datasets for the Nigerian Rice Yield Prediction and Monitoring System. 

The system fuses multi-sensor satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical) with climate data (CHIRPS, ERA5) to predict rice harvests at the state level using a Hybrid Ensemble Machine Learning architecture (XGBoost + LSTM).

**For the Frontend Developer:** Your goal is to build an intuitive, map-centric UI (inspired by modern GIS/Google Maps aesthetics) that consumes this data to display real-time crop health, weather metrics, and final yield predictions.

---

## Repository File Manifest

### 1. `national_processed_v2.csv` (The Core Dataset)
This is the primary data source for the frontend UI. It contains the fully cleaned, gap-filled, and mathematically smoothed (Savitzky-Golay) time-series data for 6 Nigerian states. 
* **Frontend Use Case:** Use this file to populate the interactive charts (e.g., NDVI phenology curves over time) and to display current temperature/rainfall metrics in the UI cards.

### 2. `xgboost_model_v2.json`
The saved state of the XGBoost Regressor model. This model analyzes the "Tabular" aggregated features (e.g., Mean NDVI, Total Rain) to predict crop yield.
* **Frontend Use Case:** If building a Python-based backend (like Streamlit/FastAPI), this file is loaded via `xgboost.XGBRegressor().load_model()`.

### 3. `dl_model_v2.pth`
The saved PyTorch weights for the Deep Learning (LSTM) model. This neural network analyzes the sequential, month-by-month time-series data to understand the temporal growth phases of the rice.
* **Frontend Use Case:** Loaded via `torch.load()` into the `RiceLSTM` class architecture to generate the temporal yield prediction.

### 4. `ensemble_weights_v2.json`
A simple JSON file containing the dynamically calculated trust weights for the two models. 
* **Current Weights:** XGBoost (`~52.1%`), LSTM (`~47.8%`).
* **Frontend Use Case:** The final yield displayed to the user MUST be calculated by multiplying the XGBoost prediction and LSTM prediction by these respective weights and adding them together.

### 5. `ensemble_v2.py`
The master backend script. This is for reference. It contains the exact data interpolation logic (`scipy.interpolate`), feature engineering math, model architectures, and training loops used to generate the `.json` and `.pth` files.

---

##  Data Dictionary (For UI Tooltips & Labels)

When building the dashboard, you will map the columns from `national_processed_v2.csv` to the UI. Here is what they mean:

| Column Name | UI Display Label | Description for UI Tooltips / Logic |
| :--- | :--- | :--- |
| `date` | **Date** | The specific day of the growing season (May to December). Use this for the X-axis on all time-series charts. |
| `State` | **Region** | The Nigerian state being analyzed (Kebbi, Niger, Kano, Jigawa, Ebonyi, Taraba). |
| `NDVI` | **Crop Health (NDVI)** | *Normalized Difference Vegetation Index.* Measures general plant health and greenness. Values range from -1 to 1. Higher is greener. Primary metric for the main phenology chart. |
| `EVI` | **Canopy Density (EVI)** | *Enhanced Vegetation Index.* Similar to NDVI but corrects for atmospheric noise and handles highly dense, thick rice canopies without maxing out. |
| `NDWI` | **Water Index (NDWI)** | *Normalized Difference Water Index.* Extremely important for rice. Measures the presence of water in the paddy. High values = well flooded; Low values = drought stress. |
| `VV` & `VH` | **SAR Backscatter** | Radar data from Sentinel-1. Measures the physical thickness, structure, and volume of the rice plants, penetrating through cloud cover. |
| `precipitation` | **Rainfall (mm)** | Daily rainfall measured in millimeters. |
| `temperature_2m`| **Temperature (°C)** | Daily average surface temperature measured in Celsius. |

---

## Integration Notes for Frontend

1. **The Phenology Chart:** The primary visual for the dashboard should be an Area/Line chart plotting the `date` (X-axis) against the `NDVI` (Y-axis) for the selected state. This will draw a natural "bell curve" showing the crop growing, peaking, and being harvested.
2. **Weather Chips:** The UI should feature quick-glance chips for Weather. To get the "Current" weather, pull the `temperature_2m` and `precipitation` values from the *most recent date row* for the selected state in the CSV.
3. **Map Coordinates:** You will need to hardcode the lat/long coordinates for the Folium/Mapbox map pins, as they are not in the CSV. 
   * Kebbi: `[11.4836, 4.1953]`
   * Kano: `[12.0022, 8.5920]`
   * Ebonyi: `[6.2649, 8.1137]`
   * Niger: `[9.9309, 6.5569]`
   * Jigawa: `[12.2280, 9.5616]`
   * Taraba: `[8.8937, 10.8198]`
