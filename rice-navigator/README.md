# Frontend Documentation

## Frontend Application URL
[Frontend App](https://rice-navigator.lovable.app) *(Update this link with the actual deployed URL if different)*

## Environment Setup & Run Commands

### Install Node.js/npm and frontend dependencies

Install Node.js (npm is included): https://nodejs.org/

From the repository root:
```bash
npm install
```

Then:
```bash
cd rice-navigator
npm install
npm run dev
```

## Data Dictionary (For UI Tooltips & Labels)

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
