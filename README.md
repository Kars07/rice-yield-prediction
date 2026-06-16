# AgroSense Intelligence: Integrated Rice Yield Prediction System

## Project Overview
This repository contains the data pipelines, pre-trained machine learning models, and processed datasets for the Nigerian Rice Yield Prediction and Monitoring System. 

The system fuses multi-sensor satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical) with climate data (CHIRPS, ERA5) to predict rice harvests at the state level using a Hybrid Ensemble Machine Learning architecture (XGBoost + LSTM).

**For the Frontend Developer:** Your goal is to build an intuitive, map-centric UI (inspired by modern GIS/Google Maps aesthetics) that consumes this data to display real-time crop health, weather metrics, and final yield predictions.

---

## Environment Setup & Run Commands

### 1) Install Python and create a virtual environment (`.venv`)

Install Python from the official website: https://www.python.org/downloads/

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install backend requirements

```bash
pip install -r requirements.txt
```

### 3) Run the backend API

```bash
python api.py
```

### 4) Run the ensemble script

```bash
python model_development/ensemble_v2.py
```

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


## Frontend Documentation & Application

The frontend user interface is built as a separate application inside the `rice-navigator/` directory.

- **Frontend Application URL**: [Rice Navigator Frontend](https://rice-navigator.lovable.app)
- **Frontend Documentation**: See [rice-navigator/README.md](rice-navigator/README.md) for UI data dictionary, integration notes, and run commands.
