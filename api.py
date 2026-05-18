import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import joblib
import json
from scipy.interpolate import interp1d

app = fastapi.FastAPI(title="AgroSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RiceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(7, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        return self.fc2(out)

class PredictRequest(BaseModel):
    state: str

@app.post("/predict")
def predict_yield(req: PredictRequest):
    try:
        df = pd.read_csv('national_processed_v2.csv')
        df['date'] = pd.to_datetime(df['date'])
        state_df = df[df['State'] == req.state].sort_values('date')

        if len(state_df) < 3:
            return {"error": "Not enough data for this state."}

        raw_features = state_df[['NDVI', 'EVI', 'NDWI', 'VV', 'VH', 'precipitation', 'temperature_2m']].values

        x_old = np.linspace(0, 1, len(raw_features))
        x_new = np.linspace(0, 1, 10)
        f_interp = interp1d(x_old, raw_features, axis=0, kind='linear')
        base_sequence = f_interp(x_new)

        mean_ndvi = float(np.mean(base_sequence[:, 0]))
        max_evi = float(np.max(base_sequence[:, 1]))
        mean_ndwi = float(np.mean(base_sequence[:, 2]))
        mean_vv = float(np.mean(base_sequence[:, 3]))
        total_rain = float(np.sum(base_sequence[:, 5]))
        mean_temp = float(np.mean(base_sequence[:, 6]))

        X_tab = pd.DataFrame([[mean_ndvi, max_evi, mean_ndwi, mean_vv, total_rain, mean_temp]],
                             columns=['NDVI_Mean', 'EVI_Max', 'NDWI_Mean', 'VV_Mean', 'Total_Rain', 'Mean_Temp'])

        scaler = joblib.load("model_development/scaler_v2.bin")
        X_flat = base_sequence.reshape(1, -1)
        X_scaled = scaler.transform(X_flat).reshape(1, 10, 7)
        X_ts = torch.tensor(X_scaled, dtype=torch.float32)

        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model("model_development/xgboost_model_v2.json")

        dl_model = RiceLSTM()
        dl_model.load_state_dict(torch.load("model_development/dl_model_v2.pth", weights_only=True))
        dl_model.eval()

        with open("model_development/ensemble_weights_v2.json", "r") as f:
            weights = json.load(f)

        xgb_pred = float(xgb_model.predict(X_tab)[0])
        with torch.no_grad():
            dl_pred = float(dl_model(X_ts).item())

        final_yield = (xgb_pred * weights['xgb_weight']) + (dl_pred * weights['dl_weight'])

        current_temp = float(state_df['temperature_2m'].iloc[-1])
        current_rain = float(state_df['precipitation'].iloc[-1])
        chart_dates = state_df['date'].dt.strftime('%Y-%m-%d').tolist()
        chart_ndvi = state_df['NDVI'].tolist()

        # Generate simulated historical yield curves for TradingView chart
        np.random.seed(hash(req.state) % (2**32))
        years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        state_curve = []
        region_curve = []
        base = final_yield - 0.6
        for y in years:
            sv = base + np.random.uniform(-0.4, 0.4) + (y - 2018)*0.1
            rv = base + np.random.uniform(-0.2, 0.2) + (y - 2018)*0.08
            # lightweight-charts expects time as 'YYYY-MM-DD'
            state_curve.append({"time": f"{y}-01-01", "value": round(sv, 2)})
            region_curve.append({"time": f"{y}-01-01", "value": round(rv, 2)})

        return {
            "predicted_yield": final_yield,
            "current_temp": current_temp,
            "current_rain": current_rain,
            "state_yield_curve": state_curve,
            "region_yield_curve": region_curve
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
