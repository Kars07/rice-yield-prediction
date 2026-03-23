import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from scipy.interpolate import interp1d
import os

os.makedirs('model_development', exist_ok=True)

# --- 1. LOAD REAL V2 SATELLITE DATA ---
print("Loading V2 National Database...")
df = pd.read_csv('national_processed_v2.csv')
df['date'] = pd.to_datetime(df['date'])

SEQUENCE_LENGTH = 10
FARMS_PER_REGION = 30

X_series_list = []
X_tabular = []
y_list = []

# Published baseline yields (t/ha) from previous research (e.g., Ibrahim et al. 2025)
base_yields = {
    'Kebbi': 3.5,  # Irrigated
    'Niger': 3.2,
    'Ebonyi': 3.0,
    'Taraba': 2.8,
    'Kano': 2.5,   # Rainfed
    'Jigawa': 2.4
}

np.random.seed(42)

print("Applying Data Augmentation and Multi-Sensor Feature Engineering...")

for (state, year), group in df.groupby(['State', 'Year']):
    group = group.sort_values('date')
    # Using the new V2 features
    raw_features = group[['NDVI', 'EVI', 'NDWI', 'VV', 'VH', 'precipitation', 'temperature_2m']].values

    if len(raw_features) < 3: continue

    # Interpolate time-series to exactly 10 steps
    x_old = np.linspace(0, 1, len(raw_features))
    x_new = np.linspace(0, 1, SEQUENCE_LENGTH)
    f = interp1d(x_old, raw_features, axis=0, kind='linear')
    base_sequence = f(x_new)

    for _ in range(FARMS_PER_REGION):
        # Inject agricultural noise
        noise = np.random.normal(0, 0.02, (SEQUENCE_LENGTH, 7))
        farm_seq = base_sequence + noise

        X_series_list.append(farm_seq)

        # Calculate Tabular Features from the V2 dataset
        mean_ndvi = np.mean(farm_seq[:, 0])
        max_evi = np.max(farm_seq[:, 1])  # EVI is better for peak biomass
        mean_ndwi = np.mean(farm_seq[:, 2]) # Flood proxy
        mean_vv = np.mean(farm_seq[:, 3])
        total_rain = np.sum(farm_seq[:, 5]) # Total seasonal precipitation
        mean_temp = np.mean(farm_seq[:, 6])

        X_tabular.append([mean_ndvi, max_evi, mean_ndwi, mean_vv, total_rain, mean_temp, int(year)])

        # Ground Truth Calculation based on historical state baselines + satellite modifiers
        state_base = base_yields.get(state, 2.5)
        # Yield is boosted by EVI and rain, reduced by extreme heat
        simulated_yield = state_base + (max_evi * 1.5) + (total_rain * 0.001) - ((mean_temp - 25) * 0.05) + np.random.normal(0, 0.2)
        simulated_yield = max(1.0, min(simulated_yield, 7.5))
        y_list.append(simulated_yield)

X_series = np.array(X_series_list)
y = np.array(y_list).reshape(-1, 1)
X_tab_df = pd.DataFrame(X_tabular, columns=['NDVI_Mean', 'EVI_Max', 'NDWI_Mean', 'VV_Mean', 'Total_Rain', 'Mean_Temp', 'Year'])

# --- 2. TIME-SERIES AWARE SPLIT ---
sort_indices = X_tab_df.sort_values('Year').index
X_series = X_series[sort_indices]
y = y[sort_indices]
X_tab_df = X_tab_df.iloc[sort_indices].drop(columns=['Year'])

n_samples = len(y)
train_idx = int(n_samples * 0.70)
val_idx = int(n_samples * 0.85)

X_train_tab, y_train = X_tab_df.iloc[:train_idx], y[:train_idx]
X_val_tab, y_val = X_tab_df.iloc[train_idx:val_idx], y[train_idx:val_idx]
X_test_tab, y_test = X_tab_df.iloc[val_idx:], y[val_idx:]

print(f"\nData Split: {len(X_train_tab)} Train | {len(X_val_tab)} Val | {len(X_test_tab)} Test")

# --- 3. TRAIN BOOSTING MODELS ---
print("\n--- Training XGBoost & LightGBM ---")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
xgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)

lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
lgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)])

xgb_val_preds = xgb_model.predict(X_val_tab)
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))
xgb_model.save_model("model_development/xgboost_model_v2.json")

# --- 4. TRAIN DEEP LEARNING (LSTM) ---
print("\n--- Training PyTorch LSTM ---")
scaler = StandardScaler()
X_flat = X_series.reshape(n_samples, -1)
X_scaled = scaler.fit_transform(X_flat).reshape(n_samples, SEQUENCE_LENGTH, 7) # Now 7 features!
joblib.dump(scaler, "model_development/scaler_v2.bin")

X_train_ts = torch.tensor(X_scaled[:train_idx], dtype=torch.float32)
y_train_ts = torch.tensor(y[:train_idx], dtype=torch.float32)
X_val_ts = torch.tensor(X_scaled[train_idx:val_idx], dtype=torch.float32)
y_val_ts = torch.tensor(y[train_idx:val_idx], dtype=torch.float32)
X_test_ts = torch.tensor(X_scaled[val_idx:], dtype=torch.float32)

class RiceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Upgraded to 2 layers and 64 hidden nodes to handle the 7 V2 features
        self.lstm = nn.LSTM(7, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Pass the last time-step through a deeper fully connected network
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        return self.fc2(out)

dl_model = RiceLSTM()
# Lowered learning rate to 0.001 for careful, stable convergence
optimizer = torch.optim.AdamW(dl_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

best_loss = float('inf')
patience = 20
trigger_times = 0

for epoch in range(150):
    dl_model.train()
    optimizer.zero_grad()
    loss = criterion(dl_model(X_train_ts), y_train_ts)
    loss.backward()
    optimizer.step()

    dl_model.eval()
    with torch.no_grad():
        val_loss = criterion(dl_model(X_val_ts), y_val_ts)

    scheduler.step(val_loss)

    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        trigger_times = 0
        torch.save(dl_model.state_dict(), "model_development/dl_model_v2.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            break

dl_model.load_state_dict(torch.load("model_development/dl_model_v2.pth", weights_only=True))
dl_model.eval()
with torch.no_grad():
    dl_val_preds = dl_model(X_val_ts).numpy().flatten()
dl_val_rmse = np.sqrt(mean_squared_error(y_val, dl_val_preds))

# --- 5. ENSEMBLE ---
weight_xgb = (1.0 / xgb_val_rmse) / ((1.0 / xgb_val_rmse) + (1.0 / dl_val_rmse))
weight_dl = (1.0 / dl_val_rmse) / ((1.0 / xgb_val_rmse) + (1.0 / dl_val_rmse))

with open("model_development/ensemble_weights_v2.json", "w") as f:
    json.dump({"xgb_weight": float(weight_xgb), "dl_weight": float(weight_dl)}, f)

# --- 6. METRICS ---
xgb_test_preds = xgb_model.predict(X_test_tab)
lgb_test_preds = lgb_model.predict(X_test_tab)
with torch.no_grad():
    dl_test_preds = dl_model(X_test_ts).numpy().flatten()

ensemble_test_preds = (xgb_test_preds * weight_xgb) + (dl_test_preds * weight_dl)

def calculate_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [name, rmse, mae, r2]

results = [
    calculate_metrics(y_test, xgb_test_preds, "XGBoost"),
    calculate_metrics(y_test, lgb_test_preds, "LightGBM"),
    calculate_metrics(y_test, dl_test_preds, "LSTM"),
    calculate_metrics(y_test, ensemble_test_preds, "Weighted Ensemble")
]

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R²"])
print("\n------------------------------------------------")
print("FINAL DEPLOYMENT RESULTS (V2 Multi-Sensor Pipeline)")
print(results_df.to_string(index=False))
print("------------------------------------------------")
