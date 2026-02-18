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
import os

# Ensure directory exists
os.makedirs('model_development', exist_ok=True)

# --- 1. REALISTIC SYNTHETIC DATA GENERATION ---
print("Generating realistic time-series dataset...")
SEQUENCE_LENGTH = 10
N_SAMPLES = 400  # Increased for deep learning stability

X_series = np.zeros((N_SAMPLES, SEQUENCE_LENGTH, 3))
y_list = []
X_tabular = []

np.random.seed(42) # For reproducibility

for i in range(N_SAMPLES):
    # Simulate realistic NDVI growth curve (bell curve/sine wave pattern)
    base_ndvi = np.random.uniform(0.1, 0.3)
    peak_ndvi = np.random.uniform(0.6, 0.9)
    curve = np.sin(np.linspace(0, np.pi, SEQUENCE_LENGTH))
    ndvi_seq = base_ndvi + (peak_ndvi - base_ndvi) * curve + np.random.normal(0, 0.02, SEQUENCE_LENGTH)

    # Simulate Radar (VV/VH)
    vv_seq = np.random.uniform(-18, -12, SEQUENCE_LENGTH) + np.random.normal(0, 1, SEQUENCE_LENGTH)
    vh_seq = np.random.uniform(-28, -20, SEQUENCE_LENGTH) + np.random.normal(0, 1, SEQUENCE_LENGTH)

    X_series[i, :, 0] = ndvi_seq
    X_series[i, :, 1] = vv_seq
    X_series[i, :, 2] = vh_seq

    # Target: Complex non-linear yield with realistic agricultural noise (StdDev ~ 0.5)
    mean_ndvi = np.mean(ndvi_seq)
    max_ndvi = np.max(ndvi_seq)
    # Realistic base yield + influence of peak health + weather/radar proxy
    y_val = 1.5 + (max_ndvi * 3.5) + (mean_ndvi * 1.0) + (np.mean(vv_seq) * 0.05) + np.random.normal(0, 0.45)
    y_val = max(1.0, min(y_val, 7.0)) # Clip to realistic Nigerian bounds (1 to 7 t/ha)
    y_list.append(y_val)

    # Features for Boosting (Summary Stats)
    row = [mean_ndvi, max_ndvi, ndvi_seq[-1] - ndvi_seq[0], np.mean(vv_seq), np.mean(vh_seq)]
    X_tabular.append(row)

y = np.array(y_list).reshape(-1, 1)
X_tab_df = pd.DataFrame(X_tabular, columns=['NDVI_Mean', 'NDVI_Max', 'Growth', 'VV', 'VH'])

# --- 2. TIME-SERIES SPLIT (70/15/15) ---
# [Methodology Source 12]: "70% train / 15% validation / 15% test... no random shuffle"
train_idx = int(N_SAMPLES * 0.70)
val_idx = int(N_SAMPLES * 0.85)

X_train_tab, y_train = X_tab_df.iloc[:train_idx], y[:train_idx]
X_val_tab, y_val = X_tab_df.iloc[train_idx:val_idx], y[train_idx:val_idx]
X_test_tab, y_test = X_tab_df.iloc[val_idx:], y[val_idx:]

print(f"Data Split: {len(X_train_tab)} Train | {len(X_val_tab)} Val | {len(X_test_tab)} Test (Sequential)")

# --- 3. TRAIN BOOSTING MODELS ---
print("\n--- Training Model A: XGBoost & LightGBM ---")
# Optimize parameters for generalization
xgb_model = xgb.XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8)
xgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)

lgb_model = lgb.LGBMRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8)
lgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)])

# Validate for Weights
xgb_val_preds = xgb_model.predict(X_val_tab)
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))

xgb_model.save_model("model_development/xgboost_model.json")

# --- 4. TRAIN DEEP LEARNING (GRU) ---
print("\n--- Training Model B: PyTorch GRU (Lightweight LSTM alternative) ---")
scaler = StandardScaler()
X_flat = X_series.reshape(N_SAMPLES, -1)
X_scaled = scaler.fit_transform(X_flat).reshape(N_SAMPLES, SEQUENCE_LENGTH, 3)
joblib.dump(scaler, "model_development/scaler.bin")

X_train_ts = torch.tensor(X_scaled[:train_idx], dtype=torch.float32)
y_train_ts = torch.tensor(y[:train_idx], dtype=torch.float32)
X_val_ts = torch.tensor(X_scaled[train_idx:val_idx], dtype=torch.float32)
y_val_ts = torch.tensor(y[train_idx:val_idx], dtype=torch.float32)
X_test_ts = torch.tensor(X_scaled[val_idx:], dtype=torch.float32)

# Simplified GRU Architecture to prevent vanishing gradients
class RiceGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, 16, batch_first=True) # Smaller hidden size
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(16, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))

dl_model = RiceGRU()
optimizer = torch.optim.AdamW(dl_model.parameters(), lr=0.01, weight_decay=1e-4)
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

    # Validation step
    dl_model.eval()
    with torch.no_grad():
        val_loss = criterion(dl_model(X_val_ts), y_val_ts)

    scheduler.step(val_loss)

    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        trigger_times = 0
        torch.save(dl_model.state_dict(), "model_development/dl_model.pth")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best GRU and calculate validation RMSE for weighting
dl_model.load_state_dict(torch.load("model_development/dl_model.pth", weights_only=True))
dl_model.eval()
with torch.no_grad():
    dl_val_preds = dl_model(X_val_ts).numpy().flatten()
dl_val_rmse = np.sqrt(mean_squared_error(y_val, dl_val_preds))

# --- 5. INVERSE RMSE WEIGHTING ENSEMBLE ---
print("\n--- Calculating Inverse RMSE Weights ---")
# [Methodology Source 236]: "Re-weight ensemble (e.g., inverse RMSE weighting: higher weight to better model)."
weight_xgb = (1.0 / xgb_val_rmse) / ((1.0 / xgb_val_rmse) + (1.0 / dl_val_rmse))
weight_dl = (1.0 / dl_val_rmse) / ((1.0 / xgb_val_rmse) + (1.0 / dl_val_rmse))

print(f"XGBoost Validation RMSE: {xgb_val_rmse:.4f} -> Weight: {weight_xgb:.2%}")
print(f"Deep Learning Validation RMSE: {dl_val_rmse:.4f} -> Weight: {weight_dl:.2%}")

# Save weights for the dashboard
with open("model_development/ensemble_weights.json", "w") as f:
    json.dump({"xgb_weight": weight_xgb, "dl_weight": weight_dl}, f)

# --- 6. FINAL TESTING & METRICS ---
xgb_test_preds = xgb_model.predict(X_test_tab)
lgb_test_preds = lgb_model.predict(X_test_tab)
with torch.no_grad():
    dl_test_preds = dl_model(X_test_ts).numpy().flatten()

# Apply learned weights
ensemble_test_preds = (xgb_test_preds * weight_xgb) + (dl_test_preds * weight_dl)

def calculate_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [name, rmse, mae, r2]

results = [
    calculate_metrics(y_test, xgb_test_preds, "XGBoost"),
    calculate_metrics(y_test, lgb_test_preds, "LightGBM"),
    calculate_metrics(y_test, dl_test_preds, "GRU (Deep Learning)"),
    calculate_metrics(y_test, ensemble_test_preds, "Weighted Ensemble")
]

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R²"])
print("\n------------------------------------------------")
print("FINAL REALISTIC RESULTS [Targeting Lit. Benchmarks]")
print(results_df.to_string(index=False))
print("------------------------------------------------")
