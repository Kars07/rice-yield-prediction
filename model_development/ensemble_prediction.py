import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# SETUP SYNTHETIC DATA
print("Generating standardized dataset...")
SEQUENCE_LENGTH = 10
N_SAMPLES = 200

X_series = np.random.rand(N_SAMPLES, SEQUENCE_LENGTH, 3)
y_list = []
X_tabular = []

for i in range(N_SAMPLES):
    mean_ndvi = np.mean(X_series[i, :, 0])
    # Synthetic Yield Formula (The "Ground Truth")
    y_val = 3.0 + (mean_ndvi * 5) + np.random.normal(0, 0.1)
    y_list.append(y_val)

    # Feature Engineering for XGBoost
    row = [
        mean_ndvi,
        np.max(X_series[i, :, 0]),
        X_series[i, -1, 0] - X_series[i, 0, 0], # Growth
        np.mean(X_series[i, :, 1]), # VV
        np.mean(X_series[i, :, 2])  # VH
    ]
    X_tabular.append(row)

y = np.array(y_list).reshape(-1, 1)
X_tab_df = pd.DataFrame(X_tabular, columns=['NDVI_Mean', 'NDVI_Max', 'Growth', 'VV', 'VH'])

# Split
indices = np.arange(N_SAMPLES)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# TRAIN & SAVE XGBOOST
print("\n--- Training Model A: XGBoost ---")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
xgb_model.fit(X_tab_df.iloc[train_idx], y[train_idx])

# SAVE XGBOOST
xgb_model.save_model("model_development/xgboost_model.json")
print("Saved XGBoost model to 'model_development/xgboost_model.json'")

# TRAIN & SAVE LSTM
print("\n--- Training Model B: PyTorch LSTM ---")
scaler = StandardScaler()
X_flat = X_series.reshape(N_SAMPLES, -1)
X_scaled = scaler.fit_transform(X_flat).reshape(N_SAMPLES, SEQUENCE_LENGTH, 3)

# SAVE SCALER
joblib.dump(scaler, "model_development/scaler.bin")
print("Saved Scaler to 'model_development/scaler.bin'")

# Convert to Tensors
X_train_ts = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
y_train_ts = torch.tensor(y[train_idx], dtype=torch.float32)

class RiceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm_model = RiceLSTM()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(50):
    optimizer.zero_grad()
    outputs = lstm_model(X_train_ts)
    loss = criterion(outputs, y_train_ts)
    loss.backward()
    optimizer.step()

# SAVE LSTM WEIGHTS
torch.save(lstm_model.state_dict(), "model_development/lstm_model.pth")
print("Saved LSTM weights to 'model_development/lstm_model.pth'")

print("\n--- Training Complete & Models Saved! ---")
