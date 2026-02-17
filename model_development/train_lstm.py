import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
input_file = 'kebbi_processed_final.csv'
SEQUENCE_LENGTH = 10  # We look at 10 time steps (weeks) to predict yield
BATCH_SIZE = 16
HIDDEN_SIZE = 32      # LSTM Hidden layers
LEARNING_RATE = 0.01
EPOCHS = 50

# DATA PREPARATION CLASS ---
class RiceYieldDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DATA LOADING & AUGMENTATION (SIMULATION) ---
print("Loading data...")
df = pd.read_csv(input_file)
features = df[['NDVI', 'VV', 'VH']].values

# SIMULATION STEP: Generate 100 "fake" farms based on the Kebbi data
# In real life, you would load 100 different CSVs here.
X_list = []
y_list = []

for i in range(100):
    # Add random noise to create variation
    noise = np.random.normal(0, 0.05, features.shape)
    fake_farm_features = features + noise

    # Ensure we fit the sequence length (pad or truncate)
    if len(fake_farm_features) >= SEQUENCE_LENGTH:
        seq = fake_farm_features[:SEQUENCE_LENGTH]
        X_list.append(seq)

        # Synthetic Target: Yield correlates with mean NDVI
        # Formula: Base Yield (3.0) + (Mean NDVI * 5) + Random Noise
        avg_ndvi = np.mean(seq[:, 0])
        simulated_yield = 3.0 + (avg_ndvi * 5) + np.random.normal(0, 0.2)
        y_list.append(simulated_yield)

X_data = np.array(X_list)
y_data = np.array(y_list).reshape(-1, 1) # Reshape for regression

print(f"Dataset Shape: {X_data.shape}") # (Samples, TimeSteps, Features)

# Standardize the data (Important for LSTMs)
scaler = StandardScaler()
# Flatten, scale, then reshape back
N, L, F = X_data.shape
X_data = scaler.fit_transform(X_data.reshape(-1, F)).reshape(N, L, F)

# Split Train/Test (85/15 split)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.15, shuffle=True)

train_dataset = RiceYieldDataset(X_train, y_train)
test_dataset = RiceYieldDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# THE LSTM MODEL ARCHITECTURE  ---
class RiceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(RiceLSTM, self).__init__()
        # LSTM Layer: Captures temporal dependencies [cite: 9]
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Dropout: Regularization to prevent overfitting [cite: 15]
        self.dropout = nn.Dropout(0.2)

        # Fully Connected Layer: Maps LSTM features to Yield
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # LSTM output shape: (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(x)

        # We only care about the output of the LAST time step
        last_step_out = lstm_out[:, -1, :]

        out = self.dropout(last_step_out)
        prediction = self.fc(out)
        return prediction

# TRAINING LOOP -
model = RiceLSTM(input_size=3, hidden_size=HIDDEN_SIZE)
criterion = nn.MSELoss() # Standard MSE for Regression
# AdamW Optimizer is a variant of Adam with better weight decay handling
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Training ---")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# EVALUATION
model.eval()
with torch.no_grad():
    test_preds = model(torch.tensor(X_test, dtype=torch.float32))
    test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))
    print(f"\nFinal Test MSE Loss: {test_loss.item():.4f}")

    # Show a few examples
    print("\nExample Predictions vs Actuals:")
    for i in range(3):
        print(f"Predicted: {test_preds[i].item():.2f} tons/ha | Actual: {y_test[i][0]:.2f} tons/ha")
