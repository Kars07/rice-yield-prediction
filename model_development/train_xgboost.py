import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# DATA GENERATION
# We recreate the same synthetic dataset structure to ensure fair comparison
print("Generating Synthetic Data...")
SEQUENCE_LENGTH = 10
features_original = np.random.rand(100, SEQUENCE_LENGTH, 3) # (Farms, Weeks, Channels)

X_tabular = []
y_list = []

for i in range(100):
    # Get the time-series for one "farm"
    # Channels: 0=NDVI, 1=VV, 2=VH
    farm_ts = features_original[i]

    # --- FEATURE ENGINEERING LAYER  ---
    # Instead of feeding raw time steps, we calculate "Phenological Metrics"

    # 1. Vegetation Health Stats (NDVI)
    ndvi_mean = np.mean(farm_ts[:, 0])
    ndvi_max  = np.max(farm_ts[:, 0])

    # 2. Growth Rate (Slope of NDVI) - Simple approximation
    # (Last NDVI - First NDVI)
    ndvi_growth = farm_ts[-1, 0] - farm_ts[0, 0]

    # 3. Radar Backscatter Stats (VV/VH)
    vv_mean = np.mean(farm_ts[:, 1])
    vh_mean = np.mean(farm_ts[:, 2])

    # Create the feature row
    row = [ndvi_mean, ndvi_max, ndvi_growth, vv_mean, vh_mean]
    X_tabular.append(row)

    # Synthetic Target (Same logic as LSTM)
    simulated_yield = 3.0 + (ndvi_mean * 5) + np.random.normal(0, 0.2)
    y_list.append(simulated_yield)

# Convert to DataFrame for XGBoost
feature_names = ['NDVI_Mean', 'NDVI_Max', 'NDVI_Growth', 'VV_Mean', 'VH_Mean']
X = pd.DataFrame(X_tabular, columns=feature_names)
y = np.array(y_list)

# Split Data (70% Train, 15% Val, 15% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# XGBOOST MODEL TRAINING  ---
print("\nTraining XGBoost Model...")

# Initialize the model with parameters from methodology
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,    # Number of trees
    learning_rate=0.1,   # Step size
    max_depth=3,         # Keep trees shallow to prevent overfitting
    random_state=42
)

# Train
model.fit(X_train, y_train)

# EVALUATION
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\nXGBoost Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Compare with some actuals
print("\nExample Predictions vs Actuals:")
for i in range(3):
    print(f"Predicted: {preds[i]:.2f} tons/ha | Actual: {y_test[i]:.2f} tons/ha")

# FEATURE IMPORTANCE PLOT
# This shows WHICH factors drove the yield prediction
plt.figure(figsize=(10, 5))
xgb.plot_importance(model, importance_type='weight')
plt.title('Feature Importance: What drives Yield?')
plt.tight_layout()
plt.savefig('model_development/xgboost_importance.png')
print("\nSaved feature importance plot to 'model_development/xgboost_importance.png'")
