import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Rice Yield Prediction System", layout="wide")

# --- MODEL DEFINITION (MUST MATCH TRAINING SCRIPT) ---
class RiceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Load XGBoost
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("model_development/xgboost_model.json")

    # Load LSTM
    lstm_model = RiceLSTM()
    lstm_model.load_state_dict(torch.load("model_development/lstm_model.pth"))
    lstm_model.eval()

    # Load Scaler
    scaler = joblib.load("model_development/scaler.bin")

    return xgb_model, lstm_model, scaler

try:
    xgb_model, lstm_model, scaler = load_models()
    st.sidebar.success("AI Models Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    st.stop()

# HEADER
st.title("Integrated Rice Yield Prediction System")
st.markdown("**Location:** Kebbi State, Nigeria | **Season:** 2024")

# SIDEBAR: CONTROLS
st.sidebar.header("User Controls")
selected_state = st.sidebar.selectbox("Select State", ["Kebbi", "Niger", "Kano", "Jigawa"])
show_raw_data = st.sidebar.checkbox("Show Raw Satellite Data", value=False)

# LOAD DATA
try:
    df = pd.read_csv('kebbi_processed_final.csv')
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    st.error("Data file not found.")
    st.stop()

# PREDICTION PIPELINE
def get_prediction(df):
    # Prepare Features for XGBoost (Tabular)
    # [Mean NDVI, Max NDVI, Growth, Mean VV, Mean VH]
    xgb_features = np.array([[
        df['NDVI'].mean(),
        df['NDVI'].max(),
        df['NDVI'].iloc[-1] - df['NDVI'].iloc[0],
        df['VV'].mean(),
        df['VH'].mean()
    ]])
    xgb_df = pd.DataFrame(xgb_features, columns=['NDVI_Mean', 'NDVI_Max', 'Growth', 'VV', 'VH'])
    pred_xgb = xgb_model.predict(xgb_df)[0]

    # Prepare Features for LSTM (Time-Series)
    # Resize/Resample to 10 steps (Sequence Length used in training)
    # We take the raw values and pad/truncate to 10
    raw_seq = df[['NDVI', 'VV', 'VH']].values
    target_len = 10

    if len(raw_seq) >= target_len:
        seq = raw_seq[:target_len]
    else:
        # Pad with zeros if too short
        padding = np.zeros((target_len - len(raw_seq), 3))
        seq = np.vstack((raw_seq, padding))

    # Scale Data
    seq_flat = seq.reshape(1, -1)
    seq_scaled = scaler.transform(seq_flat).reshape(1, target_len, 3)

    # Infer LSTM
    with torch.no_grad():
        tensor_in = torch.tensor(seq_scaled, dtype=torch.float32)
        pred_lstm = lstm_model(tensor_in).item()

    # Ensemble Average
    final_pred = (pred_xgb + pred_lstm) / 2
    return final_pred, pred_xgb, pred_lstm

# Run Prediction
predicted_yield, pred_xgb, pred_lstm = get_prediction(df)
mean_ndvi = df['NDVI'].mean()

# Confidence Interval (Based on Model Divergence)
# If models disagree, confidence is lower (wider range)
divergence = abs(pred_xgb - pred_lstm)
lower_bound = predicted_yield - (0.2 + divergence)
upper_bound = predicted_yield + (0.2 + divergence)

# --- DASHBOARD METRICS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Predicted Yield", value=f"{predicted_yield:.2f} t/ha", delta="Ensemble Model")
with col2:
    st.metric(label="Confidence Range", value=f"{lower_bound:.2f} - {upper_bound:.2f} t/ha")
with col3:
    st.metric(label="Average NDVI", value=f"{mean_ndvi:.3f}", delta="Vegetation Health")
with col4:
    st.metric(label="Model Agreement", value=f"±{divergence:.2f}", delta_color="inverse")

st.caption(f"Individual Model Outputs: XGBoost ({pred_xgb:.2f} t/ha) | LSTM ({pred_lstm:.2f} t/ha)")
st.divider()

# --- VISUALIZATION ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Crop Health Trends (NDVI Time-Series)")
    fig = px.line(df, x='date', y=['NDVI', 'VV', 'VH'],
                  title='Satellite Indicators Over Time', markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Farm Location")
    kebbi_coords = [11.4836, 4.1953]
    m = folium.Map(location=kebbi_coords, zoom_start=12)
    folium.Marker(kebbi_coords, popup=f"Yield: {predicted_yield:.2f} t/ha").add_to(m)
    st_folium(m, width=350, height=350)

if show_raw_data:
    st.subheader("Raw Sensor Data")
    st.dataframe(df)
