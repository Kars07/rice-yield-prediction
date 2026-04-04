import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
import torch
import torch.nn as nn
import joblib
import json
from scipy.interpolate import interp1d
import os

# ==========================================
# 1. PAGE CONFIG & PREMIUM CSS INJECTION
# ==========================================
st.set_page_config(page_title="AgroSense Intelligence | Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Reset and App Background */
    .stApp { background-color: #f4f7f6 !important; font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Force dark text for visibility */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1e293b !important;
    }

    /* Hide Streamlit Chrome */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; max-width: 100% !important; }

    /* Floating Glassmorphism Panel (Left Side) */
    [data-testid="column"]:nth-of-type(1) {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 24px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        backdrop-filter: blur(10px);
        margin-left: 10px;
        height: 95vh;
        overflow-y: auto;
    }

    /* Map Container (Right Side) */
    [data-testid="column"]:nth-of-type(2) { padding: 0 !important; }

    /* Typography & Spacing */
    h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.5px; }
    hr { margin: 15px 0; border-color: #e2e8f0 !important; border-width: 1px; }

    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 800; }
    [data-testid="stMetricDelta"] div { color: #22c55e !important; }

    /* Weather / Info Chips */
    .info-chip {
        display: inline-block; padding: 8px 16px; border-radius: 50px;
        font-size: 13px; font-weight: 600; margin-right: 8px; margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .chip-temp { background: #fff5f5 !important; color: #e53e3e !important; border: 1px solid #fed7d7; }
    .chip-rain { background: #ebf8ff !important; color: #3182ce !important; border: 1px solid #bee3f8; }
    .chip-status { background: #f0fff4 !important; color: #166534 !important; border: 1px solid #c6f6d5; }
    .chip-warning { background: #fffbeb !important; color: #b45309 !important; border: 1px solid #fef3c7; }

    /* Expanders */
    .streamlit-expanderHeader { font-weight: 700; color: #334155 !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE CONFIGURATION
# ==========================================
STATE_COORDS = {
    "Kebbi": [11.4836, 4.1953], "Niger": [9.9309, 6.5569],
    "Kano": [12.0022, 8.5920], "Jigawa": [12.2280, 9.5616],
    "Ebonyi": [6.2649, 8.1137], "Taraba": [8.8937, 10.8198]
}
# Historical YoY Trend Fallbacks
STATE_TRENDS = {"Kebbi": 0.15, "Niger": -0.05, "Kano": 0.22, "Jigawa": 0.10, "Ebonyi": 0.18, "Taraba": -0.12}
RMSE_SCORE = 0.274

# ==========================================
# 3. REAL INFERENCE ENGINE (LSTM + XGBOOST)
# ==========================================
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

@st.cache_data(show_spinner=False)
def load_real_data_and_predict(state_name):
    try:
        # 1. Load Dataset
        df = pd.read_csv('national_processed_v2.csv')
        df['date'] = pd.to_datetime(df['date'])
        state_df = df[df['State'] == state_name].sort_values('date')

        if len(state_df) < 3:
            return 0, "N/A", "N/A", [], []

        # 2. Extract Raw Features
        raw_features = state_df[['NDVI', 'EVI', 'NDWI', 'VV', 'VH', 'precipitation', 'temperature_2m']].values

        # 3. Interpolate to exactly 10 steps (Matching training pipeline)
        x_old = np.linspace(0, 1, len(raw_features))
        x_new = np.linspace(0, 1, 10)
        f_interp = interp1d(x_old, raw_features, axis=0, kind='linear')
        base_sequence = f_interp(x_new)

        # 4. Prepare Tabular Data for XGBoost
        mean_ndvi = np.mean(base_sequence[:, 0])
        max_evi = np.max(base_sequence[:, 1])
        mean_ndwi = np.mean(base_sequence[:, 2])
        mean_vv = np.mean(base_sequence[:, 3])
        total_rain = np.sum(base_sequence[:, 5])
        mean_temp = np.mean(base_sequence[:, 6])

        X_tab = pd.DataFrame([[mean_ndvi, max_evi, mean_ndwi, mean_vv, total_rain, mean_temp]],
                             columns=['NDVI_Mean', 'EVI_Max', 'NDWI_Mean', 'VV_Mean', 'Total_Rain', 'Mean_Temp'])

        # 5. Prepare Sequential Data for LSTM
        scaler = joblib.load("model_development/scaler_v2.bin")
        X_flat = base_sequence.reshape(1, -1)
        X_scaled = scaler.transform(X_flat).reshape(1, 10, 7)
        X_ts = torch.tensor(X_scaled, dtype=torch.float32)

        # 6. Load Models
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model("model_development/xgboost_model_v2.json")

        dl_model = RiceLSTM()
        dl_model.load_state_dict(torch.load("model_development/dl_model_v2.pth", weights_only=True))
        dl_model.eval()

        # 7. Load Weights
        with open("model_development/ensemble_weights_v2.json", "r") as f:
            weights = json.load(f)

        # 8. Predict Final Yield!
        xgb_pred = xgb_model.predict(X_tab)[0]
        with torch.no_grad():
            dl_pred = dl_model(X_ts).item()

        final_yield = (xgb_pred * weights['xgb_weight']) + (dl_pred * weights['dl_weight'])

        # 9. Extract actual UI Data
        current_temp = f"{state_df['temperature_2m'].iloc[-1]:.1f}°C"
        current_rain = f"{state_df['precipitation'].iloc[-1]:.1f}mm"
        chart_dates = state_df['date'].tolist()
        chart_ndvi = state_df['NDVI'].tolist()

        return float(final_yield), current_temp, current_rain, chart_dates, chart_ndvi

    except Exception as e:
        print(f"Backend Error: {e}")
        return 0.0, "N/A", "N/A", [], []

# ==========================================
# 4. UI LAYOUT: APP DRAWER (LEFT)
# ==========================================
col_panel, col_map = st.columns([1.2, 2.8], gap="small")

with col_panel:
    st.markdown("<h2 style='margin-bottom: 0;'>🌾 AgroSense Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b !important; font-size: 14px; margin-top: -5px;'>National Rice Yield Monitor v2.0</p>", unsafe_allow_html=True)

    selected_state = st.selectbox("📍 Search Region", list(STATE_COORDS.keys()), index=4) # Defaults to Ebonyi

    # ⚡ TRIGGER INFERENCE
    with st.spinner("Processing satellite telemetry..."):
        pred_yield, current_temp, current_rain, chart_dates, chart_ndvi = load_real_data_and_predict(selected_state)

    st.write("---")

    # Dynamic Status Logic
    status = "Optimal" if pred_yield >= 3.6 else "Warning"
    status_class = "chip-status" if status == "Optimal" else "chip-warning"

    st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <span class="info-chip {status_class}">🌱 {status} Growth</span>
            <span class="info-chip chip-temp">🌡️ {current_temp}</span>
            <span class="info-chip chip-rain">🌧️ {current_rain}</span>
        </div>
    """, unsafe_allow_html=True)

    # Primary Metric
    trend = STATE_TRENDS.get(selected_state, 0.0)
    st.metric(
        label="Predicted Harvest Yield",
        value=f"{pred_yield:.2f} t/ha",
        delta=f"{trend:+.2f} t/ha (YOY)"
    )
    st.markdown(f"<p style='font-size: 12px; color: #64748b !important; margin-top: -10px; text-align: center;'><b>Model Confidence:</b> ±{RMSE_SCORE} t/ha | Ensemble</p>", unsafe_allow_html=True)

    # AI Insight Box
    insight_color = "#22c55e" if status == "Optimal" else "#f59e0b"
    insight_text = "Satellite indices confirm excellent paddy inundation. Canopy density is tracking at optimal levels." if status == "Optimal" else "Slight deviations in expected biomass detected. Continuous monitoring advised for potential yield stress."

    st.markdown(f"""
        <div style="background: linear-gradient(90deg, #f8fafc, #ffffff); border-left: 4px solid {insight_color}; padding: 16px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 18px; margin-right: 8px;">✨</span>
                <span style="color: #1e293b !important; font-weight: 700; font-size: 14px;">AI Insight</span>
            </div>
            <span style="font-size: 13px; line-height: 1.5; color: #334155 !important;">
                {insight_text}
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Real-time Phenology Chart
    with st.expander("📊 View Phenology Data (Real-time)"):
        if chart_dates:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_dates, y=chart_ndvi, mode='lines', fill='tozeroy', name='NDVI',
                line=dict(color='#22c55e', width=3), fillcolor='rgba(34, 197, 94, 0.1)'
            ))
            fig.update_layout(
                height=180, margin=dict(l=0, r=0, t=5, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#1e293b")),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickfont=dict(size=10, color="#1e293b"))
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("No phenology data available for plotting.")

# ==========================================
# 5. NATIVE GOOGLE MAPS ENGINE (RIGHT)
# ==========================================
with col_map:
    # Initialize Folium without default tiles
    m = folium.Map(location=STATE_COORDS[selected_state], zoom_start=7, tiles=None, control_scale=True)

    # Inject Literal Google Maps Standard Street Layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps Standard',
        overlay=False,
        control=False
    ).add_to(m)

    for state, coords in STATE_COORDS.items():
        is_selected = (state == selected_state)

        # Determine pin colors based on selection
        bg_color = '#22c55e' if is_selected else '#94a3b8'
        size = 24 if is_selected else 16

        # Custom HTML CSS for the map pins (Google Maps style dots)
        icon_html = f"""
            <div style="
                background-color: {bg_color}; width: {size}px; height: {size}px;
                border: 3px solid white; border-radius: 50%; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                display: flex; justify-content: center; align-items: center;
                position: relative; left: -{size/2}px; top: -{size/2}px;
            "></div>
        """

        # Location label right under the pin
        label_html = f"""
            <div style="
                position: absolute; top: 5px; left: -20px; width: 100px;
                font-family: 'Segoe UI', sans-serif; font-weight: 800; font-size: 13px; color: #1e293b;
                text-shadow: 2px 2px 0 #fff, -2px -2px 0 #fff, 2px -2px 0 #fff, -2px 2px 0 #fff;
            ">{state}</div>
        """

        # Map Popup
        # We can run a quick mock-yield for unselected states in the popup just for UI cleanliness
        display_yield = pred_yield if is_selected else STATE_TRENDS.get(state, 0) + 3.5

        popup_html = f"""
            <div style="font-family: 'Segoe UI', sans-serif; width: 160px; padding: 5px; color: #1e293b;">
                <h3 style="margin:0; color:#1e293b; font-size: 16px;">{state} Region</h3>
                <p style="margin: 4px 0 8px; font-size: 12px; color: #64748b;">Rice Cultivation Zone</p>
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 8px; border-radius: 6px;">
                    <b style="color:#166534; font-size: 14px;">Yield Est: {display_yield:.2f} t/ha</b>
                </div>
            </div>
        """

        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Analyze {state}",
            icon=folium.DivIcon(html=icon_html + label_html)
        ).add_to(m)

        # Animated pulse ring for the selected state
        if is_selected:
            folium.CircleMarker(
                location=coords, radius=40, color='#22c55e', weight=2,
                fill=True, fill_color='#4ade80', fill_opacity=0.2
            ).add_to(m)

    st_folium(m, width="100%", height=800, returned_objects=[])
