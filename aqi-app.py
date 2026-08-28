import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Lahore AQI Prediction — NeuroFive ML Track Week 6 Capstone
# ---------------------------------------------------------

st.set_page_config(
    page_title="Lahore AQI Predictor",
    page_icon="🌫️",
    layout="centered"
)

# Find the saved model whether this file is run from the repo root
# or from an app/ directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "best_model.pkl",
    BASE_DIR / "best_model.pkl",
    BASE_DIR.parent / "models" / "best_model.pkl",
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)

st.title("🌫️ Lahore Next-Day AQI Predictor")
st.write(
    "Predict tomorrow's Lahore Air Quality Index (AQI) using "
    "weather conditions, crop-burning conditions, and recent AQI history."
)

if MODEL_PATH is None:
    st.error(
        "Model file not found. Please make sure `best_model.pkl` is uploaded "
        "to the `models/` folder in your GitHub repository."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


# ---------------------------------------------------------
# AQI interpretation
# ---------------------------------------------------------

def aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Moderate 🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 🟠"
    elif aqi <= 200:
        return "Unhealthy 🔴"
    elif aqi <= 300:
        return "Very Unhealthy 🟣"
    return "Hazardous ⚫"


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

st.subheader("Enter Today's Conditions")

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=55.0,
    value=24.0,
    step=0.5
)

humidity = st.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=55
)

wind_speed = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    max_value=50.0,
    value=8.0,
    step=0.5
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.5
)

crop_burning = st.slider(
    "Crop Burning Index",
    min_value=0,
    max_value=100,
    value=20,
    help="0 = very low burning activity, 100 = very high burning activity."
)

current_month = st.selectbox(
    "Current Month",
    list(range(1, 13)),
    index=10,
    format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B")
)

is_weekend = st.checkbox("Today is a weekend", value=False)

is_winter = current_month in [11, 12, 1, 2]

st.subheader("Recent AQI History")

aqi_lag_1 = st.number_input(
    "Yesterday's AQI",
    min_value=0.0,
    max_value=500.0,
    value=150.0,
    step=1.0
)

aqi_lag_7 = st.number_input(
    "AQI 7 Days Ago",
    min_value=0.0,
    max_value=500.0,
    value=140.0,
    step=1.0
)

aqi_rolling_7 = st.number_input(
    "7-Day Average AQI",
    min_value=0.0,
    max_value=500.0,
    value=145.0,
    step=1.0
)


# ---------------------------------------------------------
# Build exactly the same feature structure used by notebook
# ---------------------------------------------------------

month_sin = np.sin(2 * np.pi * current_month / 12)
month_cos = np.cos(2 * np.pi * current_month / 12)

input_data = pd.DataFrame([{
    "temperature_c": temperature,
    "humidity_pct": humidity,
    "wind_speed_kmh": wind_speed,
    "rainfall_mm": rainfall,
    "crop_burning_index": crop_burning,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "is_weekend": int(is_weekend),
    "is_winter": int(is_winter),
    "aqi_lag_1": aqi_lag_1,
    "aqi_lag_7": aqi_lag_7,
    "aqi_rolling_7": aqi_rolling_7,
}])


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("🔮 Predict Next-Day AQI", use_container_width=True):
    try:
        prediction = float(model.predict(input_data)[0])
        prediction = float(np.clip(prediction, 0, 500))

        st.success(f"### Predicted Next-Day AQI: {prediction:.1f}")

        category = aqi_category(prediction)
        st.info(f"**AQI Category:** {category}")

        if prediction <= 100:
            st.write(
                "Air quality is expected to be generally acceptable, "
                "with greater caution for sensitive individuals as AQI rises."
            )
        elif prediction <= 200:
            st.warning(
                "Air quality may be unhealthy, particularly for sensitive "
                "groups. Consider reducing prolonged outdoor exposure."
            )
        else:
            st.error(
                "Air quality is predicted to be very unhealthy or hazardous. "
                "Consider limiting outdoor exposure and following local health guidance."
            )

        with st.expander("View model input"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------
# Project information
# ---------------------------------------------------------

st.divider()

st.caption(
    "NeuroFive ML Track — Week 6 Capstone | Lahore AQI Prediction"
)

st.caption(
    "This application uses the trained model produced by the capstone notebook. "
    "Predictions are estimates and should not replace official air-quality or health guidance."
)
