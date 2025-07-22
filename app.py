import streamlit as st
import pandas as pd
import numpy as np
from feature_engineering import engineer_features
from model import train_models
from predictor import predict_next_draw

# Title
st.markdown("🎱 **Powerball Lottery Predictor**")
st.markdown("Use historical data and machine learning to predict the next Powerball combination.")

# Load sample Powerball draws
draw_data = [
    {"date": "2025-07-21", "white": [8, 11, 28, 33, 42], "powerball": 2},
    {"date": "2025-07-17", "white": [10, 15, 31, 45, 46], "powerball": 6},
    {"date": "2025-07-14", "white": [19, 23, 35, 51, 54], "powerball": 12},
    {"date": "2025-07-10", "white": [4, 9, 27, 52, 59], "powerball": 16},
    {"date": "2025-07-07", "white": [33, 18, 41, 56, 63], "powerball": 24},
]

df = pd.DataFrame(draw_data)

# Display last draws
st.markdown("### 🔢 Last 5 Powerball Draws")
for row in draw_data:
    white = " ".join(map(str, row["white"]))
    power = f'🎯 Powerball: {row["powerball"]}'
    st.write(f'**{row["date"]}**: {white} {power}')

# Button to predict next draw
if st.button("🔮 Predict Next Draw"):
    try:
        # Feature Engineering
        X, y_white, y_power = engineer_features(df)

        # ✅ THIS IS THE MISSING LINE — Add It!
        model_white, model_power = train_models(X, y_white, y_power)

        # Reshape last draw for prediction
        last_draw = np.array(X.iloc[-1]).reshape(1, -1)

        # Predict next draw
        predicted_white, predicted_power = predict_next_draw(model_white, model_power, last_draw)

        # Success message
        st.success("🎯 Prediction Complete!")

        # Display predicted numbers
        st.markdown("### 🎟️ Predicted Numbers:")
        cols = st.columns(6)
        for i in range(5):
            with cols[i]:
                st.markdown(f"<div style='font-size:24px; background:#eee; border-radius:50px; width:60px; height:60px; display:flex; align-items:center; justify-content:center;'>{predicted_white[i]}</div>", unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f"<div style='font-size:24px; background:red; color:white; border-radius:50px; width:60px; height:60px; display:flex; align-items:center; justify-content:center;'>{predicted_power}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
