import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ✅ ADD HERE (MODEL TRAINING BLOCK)
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("rocket_trajectory_data.csv")

X = df[['Time', 'Altitude', 'Velocity', 'Mass']]
y = df['Thrust']

model = RandomForestRegressor(n_estimators=200)
model.fit(X, y)

# ------------------------------------

st.title("🚀 Rocket Landing Thrust Prediction App")

st.write("Enter flight parameters")

# Inputs
time = st.slider("Time (s)", 0, 500, 200)
altitude = st.slider("Altitude (m)", 0, 80000, 30000)
velocity = st.slider("Velocity (m/s)", -1000, 100, -200)
mass = st.slider("Mass (kg)", 10000, 30000, 22000)

# Prediction
if st.button("Predict Thrust"):
    input_data = pd.DataFrame([[time, altitude, velocity, mass]],
                              columns=['Time','Altitude','Velocity','Mass'])
    
    prediction = model.predict(input_data)[0]
    st.success(f"🔥 Predicted Thrust: {prediction:.2f} N")

# Graphs
st.subheader("📊 Graphs")

if st.checkbox("Velocity vs Time"):
    st.line_chart(df.set_index('Time')['Velocity'])

if st.checkbox("Altitude vs Time"):
    st.line_chart(df.set_index('Time')['Altitude'])

if st.checkbox("Thrust vs Time"):
    st.line_chart(df.set_index('Time')['Thrust'])

if st.checkbox("Show Thrust vs Time"):
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Thrust'])
    ax.set_title("Thrust vs Time")
    st.pyplot(fig)
