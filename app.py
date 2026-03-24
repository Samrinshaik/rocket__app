import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model_new.pkl")

st.title("🚀 Rocket Landing Thrust Prediction App")

st.write("Enter flight parameters to predict thrust")

# User Inputs
time = st.slider("Time (s)", 0, 500, 200)
altitude = st.slider("Altitude (m)", 0, 80000, 30000)
velocity = st.slider("Velocity (m/s)", -1000, 100, -200)
mass = st.slider("Mass (kg)", 10000, 30000, 22000)

# Create input dataframe
input_data = pd.DataFrame([[time, altitude, velocity, mass]],
                          columns=['Time', 'Altitude', 'Velocity', 'Mass'])

# Prediction
if st.button("Predict Thrust"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔥 Predicted Thrust: {prediction:.2f} N")

# Visualization section
st.subheader("📊 Dataset Visualization")

df = pd.read_csv("rocket_trajectory_data.csv")

if st.checkbox("Show Velocity vs Time"):
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Velocity'])
    ax.set_title("Velocity vs Time")
    st.pyplot(fig)

if st.checkbox("Show Altitude vs Time"):
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Altitude'])
    ax.set_title("Altitude vs Time")
    st.pyplot(fig)

if st.checkbox("Show Thrust vs Time"):
    fig, ax = plt.subplots()
    ax.plot(df['Time'], df['Thrust'])
    ax.set_title("Thrust vs Time")
    st.pyplot(fig)