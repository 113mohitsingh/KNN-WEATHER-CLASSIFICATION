import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
st.set_page_config(page_title="KNN Weather Classification", layout="wide")
st.title("🌤 K-Nearest-Neighbor Weather Classification")
st.write(
    "This app uses a K-Nearest Neighbors (KNN) model to classify weather "
    "conditions based on temperature and humidity levels."
)
X = np.array([
    [30, 65],  # Sunny
    [32, 70],  # Sunny
    [28, 60],  # Sunny
    [25, 80],  # Rainy
    [22, 85],  # Rainy
    [27, 75]   # Rainy
])
y = np.array([0, 0, 0, 1, 1, 1])
st.sidebar.header("📥 Input Parameters")
temperature = st.sidebar.slider("Temperature (°C)", 20, 35, 26)
humidity = st.sidebar.slider("Humidity (%)", 50, 90, 78)
k_value = st.sidebar.slider("K value", 1, 5, 3)
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X, y)
new_point = np.array([[temperature, humidity]])
prediction = model.predict(new_point)[0]
probability = model.predict_proba(new_point)[0]
weather_label = "Sunny" if prediction == 0 else "Rainy"
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 Classification Visualization")

    fig, ax = plt.subplots()
    for i, label in enumerate(y):
        if label == 0:
            ax.scatter(X[i][0], X[i][1], color="orange", label="Sunny" if i == 0 else "")
        else:
            ax.scatter(X[i][0], X[i][1], color="blue", label="Rainy" if i == 3 else "")
    ax.scatter(temperature, humidity, color="black", marker="*", s=200, label="New Day")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title("Weather Classification Model")
    ax.legend()
    st.pyplot(fig)
with col2:
    st.subheader("ℹ Model Information")
    st.write("### Training Data Summary")
    st.write(f"Total samples: {len(X)}")
    st.write(f"Sunny days: {sum(y == 0)}")
    st.write(f"Rainy days: {sum(y == 1)}")
    st.write(f"K neighbors: {k_value}")
    st.write("### Current Input")
    st.write(f"Temperature: {temperature}°C")
    st.write(f"Humidity: {humidity}%")
    st.write("### Prediction Result")
    st.success(f"Prediction: {weather_label}")
    st.write("### Prediction Details")
    st.write(f"Sunny Probability: {probability[0]*100:.1f}%")
    st.write(f"Rainy Probability: {probability[1]*100:.1f}%")

