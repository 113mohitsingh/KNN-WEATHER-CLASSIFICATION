import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.title("KNN Weather Classification")

# Example input fields
temperature = st.number_input("Enter Temperature")
humidity = st.number_input("Enter Humidity")

if st.button("Predict"):
    # Dummy example model
    X = [[20, 30], [30, 60], [25, 50]]
    y = ["Cold", "Hot", "Mild"]

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)

    prediction = model.predict([[temperature, humidity]])

    st.success(f"Predicted Weather: {prediction[0]}")
