import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
st.title("KNN Weather Classification")
X=np.array([[30, 70],[25, 80],[27, 60],[31, 65],[23, 85],[28, 75]]) 
y=np.array([0, 1, 0, 0, 1, 1])
knn=KNeighborsClassifier(n_neighbors=3) 
knn.fit(X, y) 
new_point=np.array ([[26, 78]]) 
prediction=knn.predict(new_point)[0] 
plt.figure(figsize=(7, 5)) 
plt.scatter(X[y==0, 0], X[y==0, 1], label="Sunny", s=100) 
plt.scatter(X[y==1, 0], X[y==1, 1], label= "Rainy", s=100)
plt.scatter( 
    new_point[0, 0],
    new_point[0, 1], 
    marker="*",
    s=300, 
    color="red",
    label="New Prediction",
)
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("KNN Weather Classification") 
plt.legend() 
plt.grid(alpha=0.3) 
st.pyplot(plt)
if prediction == 0: 
    st.success(f"Predicted Weather: Sunny")
else: 
    st.success(f"Predicted Weather: Rainy")
